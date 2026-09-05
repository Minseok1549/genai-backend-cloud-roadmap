"""EPL 승부 예측 API. /predict는 호출 시점마다 최신 시즌 경기를 반영해 팀 폼을 다시 계산한다."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import db
from data import load_matches
from features import latest_team_form, FEATURE_NAMES
from fetch_data import ensure_all_seasons_cached

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = joblib.load(MODEL_PATH)
    if "model_version" not in app.state.model_bundle:
        # 예측 기록의 감사 추적이 이 값에 의존하므로, 없는 채로 조용히 "unknown"을
        # 쓰는 것보다 기동 시점에 바로 실패하는 편이 안전하다.
        raise RuntimeError(f"{MODEL_PATH}에 model_version이 없습니다 — train.py를 다시 실행하세요")

    try:
        app.state.db_pool = db.create_pool()
        db.init_schema(app.state.db_pool)
    except Exception as e:
        # DB가 기동 시점에 죽어 있어도 예측 서빙 자체는 계속돼야 한다(기록 저장만 포기) —
        # DB 장애로 앱 전체가 뜨지 못하면 /health조차 응답하지 못하게 된다.
        print(f"warning: database unavailable at startup, prediction history will not be recorded: {e}")
        app.state.db_pool = None

    yield

    if app.state.db_pool is not None:
        app.state.db_pool.closeall()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # exc.errors()의 'ctx'에는 원본 예외 객체가 그대로 들어있어 JSON 직렬화가 안 됨 — 제거하고 반환
    # 'ctx' 제거에 더해 'input'도 제거한다 — 안 그러면 과도하게 긴 입력을 그대로 에러 응답에
    # 되돌려보내게 되어 응답 크기 증폭에 악용될 수 있다.
    errors = [{k: v for k, v in err.items() if k not in ("ctx", "input")} for err in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": jsonable_encoder(errors)})


MAX_TEAM_NAME_LENGTH = 64  # 실제 EPL 팀명은 이보다 훨씬 짧다 — 과도한 입력으로 인한 낭비 호출 방지


class PredictRequest(BaseModel):
    home_team: str = Field(max_length=MAX_TEAM_NAME_LENGTH)
    away_team: str = Field(max_length=MAX_TEAM_NAME_LENGTH)

    @field_validator("home_team", "away_team")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("팀명은 빈 값일 수 없습니다")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="홈팀과 원정팀이 같을 수 없습니다")

    try:
        ensure_all_seasons_cached()
    except Exception as e:  # 외부 API가 잠깐 죽어도 서빙은 기존 캐시로 계속되게
        print(f"warning: season cache refresh failed, using existing cache: {e}")

    # 캐시 갱신 이후의 실패(손상된 캐시, 파싱 오류, 모델 추론 오류 등)는 클라이언트 잘못이
    # 아니라 서버 쪽 일시 장애이므로 500이 아니라 503으로 알린다. 의도적으로 던진
    # HTTPException(알 수 없는 팀명/기록 부족 등 400)은 그대로 통과시킨다.
    try:
        matches = load_matches()
        known_teams = set(matches["home_team"]) | set(matches["away_team"])
        for team in (req.home_team, req.away_team):
            if team not in known_teams:
                raise HTTPException(status_code=400, detail=f"알 수 없는 팀명: {team}")

        home_form = latest_team_form(matches, req.home_team)
        away_form = latest_team_form(matches, req.away_team)
        for team, form in [(req.home_team, home_form), (req.away_team, away_form)]:
            if form is None:
                raise HTTPException(status_code=400, detail=f"{team}의 최근 경기 기록이 부족합니다")

        features = pd.DataFrame([{
            "home_form_points": home_form["points"],
            "home_form_gf": home_form["goals_for"],
            "home_form_ga": home_form["goals_against"],
            "away_form_points": away_form["points"],
            "away_form_gf": away_form["goals_for"],
            "away_form_ga": away_form["goals_against"],
        }])[FEATURE_NAMES]

        bundle = request.app.state.model_bundle
        proba = bundle["model"].predict_proba(features)[0]
        probabilities = dict(zip(bundle["classes"], proba.tolist()))
        # 이 예측이 어떤 경기 데이터를 반영했는지의 스냅샷 — 실시간으로 갱신되는
        # matches를 매번 다시 읽으므로 "가장 최근 반영된 경기 날짜"로 데이터 버전을 삼는다.
        data_version = str(matches["date"].max())
    except HTTPException:
        raise
    except Exception as e:
        print(f"error: prediction pipeline failed: {e}")
        raise HTTPException(status_code=503, detail="예측 서비스를 일시적으로 사용할 수 없습니다")

    # 기록 저장은 예측 응답의 필수 조건이 아니다 — DB가 죽어 있어도(기동 시점 장애 포함,
    # 이 경우 db_pool이 None) 예측 자체는 계속 서빙하고, 저장 실패는 로그만 남긴다.
    if request.app.state.db_pool is not None:
        try:
            db.save_prediction(
                request.app.state.db_pool,
                home_team=req.home_team,
                away_team=req.away_team,
                probabilities=probabilities,
                model_version=bundle["model_version"],
                data_version=data_version,
            )
        except Exception as e:
            print(f"warning: failed to save prediction record: {e}")

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "probabilities": probabilities,
    }
