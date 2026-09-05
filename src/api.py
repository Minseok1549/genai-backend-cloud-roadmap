"""EPL 승부 예측 API. /predict는 호출 시점마다 최신 시즌 경기를 반영해 팀 폼을 다시 계산한다."""
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import db
from data import load_matches
from fetch_data import ensure_all_seasons_cached
from logutil import log_json
from predictor import predict_match, load_model_bundle, UnknownTeamError, InsufficientFormError

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 예측 기록의 감사 추적이 model_version에 의존하므로, 없는 채로 조용히 "unknown"을
    # 쓰는 것보다 기동 시점에 바로 실패하는 편이 안전하다(load_model_bundle이 검증).
    app.state.model_bundle = load_model_bundle(MODEL_PATH)

    try:
        app.state.db_pool = db.create_pool()
        db.init_schema(app.state.db_pool)
    except Exception as e:
        # DB가 기동 시점에 죽어 있어도 예측 서빙 자체는 계속돼야 한다(기록 저장만 포기) —
        # DB 장애로 앱 전체가 뜨지 못하면 /health조차 응답하지 못하게 된다.
        log_json("warning", "database unavailable at startup, prediction history will not be recorded", error=str(e))
        app.state.db_pool = None

    yield

    if app.state.db_pool is not None:
        app.state.db_pool.closeall()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    # 클라이언트나 로드밸런서가 이미 요청 ID를 붙여왔으면 그걸 그대로 잇는다(분산 추적) —
    # 없으면 새로 발급한다. 응답 헤더에도 실어서, 호출한 쪽이 이 값을 자기 로그와
    # 대조할 수 있게 한다.
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - start) * 1000, 1)
    log_json(
        "info",
        "request completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


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
        log_json("warning", "season cache refresh failed, using existing cache", request_id=request.state.request_id, error=str(e))

    # 캐시 갱신 이후의 실패(손상된 캐시, 파싱 오류, 모델 추론 오류 등)는 클라이언트 잘못이
    # 아니라 서버 쪽 일시 장애이므로 500이 아니라 503으로 알린다. 의도적으로 던진
    # HTTPException(알 수 없는 팀명/기록 부족 등 400)은 그대로 통과시킨다.
    try:
        matches = load_matches()
        bundle = request.app.state.model_bundle
        probabilities = predict_match(req.home_team, req.away_team, matches, bundle)
        # 이 예측이 어떤 경기 데이터를 반영했는지의 스냅샷 — 실시간으로 갱신되는
        # matches를 매번 다시 읽으므로 "가장 최근 반영된 경기 날짜"로 데이터 버전을 삼는다.
        data_version = str(matches["date"].max())
    except (UnknownTeamError, InsufficientFormError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_json("error", "prediction pipeline failed", request_id=request.state.request_id, error=str(e))
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
            log_json("warning", "failed to save prediction record", request_id=request.state.request_id, error=str(e))

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "probabilities": probabilities,
    }
