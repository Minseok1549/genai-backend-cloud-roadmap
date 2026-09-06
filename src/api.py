"""EPL 승부 예측 API. /predict는 호출 시점마다 최신 시즌 경기를 반영해 팀 폼을 다시 계산한다."""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import storage
from pydantic import BaseModel, Field, field_validator

import db
from data import load_matches, load_matchday_info
from fetch_data import ensure_all_seasons_cached
from logutil import log_json
from predictor import predict_match, load_model_bundle, UnknownTeamError, InsufficientFormError

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
PREDICTIONS_BUCKET = os.environ.get("PREDICTIONS_BUCKET")
KST = ZoneInfo("Asia/Seoul")
KST_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


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


def _fetch_daily_predictions(date_str: str) -> dict | None:
    if not PREDICTIONS_BUCKET:
        return None
    try:
        blob = storage.Client().bucket(PREDICTIONS_BUCKET).blob(f"predictions/{date_str}.json")
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    except Exception as e:
        log_json("warning", "failed to read daily predictions from GCS", date=date_str, error=str(e))
        return None


def _list_available_dates(limit: int = 14) -> list[str]:
    if not PREDICTIONS_BUCKET:
        return []
    try:
        blobs = storage.Client().list_blobs(PREDICTIONS_BUCKET, prefix="predictions/")
        dates = sorted((b.name.removeprefix("predictions/").removesuffix(".json") for b in blobs), reverse=True)
        return dates[:limit]
    except Exception as e:
        log_json("warning", "failed to list daily prediction dates from GCS", error=str(e))
        return []


def _collect_matchday_predictions(dates: list[str]) -> tuple[dict[int, dict], str | None]:
    """라운드가 걸쳐 있는 날짜들의 저장된 예측 파일을 모두 읽어 match_id 기준으로 합친다.
    하루짜리 배치 파일 하나만 보면 같은 라운드의 다른 날 경기가 빠지기 때문."""
    by_id: dict[int, dict] = {}
    latest_generated_at = None
    for d in dates:
        payload = _fetch_daily_predictions(d)
        if not payload:
            continue
        for p in payload["predictions"]:
            by_id[p["match_id"]] = p
        generated_at = payload.get("generated_at")
        if generated_at and (latest_generated_at is None or generated_at > latest_generated_at):
            latest_generated_at = generated_at
    return by_id, latest_generated_at


NO_PREDICTION_REASONS = {
    "FINISHED": "경기 종료 · 예측 기록 없음",
    "POSTPONED": "경기 연기됨",
    "CANCELLED": "경기 취소됨",
    "SUSPENDED": "경기 중단됨",
}


def _merge_round_fixtures(fixtures: list[dict], predictions_by_id: dict[int, dict]) -> list[dict]:
    """라운드 전체 경기 목록에, 저장된 예측이 있으면 확률을 붙이고 없으면 이유와 함께
    None으로 남긴다 — 폼 데이터 부족으로 모델이 건너뛴 경기나 배치가 아직 안 돌았던
    경기도 라운드에서 통째로 빠지지 않고 "왜 예측이 없는지"와 함께 보이게 하기 위해서다."""
    merged = []
    for fx in fixtures:
        pred = predictions_by_id.get(fx["match_id"])
        if pred:
            merged.append({**fx, "probabilities": pred["probabilities"]})
        else:
            reason = NO_PREDICTION_REASONS.get(fx["status"], "예측 데이터 없음")
            merged.append({**fx, "probabilities": None, "no_prediction_reason": reason})
    return merged


def _format_kickoff_kst(kickoff_utc: str) -> str:
    dt_kst = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00")).astimezone(KST)
    weekday = KST_WEEKDAYS[dt_kst.weekday()]
    return dt_kst.strftime(f"%m/%d({weekday}) %H:%M")


def _match_card_html(p: dict) -> str:
    header = f"""<div class="match-time">{_format_kickoff_kst(p['kickoff_utc'])} <span class="tz">KST</span></div>
  <div class="match-teams">
    <span class="team">{p['home_team']}</span>
    <span class="vs">vs</span>
    <span class="team">{p['away_team']}</span>
  </div>"""

    probs = p.get("probabilities")
    if probs is None:
        body = f'<p class="no-pred">{p.get("no_prediction_reason", "예측 데이터 없음")}</p>'
        return f'<div class="match-card no-pred-card">{header}{body}</div>'

    home_pct = probs.get("HOME_TEAM", 0) * 100
    draw_pct = probs.get("DRAW", 0) * 100
    away_pct = probs.get("AWAY_TEAM", 0) * 100
    body = f"""<div class="prob-bar">
    <div class="prob-seg home" style="width:{home_pct:.1f}%"></div>
    <div class="prob-seg draw" style="width:{draw_pct:.1f}%"></div>
    <div class="prob-seg away" style="width:{away_pct:.1f}%"></div>
  </div>
  <div class="prob-labels">
    <span class="prob-label home">홈승 {home_pct:.1f}%</span>
    <span class="prob-label draw">무 {draw_pct:.1f}%</span>
    <span class="prob-label away">원정승 {away_pct:.1f}%</span>
  </div>"""
    return f'<div class="match-card">{header}{body}</div>'


def _render_dashboard_html(
    view_label: str, payload: dict | None, available_dates: list[str], active_date: str | None
) -> str:
    date_links = "".join(
        f'<a class="date-link{" active" if d == active_date else ""}" href="/dashboard?date={d}">{d}</a>'
        for d in available_dates
    ) or '<span class="empty">기록된 날짜가 없습니다.</span>'

    predictions = sorted(payload["predictions"], key=lambda p: p["kickoff_utc"]) if payload else []
    if not predictions:
        body = f'<p class="empty">{view_label}에 예정된 경기 예측 기록이 없습니다.</p>'
    else:
        cards = "".join(_match_card_html(p) for p in predictions)
        generated_kst = _format_kickoff_kst(payload["generated_at"]) if payload.get("generated_at") else "-"
        body = f'<div class="match-grid">{cards}</div><p class="meta">예측 생성 시각: {generated_kst} KST</p>'

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EPL 예측 대시보드</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Pretendard", "Apple SD Gothic Neo", sans-serif;
    max-width: 880px; margin: 0 auto; padding: 32px 16px 60px;
    background: linear-gradient(180deg, #f4f6fb 0%, #eef1f8 400px, #f7f8fb 100%);
    color: #1a1a2e;
  }}
  header {{
    background: linear-gradient(135deg, #3a2a6d 0%, #6a2c8c 100%);
    color: #fff; border-radius: 16px; padding: 24px 28px; margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(58, 42, 109, 0.25);
  }}
  header h1 {{ margin: 0 0 4px; font-size: 1.5rem; }}
  header p {{ margin: 0; opacity: 0.85; font-size: 0.9rem; }}
  .dates {{ margin-bottom: 24px; display: flex; flex-wrap: wrap; }}
  .date-link {{
    display: inline-block; padding: 5px 12px; margin: 0 6px 6px 0; border-radius: 999px;
    background: #fff; text-decoration: none; color: #555; font-size: 0.82rem;
    border: 1px solid #e2e2ee; transition: background 0.15s;
  }}
  .date-link.active {{ background: #3a2a6d; color: #fff; border-color: #3a2a6d; }}
  h2 {{ font-size: 1.1rem; color: #444; margin: 4px 0 16px; }}
  .match-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
  .match-card {{
    background: #fff; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(30, 20, 60, 0.06); border: 1px solid #eceef7;
  }}
  .match-time {{ font-size: 0.78rem; color: #8a8aa0; margin-bottom: 8px; letter-spacing: 0.02em; }}
  .match-time .tz {{ color: #b4b4c8; }}
  .match-teams {{ display: flex; align-items: center; justify-content: space-between; font-weight: 600; font-size: 0.95rem; margin-bottom: 12px; }}
  .match-teams .vs {{ color: #c2c2d6; font-weight: 400; font-size: 0.8rem; margin: 0 6px; }}
  .match-teams .team {{ flex: 1; }}
  .match-teams .team:last-child {{ text-align: right; }}
  .prob-bar {{ display: flex; height: 8px; border-radius: 999px; overflow: hidden; background: #f0f0f5; margin-bottom: 8px; }}
  .prob-seg.home {{ background: #4f6bed; }}
  .prob-seg.draw {{ background: #c7c7d9; }}
  .prob-seg.away {{ background: #e0526b; }}
  .prob-labels {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: #888; }}
  .prob-label.home {{ color: #4f6bed; }}
  .prob-label.draw {{ color: #999; }}
  .prob-label.away {{ color: #e0526b; }}
  .no-pred-card {{ opacity: 0.6; }}
  .no-pred {{ color: #aaa; font-size: 0.82rem; margin: 10px 0 2px; }}
  .empty {{ color: #999; padding: 24px 0; }}
  .meta {{ color: #aaa; font-size: 0.78rem; margin-top: 20px; }}
</style>
</head>
<body>
  <header>
    <h1>⚽ EPL 매일 예측 대시보드</h1>
    <p>매일 자동으로 갱신되는 프리미어리그 승부 예측 (한국 시간 기준)</p>
  </header>
  <div class="dates">{date_links}</div>
  <h2>{view_label}</h2>
  {body}
</body>
</html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(date: str | None = None):
    available_dates = _list_available_dates()

    if date:
        payload = _fetch_daily_predictions(date)
        return _render_dashboard_html(date, payload, available_dates, active_date=date)

    matchday_info = load_matchday_info()
    predictions_by_id, generated_at = _collect_matchday_predictions(matchday_info["dates"])
    merged = _merge_round_fixtures(matchday_info["fixtures"], predictions_by_id)
    payload = {"predictions": merged, "generated_at": generated_at} if merged else None
    view_label = f"{matchday_info['matchday']}라운드" if matchday_info["matchday"] else "오늘"
    return _render_dashboard_html(view_label, payload, available_dates, active_date=None)
