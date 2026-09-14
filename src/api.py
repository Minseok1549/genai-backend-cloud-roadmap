"""EPL 승부 예측 API. /predict는 호출 시점마다 최신 시즌 경기를 반영해 팀 폼을 다시 계산한다."""
import html
import json
import os
import threading
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
from google.api_core.exceptions import NotFound
from google.cloud import storage
from pydantic import BaseModel, Field, field_validator

import db
from context_stats import match_context
from data import load_matches, load_matchday_info, load_all_fixtures_on_date, load_season_teams
from fetch_data import ensure_all_seasons_cached, CURRENT_SEASON
from scorecard import SERIES_LABELS
from odds import fetch_upcoming_odds
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

MAX_REQUEST_ID_LENGTH = 200  # 분산 추적 ID로 쓰이는 형식(UUID 36자, W3C traceparent 55자)에 넉넉한 상한


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    # 클라이언트나 로드밸런서가 이미 요청 ID를 붙여왔으면 그걸 그대로 잇는다(분산 추적) —
    # 없으면 새로 발급한다. 응답 헤더에도 실어서, 호출한 쪽이 이 값을 자기 로그와
    # 대조할 수 있게 한다.
    # 길이는 제한한다. 이 값은 클라이언트가 정하는데 로그 한 줄과 응답 헤더에 그대로 실리므로,
    # 10만 자를 보내면 요청 하나가 Cloud Logging에 10만 자를 쓰고 응답에도 되돌려준다(실측).
    # 추적 ID로 쓰이는 UUID는 36자면 충분해서, 넘는 값은 클라이언트 것을 버리고 새로 발급한다.
    # 개행을 섞은 헤더 주입은 HTTP 계층에서 이미 막히는 걸 확인했으므로 따로 검사하지 않는다.
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if 0 < len(incoming) <= MAX_REQUEST_ID_LENGTH else str(uuid.uuid4())
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

    # 팀명 검증을 배당률 API 호출보다 먼저 한다. 전에는 검증이 predict_match() 안에서만
    # 이뤄져서, 오타나 존재하지 않는 팀명으로 들어온 요청도 유료 쿼터(무료 티어 월 500회)를
    # 한 번 태운 뒤에야 400으로 거절됐다. 일정표가 아직 캐시되지 않아 목록이 비면 검증을
    # 건너뛰고 predict_match()의 기존 판정에 맡긴다.
    try:
        season_teams = load_season_teams()
    except Exception as e:
        # 캐시가 손상돼 팀 목록을 못 만드는 경우 — 조기 검증만 포기하고 아래 예측 파이프라인의
        # 예외 처리(503)에 맡긴다. 여기서 그냥 터지면 서버 장애가 500으로 나간다.
        log_json("warning", "failed to load season teams, skipping early validation",
                 request_id=request.state.request_id, error=str(e))
        season_teams = set()
    if season_teams:
        for team in (req.home_team, req.away_team):
            if team not in season_teams:
                raise HTTPException(status_code=400, detail=str(UnknownTeamError(team)))

    # 캐시 갱신 이후의 실패(손상된 캐시, 파싱 오류, 모델 추론 오류 등)는 클라이언트 잘못이
    # 아니라 서버 쪽 일시 장애이므로 500이 아니라 503으로 알린다. 의도적으로 던진
    # HTTPException(알 수 없는 팀명/기록 부족 등 400)은 그대로 통과시킨다.
    try:
        upcoming_odds = fetch_upcoming_odds()
    except Exception as e:  # 배당률 API 장애/쿼터 소진이어도 폼 기반 모델로 계속 서빙
        log_json("warning", "live odds fetch failed, falling back to form model", request_id=request.state.request_id, error=str(e))
        upcoming_odds = {}

    try:
        matches = load_matches()
        bundle = request.app.state.model_bundle
        fixture_odds = upcoming_odds.get((req.home_team, req.away_team))
        # TTL이 지난 캐시를 API 장애 때문에 어쩔 수 없이 재사용한 배당률(stale=True)은 쓰지
        # 않는다 — 매일 배치도 같은 이유로 버린다. 몇 시간 전 시장 가격을 지금 가격처럼
        # 확률로 환산하면 그 사이의 부상·라인업 발표가 전혀 반영되지 않은 숫자를 "최신
        # 예측"으로 내놓게 된다. 이 경우 폼 기반 모델로 내려가는 쪽이 정직하다.
        if fixture_odds is not None and fixture_odds.get("stale"):
            log_json("info", "discarding stale odds, falling back to form model",
                     request_id=request.state.request_id, home_team=req.home_team, away_team=req.away_team)
            fixture_odds = None
        probabilities = predict_match(
            req.home_team, req.away_team, matches, bundle, odds=fixture_odds, known_teams=season_teams
        )
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


_storage_client = None
_storage_client_lock = threading.Lock()


def _get_storage_client() -> storage.Client:
    """GCS 클라이언트를 프로세스당 하나만 만들어 재사용한다. storage.Client()는 생성할 때마다
    인증 자격증명을 다시 확인하는데(Cloud Run에서는 메타데이터 서버 호출), 대시보드 한 번에
    날짜 여러 개를 읽으므로 호출마다 새로 만들면 그 왕복이 날짜 수만큼 반복된다."""
    global _storage_client
    if _storage_client is None:
        with _storage_client_lock:
            if _storage_client is None:
                _storage_client = storage.Client()
    return _storage_client


def _fetch_daily_predictions(date_str: str) -> dict | None:
    if not PREDICTIONS_BUCKET:
        return None
    try:
        blob = _get_storage_client().bucket(PREDICTIONS_BUCKET).blob(f"predictions/{date_str}.json")
        # exists() + download_as_text()는 GCS를 두 번 부른다 — 없으면 download가 NotFound를
        # 던지므로 그걸 잡는 쪽이 한 번으로 끝난다.
        return json.loads(blob.download_as_text())
    except NotFound:
        return None
    except Exception as e:
        log_json("warning", "failed to read daily predictions from GCS", date=date_str, error=str(e))
        return None


def _fetch_scorecard() -> dict | None:
    """배치가 올려둔 성적표를 읽는다. 여기서 채점을 직접 하지 않는 이유는, 전 기간 예측
    파일을 다 읽어야 하는 작업이라 요청마다 하면 대시보드가 느려지고 GCS 읽기도 그만큼
    늘어나기 때문이다 — 결과는 하루 단위로만 바뀐다."""
    if not PREDICTIONS_BUCKET:
        return None
    try:
        blob = _get_storage_client().bucket(PREDICTIONS_BUCKET).blob("scorecard.json")
        return json.loads(blob.download_as_text())
    except NotFound:
        return None
    except Exception as e:
        log_json("warning", "failed to read scorecard from GCS", error=str(e))
        return None


def _list_available_dates(limit: int = 14) -> list[str]:
    if not PREDICTIONS_BUCKET:
        return []
    try:
        blobs = _get_storage_client().list_blobs(PREDICTIONS_BUCKET, prefix="predictions/")
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
    경기도 라운드에서 통째로 빠지지 않고 "왜 예측이 없는지"와 함께 보이게 하기 위해서다.

    확률의 출처 정보(북메이커 수, 모델 버전, 계산 시각)도 같이 넘긴다 — 같은 크기의 막대로
    보이는 두 확률이 실제로는 근거와 시점이 다르므로, 화면에서 구분할 수 있어야 한다."""
    merged = []
    for fx in fixtures:
        pred = predictions_by_id.get(fx["match_id"])
        if pred:
            merged.append({
                **fx,
                "probabilities": pred["probabilities"],
                "genai_prediction": pred.get("genai_prediction"),
                "bookmaker_count": pred.get("bookmaker_count"),
                "model_version": pred.get("model_version"),
                "computed_at": pred.get("computed_at"),
            })
        else:
            reason = NO_PREDICTION_REASONS.get(fx["status"], "예측 데이터 없음")
            merged.append({**fx, "probabilities": None, "no_prediction_reason": reason})
    return merged


def _attach_context(fixtures: list[dict]) -> list[dict]:
    """각 경기에 순위·최근 5경기·홈/원정 경기당 승점을 붙인다.

    확률만 있는 카드는 "왜 그 숫자인지"를 볼 수 없다. 전부 이미 받아둔 경기 데이터로
    계산하므로 외부 API 호출은 늘지 않는다. 경기 데이터 파싱은 한 번만 하고 모든 카드가
    공유한다 — 카드마다 다시 읽으면 한 페이지에 같은 파일을 열 번 파싱하게 된다.
    실패하면 맥락만 비운다: 부가 정보 때문에 대시보드 전체가 죽어선 안 된다."""
    if not fixtures:
        return fixtures
    try:
        matches = load_matches()
    except Exception as e:
        log_json("warning", "failed to load matches for context stats", error=str(e))
        return fixtures
    for fx in fixtures:
        try:
            fx["context"] = match_context(matches, fx["home_team"], fx["away_team"], CURRENT_SEASON)
        except Exception as e:
            log_json("warning", "failed to build match context", fixture_id=fx.get("match_id"), error=str(e))
    return fixtures


def _format_kickoff_kst(kickoff_utc: str) -> str:
    dt_kst = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00")).astimezone(KST)
    weekday = KST_WEEKDAYS[dt_kst.weekday()]
    return dt_kst.strftime(f"%m/%d({weekday}) %H:%M")


def _safe_url(url: str) -> str:
    """출처 링크는 Gemini의 grounding 응답에서 온 값이라 속성 탈출은 escape로 막아도
    javascript: 같은 스킴까지 그대로 넣어줄 순 없다 — http(s)가 아니면 무해한 값으로 대체한다."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return "#"
    return html.escape(url)


_TAG_SLUGS = {"부상": "injury", "폼": "form", "전술": "tactics", "주심": "referee", "기타": "other"}
_TAG_ICONS = {"부상": "🩹", "폼": "📈", "전술": "♟️", "주심": "🟨", "기타": "💡"}


def _sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""
    items = "".join(
        f'<li><a href="{_safe_url(s["url"])}" target="_blank" rel="noopener noreferrer">{html.escape(s["title"])}</a></li>'
        for s in sources
    )
    return f'<ul class="ai-report-sources">{items}</ul>'


def _points_html(points: list[dict]) -> str:
    """핵심 포인트를 기존 승부 예측 서비스(SofaScore AI Insights 등)처럼 태그별 독립
    카드 그리드로 렌더링한다 — 세로로 늘어선 리스트보다 한눈에 카테고리별로 스캔하기
    쉽고, 줄글 문단보다 플랫폼 서비스에 가까운 인상을 준다."""
    items = "".join(
        f'<li class="ai-point ai-point-{_TAG_SLUGS.get(p["tag"], "other")}">'
        f'<span class="ai-tag">{_TAG_ICONS.get(p["tag"], "💡")} {html.escape(p["tag"])}</span>'
        f'<span class="ai-point-text">{html.escape(p["text"])}</span></li>'
        for p in points
    )
    return f'<ul class="ai-points">{items}</ul>'


def _ai_details_html(label: str, headline: str | None, points: list[dict] | None, prose_text: str | None, sources: list[dict]) -> str:
    """headline은 카드에서 바로 보이는 한 줄 총평으로(탭 없이), points는 태그별 카드
    그리드로 접어서(details) 렌더링한다 — 없으면 과거에 저장된 문단 텍스트(prose_text)로
    폴백 렌더링한다. GCS에 이미 구버전 형식으로 저장된 과거 예측 기록도 계속 보여야
    하기 때문."""
    if headline or points:
        headline_html = (
            f'<div class="ai-teaser"><span class="ai-badge">AI</span>'
            f'<p class="ai-headline">{html.escape(headline)}</p></div>'
            if headline else ""
        )
        if not points:
            return headline_html
        body_html = _points_html(points)
    elif prose_text:
        headline_html = ""
        # text/reasoning은 Gemini가 생성한 텍스트라 html.escape 없이 그대로 꽂으면 삽입된
        # HTML/스크립트가 대시보드에서 그대로 실행될 수 있다(모델이 마크업을 흉내 내는
        # 경우가 실제로 있다).
        body_html = f'<div class="ai-report-prose">{html.escape(prose_text)}</div>'
    else:
        return ""
    return (
        f'{headline_html}'
        '<details class="ai-report">'
        f'<summary class="ai-report-label">{label}</summary>'
        f'<div class="ai-report-body">{body_html}</div>'
        f'{_sources_html(sources)}'
        '</details>'
    )


def _crest_html(url: str | None) -> str:
    if not url:
        return '<span class="crest crest-empty"></span>'
    return f'<img class="crest" src="{html.escape(url)}" alt="" loading="lazy">'


def _prob_bar_html(probabilities: dict, heading: str) -> str:
    home_pct = probabilities.get("HOME_TEAM", 0) * 100
    draw_pct = probabilities.get("DRAW", 0) * 100
    away_pct = probabilities.get("AWAY_TEAM", 0) * 100
    favorite = max(("home", home_pct), ("draw", draw_pct), ("away", away_pct), key=lambda t: t[1])[0]
    heading_html = f'<div class="prob-heading">{heading}</div>' if heading else ""
    return f"""{heading_html}<div class="prob-bar">
    <div class="prob-seg home{' fav' if favorite == 'home' else ''}" style="width:{home_pct:.1f}%"></div>
    <div class="prob-seg draw{' fav' if favorite == 'draw' else ''}" style="width:{draw_pct:.1f}%"></div>
    <div class="prob-seg away{' fav' if favorite == 'away' else ''}" style="width:{away_pct:.1f}%"></div>
  </div>
  <div class="prob-labels">
    <span class="prob-label home{' fav' if favorite == 'home' else ''}">홈승 {home_pct:.1f}%</span>
    <span class="prob-label draw{' fav' if favorite == 'draw' else ''}">무 {draw_pct:.1f}%</span>
    <span class="prob-label away{' fav' if favorite == 'away' else ''}">원정승 {away_pct:.1f}%</span>
  </div>"""


def _form_strip_html(recent: list[str]) -> str:
    """최근 경기를 W/D/L 칩으로 그린다. 왼쪽이 오래된 경기다."""
    if not recent:
        return '<span class="form-empty">기록 없음</span>'
    return "".join(f'<span class="form-chip form-{r.lower()}">{r}</span>' for r in recent)


def _team_context_html(side: dict, label: str, venue: str) -> str:
    table = side.get("table")
    rank_html = (
        f'<b>{table["rank"]}위</b> <span class="ctx-dim">{table["points"]}점</span>'
        if table else '<span class="ctx-dim">순위 기록 없음</span>'
    )
    ppg = side.get("ppg")
    ppg_html = f'{venue} <b>{ppg:.2f}</b>' if ppg is not None else f'{venue} <span class="ctx-dim">-</span>'
    return f"""<div class="ctx-team">
      <div class="ctx-label">{label}</div>
      <div class="ctx-rank">{rank_html}</div>
      <div class="ctx-form">{_form_strip_html(side.get("recent") or [])}</div>
      <div class="ctx-ppg">{ppg_html}</div>
    </div>"""


def _context_html(context: dict | None) -> str:
    """확률 아래에 맥락 세 가지(순위, 최근 5경기, 홈/원정 경기당 승점)를 붙인다.
    경기당 승점을 홈/원정으로 나눠 보여주는 이유는 홈에서 강하고 원정에서 무너지는 팀이
    흔하고, 이 경기에서 참고해야 할 숫자가 그 둘 중 하나이기 때문이다."""
    if not context:
        return ""
    return (
        '<div class="ctx-row">'
        + _team_context_html(context["home"], "홈", "홈 경기당")
        + _team_context_html(context["away"], "원정", "원정 경기당")
        + "</div>"
    )


def _provenance_html(p: dict) -> str:
    """이 확률이 어디서 나온 것인지 카드 맨 아래에 작게 남긴다.

    북메이커 3곳 평균과 15곳 평균은 신뢰도가 다르고, 모델 버전이 바뀌면 같은 경기 확률도
    달라진다. 계산 시각은 "이 숫자가 언제 기준인지"다 — 페이지 하단의 배치 생성 시각은
    파일 단위라, 그 안의 개별 경기가 며칠 전에 계산된 값일 수 있다."""
    bits = []
    count = p.get("bookmaker_count")
    if count:
        bits.append(f"북메이커 {count}곳 평균")
    if p.get("model_version"):
        bits.append(f"모델 {html.escape(str(p['model_version']))}")
    if p.get("computed_at"):
        try:
            bits.append(f"{_format_kickoff_kst(p['computed_at'])} 계산")
        except (ValueError, TypeError):
            pass
    if not bits:
        return ""
    return f'<div class="provenance">{" · ".join(bits)}</div>'


def _genai_block_html(genai: dict, heading: str = "뉴스 시나리오 (GenAI)") -> str:
    """GenAI 예측을 통계 모델 예측 아래에 나란히 붙인다 — 대체가 아니라 병기용이라
    자체 확률 바 + 근거를 별도 블록으로 감싼다. 통계 예측 자체가 없는 경기(배당률
    미공개)에서는 heading을 다르게 줘서 "이게 유일한 예측"이라는 걸 구분해준다."""
    bar_html = _prob_bar_html(genai["probabilities"], heading)
    reasoning_html = _ai_details_html(
        "근거 상세 보기",
        genai.get("headline"),
        genai.get("points"),
        genai.get("reasoning"),
        genai.get("sources") or [],
    )
    return f'<div class="genai-block">{bar_html}{reasoning_html}</div>'


def _match_card_html(p: dict) -> str:
    score = p.get("score")
    has_score = bool(score and score.get("home") is not None and score.get("away") is not None)
    home_team = html.escape(p["home_team"])
    away_team = html.escape(p["away_team"])
    home_crest = _crest_html(p.get("home_crest"))
    away_crest = _crest_html(p.get("away_crest"))

    if has_score:
        center_html = f'<div class="score">{score["home"]} : {score["away"]}</div>'
        time_suffix = ' <span class="ft-badge">종료</span>'
    else:
        center_html = '<div class="vs">vs</div>'
        time_suffix = ""

    teams_html = (
        f'<div class="team home">{home_crest}<span class="team-name" title="{home_team}">{home_team}</span></div>'
        f'{center_html}'
        f'<div class="team away"><span class="team-name" title="{away_team}">{away_team}</span>{away_crest}</div>'
    )

    header = f"""<div class="match-time">{_format_kickoff_kst(p['kickoff_utc'])} <span class="tz">KST</span>{time_suffix}</div>
  <div class="match-teams">{teams_html}</div>"""

    probs = p.get("probabilities")
    genai = p.get("genai_prediction")

    if probs is None and not genai:
        reason = "사전 예측 없음" if has_score else p.get("no_prediction_reason", "예측 데이터 없음")
        # 예측이 없는 경기에도 맥락은 붙인다 — 순위·최근 폼은 예측과 무관하게 유효한 정보고,
        # 카드가 "예측 없음" 한 줄로 비어 보이는 것보다 낫다.
        body = f'<p class="no-pred">{reason}</p>' + _context_html(p.get("context"))
        card_class = "match-card finished-card state-done" if has_score else "match-card no-pred-card state-nopred"
        return f'<div class="{card_class}">{header}{body}</div>'

    if probs is not None:
        if has_score:
            stat_heading = "경기 전 예측"
        elif genai:
            # "통계 모델 예측"이라고 쓰면 우리가 자체 통계로 계산한 독자 예측처럼 읽히는데,
            # 실제로 이 확률은 여러 북메이커 배당률의 내재확률을 평균·재보정한 값이다(배당률이
            # 없는 경기에는 배치가 아예 통계 예측을 만들지 않는다). 근거를 이름에 드러낸다.
            stat_heading = "배당률 기반 예측"
        else:
            stat_heading = ""
        body = _prob_bar_html(probs, stat_heading)
    else:
        body = ""

    if genai:
        # 종료된 경기도 "당시 예측을 어떤 근거로 했는지"는 결과와 함께 유지해야 하므로 생략하지
        # 않는다 — 특히 배당률이 끝까지 안 열려 GenAI가 유일한 예측이었던 경기는, 숨기면 예측
        # 기록 자체가 통째로 사라진다. 대신 지난 예측이라는 걸 heading으로 구분해준다.
        # "GenAI 예측"이라고만 쓰면 위 막대와 독립된 두 번째 모델의 의견처럼 읽힌다. 실제로는
        # 위 배당률 기반 확률을 프롬프트로 받아 뉴스·선수단 정보로 조정한 값이고, 확률이 실제
        # 빈도와 맞는지 검증된 적이 없다 — 그래서 "예측"이 아니라 "시나리오"로 부른다.
        if has_score:
            genai_heading = "경기 전 뉴스 시나리오"
        elif probs is not None:
            genai_heading = "뉴스 시나리오 (위 확률을 뉴스로 조정)"
        else:
            genai_heading = "뉴스 시나리오 (배당률 공개 전, 뉴스만 근거)"
        body += _genai_block_html(genai, genai_heading)

    body += _context_html(p.get("context")) + _provenance_html(p)

    card_class = "match-card finished-card state-done" if has_score else "match-card state-upcoming"
    return f'<div class="{card_class}">{header}{body}</div>'


def _date_chip_label(date_str: str) -> str:
    """날짜 칩에 요일과 오늘/내일을 붙인다. ISO 날짜만 있으면 어느 게 오늘인지 세어봐야
    알 수 있다. 기준은 KST — 카드에 표시되는 킥오프 시각과 같은 기준이어야 한다."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return html.escape(date_str)  # 예상 밖 형식이면 있는 그대로 보여준다
    today = datetime.now(KST).date()
    delta = (d - today).days
    if delta == 0:
        return "오늘"
    if delta == 1:
        return "내일"
    if delta == -1:
        return "어제"
    return f"{d.month}/{d.day}({KST_WEEKDAYS[d.weekday()]})"


def _render_dashboard_html(
    view_label: str,
    payload: dict | None,
    available_dates: list[str],
    active_date: str | None,
    round_summary: dict | None = None,
    scorecard: dict | None = None,
) -> str:
    view_label = html.escape(view_label)  # date 쿼리 파라미터가 그대로 들어올 수 있음 (반사형 XSS 방지)
    # 날짜를 고른 상태에서 현재 라운드로 돌아갈 방법이 없으면 주소를 직접 지워야 한다.
    round_link = (
        f'<a class="date-link{"" if active_date else " active"}" href="/dashboard">현재 라운드</a>'
    )
    date_links = round_link + "".join(
        f'<a class="date-link{" active" if d == active_date else ""}" href="/dashboard?date={d}">'
        f'{_date_chip_label(d)}</a>'
        for d in available_dates
    )

    chips = []
    if round_summary:
        upcoming = round_summary["total"] - round_summary["finished"]
        chips.append(f'<span class="summary-chip"><b>{round_summary["total"]}</b>경기</span>')
        chips.append(f'<span class="summary-chip done"><b>{round_summary["finished"]}</b>종료</span>')
        chips.append(f'<span class="summary-chip live"><b>{upcoming}</b>예정</span>')
    # 누적 성적을 첫 화면에 같이 둔다. 성적표를 따로 들어가야만 볼 수 있으면, 확률을 믿을
    # 근거가 있는지 모르는 채로 숫자만 읽게 된다.
    # 단, 표본이 적을 때는 적중률 숫자를 첫 화면에 걸지 않는다. 14경기짜리 "적중률 14%"는
    # 모델이 그 정도라는 뜻이 아니라 아직 알 수 없다는 뜻인데, 칩으로 보면 전자로 읽힌다.
    market = ((scorecard or {}).get("series") or {}).get("market")
    if market:
        if market["n"] >= MIN_COMPARABLE_MATCHES:
            label = f'적중률 <b>{market["accuracy"] * 100:.0f}%</b> · {market["n"]}경기 채점'
        else:
            label = f'성적표 · {market["n"]}경기 채점 (표본 부족)'
        chips.append(f'<a class="summary-chip scorecard-chip" href="/scorecard">{label}</a>')
    summary_html = f'<div class="summary-strip">{"".join(chips)}</div>' if chips else ""

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
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="stylesheet" as="style" crossorigin
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.css">
<style>
  :root {{
    --bg: #0b0e14;
    --surface: #141924;
    --surface-2: #1b2130;
    --border: #262d3d;
    --text: #eef1f8;
    --text-dim: #9099ad;
    --text-faint: #5c637a;
    --brand: #8b7bff;
    --home: #5b8def;
    --draw: #9099ad;
    --away: #ef5c8a;
    --win: #34d399;
  }}
  * {{ box-sizing: border-box; }}
  html {{ background: var(--bg); }}
  body {{
    font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif;
    max-width: 920px; margin: 0 auto; padding: 28px 16px 60px;
    background: var(--bg); color: var(--text);
    font-variant-numeric: tabular-nums;
  }}
  header {{
    display: flex; align-items: center; gap: 14px;
    padding: 18px 4px 22px; margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
  }}
  header img.emblem {{ width: 40px; height: 40px; flex-shrink: 0; }}
  header .heading {{ min-width: 0; }}
  header h1 {{ margin: 0 0 3px; font-size: 1.3rem; font-weight: 800; letter-spacing: -0.01em; }}
  header p {{ margin: 0; color: var(--text-dim); font-size: 0.84rem; }}
  .dates {{
    display: flex; gap: 8px; margin-bottom: 20px; overflow-x: auto; padding-bottom: 4px;
    scrollbar-width: thin;
  }}
  .date-link {{
    flex-shrink: 0; display: inline-block; padding: 7px 14px; border-radius: 999px;
    background: var(--surface); text-decoration: none; color: var(--text-dim); font-size: 0.82rem;
    font-weight: 600; border: 1px solid var(--border); transition: border-color 0.15s, color 0.15s;
  }}
  .date-link:hover {{ color: var(--text); border-color: var(--text-faint); }}
  .date-link.active {{ background: var(--brand); color: #fff; border-color: var(--brand); }}
  h2 {{ font-size: 1.05rem; color: var(--text); margin: 4px 0 14px; font-weight: 700; }}
  .summary-strip {{ display: flex; gap: 10px; margin-bottom: 20px; }}
  .summary-chip {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 8px 16px;
    font-size: 0.78rem; color: var(--text-dim);
  }}
  .summary-chip b {{ color: var(--text); font-size: 1rem; margin-right: 4px; }}
  .summary-chip.done b {{ color: var(--win); }}
  .summary-chip.live b {{ color: var(--home); }}
  .match-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; }}
  .match-card {{
    background: var(--surface); border-radius: 14px; padding: 16px 18px;
    border: 1px solid var(--border); border-top: 3px solid var(--text-faint);
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}
  .match-card:hover {{ transform: translateY(-2px); border-color: var(--text-faint); }}
  .match-card.state-upcoming {{ border-top-color: var(--home); }}
  .match-card.state-done {{ border-top-color: var(--win); }}
  .match-card.state-nopred {{ border-top-color: #d8a44c; }}
  .match-time {{ font-size: 0.74rem; color: var(--text-dim); margin-bottom: 12px; letter-spacing: 0.02em; }}
  .match-time .tz {{ color: var(--text-faint); }}
  .match-teams {{
    display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 8px;
    font-weight: 700; font-size: 0.92rem; margin-bottom: 14px;
  }}
  .match-teams .team {{ display: flex; align-items: center; gap: 8px; min-width: 0; }}
  .match-teams .team.away {{ justify-content: flex-end; }}
  .match-teams .team-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .crest {{ width: 22px; height: 22px; object-fit: contain; flex-shrink: 0; }}
  .crest-empty {{ width: 22px; height: 22px; border-radius: 50%; background: var(--surface-2); flex-shrink: 0; }}
  .match-teams .vs {{ color: var(--text-faint); font-weight: 500; font-size: 0.76rem; padding: 0 4px; }}
  .match-teams .score {{ font-weight: 800; font-size: 1.1rem; color: var(--text); padding: 0 6px; white-space: nowrap; }}
  .ft-badge {{ background: rgba(52, 211, 153, 0.15); color: var(--win); font-size: 0.62rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; margin-left: 6px; }}
  .prob-heading {{ font-size: 0.68rem; color: var(--text-faint); margin-bottom: 6px; }}
  .prob-bar {{ display: flex; height: 8px; border-radius: 999px; overflow: hidden; background: var(--surface-2); margin-bottom: 9px; }}
  .prob-seg {{ opacity: 0.35; }}
  .prob-seg.fav {{ opacity: 1; }}
  .prob-seg.home {{ background: var(--home); }}
  .prob-seg.draw {{ background: var(--draw); }}
  .prob-seg.away {{ background: var(--away); }}
  .prob-labels {{ display: flex; justify-content: space-between; font-size: 0.74rem; color: var(--text-faint); }}
  .prob-label.fav {{ font-weight: 800; }}
  .prob-label.fav.home {{ color: var(--home); }}
  .prob-label.fav.draw {{ color: var(--text-dim); }}
  .prob-label.fav.away {{ color: var(--away); }}
  .genai-block {{ margin-top: 14px; padding-top: 14px; border-top: 1px dashed var(--border); }}
  .genai-block .ai-report {{ border-top: none; padding-top: 8px; margin-top: 0; }}
  .ai-report {{ margin-top: 12px; padding-top: 12px; border-top: 1px dashed var(--border); font-size: 0.82rem; }}
  .ai-report-label {{ cursor: pointer; font-size: 0.64rem; font-weight: 700; color: var(--text-dim); letter-spacing: 0.04em; list-style: none; }}
  .ai-report-label::-webkit-details-marker {{ display: none; }}
  .ai-report-label::after {{ content: "▾"; float: right; }}
  .ai-report[open] .ai-report-label::after {{ content: "▴"; }}
  .ai-report-body {{ line-height: 1.6; color: var(--text-dim); margin-top: 9px; }}
  .ai-report-prose {{ line-height: 1.6; color: var(--text-dim); }}
  .ai-teaser {{ display: flex; align-items: baseline; gap: 8px; margin: 12px 0 0; }}
  .ai-badge {{
    flex-shrink: 0; font-size: 0.6rem; font-weight: 800; letter-spacing: 0.04em;
    padding: 2px 6px; border-radius: 5px; background: rgba(139, 123, 255, 0.16);
    color: var(--brand); border: 1px solid rgba(139, 123, 255, 0.35);
  }}
  .ai-headline {{ margin: 0; font-weight: 700; color: var(--text); line-height: 1.4; font-size: 0.86rem; }}
  .ai-points {{
    list-style: none; margin: 0; padding: 0; display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px;
  }}
  .ai-point {{
    display: flex; flex-direction: column; gap: 5px; background: var(--surface-2);
    border: 1px solid var(--border); border-left: 3px solid var(--text-faint);
    border-radius: 8px; padding: 9px 11px;
  }}
  .ai-point-text {{ color: var(--text-dim); line-height: 1.5; font-size: 0.8rem; }}
  .ai-tag {{ font-size: 0.66rem; font-weight: 700; letter-spacing: 0.01em; color: var(--text-faint); }}
  .ai-point-injury {{ border-left-color: var(--away); }}
  .ai-point-injury .ai-tag {{ color: var(--away); }}
  .ai-point-form {{ border-left-color: var(--home); }}
  .ai-point-form .ai-tag {{ color: var(--home); }}
  .ai-point-tactics {{ border-left-color: #d8a44c; }}
  .ai-point-tactics .ai-tag {{ color: #d8a44c; }}
  .ai-point-referee {{ border-left-color: var(--win); }}
  .ai-point-referee .ai-tag {{ color: var(--win); }}
  .ai-report-sources {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 0; padding: 0; list-style: none; }}
  .ai-report-sources li {{ display: contents; }}
  .ai-report-sources a {{
    display: inline-block; max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    color: var(--text-dim); background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 999px; padding: 3px 10px; font-size: 0.68rem; text-decoration: none;
  }}
  .ai-report-sources a:hover {{ color: var(--brand); border-color: var(--brand); }}
  .no-pred-card {{ opacity: 0.6; }}
  .no-pred {{ color: #d8a44c; font-size: 0.82rem; margin: 10px 0 2px; }}
  .state-done .no-pred {{ color: var(--text-faint); }}
  .empty {{ color: var(--text-faint); padding: 24px 0; }}
  .meta {{ color: var(--text-faint); font-size: 0.76rem; margin-top: 20px; }}
  .ctx-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    margin-top: 14px; padding-top: 12px; border-top: 1px dashed var(--border);
  }}
  .ctx-team {{ display: flex; flex-direction: column; gap: 5px; min-width: 0; }}
  .ctx-team:last-child {{ text-align: right; align-items: flex-end; }}
  .ctx-label {{ font-size: 0.66rem; color: var(--text-faint); letter-spacing: 0.04em; }}
  .ctx-rank {{ font-size: 0.8rem; }}
  .ctx-dim {{ color: var(--text-faint); font-weight: 400; }}
  .ctx-ppg {{ font-size: 0.72rem; color: var(--text-dim); }}
  .ctx-form {{ display: flex; gap: 3px; }}
  .form-chip {{
    width: 17px; height: 17px; border-radius: 4px; font-size: 0.62rem; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center; color: #0b0e14;
  }}
  .form-w {{ background: var(--win); }}
  .form-d {{ background: var(--draw); }}
  .form-l {{ background: var(--away); }}
  .form-empty {{ font-size: 0.72rem; color: var(--text-faint); }}
  .provenance {{ margin-top: 12px; font-size: 0.68rem; color: var(--text-faint); }}
  .scorecard-link {{
    margin-left: auto; flex-shrink: 0; padding: 7px 13px; border-radius: 999px;
    background: var(--surface); border: 1px solid var(--border); color: var(--brand);
    text-decoration: none; font-size: 0.78rem; font-weight: 600;
  }}
  .scorecard-chip {{ color: var(--brand); text-decoration: none; }}
</style>
</head>
<body>
  <header>
    <img class="emblem" src="https://crests.football-data.org/PL.png" alt="" loading="lazy">
    <div class="heading">
      <h1>EPL 승부 예측</h1>
      <p>매일 자동으로 갱신되는 프리미어리그 승부 예측 · AI 프리뷰 (한국 시간 기준)</p>
    </div>
    <a class="scorecard-link" href="/scorecard">성적표</a>
  </header>
  <div class="dates">{date_links}</div>
  <h2>{view_label}</h2>
  {summary_html}
  {body}
</body>
</html>"""


# 성적표에 표시할 순서. 우리 예측을 먼저, baseline을 아래에 둬서 "이것보다 나은가"로 읽히게 한다.
SCORECARD_ORDER = ["market", "genai", "bookmaker", "base_rate", "always_home"]

# 이 경기 수 아래에서는 모델 간 우열을 가리지 않는다. 축구 한 경기의 결과는 대부분 운이라,
# 수십 경기 규모에서는 어느 모델이 앞서는지가 라운드마다 뒤집힌다. 실제로 첫 채점(14경기)에서
# "무조건 홈 승"이 1/14를 기록했는데, 같은 시즌 전체 홈 승률은 40%대다 — 그 두 라운드가
# 무승부로 쏠린 우연이었을 뿐이다. 그런 표에 최고값을 초록색으로 칠하면, 우연을 실력으로
# 읽게 만든다. 50경기도 확실한 판단선은 아니지만, 한 라운드가 순위를 뒤집지는 못하는 최소선이다.
MIN_COMPARABLE_MATCHES = 50

_SCORECARD_CSS = """
  :root {
    --bg:#0b0e14; --surface:#141924; --border:#262d3d;
    --text:#eef1f8; --text-dim:#9099ad; --text-faint:#5c637a; --brand:#8b7bff; --win:#34d399;
    --warn:#fbbf24;
  }
  * { box-sizing: border-box; }
  html { background: var(--bg); }
  body {
    font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont,
                 "Apple SD Gothic Neo", sans-serif;
    max-width: 920px; margin: 0 auto; padding: 28px 16px 60px;
    background: var(--bg); color: var(--text); font-variant-numeric: tabular-nums;
  }
  h1 { margin: 0 0 4px; font-size: 1.3rem; font-weight: 800; }
  h2 { margin: 32px 0 10px; font-size: 1rem; font-weight: 700; }
  a { color: var(--brand); }
  .sub { margin: 0 0 22px; color: var(--text-dim); font-size: 0.84rem; }
  .note { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
          padding: 14px 16px; margin-bottom: 22px; color: var(--text-dim);
          font-size: 0.82rem; line-height: 1.7; }
  .note b { color: var(--text); }
  .note.warn { border-color: var(--warn); }
  .note.warn b { color: var(--warn); }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; min-width: 480px; font-size: 0.85rem; }
  th, td { padding: 9px 12px; text-align: right; border-bottom: 1px solid var(--border); }
  th:first-child, td:first-child { text-align: left; }
  thead th { color: var(--text-dim); font-weight: 600; font-size: 0.78rem; }
  tbody tr.ours td { font-weight: 700; }
  tbody tr.baseline td { color: var(--text-dim); }
  td.best { color: var(--win); }
  .empty { color: var(--text-faint); font-size: 0.86rem; }
"""


DASHBOARD_CACHE_SECONDS = 300  # 예측은 하루 한 번 배치로만 갱신되고 스코어는 6h TTL 캐시에서 온다 — 5분은 안전하게 짧은 값


def _dashboard_response(body: str) -> HTMLResponse:
    """대시보드 응답에 캐시 헤더를 붙인다. 이 페이지를 한 번 그리려면 GCS에서 날짜별 예측
    파일을 여러 개 읽고 시즌 캐시도 확인해야 하는데, 원본 데이터는 하루 한 번(배치)만
    바뀐다 — 헤더가 없으면 새로고침·링크 공유·크롤러 방문이 전부 그 작업을 처음부터
    다시 시키고, 스케일 투 제로 때문에 콜드스타트와 외부 API 호출까지 같이 유발한다."""
    return HTMLResponse(
        body, headers={"Cache-Control": f"public, max-age={DASHBOARD_CACHE_SECONDS}"}
    )


def _metric_cell(value, best: bool) -> str:
    if value is None:
        return '<td class="dim">-</td>'
    return f'<td class="{"best" if best else ""}">{value:.3f}</td>'


def _scorecard_table_html(series: dict, comparable: bool = True) -> str:
    rows_data = [(name, series[name]) for name in SCORECARD_ORDER if name in series]
    if not rows_data:
        return '<p class="empty">채점할 예측이 아직 없습니다.</p>'

    # 각 지표에서 가장 좋은 값을 찾아 강조한다. 정확도는 높을수록, log loss와 RPS는 낮을수록
    # 좋다. "무조건 홈 승"은 log loss·RPS를 내놓지 않으므로 비교에서 자연히 빠진다.
    # 표본이 아직 작으면 강조를 아예 하지 않는다 — 숫자는 그대로 보여주되, 그 차이를 결론으로
    # 읽으라고 권하지는 않는다.
    best = {}
    if comparable:
        for metric, better in (("accuracy", max), ("log_loss", min), ("rps", min)):
            values = [m[metric] for _, m in rows_data if m.get(metric) is not None]
            if values:
                best[metric] = better(values)

    rows = []
    for name, m in rows_data:
        css = "baseline" if name in ("bookmaker", "base_rate", "always_home") else "ours"
        cells = "".join(
            _metric_cell(m.get(metric), m.get(metric) is not None and m.get(metric) == best.get(metric))
            for metric in ("accuracy", "log_loss", "rps")
        )
        rows.append(
            f'<tr class="{css}"><td>{html.escape(SERIES_LABELS.get(name, name))}</td>'
            f'<td>{m["n"]}</td>{cells}</tr>'
        )
    return f"""<div class="scroll"><table>
  <thead><tr><th>모델</th><th>채점 경기</th><th>적중률</th><th>log loss</th><th>RPS</th></tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table></div>"""


def _calibration_table_html(series: dict) -> str:
    blocks = []
    for name in ("market", "genai"):
        bins = (series.get(name) or {}).get("calibration")
        if not bins:
            continue
        rows = "".join(
            f'<tr><td>{html.escape(b["range"])}</td><td>{b["n"]}</td>'
            f'<td>{b["predicted"] * 100:.1f}%</td><td>{b["actual"] * 100:.1f}%</td>'
            f'<td>{(b["actual"] - b["predicted"]) * 100:+.1f}%p</td></tr>'
            for b in bins
        )
        blocks.append(f"""<h2>{html.escape(SERIES_LABELS[name])}</h2>
<div class="scroll"><table>
  <thead><tr><th>우리가 말한 확률</th><th>표본</th><th>평균 주장</th><th>실제 발생</th><th>차이</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>""")
    return "".join(blocks)


def _render_scorecard_html(card: dict | None) -> str:
    if not card or not card.get("series"):
        body = '<p class="empty">아직 채점된 예측이 없습니다. 경기 결과가 쌓이면 여기에 성적이 표시됩니다.</p>'
        sub = ""
    else:
        date_range = card.get("date_range")
        span = f"{date_range[0]} ~ {date_range[1]}" if date_range else "-"
        generated = card.get("generated_at")
        generated_kst = _format_kickoff_kst(generated) if generated else "-"
        sub = f'<p class="sub">{html.escape(span)} · 채점 경기 {card["graded_matches"]}경기 · 갱신 {generated_kst} KST</p>'
        graded = card.get("graded_matches") or 0
        comparable = graded >= MIN_COMPARABLE_MATCHES
        warning = "" if comparable else f"""<p class="note warn">
  <b>아직 {graded}경기만 채점됐습니다.</b> 이 정도 표본에서는 모델 간 점수 차이가 실력 차이가
  아니라 대부분 운입니다. 한 라운드가 무승부로 쏠리기만 해도 순위가 뒤집히기 때문에,
  {MIN_COMPARABLE_MATCHES}경기가 쌓이기 전까지는 어느 쪽이 낫다고 표시하지 않습니다.
  아래 숫자는 참고용 기록입니다.
</p>"""
        body = warning + _scorecard_table_html(card["series"], comparable) + f"""
<h2>확률이 실제와 맞는지 (calibration)</h2>
<p class="note">
  "70%라고 말한 경기들에서 실제로 70%쯤 일어났는가"를 봅니다.
  <b>차이</b>가 0에 가까울수록 확률을 정직하게 말한 것이고,
  음수면 <b>과신</b>(말한 것보다 덜 일어남), 양수면 <b>과소평가</b>입니다.<br>
  홈 승·무승부·원정 승 확률을 한 구간에 모아서 세므로, 차이가 0이라도 홈 승 과신과
  원정 승 과소평가가 서로 상쇄된 결과일 수 있습니다 — 결과별로 맞는다는 뜻은 아닙니다.
</p>""" + _calibration_table_html(card["series"])

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EPL 예측 성적표</title>
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="stylesheet" as="style" crossorigin
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.css">
<style>{_SCORECARD_CSS}</style>
</head>
<body>
<h1>예측 성적표</h1>
{sub}
<p class="note">
  경기가 시작되기 전에 저장한 확률만 채점합니다. 틀린 예측도 지우지 않습니다.<br>
  <b>적중률</b>은 가장 확률이 높다고 본 결과가 실제로 나온 비율입니다. 다만 이것만으로는
  부족합니다 — 항상 강팀만 고르는 예측기도 적중률은 높게 나오기 때문입니다.<br>
  <b>log loss</b>와 <b>RPS</b>는 확률을 얼마나 정직하게 말했는지를 재는 점수로,
  <b>둘 다 낮을수록 좋습니다</b>. 자신 없는 경기에 과한 확신을 실으면 점수가 나빠집니다.<br>
  아래쪽 <b>baseline</b>들은 비교 기준입니다. 우리 모델이 북메이커 평균이나
  "무조건 홈 승" 같은 단순 규칙보다 나은지가 실제 판단 기준입니다.<br>
  단, <b>채점 경기 수가 서로 다른 줄은 직접 비교할 수 없습니다</b>. 뉴스 시나리오는 킥오프
  하루 전부터만 만들어져 다른 줄보다 표본이 적고, 그만큼 다른 경기들을 채점한 점수입니다.
</p>
{body}
<p class="sub" style="margin-top:32px"><a href="/dashboard">← 대시보드로</a></p>
</body>
</html>"""


@app.get("/scorecard", response_class=HTMLResponse)
def scorecard_page():
    return _dashboard_response(_render_scorecard_html(_fetch_scorecard()))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(date: str | None = None):
    available_dates = _list_available_dates()

    try:
        ensure_all_seasons_cached()
    except Exception as e:
        # 로컬에 캐시된 원본 경기 데이터를 직접 읽는 두 뷰(날짜별/라운드) 모두 여기서
        # 갱신하지 않으면 방금 끝난 경기의 스코어가 컨테이너가 재기동될 때까지 대시보드에
        # 반영되지 않는다. 외부 API가 잠깐 죽어도 기존 캐시로 계속 서빙한다.
        log_json("warning", "season cache refresh failed, dashboard using existing cache", error=str(e))

    if date:
        try:
            fixtures = load_all_fixtures_on_date(date)
        except (ValueError, TypeError):
            # date는 쿼리 파라미터라 임의 문자열이 들어올 수 있다 — 날짜로 파싱 안 되면
            # 경기 목록이 없는 것으로 취급한다(반사형 XSS 방지 escape는 렌더링 쪽에서 처리).
            fixtures = []
        stored = _fetch_daily_predictions(date)
        predictions_by_id = {p["match_id"]: p for p in stored["predictions"]} if stored else {}
        merged = _attach_context(_merge_round_fixtures(fixtures, predictions_by_id))
        generated_at = stored.get("generated_at") if stored else None
        payload = {"predictions": merged, "generated_at": generated_at} if merged else None
        return _dashboard_response(_render_dashboard_html(
            date, payload, available_dates, active_date=date, scorecard=_fetch_scorecard(),
        ))

    matchday_info = load_matchday_info()
    predictions_by_id, generated_at = _collect_matchday_predictions(matchday_info["dates"])
    merged = _attach_context(_merge_round_fixtures(matchday_info["fixtures"], predictions_by_id))
    payload = {"predictions": merged, "generated_at": generated_at} if merged else None
    view_label = f"{matchday_info['matchday']}라운드" if matchday_info["matchday"] else "오늘"
    round_summary = {
        "total": len(matchday_info["fixtures"]),
        "finished": sum(1 for f in matchday_info["fixtures"] if f["status"] == "FINISHED"),
    } if matchday_info["fixtures"] else None
    return _dashboard_response(_render_dashboard_html(
        view_label, payload, available_dates, active_date=None,
        round_summary=round_summary, scorecard=_fetch_scorecard(),
    ))
