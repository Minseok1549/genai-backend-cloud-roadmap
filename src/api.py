"""EPL 승부 예측 API. /predict는 호출 시점마다 최신 시즌 경기를 반영해 팀 폼을 다시 계산한다."""
import html
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
from data import load_matches, load_matchday_info, load_all_fixtures_on_date
from fetch_data import ensure_all_seasons_cached
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
        upcoming_odds = fetch_upcoming_odds()
    except Exception as e:  # 배당률 API 장애/쿼터 소진이어도 폼 기반 모델로 계속 서빙
        log_json("warning", "live odds fetch failed, falling back to form model", request_id=request.state.request_id, error=str(e))
        upcoming_odds = {}

    try:
        matches = load_matches()
        bundle = request.app.state.model_bundle
        fixture_odds = upcoming_odds.get((req.home_team, req.away_team))
        probabilities = predict_match(req.home_team, req.away_team, matches, bundle, odds=fixture_odds)
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
            merged.append({
                **fx,
                "probabilities": pred["probabilities"],
                "report": pred.get("report"),
                "genai_prediction": pred.get("genai_prediction"),
            })
        else:
            reason = NO_PREDICTION_REASONS.get(fx["status"], "예측 데이터 없음")
            merged.append({**fx, "probabilities": None, "no_prediction_reason": reason})
    return merged


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


def _sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""
    items = "".join(
        f'<li><a href="{_safe_url(s["url"])}" target="_blank" rel="noopener noreferrer">{html.escape(s["title"])}</a></li>'
        for s in sources
    )
    return f'<ul class="ai-report-sources">{items}</ul>'


def _points_html(points: list[dict]) -> str:
    """핵심 포인트를 기존 승부 예측 서비스처럼 태그 칩 + 짧은 한 줄 리스트로 렌더링한다
    — Gemini가 통으로 뽑아주는 긴 문단을 그대로 꽂으면 카드마다 줄글이 늘어져 플랫폼
    서비스보다는 블로그 포스트처럼 보인다."""
    items = "".join(
        f'<li class="ai-point"><span class="ai-tag ai-tag-{_TAG_SLUGS.get(p["tag"], "other")}">{html.escape(p["tag"])}</span>'
        f'<span class="ai-point-text">{html.escape(p["text"])}</span></li>'
        for p in points
    )
    return f'<ul class="ai-points">{items}</ul>'


def _ai_details_html(label: str, headline: str | None, points: list[dict] | None, prose_text: str | None, sources: list[dict]) -> str:
    """headline/points(신규 저장 형식)가 있으면 칩+리스트로, 없으면 과거에 저장된
    문단 텍스트(prose_text)로 폴백 렌더링한다 — GCS에 이미 구버전 형식으로 저장된
    과거 예측 기록도 계속 보여야 하기 때문."""
    if headline or points:
        body_html = (f'<p class="ai-headline">{html.escape(headline)}</p>' if headline else "") + (
            _points_html(points) if points else ""
        )
    elif prose_text:
        # text/reasoning은 Gemini가 생성한 텍스트라 html.escape 없이 그대로 꽂으면 삽입된
        # HTML/스크립트가 대시보드에서 그대로 실행될 수 있다(모델이 마크업을 흉내 내는
        # 경우가 실제로 있다).
        body_html = f'<div class="ai-report-prose">{html.escape(prose_text)}</div>'
    else:
        return ""
    return (
        '<details class="ai-report">'
        f'<summary class="ai-report-label">{label}</summary>'
        f'<div class="ai-report-body">{body_html}</div>'
        f'{_sources_html(sources)}'
        '</details>'
    )


def _report_html(report: dict | str | None) -> str:
    if not report:
        return ""
    if isinstance(report, str):
        # GCS에 이미 저장된 과거 형식(리포트가 문자열 하나였던 버전) 호환
        report = {"text": report, "sources": []}
    # 카드마다 리포트 길이가 들쭉날쭉해 그리드가 깨지는 걸 막기 위해 기본은 접어두고,
    # 근거 링크를 같이 보여줘 "AI가 최신 정보를 반영했다"는 말을 사용자가 직접 검증할 수 있게 한다.
    return _ai_details_html(
        "AI 프리뷰 (탭하여 펼치기)",
        report.get("headline"),
        report.get("points"),
        report.get("text"),
        report.get("sources") or [],
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


def _genai_block_html(genai: dict, heading: str = "GenAI 예측 (최신 뉴스 반영)") -> str:
    """GenAI 예측을 통계 모델 예측 아래에 나란히 붙인다 — 대체가 아니라 병기용이라
    자체 확률 바 + 근거를 별도 블록으로 감싼다. 통계 예측 자체가 없는 경기(배당률
    미공개)에서는 heading을 다르게 줘서 "이게 유일한 예측"이라는 걸 구분해준다."""
    bar_html = _prob_bar_html(genai["probabilities"], heading)
    reasoning_html = _ai_details_html(
        "GenAI 예측 근거 (탭하여 펼치기)",
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
        body = f'<p class="no-pred">{reason}</p>'
        card_class = "match-card finished-card state-done" if has_score else "match-card no-pred-card state-nopred"
        return f'<div class="{card_class}">{header}{body}</div>'

    if probs is not None:
        if has_score:
            stat_heading = "경기 전 예측"
        elif genai:
            stat_heading = "통계 모델 예측"
        else:
            stat_heading = ""
        body = _prob_bar_html(probs, stat_heading)
    else:
        body = ""

    if genai:
        # 종료된 경기도 "당시 예측을 어떤 근거로 했는지"는 결과와 함께 유지해야 하므로 생략하지
        # 않는다 — 특히 배당률이 끝까지 안 열려 GenAI가 유일한 예측이었던 경기는, 숨기면 예측
        # 기록 자체가 통째로 사라진다. 대신 지난 예측이라는 걸 heading으로 구분해준다.
        if has_score:
            genai_heading = "경기 전 GenAI 예측"
        elif probs is not None:
            genai_heading = "GenAI 예측 (최신 뉴스 반영)"
        else:
            genai_heading = "GenAI 예측 (배당률 공개 전, 뉴스 기반)"
        body += _genai_block_html(genai, genai_heading)

    body += _report_html(p.get("report"))
    card_class = "match-card finished-card state-done" if has_score else "match-card state-upcoming"
    return f'<div class="{card_class}">{header}{body}</div>'


def _render_dashboard_html(
    view_label: str,
    payload: dict | None,
    available_dates: list[str],
    active_date: str | None,
    round_summary: dict | None = None,
) -> str:
    view_label = html.escape(view_label)  # date 쿼리 파라미터가 그대로 들어올 수 있음 (반사형 XSS 방지)
    date_links = "".join(
        f'<a class="date-link{" active" if d == active_date else ""}" href="/dashboard?date={d}">{d}</a>'
        for d in available_dates
    ) or '<span class="empty">기록된 날짜가 없습니다.</span>'

    summary_html = ""
    if round_summary:
        upcoming = round_summary["total"] - round_summary["finished"]
        summary_html = f"""<div class="summary-strip">
    <span class="summary-chip"><b>{round_summary['total']}</b>경기</span>
    <span class="summary-chip done"><b>{round_summary['finished']}</b>종료</span>
    <span class="summary-chip live"><b>{upcoming}</b>예정</span>
  </div>"""

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
  .ai-headline {{ margin: 0 0 10px; font-weight: 700; color: var(--text); line-height: 1.45; }}
  .ai-points {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 7px; }}
  .ai-point {{ display: flex; align-items: baseline; gap: 8px; }}
  .ai-point-text {{ color: var(--text-dim); line-height: 1.5; }}
  .ai-tag {{
    flex-shrink: 0; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.01em;
    padding: 2px 8px; border-radius: 999px; background: var(--surface-2);
    color: var(--text-faint); border: 1px solid var(--border);
  }}
  .ai-tag-injury {{ color: var(--away); border-color: rgba(239, 92, 138, 0.35); }}
  .ai-tag-form {{ color: var(--home); border-color: rgba(91, 141, 239, 0.35); }}
  .ai-tag-tactics {{ color: #d8a44c; border-color: rgba(216, 164, 76, 0.35); }}
  .ai-tag-referee {{ color: var(--win); border-color: rgba(52, 211, 153, 0.35); }}
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
</style>
</head>
<body>
  <header>
    <img class="emblem" src="https://crests.football-data.org/PL.png" alt="" loading="lazy">
    <div class="heading">
      <h1>EPL 승부 예측</h1>
      <p>매일 자동으로 갱신되는 프리미어리그 승부 예측 · AI 프리뷰 (한국 시간 기준)</p>
    </div>
  </header>
  <div class="dates">{date_links}</div>
  <h2>{view_label}</h2>
  {summary_html}
  {body}
</body>
</html>"""


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
        merged = _merge_round_fixtures(fixtures, predictions_by_id)
        generated_at = stored.get("generated_at") if stored else None
        payload = {"predictions": merged, "generated_at": generated_at} if merged else None
        return _render_dashboard_html(date, payload, available_dates, active_date=date)

    matchday_info = load_matchday_info()
    predictions_by_id, generated_at = _collect_matchday_predictions(matchday_info["dates"])
    merged = _merge_round_fixtures(matchday_info["fixtures"], predictions_by_id)
    payload = {"predictions": merged, "generated_at": generated_at} if merged else None
    view_label = f"{matchday_info['matchday']}라운드" if matchday_info["matchday"] else "오늘"
    round_summary = {
        "total": len(matchday_info["fixtures"]),
        "finished": sum(1 for f in matchday_info["fixtures"] if f["status"] == "FINISHED"),
    } if matchday_info["fixtures"] else None
    return _render_dashboard_html(view_label, payload, available_dates, active_date=None, round_summary=round_summary)
