"""Gemini API로 킥오프 임박 시점의 독립적인 승부 예측(확률 + 근거)을 생성한다. Google Search
grounding으로 학습 데이터 컷오프 이후의 최신 선수/감독 소식을 반영하고, 우리 통계 모델이
계산한 확률을 참고값으로 프롬프트에 함께 넣는다.

Pro가 아니라 Flash 계열을 쓰는 이유: gemini-3.6-flash는 토큰 단가($0.75/$3.75 per 1M, 2026년
말까지 프로모션가)로 Pro(2.5.1-pro 기준 $2/$12대)보다 훨씬 싸면서 grounding·리포트 종합 품질은
충분하다. 애초 기본값이던 gemini-2.5-flash는 신규 API 키에는 더 이상 제공되지 않아(2026-09
확인, 404) 이 모델로 교체했다 — Google 권장 대체 모델이기도 하다. 필요하면 GEMINI_MODEL
환경변수로 다른 모델을 쓸 수 있다.
"""
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
REQUEST_TIMEOUT = 45  # Google Search grounding 왕복까지 포함하므로 일반 API 호출보다 여유를 둔다


def load_gemini_api_key() -> str:
    # 클라우드 배포 시 시크릿은 보통 환경변수로 주입된다 — .env는 로컬 개발용 fallback으로만 쓴다.
    # strip 이유: 시크릿에 파일 끝 개행이 같이 들어가는 일이 흔한데, .env 경로만 strip하고
    # 있으면 로컬에서는 멀쩡하고 배포 환경에서만 인증이 깨진다(odds.py에서 실제로 발생).
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GEMINI_API_KEY not found in environment or .env")


# 기존 승부 예측 서비스(Sofascore/FotMob류)를 따라 프리뷰를 긴 문단이 아니라 한 줄 총평 +
# 태그가 붙은 핵심 포인트 목록으로 받는다 — 대시보드에서 줄글 대신 칩/리스트로 렌더링하기 위함.
_POINT_TAGS = ("부상", "폼", "전술", "주심", "기타")


def _extract_sources(candidate: dict) -> list[dict]:
    """grounding에 실제로 쓰인 웹 출처를 뽑는다 — 사용자가 AI 코멘트를 검증할 수 있으려면
    "최신 정보를 반영했다"는 말뿐 아니라 어디서 가져왔는지가 화면에 보여야 한다."""
    chunks = (candidate.get("groundingMetadata") or {}).get("groundingChunks") or []
    seen_urls = set()
    sources = []
    for chunk in chunks:
        web = chunk.get("web") or {}
        url = web.get("uri")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append({"title": web.get("title") or url, "url": url})
    return sources[:5]


def _extract_json_trailer(text: str, prefix: str) -> dict:
    """응답에 붙는 `{prefix}: {...}` 한 줄을 파싱한다. Gemini의 responseSchema(구조화
    출력)는 google_search tool과 함께 쓸 수 없어서, 프롬프트로 형식을 지정하고 텍스트에서
    뽑아내는 방식을 쓴다. 괄호 depth를 문자 단위로 세면 headline/point 텍스트 안에 우연히
    '{'나 '}'가 섞였을 때 어긋나므로, JSON 문자열 문법을 아는 JSONDecoder.raw_decode로
    닫는 지점을 찾는다. prefix도 headline/point 텍스트 안에 그대로 인용될 수 있어 아무 데서나
    찾지 않고, 프롬프트 지시대로 줄 맨 앞에서 시작하는 마지막 등장만 실제 trailer로 본다."""
    marker = f"{prefix}:"
    start = None
    search_from = 0
    while True:
        idx = text.find(marker, search_from)
        if idx == -1:
            break
        line_begin = text.rfind("\n", 0, idx) + 1
        if text[line_begin:idx].strip() == "":
            start = idx
        search_from = idx + 1
    if start is None:
        raise ValueError(f"Gemini 응답에서 {prefix}를 찾을 수 없습니다")
    brace_start = text.find("{", start)
    if brace_start == -1:
        raise ValueError(f"Gemini 응답에서 {prefix}를 찾을 수 없습니다")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, brace_start)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini 응답의 {prefix} JSON을 파싱할 수 없습니다: {e}") from e
    return obj


def _normalize_points(raw_points) -> list[dict]:
    """Gemini가 뽑아준 핵심 포인트를 태그가 유효한 값만 남겨 최대 4개로 정리한다.
    태그가 정해진 값이 아니면(Gemini가 지시를 어긴 경우) '기타'로 묶어 UI 렌더링이
    깨지지 않게 한다. raw_points가 지시를 무시하고 리스트가 아닌 형태(dict/숫자 등)로
    와도 예외 없이 빈 목록으로 처리해야 한다 — 이게 터지면 이미 파싱에 성공한 확률까지
    호출부에서 함께 버려지기 때문이다. 유효성 검사를 먼저 끝내고 나서 4개로 자른다 —
    앞쪽에 무효한 항목이 섞여 있다고 뒤쪽의 유효한 항목이 밀려나면 안 된다."""
    if not isinstance(raw_points, list):
        return []
    points = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        tag = item.get("tag") if item.get("tag") in _POINT_TAGS else "기타"
        points.append({"tag": tag, "text": text})
        if len(points) == 4:
            break
    return points


def _build_prediction_prompt(
    home_team: str, away_team: str, kickoff_utc: str, referee: str | None, stat_probabilities: dict | None
) -> str:
    referee_line = f"배정된 주심: {referee}" if referee else "배정된 주심 정보 없음"
    if stat_probabilities:
        home_pct = stat_probabilities["HOME_TEAM"] * 100
        draw_pct = stat_probabilities["DRAW"] * 100
        away_pct = stat_probabilities["AWAY_TEAM"] * 100
        stat_line = f"참고용 배당률 기반 확률(뉴스 반영 전): 홈승 {home_pct:.1f}% / 무 {draw_pct:.1f}% / 원정승 {away_pct:.1f}%"
    else:
        stat_line = "참고용 배당률 기반 확률: 아직 배당률이 공개되지 않아 없음 — 뉴스와 선수단 정보만으로 판단해야 함"
    tags = "/".join(_POINT_TAGS)
    return f"""다음 EPL 경기의 승부 확률을 킥오프 직전 시점 기준 최신 정보로 예측해줘.

경기: {home_team} vs {away_team}
킥오프(UTC): {kickoff_utc}
{referee_line}
{stat_line}

Google 검색으로 아래 내용을 최신 상태로 확인해서 반영해:
- 양 팀의 예상 선발 라인업, 주요 선수 부상/징계/컨디션 이슈
- 감독 관련 이슈(경질설, 전술 변화, 최근 발언 등)
- 최근 경기력 흐름(연승/연패, 최근 폼)

위 정보를 반영해 승부 확률을 예측하고, 15단어 이내의 짧은 한 줄 총평(headline)과 근거가
되는 핵심 포인트 2~4개를 뽑아줘. 각 포인트는 tag({tags} 중 하나)와 15단어 이내의 짧은
text로 구성해. 확인 안 되는 내용은 추측하지 말고 생략해.

다른 설명 문장 없이 아래 형식 그대로 한 줄로만 응답해:
PREDICTION_JSON: {{"home_win": 0.00, "draw": 0.00, "away_win": 0.00, "headline": "...", "points": [{{"tag": "...", "text": "..."}}]}}
확률 세 값은 0~1 사이 소수이고 합이 1이 되어야 해."""


def _extract_prediction(text: str) -> tuple[dict, str, list[dict]]:
    """응답의 PREDICTION_JSON을 파싱해 확률·총평(headline)·근거 포인트를 뽑는다."""
    raw = _extract_json_trailer(text, "PREDICTION_JSON")
    home, draw, away = float(raw["home_win"]), float(raw["draw"]), float(raw["away_win"])
    for name, value in (("home_win", home), ("draw", draw), ("away_win", away)):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Gemini 확률 값이 0~1 범위를 벗어났습니다: {name}={value}")
    total = home + draw + away
    if total <= 0 or not (0.9 - 1e-9 <= total <= 1.1 + 1e-9):
        raise ValueError(f"Gemini 확률 합이 비정상입니다: {total}")
    probabilities = {"HOME_TEAM": home / total, "DRAW": draw / total, "AWAY_TEAM": away / total}
    headline = str(raw.get("headline") or "").strip()
    points = _normalize_points(raw.get("points"))
    return probabilities, headline, points


def generate_genai_prediction(
    home_team: str, away_team: str, kickoff_utc: str, referee: str | None, stat_probabilities: dict | None
) -> dict:
    """킥오프 임박 시점에, 최신 뉴스/선수단 정보를 검색해 반영한 독립적인 승부 확률을
    만든다. 통계 예측이 있으면(stat_probabilities) 대시보드에 나란히 병기해서 보여주고,
    없으면(배당률이 아직 안 열린 경기) 이 예측이 유일한 예측으로 쓰인다 — 그 경우도
    프롬프트는 뉴스/선수단 정보만으로 판단하도록 안내한다."""
    api_key = load_gemini_api_key()
    prompt = _build_prediction_prompt(home_team, away_team, kickoff_utc, referee, stat_probabilities)
    resp = requests.post(
        API_URL,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    candidate = resp.json()["candidates"][0]
    text = "".join(p["text"] for p in candidate["content"]["parts"] if "text" in p).strip()
    if not text:
        raise RuntimeError("Gemini 응답에 텍스트가 없습니다")
    probabilities, headline, points = _extract_prediction(text)
    return {"probabilities": probabilities, "headline": headline, "points": points, "sources": _extract_sources(candidate)}
