"""배당률 기반 예측 신호.

실험으로 확인한 사실: 북메이커 배당률은 이미 부상·폼·전술 등 공개정보를 우리가 만든
통계모델(팀 폼 롤링 평균, Elo)보다 훨씬 효율적으로 종합한 결과다. 우리 자체 피처와
결합해서 다시 학습하면 오히려 정확도가 떨어지고(노이즈 추가), 배당률의 내재확률을
그대로/가볍게 재보정해서 쓰는 쪽이 낫다 — 전체 EPL 26년 이력(2000~2026) 기준
accuracy 55%대, 기존 폼 기반 모델은 40%대.

과거(학습용)와 미래(예측용) 배당률은 출처가 다르다:
- 과거: GitHub에 미러링된 football-data.co.uk 데이터(2000~현재, Bet365 배당률 결측
  거의 없음)를 한 번 받아 로컬에 캐시한다. 결과가 이미 확정된 과거 경기라 팀명을
  프로젝트 정식 명칭으로 맞출 필요가 없다 — odds_p_*와 result만으로 재보정 모델을
  학습하면 되고, 다른 모듈의 team_name 체계와 조인하지 않는다.
- 미래: The Odds API에서 아직 열리지 않은 경기의 실시간 배당률을 받는다. 여러
  북메이커의 배당률을 평균해 단일 북메이커보다 안정적인 확률을 만든다. 이쪽은
  predict_match() 호출자(canonical 팀명)와 맞춰야 하므로 이름 매핑이 필요하다.
"""
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ODDS_HISTORY_PATH = RAW_DIR / "odds_history.csv"
ODDS_LIVE_CACHE_PATH = RAW_DIR / "odds_live_cache.json"
LIVE_CACHE_TTL_SECONDS = 6 * 3600  # fetch_data.py의 CACHE_TTL_SECONDS와 동일 — 무료 티어(월 500회) 보호
FAILURE_BACKOFF_SECONDS = 15 * 60  # API 호출이 실패한 뒤 이 시간 동안은 다시 부르지 않는다
_fetch_lock = threading.Lock()  # 같은 프로세스 내 동시 요청이 캐시 미스 시 각자 API를 부르는 걸 방지
_last_failure_at = 0.0  # 마지막 API 실패 시각(monotonic 아님 — 프로세스 재시작 시 자연히 초기화)

HISTORY_SOURCE_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

ODDS_FEATURE_NAMES = ["odds_p_home", "odds_p_draw", "odds_p_away"]

# The Odds API의 팀명 표기를 프로젝트 정식 명칭(football-data.org 기준)으로 맞춘다.
# data.py로 캐시된 팀명과 다르게 쓰는 것만 나열 — 나머지는 이미 동일 표기.
ODDS_API_TEAM_MAP = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton and Hove Albion": "Brighton & Hove Albion FC",
    "Sunderland": "Sunderland AFC",
    "Hull City": "Hull City AFC",
}
_FC_SUFFIX_TEAMS = {
    "Arsenal", "Aston Villa", "Brentford", "Burnley", "Chelsea", "Coventry City",
    "Crystal Palace", "Everton", "Fulham", "Ipswich Town", "Leeds United",
    "Leicester City", "Liverpool", "Luton Town", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Sheffield United", "Southampton",
    "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers",
}
ODDS_API_TEAM_MAP.update({t: f"{t} FC" for t in _FC_SUFFIX_TEAMS})


def load_odds_api_key() -> str:
    """환경 변수 쪽 값도 strip한다. Secret Manager에 키를 넣을 때 파일 끝 개행이 같이 들어가는
    일이 흔하고, 이 키는 쿼리 문자열에 붙기 때문에 개행이 %0A로 인코딩돼 401을 받는다.
    아래 .env 경로는 이미 strip을 하고 있어서 로컬에서는 증상이 안 나타나고, 배포 환경에서만
    조용히 깨진다 — 실제로 프로덕션이 이 이유로 배당률을 못 받고 있었다."""
    env_key = os.environ.get("ODDS_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("ODDS_API_KEY not found in environment or .env")


def _devig(home, draw, away):
    """북메이커 마진(overround)을 제거해 세 확률의 합이 1이 되도록 정규화한다."""
    inv_h, inv_d, inv_a = 1 / home, 1 / draw, 1 / away
    total = inv_h + inv_d + inv_a
    return inv_h / total, inv_d / total, inv_a / total


def ensure_odds_history_cached() -> None:
    """과거 배당률 캐시가 없으면 GitHub 미러에서 한 번 받아온다. 완결 시즌 결과처럼
    바뀌지 않는 과거 데이터라 TTL 없이 존재 여부만 확인한다(fetch_data.py의
    COMPLETED_SEASONS와 같은 방침)."""
    if ODDS_HISTORY_PATH.exists():
        return

    resp = requests.get(HISTORY_SOURCE_URL, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    epl = df[df["Division"] == "E0"].copy()
    epl = epl.dropna(subset=["OddHome", "OddDraw", "OddAway", "FTResult"])

    p_home, p_draw, p_away = _devig(epl["OddHome"], epl["OddDraw"], epl["OddAway"])
    out = pd.DataFrame({
        "date": pd.to_datetime(epl["MatchDate"]),
        "result": epl["FTResult"].map({"H": "HOME_TEAM", "D": "DRAW", "A": "AWAY_TEAM"}),
        "odds_p_home": p_home,
        "odds_p_draw": p_draw,
        "odds_p_away": p_away,
    })
    out = out.sort_values("date").reset_index(drop=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = ODDS_HISTORY_PATH.with_suffix(f".tmp{os.getpid()}-{uuid.uuid4().hex}")
    out.to_csv(tmp_path, index=False)
    os.replace(tmp_path, ODDS_HISTORY_PATH)


def load_odds_history() -> pd.DataFrame:
    ensure_odds_history_cached()
    df = pd.read_csv(ODDS_HISTORY_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _fetch_live_odds_from_api(api_key: str) -> list[dict]:
    resp = requests.get(
        ODDS_API_URL,
        params={"apiKey": api_key, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=20,
    )
    # requests의 HTTPError 메시지에는 요청 URL이 그대로 들어간다. 이 API는 키를 쿼리
    # 문자열로 받으므로, 그 예외를 로그에 남기면 API 키가 평문으로 로그에 적힌다 — 실제로
    # Cloud Logging에 키가 남았다. 상태 코드만 남기고 URL은 버린다.
    if not resp.ok:
        raise RuntimeError(f"odds API returned HTTP {resp.status_code}")

    results = []
    for m in resp.json():
        home_raw, away_raw = m["home_team"], m["away_team"]
        probs = []
        for bk in m.get("bookmakers", []):
            outcomes = {}
            for market in bk.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for o in market["outcomes"]:
                    outcomes[o["name"]] = o["price"]
            if home_raw in outcomes and away_raw in outcomes and "Draw" in outcomes:
                probs.append(_devig(outcomes[home_raw], outcomes["Draw"], outcomes[away_raw]))
        if not probs:
            continue
        avg = pd.DataFrame(probs, columns=["p_home", "p_draw", "p_away"]).mean()
        results.append({
            "home_team": ODDS_API_TEAM_MAP.get(home_raw, home_raw),
            "away_team": ODDS_API_TEAM_MAP.get(away_raw, away_raw),
            "odds_p_home": avg["p_home"],
            "odds_p_draw": avg["p_draw"],
            "odds_p_away": avg["p_away"],
            "bookmaker_count": len(probs),
        })
    return results


def _read_live_cache() -> list[dict] | None:
    if not ODDS_LIVE_CACHE_PATH.exists():
        return None
    return json.loads(ODDS_LIVE_CACHE_PATH.read_text())


def _to_odds_map(records: list[dict], stale: bool) -> dict[tuple[str, str], dict]:
    """레코드 리스트를 (home_team, away_team) 키 dict로 바꾸면서 stale 여부를 얹는다.
    stale=True는 "TTL 지난 캐시를 API 장애로 어쩔 수 없이 재사용했다"는 뜻 — 호출자가
    이걸 신선한 배당률과 구분해서, 오래된 숫자로 이미 저장된 최신 예측을 되돌리는 걸
    막을 수 있게 한다."""
    return {(r["home_team"], r["away_team"]): {**r, "stale": stale} for r in records}


def fetch_upcoming_odds(api_key: str | None = None) -> dict[tuple[str, str], dict]:
    """예정된 EPL 경기의 배당률을 (home_team, away_team) -> 확률 dict로 반환한다.
    캐시가 신선하면 API를 호출하지 않는다 — 무료 티어(월 500회) 보호.

    주의: 이 캐시는 프로세스(컨테이너) 로컬이다. Cloud Run이 여러 인스턴스로 스케일
    아웃하면 인스턴스별로 각자 캐시를 채우므로, TTL이 보장하는 건 "인스턴스 하나가
    같은 시간대에 중복 호출하지 않는다"까지다 — 인스턴스 전체의 총 호출량까지 이
    캐시만으로 제한할 수는 없다. 트래픽이 늘면 공유 캐시(Firestore/Redis 등)가 필요하다.
    """
    _lock_free_cached = _read_live_cache()
    if _lock_free_cached is not None:
        age = time.time() - ODDS_LIVE_CACHE_PATH.stat().st_mtime
        if age < LIVE_CACHE_TTL_SECONDS:
            return _to_odds_map(_lock_free_cached, stale=False)

    global _last_failure_at
    with _fetch_lock:
        # 락을 얻는 동안 다른 스레드가 이미 갱신했을 수 있으니 다시 확인한다.
        cached = _read_live_cache()
        if cached is not None:
            age = time.time() - ODDS_LIVE_CACHE_PATH.stat().st_mtime
            if age < LIVE_CACHE_TTL_SECONDS:
                return _to_odds_map(cached, stale=False)

        # 직전 호출이 실패했다면 backoff 동안은 아예 부르지 않는다. 실패해도 캐시 파일의
        # mtime은 그대로라 TTL이 계속 만료 상태로 남는데, 이 게이트가 없으면 API가 죽어
        # 있는 동안 들어오는 모든 요청이 각각 한 번씩 API를 다시 때려 쿼터를 태운다
        # (무료 티어 월 500회) — "재시도 대신 오래된 캐시를 쓴다"는 게 실제로 성립하려면
        # 실패 사실 자체를 기억해야 한다. 캐시가 아예 없으면 돌려줄 값이 없어 예외를 내지만,
        # 그것도 API를 다시 부르지 않고 내는 쪽이 쿼터를 아낀다.
        if time.time() - _last_failure_at < FAILURE_BACKOFF_SECONDS:
            if cached is not None:
                return _to_odds_map(cached, stale=True)
            raise RuntimeError("배당률 API 호출이 최근 실패해 backoff 중입니다")

        try:
            api_key = api_key or load_odds_api_key()
            results = _fetch_live_odds_from_api(api_key)
        except Exception:
            _last_failure_at = time.time()
            if cached is not None:
                # API 장애/쿼터 소진 시 새로 호출을 반복하는 대신 오래된 캐시라도 쓴다
                # — 매 요청마다 재시도해서 쿼터를 더 태우는 것보다 낫다. 다만 이 숫자는
                # TTL이 지난 걸 알고 쓰는 거라 stale=True로 표시해 호출자가 구분하게 한다.
                return _to_odds_map(cached, stale=True)
            raise

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = ODDS_LIVE_CACHE_PATH.with_suffix(f".tmp{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}")
        tmp_path.write_text(json.dumps(results))
        os.replace(tmp_path, ODDS_LIVE_CACHE_PATH)
        return _to_odds_map(results, stale=False)
