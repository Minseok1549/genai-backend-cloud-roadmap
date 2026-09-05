"""football-data.org에서 EPL(PL) 시즌별 경기 데이터를 받아 data/raw/에 캐시한다.

완결 시즌(COMPLETED_SEASONS)은 결과가 절대 바뀌지 않으므로 한 번 받으면 재수신하지 않는다.
진행 중인 시즌(CURRENT_SEASON)은 새 경기가 계속 끝나므로 캐시가 CACHE_TTL_SECONDS보다
오래되면 다시 받아온다 — /predict가 항상 최신 팀 폼을 반영하도록 하기 위함.
"""
import json
import os
import threading
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
COMPLETED_SEASONS = [2023, 2024, 2025]  # 무료 티어에서 접근 가능한 완결 시즌
CURRENT_SEASON = 2026  # 진행 중 — 계속 갱신 필요
CACHE_TTL_SECONDS = 6 * 3600
API_URL = "https://api.football-data.org/v4/competitions/PL/matches"
VALID_RESULTS = {"HOME_TEAM", "AWAY_TEAM", "DRAW"}


def load_api_key() -> str:
    # 클라우드 배포 시 시크릿은 보통 환경변수로 주입된다 — .env는 로컬 개발용 fallback으로만 쓴다.
    env_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if env_key:
        return env_key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("FOOTBALL_DATA_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("FOOTBALL_DATA_API_KEY not found in environment or .env")


def fetch_season(season: int, api_key: str) -> dict:
    resp = requests.get(
        API_URL,
        headers={"X-Auth-Token": api_key},
        params={"season": season},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def validate_matches_payload(data: dict) -> None:
    """API 응답을 캐시에 쓰기 전에 검증한다. 여기서 걸러야 손상된 캐시가 이후
    load_matches()/latest_team_form()에서 500(KeyError/NaN)으로 터지는 걸 막는다."""
    if not isinstance(data, dict) or not isinstance(data.get("matches"), list):
        raise ValueError("API 응답에 유효한 matches 목록이 없습니다")
    for m in data["matches"]:
        if m.get("status") != "FINISHED":
            continue
        score = m.get("score", {}).get("fullTime", {})
        if score.get("home") is None or score.get("away") is None:
            raise ValueError(f"FINISHED 경기에 score가 없습니다: match_id={m.get('id')}")
        if m.get("score", {}).get("winner") not in VALID_RESULTS:
            raise ValueError(f"FINISHED 경기에 유효한 winner가 없습니다: match_id={m.get('id')}")


def is_cached_file_valid(path: Path) -> bool:
    try:
        validate_matches_payload(json.loads(path.read_text()))
        return True
    except (ValueError, json.JSONDecodeError):
        return False


def ensure_season_cached(season: int, api_key: str) -> bool:
    """캐시가 없거나(완결/진행 시즌 공통) 오래됐으면(진행 시즌만) 새로 받아온다.

    반환값: 실제로 네트워크 fetch를 했으면 True (호출자가 rate-limit sleep 여부를 판단하는 데 씀).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"matches_{season}.json"

    if out_path.exists() and is_cached_file_valid(out_path):
        if season in COMPLETED_SEASONS:
            return False  # 완결 시즌은 결과가 바뀌지 않음
        age = time.time() - out_path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return False

    data = fetch_season(season, api_key)
    validate_matches_payload(data)  # 손상된 응답이면 여기서 예외 -> 기존 캐시 보존

    # 동시 요청이 같은 파일을 읽는 도중에 truncate된 내용을 보지 않도록 같은 디렉터리에
    # 임시로 쓴 뒤 os.replace()로 원자적 교체한다. FastAPI의 동기 핸들러는 스레드풀에서
    # 병렬 실행되므로 PID만으로는 같은 프로세스 내 동시 호출이 같은 임시 파일에 쓸 수
    # 있다 — 스레드 ID와 uuid를 더해 호출마다 고유한 파일명을 보장한다.
    tmp_path = out_path.with_suffix(f".tmp{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}")
    tmp_path.write_text(json.dumps(data))
    os.replace(tmp_path, out_path)
    return True


def ensure_all_seasons_cached(api_key: str | None = None) -> None:
    api_key = api_key or load_api_key()
    for season in COMPLETED_SEASONS + [CURRENT_SEASON]:
        fetched = ensure_season_cached(season, api_key)
        if fetched:
            time.sleep(1)  # free tier: 10 calls/min — 실제로 호출했을 때만 대기


def main() -> None:
    ensure_all_seasons_cached()
    for season in COMPLETED_SEASONS + [CURRENT_SEASON]:
        path = RAW_DIR / f"matches_{season}.json"
        count = len(json.loads(path.read_text()).get("matches", []))
        print(f"season={season}: {count} matches cached at {path}")


if __name__ == "__main__":
    main()
