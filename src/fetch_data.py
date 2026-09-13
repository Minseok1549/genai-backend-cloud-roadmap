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
FAILURE_BACKOFF_SECONDS = 15 * 60  # 호출이 실패한 뒤 이 시간 동안은 다시 부르지 않는다
_fetch_lock = threading.Lock()  # 같은 프로세스의 동시 요청이 각자 같은 시즌을 받아오는 걸 방지
_last_failure_at = 0.0  # 마지막 API 실패 시각(프로세스 재시작 시 자연히 초기화)


def load_api_key() -> str:
    # 클라우드 배포 시 시크릿은 보통 환경변수로 주입된다 — .env는 로컬 개발용 fallback으로만 쓴다.
    # strip 이유: 시크릿에 파일 끝 개행이 같이 들어가는 일이 흔한데, .env 경로만 strip하고
    # 있으면 로컬에서는 멀쩡하고 배포 환경에서만 인증이 깨진다(odds.py에서 실제로 발생).
    env_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
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
    if not resp.ok:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} for url {resp.url}: {resp.text[:500]}", response=resp
        )
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


_validated_cache: dict[Path, tuple[int, int]] = {}  # path -> 검증을 통과했을 때의 (mtime_ns, size)


def is_cached_file_valid(path: Path) -> bool:
    """캐시 파일이 온전한지 검사한다. 검사는 파일을 통째로 파싱하는데(시즌당 약 400KB, 4시즌),
    /predict·/dashboard가 요청마다 ensure_all_seasons_cached()를 부르므로 매번 다시 파싱하면
    요청 하나가 수 MB를 파싱하게 된다 — Cloud Run은 CPU 시간으로 과금되니 그대로 비용이다.
    파일이 (mtime, size) 그대로면 지난번 검증 결과를 재사용한다. 우리가 캐시를 쓸 때는 항상
    os.replace로 교체하므로 내용이 바뀌면 mtime도 반드시 바뀐다."""
    try:
        stat = path.stat()
    except OSError:
        return False
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    if _validated_cache.get(path) == fingerprint:
        return True
    try:
        validate_matches_payload(json.loads(path.read_text()))
    except (ValueError, json.JSONDecodeError):
        _validated_cache.pop(path, None)
        return False
    _validated_cache[path] = fingerprint
    return True


def current_season_cache_age_hours() -> float | None:
    """진행 중 시즌 캐시가 몇 시간 전 것인지 반환한다(캐시가 아예 없으면 None).
    갱신 실패 로그에 붙여서, 로그만 보고 "얼마나 오래된 데이터로 서빙 중인지" 알 수 있게 한다.

    파일 접근 오류도 None으로 삼킨다. 이 함수는 갱신 실패를 기록하는 except 블록에서 로그 인자로
    불리므로, 여기서 예외가 새어나가면 원래 실패 원인이 그 예외에 가려지고 "기존 캐시로 계속
    진행"해야 할 배치가 중단된다. 진단용으로 붙인 값이 진단 대상을 죽여서는 안 된다."""
    path = RAW_DIR / f"matches_{CURRENT_SEASON}.json"
    try:
        return round((time.time() - path.stat().st_mtime) / 3600, 1)
    except OSError:
        return None


def _in_backoff() -> bool:
    return time.time() - _last_failure_at < FAILURE_BACKOFF_SECONDS


def needs_fetch(season: int) -> bool:
    """이 시즌을 네트워크에서 받아와야 하는 상태인지만 판단한다(부수효과 없음).
    완결 시즌은 캐시가 온전하면 영구 재사용, 진행 시즌은 TTL이 지나면 갱신 대상이다."""
    out_path = RAW_DIR / f"matches_{season}.json"
    if not out_path.exists() or not is_cached_file_valid(out_path):
        return True
    if season in COMPLETED_SEASONS:
        return False  # 완결 시즌은 결과가 바뀌지 않음
    return time.time() - out_path.stat().st_mtime >= CACHE_TTL_SECONDS


def ensure_season_cached(season: int, api_key: str) -> bool:
    """캐시가 없거나(완결/진행 시즌 공통) 오래됐으면(진행 시즌만) 새로 받아온다.

    반환값: 실제로 네트워크 fetch를 했으면 True (호출자가 rate-limit sleep 여부를 판단하는 데 씀).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"matches_{season}.json"

    if not needs_fetch(season):
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
    """필요한 시즌만 받아온다. /predict·/dashboard가 요청마다 이 함수를 부르므로 두 가지를
    지킨다.

    1) 동시 요청 직렬화: 캐시가 만료된 순간 여러 요청이 동시에 들어오면 각자 같은 시즌을
       받아와 무료 티어 한도(분당 10회)를 쓸데없이 소모했다. 락 안에서 다시 확인해, 먼저
       들어온 요청이 갱신을 끝내면 나머지는 호출 없이 통과한다.
    2) 실패 backoff: 토큰 만료·장애로 호출이 실패하면 캐시 파일은 그대로라 계속 "갱신
       필요" 상태로 남는다. 이 게이트가 없으면 API가 죽어 있는 동안 들어오는 모든 요청이
       시즌 4개를 각각 다시 때리고, 실패한 호출마다 1초씩 sleep까지 해서 Cloud Run 과금
       시간(요청 지연)도 같이 늘어난다. 예외는 그대로 올려서 호출자가 기존 캐시로 서빙하며
       경고를 남길 수 있게 한다.
    """
    global _last_failure_at
    seasons = COMPLETED_SEASONS + [CURRENT_SEASON]
    if not any(needs_fetch(season) for season in seasons):
        return  # 정상 경로 — 전부 신선하면 락도 잡지 않는다

    if _in_backoff():
        raise RuntimeError("football-data API 호출이 최근 실패해 backoff 중입니다")

    with _fetch_lock:
        if not any(needs_fetch(season) for season in seasons):
            return  # 락을 기다리는 동안 다른 요청이 갱신을 끝냈다
        # backoff는 락 안에서 다시 확인한다. 락 밖 검사만 있으면, 첫 실패가 기록되기 전에
        # 검사를 통과해 락에서 대기하던 요청들이 실패 직후 차례로 API를 다시 부른다
        # (동시 요청 수만큼 직렬 재호출) — 실패를 기억하는 의미가 없어진다.
        if _in_backoff():
            raise RuntimeError("football-data API 호출이 최근 실패해 backoff 중입니다")
        try:
            api_key = api_key or load_api_key()
            for season in seasons:
                if ensure_season_cached(season, api_key):
                    time.sleep(1)  # free tier: 10 calls/min — 실제로 호출했을 때만 대기
        except Exception:
            # load_api_key() 실패(키 자체가 없음)도 여기 포함한다 — 이것도 요청마다 반복해서
            # 재시도할 이유가 없는 실패다.
            _last_failure_at = time.time()
            raise


def main() -> None:
    ensure_all_seasons_cached()
    for season in COMPLETED_SEASONS + [CURRENT_SEASON]:
        path = RAW_DIR / f"matches_{season}.json"
        count = len(json.loads(path.read_text()).get("matches", []))
        print(f"season={season}: {count} matches cached at {path}")


if __name__ == "__main__":
    main()
