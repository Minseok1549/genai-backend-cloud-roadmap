import json
import sys
import time as real_time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fetch_data  # noqa: E402


class _NoSleepTime:
    """time.sleep만 무력화하고 time()은 그대로 쓰는 얇은 대역 — rate-limit 대기(시즌당 1초)로
    테스트가 느려지는 걸 막는다."""

    def sleep(self, seconds):
        pass

    def time(self):
        return real_time.time()


def _valid_payload() -> dict:
    return {"matches": [{"id": 1, "status": "SCHEDULED", "utcDate": "2026-09-20T14:00:00Z",
                         "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}}]}


def test_ensure_all_seasons_cached_backs_off_after_failure(tmp_path, monkeypatch):
    """호출이 실패하면 일정 시간 동안 다시 부르지 않는다. /predict·/dashboard가 요청마다
    이 함수를 부르므로, 게이트가 없으면 토큰이 만료된 동안 들어오는 모든 요청이 시즌 4개를
    각각 다시 때리고 실패당 1초씩 sleep까지 해서 Cloud Run 과금 시간도 같이 늘어난다."""
    monkeypatch.setattr(fetch_data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch_data, "_last_failure_at", 0.0)
    monkeypatch.setattr(fetch_data, "time", _NoSleepTime())

    calls = []

    def failing_fetch(season, api_key):
        calls.append(season)
        raise RuntimeError("Your API token is invalid.")

    monkeypatch.setattr(fetch_data, "fetch_season", failing_fetch)

    with pytest.raises(RuntimeError):
        fetch_data.ensure_all_seasons_cached(api_key="k")
    assert len(calls) == 1  # 첫 시즌에서 실패하면 나머지 시즌까지 이어서 때리지 않는다

    with pytest.raises(RuntimeError, match="backoff"):
        fetch_data.ensure_all_seasons_cached(api_key="k")
    assert len(calls) == 1  # backoff 중에는 API를 아예 부르지 않는다


def test_ensure_all_seasons_cached_skips_api_when_all_fresh(tmp_path, monkeypatch):
    """캐시가 전부 신선하면 네트워크를 건드리지 않는다 — 요청마다 불리는 함수라 이 경로가
    기본이어야 한다."""
    monkeypatch.setattr(fetch_data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch_data, "_last_failure_at", 0.0)
    for season in fetch_data.COMPLETED_SEASONS + [fetch_data.CURRENT_SEASON]:
        (tmp_path / f"matches_{season}.json").write_text(json.dumps(_valid_payload()))

    def fail_fetch(season, api_key):
        raise AssertionError("캐시가 신선하면 API를 부르면 안 된다")

    monkeypatch.setattr(fetch_data, "fetch_season", fail_fetch)

    fetch_data.ensure_all_seasons_cached(api_key="k")


def test_is_cached_file_valid_parses_only_once_for_unchanged_file(tmp_path, monkeypatch):
    """검증은 파일을 통째로 파싱한다(시즌당 약 400KB, 4시즌). 요청마다 다시 파싱하면 요청
    하나가 수 MB를 파싱하게 되고 Cloud Run은 CPU 시간으로 과금되니 그대로 비용이다."""
    path = tmp_path / "matches_2026.json"
    path.write_text(json.dumps(_valid_payload()))
    fetch_data._validated_cache.pop(path, None)

    parse_count = 0
    real_loads = fetch_data.json.loads

    def counting_loads(s):
        nonlocal parse_count
        parse_count += 1
        return real_loads(s)

    monkeypatch.setattr(fetch_data.json, "loads", counting_loads)

    assert fetch_data.is_cached_file_valid(path)
    assert fetch_data.is_cached_file_valid(path)
    assert parse_count == 1

    # 내용이 바뀌면(캐시 교체는 항상 os.replace라 mtime도 바뀐다) 다시 검증한다.
    real_time.sleep(0.01)
    path.write_text(json.dumps(_valid_payload()))
    assert fetch_data.is_cached_file_valid(path)
    assert parse_count == 2


def test_missing_api_key_is_recorded_as_failure(tmp_path, monkeypatch):
    """키 자체가 없어서 실패한 경우도 backoff에 기록해야 한다 — 이것도 요청마다 반복해서
    재시도할 이유가 없는 실패다."""
    monkeypatch.setattr(fetch_data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch_data, "_last_failure_at", 0.0)

    def no_key():
        raise RuntimeError("FOOTBALL_DATA_API_KEY not found in environment or .env")

    monkeypatch.setattr(fetch_data, "load_api_key", no_key)

    with pytest.raises(RuntimeError, match="not found"):
        fetch_data.ensure_all_seasons_cached()
    with pytest.raises(RuntimeError, match="backoff"):
        fetch_data.ensure_all_seasons_cached()


def test_concurrent_requests_call_api_once_when_it_fails(tmp_path, monkeypatch):
    """동시 요청이 한꺼번에 들어와도 실패한 API는 한 번만 부른다. backoff 검사가 락 밖에만
    있으면, 첫 실패가 기록되기 전에 검사를 통과해 락에서 대기하던 요청들이 실패 직후 차례로
    다시 부른다 — 동시 요청 수만큼 직렬 재호출된다."""
    import threading

    monkeypatch.setattr(fetch_data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch_data, "_last_failure_at", 0.0)
    monkeypatch.setattr(fetch_data, "time", _NoSleepTime())

    calls = []
    # 스레드를 그냥 동시에 풀어놓으면 재현이 타이밍에 달린다. 첫 스레드가 락을 잡고 API
    # 호출 중인 상태로 붙잡아두고, 그 사이 나머지 두 스레드가 "락 밖 backoff 검사를 통과해
    # 락에서 대기"하는 상태까지 간 것을 확인한 뒤에 실패를 내게 한다 — 옛 코드(락 안 검사
    # 없음)에서는 이 상태에서 대기하던 두 스레드가 반드시 API를 다시 부른다.
    in_fetch = threading.Event()       # 첫 스레드가 API 호출에 진입함
    release_fetch = threading.Event()  # 첫 스레드의 실패를 여기서 풀어준다
    gate_lock = threading.Lock()
    passed_outer_gate = set()          # 락 밖 backoff 검사를 통과한 스레드들

    real_in_backoff = fetch_data._in_backoff

    def tracking_in_backoff():
        # 반환 뒤에 기록한다 — 기록된 스레드는 "이 시점의 _last_failure_at으로 검사를 마쳤다"가
        # 보장돼야 하고, 첫 실패는 세 스레드가 모두 기록된 뒤에야 일어난다.
        result = real_in_backoff()
        with gate_lock:
            passed_outer_gate.add(threading.get_ident())
        return result

    def failing_fetch(season, api_key):
        calls.append(season)
        in_fetch.set()
        release_fetch.wait(timeout=10)
        raise RuntimeError("Your API token is invalid.")

    monkeypatch.setattr(fetch_data, "_in_backoff", tracking_in_backoff)
    monkeypatch.setattr(fetch_data, "fetch_season", failing_fetch)

    def worker():
        try:
            fetch_data.ensure_all_seasons_cached(api_key="k")
        except RuntimeError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(3)]
    threads[0].start()
    assert in_fetch.wait(timeout=10), "첫 스레드가 API 호출까지 가지 못했다"
    for t in threads[1:]:
        t.start()

    deadline = real_time.monotonic() + 10
    while real_time.monotonic() < deadline:
        with gate_lock:
            if len(passed_outer_gate) == 3:
                break
        real_time.sleep(0.01)
    with gate_lock:
        assert len(passed_outer_gate) == 3, "나머지 스레드가 락 밖 검사까지 오지 않았다"

    release_fetch.set()
    for t in threads:
        t.join(timeout=10)

    assert len(calls) == 1
