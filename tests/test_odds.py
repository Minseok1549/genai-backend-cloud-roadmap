import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import odds as odds_module  # noqa: E402


RECORDS = [{"home_team": "A", "away_team": "B", "odds_p_home": 0.5, "odds_p_draw": 0.3,
            "odds_p_away": 0.2, "bookmaker_count": 3}]


def test_devig_removes_overround(tmp_path):
    """북메이커 마진을 제거해 세 확률의 합이 정확히 1이 되어야 한다 — 이 정규화가 틀리면
    모든 예측 확률이 조용히 1을 넘거나 못 미친다."""
    p_home, p_draw, p_away = odds_module._devig(2.0, 3.5, 4.0)
    assert abs(p_home + p_draw + p_away - 1.0) < 1e-9
    assert p_home > p_draw > p_away  # 배당률이 낮을수록 확률이 높다


def test_fetch_upcoming_odds_backs_off_and_reuses_stale_cache(tmp_path, monkeypatch):
    """API가 실패하면 TTL 지난 캐시를 stale로 표시해 돌려주고, backoff 동안은 다시 부르지
    않는다. 실패해도 캐시 파일의 mtime은 그대로라 계속 "만료" 상태로 남으므로, 실패 사실을
    기억하지 않으면 API가 죽은 동안 들어오는 모든 요청이 각각 쿼터(무료 티어 월 500회)를
    한 번씩 태운다."""
    cache = tmp_path / "odds_live_cache.json"
    cache.write_text(json.dumps(RECORDS))
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", cache)
    monkeypatch.setattr(odds_module, "LIVE_CACHE_TTL_SECONDS", 0)  # 즉시 만료 상태로 만든다
    monkeypatch.setattr(odds_module, "_last_failure_at", 0.0)

    calls = []

    def failing_fetch(api_key):
        calls.append(api_key)
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(odds_module, "_fetch_live_odds_from_api", failing_fetch)

    first = odds_module.fetch_upcoming_odds(api_key="k")
    assert first[("A", "B")]["stale"] is True
    assert len(calls) == 1

    second = odds_module.fetch_upcoming_odds(api_key="k")
    assert second[("A", "B")]["stale"] is True
    assert len(calls) == 1  # backoff 중에는 API를 부르지 않는다


def test_fetch_upcoming_odds_raises_during_backoff_without_cache(tmp_path, monkeypatch):
    """돌려줄 캐시가 아예 없으면 예외를 내지만, 그것도 API를 다시 부르지 않고 내야 한다."""
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", tmp_path / "none.json")
    monkeypatch.setattr(odds_module, "_last_failure_at", 0.0)

    calls = []

    def failing_fetch(api_key):
        calls.append(api_key)
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(odds_module, "_fetch_live_odds_from_api", failing_fetch)

    with pytest.raises(RuntimeError):
        odds_module.fetch_upcoming_odds(api_key="k")
    assert len(calls) == 1

    with pytest.raises(RuntimeError, match="backoff"):
        odds_module.fetch_upcoming_odds(api_key="k")
    assert len(calls) == 1


def test_api_key_from_environment_is_stripped(monkeypatch):
    """Secret Manager에 키를 넣을 때 파일 끝 개행이 같이 들어가는 일이 흔하다. 이 키는
    쿼리 문자열에 붙기 때문에 개행이 %0A로 인코딩돼 401이 된다 — .env 경로만 strip하고
    있어서 로컬에서는 멀쩡하고 배포 환경에서만 깨졌다."""
    monkeypatch.setenv("ODDS_API_KEY", "abc123\n")
    assert odds_module.load_odds_api_key() == "abc123"


def test_api_key_error_does_not_leak_the_key(monkeypatch):
    """requests의 HTTPError 메시지에는 요청 URL이 그대로 들어간다. 그 예외를 로그에 남기면
    쿼리 문자열에 실린 API 키가 평문으로 로그에 적힌다."""
    class FakeResp:
        ok = False
        status_code = 401

    monkeypatch.setattr(odds_module.requests, "get", lambda *a, **kw: FakeResp())

    with pytest.raises(Exception) as excinfo:
        odds_module._fetch_live_odds_from_api("super-secret-key")
    assert "super-secret-key" not in str(excinfo.value)
    assert "401" in str(excinfo.value)
