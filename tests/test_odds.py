import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import odds as odds_module  # noqa: E402


# commence_time은 먼 미래로 둔다 — 이미 시작한 경기의 배당률은 걸러지기 때문에, 시각이
# 지나간 값이면 이 레코드를 쓰는 모든 테스트가 시간이 흐른 뒤 조용히 빈 결과를 받는다.
RECORDS = [{"home_team": "A", "away_team": "B", "odds_p_home": 0.5, "odds_p_draw": 0.3,
            "odds_p_away": 0.2, "bookmaker_count": 3, "commence_time": "2099-01-01T00:00:00Z"}]


def cache_state(records, fetched_at=None, failed_at=None):
    """캐시 파일/공유 객체 한 건의 내용. 신선도를 파일 수정 시각이 아니라 내용 안의 '받아온
    시각'으로 판단하는 형식이라, 테스트도 같은 형식으로 써 준다."""
    return {
        "fetched_at": time.time() if fetched_at is None else fetched_at,
        "records": records,
        "failed_at": failed_at,
    }


class FakeSharedCache:
    """GCS 공유 캐시 대역. 실제 버킷 없이 "다른 인스턴스가 남겨둔 내용"을 흉내내고, 읽기/쓰기
    횟수를 세서 로컬 캐시가 GCS를 불필요하게 왕복하지 않는지도 확인할 수 있게 한다."""

    def __init__(self, state=None):
        self.state = state
        self.reads = 0
        self.writes = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(odds_module, "_read_shared_cache", self._read)
        monkeypatch.setattr(odds_module, "_write_shared_cache", self._write)
        return self

    def _read(self):
        self.reads += 1
        return self.state

    def _write(self, state):
        self.writes += 1
        self.state = state


@pytest.fixture(autouse=True)
def no_real_shared_cache(tmp_path, monkeypatch):
    """기본값은 "공유 캐시 설정이 없는 환경"이다. 버킷 이름이 없으면 GCS 경로가 통째로
    건너뛰어지므로, 이 환경 변수가 설정된 머신에서 테스트를 돌려도 실제 버킷을 읽지 않는다."""
    monkeypatch.delenv("PREDICTIONS_BUCKET", raising=False)
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", tmp_path / "odds_live_cache.json")


def test_devig_removes_overround(tmp_path):
    """북메이커 마진을 제거해 세 확률의 합이 정확히 1이 되어야 한다 — 이 정규화가 틀리면
    모든 예측 확률이 조용히 1을 넘거나 못 미친다."""
    p_home, p_draw, p_away = odds_module._devig(2.0, 3.5, 4.0)
    assert abs(p_home + p_draw + p_away - 1.0) < 1e-9
    assert p_home > p_draw > p_away  # 배당률이 낮을수록 확률이 높다


def test_fetch_upcoming_odds_backs_off_and_reuses_stale_cache(tmp_path, monkeypatch):
    """API가 실패하면 TTL 지난 캐시를 stale로 표시해 돌려주고, backoff 동안은 다시 부르지
    않는다. 실패해도 캐시의 '받아온 시각'은 그대로라 계속 "만료" 상태로 남으므로, 실패 사실을
    기억하지 않으면 API가 죽은 동안 들어오는 모든 요청이 각각 쿼터(무료 티어 월 500회)를
    한 번씩 태운다."""
    cache = tmp_path / "odds_live_cache.json"
    cache.write_text(json.dumps(cache_state(RECORDS)))
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", cache)
    monkeypatch.setattr(odds_module, "LIVE_CACHE_TTL_SECONDS", 0)  # 즉시 만료 상태로 만든다

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


def _payload(home_price):
    """h2h 응답 하나. 첫 북메이커의 홈 가격만 바꿔 이상치를 주입할 수 있게 한다."""
    def book(h, d, a):
        return {"markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": h}, {"name": "Draw", "price": d},
            {"name": "Chelsea", "price": a}]}]}
    return [{"home_team": "Arsenal", "away_team": "Chelsea",
             "bookmakers": [book(home_price, 3.5, 4.0), book(1.8, 3.6, 4.2)]}]


@pytest.mark.parametrize("bad_price", [0, None, 1.0, True, "1.8", float("inf")])
def test_one_bad_bookmaker_price_does_not_discard_the_others(monkeypatch, bad_price):
    """북메이커 한 곳의 가격이 이상해도 나머지 북메이커의 배당률은 살아야 한다.

    _devig는 1/가격을 계산하므로 0이나 null이 들어오면 예외가 난다. 그 예외가 루프 밖으로
    새면 이 경기뿐 아니라 같은 응답에 실린 모든 경기의 배당률이 함께 사라지고 15분 backoff까지
    걸린다 — 정상 북메이커 20곳의 가격이 이상치 하나 때문에 버려진다. 십진 배당률 1.0과
    문자열, int의 하위 타입이라 1로 통과해버리는 True, 그리고 "1보다 크다"를 통과하지만 확률
    합을 0으로 만들어 0으로 나누기를 일으키는 무한대도 같이 걸러야 한다."""
    class Resp:
        ok, status_code = True, 200
        def json(self): return _payload(bad_price)

    monkeypatch.setattr(odds_module.requests, "get", lambda *a, **k: Resp())
    results = odds_module._fetch_live_odds_from_api("key")

    assert len(results) == 1
    assert results[0]["bookmaker_count"] == 1  # 정상 북메이커 한 곳만 채택
    assert abs(sum(results[0][k] for k in odds_module.ODDS_FEATURE_NAMES) - 1.0) < 1e-9


def test_corrupt_live_cache_is_refetched_instead_of_raising(tmp_path, monkeypatch):
    """캐시 파일이 깨져 있으면 "캐시 없음"으로 보고 새로 받아와야 한다.

    이 읽기는 fetch_upcoming_odds의 try 블록 밖에서 먼저 일어나므로, 깨진 파일에서 예외를
    올리면 배당률 경로가 통째로 실패하고 새로 받아오는 시도조차 하지 않는다 — 파일이 한 번
    깨지면 스스로 복구되지 않는다."""
    cache = tmp_path / "odds_live_cache.json"
    cache.write_text("{truncated")
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", cache)
    monkeypatch.setattr(odds_module, "_fetch_live_odds_from_api", lambda api_key: RECORDS)

    result = odds_module.fetch_upcoming_odds(api_key="k")

    assert result[("A", "B")]["stale"] is False
    assert json.loads(cache.read_text())["records"] == RECORDS  # 깨진 내용이 정상 값으로 덮어써졌다


def test_fetched_records_keep_the_kickoff_time(monkeypatch):
    """받아온 레코드에는 킥오프 시각이 함께 남아야 한다 — 캐시는 최대 6시간(장애 시 그보다
    오래) 살아있어서, 저장 당시엔 시작 전이던 경기가 쓰는 시점엔 이미 진행 중일 수 있다."""
    class Resp:
        ok, status_code = True, 200
        def json(self):
            payload = _payload(1.9)
            payload[0]["commence_time"] = "2026-09-20T14:00:00Z"
            return payload

    monkeypatch.setattr(odds_module.requests, "get", lambda *a, **k: Resp())

    assert odds_module._fetch_live_odds_from_api("key")[0]["commence_time"] == "2026-09-20T14:00:00Z"


@pytest.mark.parametrize("commence_time", ["2020-01-01T00:00:00Z", None, "어제", "2026-09-20T14:00:00"])
def test_odds_for_matches_that_already_started_are_not_served(tmp_path, monkeypatch, commence_time):
    """이미 킥오프한 경기의 배당률은 예측에 쓰면 안 된다.

    배당률 제공사는 진행 중인 경기의 배당률도 함께 내려주는데, 그 숫자에는 현재 스코어가 이미
    반영돼 있다 — 후반에 0-2로 지고 있는 팀의 승리 확률이 낮게 나오는 건 예측이 아니라 중간
    결과 요약이다. 실제로 13시에 시작한 경기의 배당률이 13시 22분 캐시에 들어있었다.
    킥오프 시각이 없거나 형식이 깨져(시간대 없음 등) 시작 전인지 확인할 수 없는 레코드도
    같이 버린다 — 확인할 수 없는 배당률을 통과시키는 쪽이 더 위험하다."""
    cache = tmp_path / "odds_live_cache.json"
    record = {**RECORDS[0], "commence_time": commence_time}
    cache.write_text(json.dumps(cache_state([record])))
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", cache)

    assert odds_module.fetch_upcoming_odds(api_key="k") == {}


def test_cold_instance_reads_the_shared_cache_instead_of_calling_the_api(tmp_path, monkeypatch):
    """로컬 캐시가 빈 인스턴스는 API가 아니라 공유 캐시를 먼저 봐야 한다.

    이 서비스는 요청이 없으면 인스턴스가 0으로 줄어드는 구성이라, 콜드 스타트마다 빈 로컬
    캐시에서 시작한다. 공유 캐시가 없으면 그때마다 API를 한 번씩 부르게 되고, 그 횟수는
    트래픽에 따라 정해져서 무료 티어(월 500회)에 상한이 없어진다."""
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", tmp_path / "absent.json")
    shared = FakeSharedCache(cache_state(RECORDS)).install(monkeypatch)
    monkeypatch.setattr(odds_module, "_fetch_live_odds_from_api",
                        lambda api_key: pytest.fail("공유 캐시가 신선한데 API를 불렀다"))

    result = odds_module.fetch_upcoming_odds(api_key="k")

    assert result[("A", "B")]["stale"] is False
    assert shared.reads == 1
    # 다음 요청은 로컬 캐시로 처리돼야 한다 — 같은 인스턴스가 매 요청마다 GCS를 왕복할 이유는 없다.
    odds_module.fetch_upcoming_odds(api_key="k")
    assert shared.reads == 1


def test_copying_the_shared_cache_does_not_renew_its_freshness(tmp_path, monkeypatch):
    """공유 캐시를 로컬로 복사할 때 "받아온 시각"이 "복사한 시각"으로 바뀌면 안 된다.

    바뀌면 5시간 59분 된 배당률이 로컬에서 다시 6시간 신선한 것으로 되살아나, 최대 12시간
    묵은 숫자가 신선한 예측으로 저장된다. 시각을 파일 수정 시각이 아니라 내용 안에 담는
    이유가 이것이다."""
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", tmp_path / "local.json")
    five_hours_ago = time.time() - 5 * 3600
    FakeSharedCache(cache_state(RECORDS, fetched_at=five_hours_ago)).install(monkeypatch)
    monkeypatch.setattr(odds_module, "_fetch_live_odds_from_api", lambda api_key: RECORDS)

    odds_module.fetch_upcoming_odds(api_key="k")
    local = odds_module._read_local_cache()

    assert local["fetched_at"] == pytest.approx(five_hours_ago)
    assert odds_module._cache_age(local) > 5 * 3600


def test_a_failure_recorded_by_another_instance_stops_this_one_from_retrying(tmp_path, monkeypatch):
    """다른 인스턴스가 남긴 실패 기록도 backoff에 반영돼야 한다.

    실패 사실을 프로세스 변수로만 기억하면 인스턴스가 새로 뜰 때마다 backoff가 초기화된다 —
    쿼터가 소진된 상태에서 "15분에 한 번 재시도"가 "인스턴스마다 15분에 한 번"이 되고,
    스케일 아웃과 콜드 스타트가 그 배수를 정한다."""
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", tmp_path / "absent.json")
    just_failed = cache_state(RECORDS, fetched_at=time.time() - 7 * 3600, failed_at=time.time())
    FakeSharedCache(just_failed).install(monkeypatch)
    monkeypatch.setattr(odds_module, "_fetch_live_odds_from_api",
                        lambda api_key: pytest.fail("backoff 중인데 API를 불렀다"))

    result = odds_module.fetch_upcoming_odds(api_key="k")

    assert result[("A", "B")]["stale"] is True


def test_a_successful_fetch_updates_both_cache_layers(tmp_path, monkeypatch):
    """새로 받아온 배당률은 로컬과 공유 캐시 양쪽에 기록돼야 한다 — 공유 쪽에 안 남기면
    다른 인스턴스가 같은 6시간 안에 또 API를 부른다."""
    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", tmp_path / "local.json")
    shared = FakeSharedCache(None).install(monkeypatch)
    monkeypatch.setattr(odds_module, "_fetch_live_odds_from_api", lambda api_key: RECORDS)

    odds_module.fetch_upcoming_odds(api_key="k")

    assert shared.writes == 1
    assert shared.state["records"] == RECORDS
    assert shared.state["failed_at"] is None  # 성공했으니 이전 실패 기록은 해제된다
    assert odds_module._read_local_cache()["records"] == RECORDS


def test_shared_cache_is_skipped_without_a_bucket(tmp_path, monkeypatch):
    """버킷 설정이 없는 로컬 개발 환경에서는 공유 캐시 경로를 아예 타지 않아야 한다 —
    자격증명도 없는 곳에서 GCS 클라이언트를 만들면 그 자체로 예외가 난다."""
    monkeypatch.setattr(odds_module, "_get_storage_client",
                        lambda: pytest.fail("버킷 설정이 없는데 GCS 클라이언트를 만들었다"))

    assert odds_module._read_shared_cache() is None
    odds_module._write_shared_cache(cache_state(RECORDS))  # 조용히 넘어가야 한다


def test_history_cache_is_refetched_once_it_falls_behind(tmp_path, monkeypatch):
    """과거 배당률 캐시가 한 달 넘게 뒤처지면 다시 받아와야 한다.

    이 파일에는 완결 시즌뿐 아니라 진행 중인 시즌의 경기도 들어있다. 존재 여부만 확인하면
    처음 받은 시점 이후에 치러진 경기가 영원히 들어오지 않아, 해가 바뀌어 재학습을 돌려도
    계속 같은 시점의 데이터로 학습한다."""
    history = tmp_path / "odds_history.csv"
    monkeypatch.setattr(odds_module, "ODDS_HISTORY_PATH", history)

    fresh = pd.Timestamp.now().normalize()
    pd.DataFrame({"date": [fresh], "result": ["HOME_TEAM"],
                  "odds_p_home": [0.5], "odds_p_draw": [0.3], "odds_p_away": [0.2]}).to_csv(history, index=False)
    assert odds_module._history_needs_refresh() is False

    stale = fresh - pd.Timedelta(days=odds_module.HISTORY_MAX_STALENESS_DAYS + 1)
    pd.DataFrame({"date": [stale], "result": ["HOME_TEAM"],
                  "odds_p_home": [0.5], "odds_p_draw": [0.3], "odds_p_away": [0.2]}).to_csv(history, index=False)
    assert odds_module._history_needs_refresh() is True

    history.unlink()
    assert odds_module._history_needs_refresh() is True
