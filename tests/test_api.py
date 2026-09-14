import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fastapi.testclient import TestClient
from api import app, MAX_REQUEST_ID_LENGTH  # noqa: E402


@pytest.fixture
def client():
    # TestClient를 with 블록으로 써야 lifespan(모델 로드)이 실제로 실행된다.
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_returns_probabilities_summing_to_one(client):
    resp = client.post("/predict", json={"home_team": "Arsenal FC", "away_team": "Aston Villa FC"})
    assert resp.status_code == 200
    body = resp.json()
    probs = body["probabilities"]
    assert set(probs.keys()) == {"HOME_TEAM", "DRAW", "AWAY_TEAM"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_predict_unknown_team_returns_400_not_500(client):
    resp = client.post("/predict", json={"home_team": "Arsnal Typo", "away_team": "Aston Villa FC"})
    assert resp.status_code == 400


def test_predict_blank_team_returns_400_not_500(client):
    resp = client.post("/predict", json={"home_team": "", "away_team": "Aston Villa FC"})
    assert resp.status_code == 400


def test_predict_missing_field_returns_400(client):
    resp = client.post("/predict", json={"home_team": "Arsenal FC"})
    assert resp.status_code == 400


def test_predict_same_team_twice_returns_400(client):
    resp = client.post("/predict", json={"home_team": "Arsenal FC", "away_team": "Arsenal FC"})
    assert resp.status_code == 400


def test_predict_insufficient_history_returns_400_not_500(client):
    # 이번 시즌 데이터상 콤바인 경기 기록이 5경기 미만인 팀 (승격팀 등)
    resp = client.post("/predict", json={"home_team": "Coventry City FC", "away_team": "Arsenal FC"})
    assert resp.status_code == 400


def test_predict_unknown_team_does_not_call_odds_api(client, monkeypatch):
    """존재하지 않는 팀명 요청은 유료 배당률 API를 부르기 전에 400으로 끝나야 한다.
    전에는 검증이 predict_match() 안에서만 이뤄져 쿼터(무료 티어 월 500회)를 먼저 태웠다."""
    import api

    # 예외로 단정하면 안 된다 — api.py가 배당률 조회 실패를 모두 잡아 폼 모델로 넘기므로,
    # 호출이 다시 생겨도 뒤이은 팀 검증이 400을 내서 테스트가 통과해버린다. 호출 횟수를 센다.
    calls = []
    monkeypatch.setattr(api, "fetch_upcoming_odds", lambda: calls.append(1) or {})

    resp = client.post("/predict", json={"home_team": "Nonexistent FC", "away_team": "Arsenal FC"})
    assert resp.status_code == 400
    assert calls == []


def test_predict_ignores_stale_odds_and_uses_form_model(client, monkeypatch):
    """API 장애로 TTL 지난 캐시를 재사용한 배당률(stale=True)은 통계 예측에 쓰지 않고
    폼 모델로 내려가야 한다 — 매일 배치도 같은 규칙을 쓴다."""
    import api

    home, away = "Arsenal FC", "Aston Villa FC"
    monkeypatch.setattr(api, "fetch_upcoming_odds", lambda: {
        (home, away): {"odds_p_home": 0.9, "odds_p_draw": 0.05, "odds_p_away": 0.05, "stale": True},
    })

    captured = {}
    real_predict = api.predict_match

    def spy_predict(*a, **kw):
        captured["odds"] = kw.get("odds")
        return real_predict(*a, **kw)

    monkeypatch.setattr(api, "predict_match", spy_predict)

    resp = client.post("/predict", json={"home_team": home, "away_team": away})
    assert resp.status_code == 200
    assert captured["odds"] is None  # stale 배당률은 버려졌다


def test_request_id_is_passed_through_for_tracing(client):
    """클라이언트가 보낸 추적 ID는 그대로 이어받아 응답에 되돌려줘야 한다(분산 추적)."""
    resp = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert resp.headers["X-Request-ID"] == "trace-abc-123"


def test_overlong_request_id_is_replaced_instead_of_echoed(client):
    """과도하게 긴 추적 ID는 버리고 새로 발급한다.

    이 값은 클라이언트가 정하는데 로그 한 줄과 응답 헤더에 그대로 실린다 — 제한이 없으면
    요청 하나로 Cloud Logging에 10만 자를 쓰게 만들 수 있고(과금), 응답 크기도 같이 부풀린다."""
    resp = client.get("/health", headers={"X-Request-ID": "A" * 100_000})
    returned = resp.headers["X-Request-ID"]
    assert len(returned) <= MAX_REQUEST_ID_LENGTH
    assert "A" * 100 not in returned
