import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fastapi.testclient import TestClient
from api import app  # noqa: E402


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
