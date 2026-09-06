import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fastapi.testclient import TestClient
import api  # noqa: E402
from api import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_dashboard_renders_predictions_table(client, monkeypatch):
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06", "2026-09-05"])
    monkeypatch.setattr(
        api,
        "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {
                    "match_id": 1,
                    "kickoff_utc": "2026-09-06T15:30:00Z",
                    "home_team": "Arsenal FC",
                    "away_team": "Chelsea FC",
                    "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
                }
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert "Arsenal FC" in resp.text
    assert "60.0%" in resp.text


def test_dashboard_shows_empty_state_when_no_predictions(client, monkeypatch):
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: [])
    monkeypatch.setattr(api, "_fetch_daily_predictions", lambda date_str: None)

    resp = client.get("/dashboard?date=2099-01-01")
    assert resp.status_code == 200
    assert "예측 기록이 없습니다" in resp.text
