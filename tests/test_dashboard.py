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


def test_dashboard_default_view_aggregates_full_matchday_across_dates(client, monkeypatch):
    """하나의 라운드가 여러 날짜에 걸쳐 저장돼도, 날짜 파라미터 없이 들어오면
    라운드 전체 경기가 한 화면에 모여야 한다(과거 버그: 한 날짜 파일만 보여줌).
    예측이 저장 안 된 경기(폼 데이터 부족으로 스킵됨 등)도 라운드에서 빠지지 않고
    이유와 함께 나와야 한다(과거 버그: 예측 있는 경기만 보여서 라운드 일부가 통째로 빠짐)."""
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06", "2026-09-05"])
    monkeypatch.setattr(
        api,
        "load_matchday_info",
        lambda: {
            "matchday": 3,
            "dates": ["2026-09-05", "2026-09-06"],
            "fixtures": [
                {
                    "match_id": 1,
                    "kickoff_utc": "2026-09-05T14:00:00Z",
                    "home_team": "Fulham FC",
                    "away_team": "Crystal Palace FC",
                    "status": "FINISHED",
                },
                {
                    "match_id": 99,
                    "kickoff_utc": "2026-09-05T14:00:00Z",
                    "home_team": "Manchester City FC",
                    "away_team": "Coventry City FC",
                    "status": "FINISHED",
                },
                {
                    "match_id": 2,
                    "kickoff_utc": "2026-09-06T15:30:00Z",
                    "home_team": "Arsenal FC",
                    "away_team": "Chelsea FC",
                    "status": "TIMED",
                },
            ],
        },
    )

    def fake_fetch(date_str):
        if date_str == "2026-09-05":
            return {
                "generated_at": "2026-09-05T06:00:00Z",
                "predictions": [
                    {
                        "match_id": 1,
                        "kickoff_utc": "2026-09-05T14:00:00Z",
                        "home_team": "Fulham FC",
                        "away_team": "Crystal Palace FC",
                        "probabilities": {"HOME_TEAM": 0.4, "DRAW": 0.3, "AWAY_TEAM": 0.3},
                    }
                ],
            }
        if date_str == "2026-09-06":
            return {
                "generated_at": "2026-09-06T06:00:00Z",
                "predictions": [
                    {
                        "match_id": 2,
                        "kickoff_utc": "2026-09-06T15:30:00Z",
                        "home_team": "Arsenal FC",
                        "away_team": "Chelsea FC",
                        "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
                    }
                ],
            }
        return None

    monkeypatch.setattr(api, "_fetch_daily_predictions", fake_fetch)

    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "3라운드" in resp.text
    assert "Fulham FC" in resp.text
    assert "Arsenal FC" in resp.text
    assert "Coventry City FC" in resp.text
    assert "경기 종료 · 예측 기록 없음" in resp.text
