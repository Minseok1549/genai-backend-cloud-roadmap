import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data as data_module  # noqa: E402
from data import load_fixtures_on_date  # noqa: E402
import daily_predict  # noqa: E402
from predictor import UnknownTeamError  # noqa: E402


def test_load_fixtures_on_date_filters_status_and_date(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "RAW_DIR", tmp_path)
    payload = {
        "matches": [
            {"id": 1, "status": "TIMED", "utcDate": "2026-09-05T11:30:00Z",
             "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}},
            {"id": 2, "status": "SCHEDULED", "utcDate": "2026-09-06T11:30:00Z",
             "homeTeam": {"name": "C"}, "awayTeam": {"name": "D"}},
            {"id": 3, "status": "FINISHED", "utcDate": "2026-09-05T09:00:00Z",
             "homeTeam": {"name": "E"}, "awayTeam": {"name": "F"},
             "score": {"fullTime": {"home": 1, "away": 0}, "winner": "HOME_TEAM"}},
        ]
    }
    (tmp_path / "matches_2026.json").write_text(json.dumps(payload))

    fixtures = load_fixtures_on_date("2026-09-05", season=2026)
    assert len(fixtures) == 1
    assert fixtures[0]["match_id"] == 1


def test_build_daily_predictions_skips_failed_fixture_and_continues(monkeypatch):
    """경기 하나의 예측이 실패해도(폼 기록 부족 등) 나머지 경기 배치는 계속돼야 한다."""
    monkeypatch.setattr(daily_predict, "ensure_all_seasons_cached", lambda: None)
    monkeypatch.setattr(
        daily_predict,
        "load_fixtures_on_date",
        lambda date: [
            {"match_id": 1, "kickoff_utc": "2026-09-05T11:30:00Z", "home_team": "Unknown FC", "away_team": "B"},
            {"match_id": 2, "kickoff_utc": "2026-09-05T14:00:00Z", "home_team": "A", "away_team": "B"},
        ],
    )
    monkeypatch.setattr(daily_predict, "load_matches", lambda: pd.DataFrame())
    monkeypatch.setattr(daily_predict, "load_model_bundle", lambda path: {"model_version": "v1"})

    def fake_predict_match(home, away, matches, bundle):
        if home == "Unknown FC":
            raise UnknownTeamError(home)
        return {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2}

    monkeypatch.setattr(daily_predict, "predict_match", fake_predict_match)

    payload = daily_predict.build_daily_predictions("2026-09-05")
    assert len(payload["predictions"]) == 1
    assert payload["predictions"][0]["match_id"] == 2
    assert payload["predictions"][0]["model_version"] == "v1"


def test_build_daily_predictions_empty_when_no_fixtures(monkeypatch):
    monkeypatch.setattr(daily_predict, "ensure_all_seasons_cached", lambda: None)
    monkeypatch.setattr(daily_predict, "load_fixtures_on_date", lambda date: [])
    monkeypatch.setattr(daily_predict, "load_matches", lambda: pd.DataFrame())
    monkeypatch.setattr(daily_predict, "load_model_bundle", lambda path: {"model_version": "v1"})

    payload = daily_predict.build_daily_predictions("2026-09-05")
    assert payload["predictions"] == []
