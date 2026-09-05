import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predictor import predict_match, load_model_bundle, UnknownTeamError  # noqa: E402


def _match(match_id, date, season, home, away, hg, ag):
    result = "HOME_TEAM" if hg > ag else ("AWAY_TEAM" if ag > hg else "DRAW")
    return {
        "match_id": match_id, "date": pd.Timestamp(date), "season": season,
        "home_team": home, "away_team": away,
        "home_goals": hg, "away_goals": ag, "result": result,
    }


class _FakeModel:
    def __init__(self, classes):
        self.classes_ = classes

    def predict_proba(self, features):
        return [[0.5, 0.3, 0.2]]


def test_predict_match_rejects_team_not_in_latest_season():
    """강등된 팀이 과거 시즌 로그에 남아있다는 이유만으로 known 팀 취급되면 안 된다."""
    rows = [
        _match(0, "2023-01-01", 2023, "OldClub", "A", 2, 0),
        _match(1, "2023-01-02", 2023, "A", "OldClub", 2, 0),
        _match(2, "2023-01-03", 2023, "OldClub", "A", 2, 0),
        _match(3, "2023-01-04", 2023, "A", "OldClub", 2, 0),
        _match(4, "2023-01-05", 2023, "OldClub", "A", 2, 0),
        _match(5, "2024-01-01", 2024, "A", "B", 1, 1),
        _match(6, "2024-01-02", 2024, "B", "A", 1, 1),
        _match(7, "2024-01-03", 2024, "A", "B", 1, 1),
        _match(8, "2024-01-04", 2024, "B", "A", 1, 1),
        _match(9, "2024-01-05", 2024, "A", "B", 1, 1),
    ]
    matches = pd.DataFrame(rows)
    bundle = {"model": _FakeModel(["AWAY_TEAM", "DRAW", "HOME_TEAM"]), "classes": ["AWAY_TEAM", "DRAW", "HOME_TEAM"]}

    with pytest.raises(UnknownTeamError):
        predict_match("OldClub", "B", matches, bundle)


def test_load_model_bundle_rejects_class_mismatch(tmp_path):
    bad_bundle = {
        "model": _FakeModel(["AWAY_TEAM", "DRAW", "HOME_TEAM"]),
        "classes": ["HOME_TEAM", "DRAW", "AWAY_TEAM"],  # 모델의 classes_와 순서가 다름
        "model_version": "v-test",
    }
    path = tmp_path / "model.joblib"
    joblib.dump(bad_bundle, path)

    with pytest.raises(RuntimeError, match="classes가 모델과 불일치"):
        load_model_bundle(path)
