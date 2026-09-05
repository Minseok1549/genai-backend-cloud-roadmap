"""단일 경기 예측 로직. /predict(api.py)와 매일 배치(daily_predict.py)가 이 함수를 공유해
같은 팀 검증·피처 계산 규칙을 쓰게 한다 — 로직이 두 곳에서 따로 관리되며 어긋나는 걸 방지."""
from pathlib import Path

import joblib
import pandas as pd

from features import latest_team_form, FEATURE_NAMES


class UnknownTeamError(Exception):
    def __init__(self, team: str):
        self.team = team
        super().__init__(f"알 수 없는 팀명: {team}")


class InsufficientFormError(Exception):
    def __init__(self, team: str):
        self.team = team
        super().__init__(f"{team}의 최근 경기 기록이 부족합니다")


def load_model_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    if "model_version" not in bundle:
        raise RuntimeError(f"{path}에 model_version이 없습니다 — train.py를 다시 실행하세요")
    return bundle


def predict_match(home_team: str, away_team: str, matches: pd.DataFrame, model_bundle: dict) -> dict:
    known_teams = set(matches["home_team"]) | set(matches["away_team"])
    for team in (home_team, away_team):
        if team not in known_teams:
            raise UnknownTeamError(team)

    home_form = latest_team_form(matches, home_team)
    away_form = latest_team_form(matches, away_team)
    for team, form in [(home_team, home_form), (away_team, away_form)]:
        if form is None:
            raise InsufficientFormError(team)

    features = pd.DataFrame([{
        "home_form_points": home_form["points"],
        "home_form_gf": home_form["goals_for"],
        "home_form_ga": home_form["goals_against"],
        "away_form_points": away_form["points"],
        "away_form_gf": away_form["goals_for"],
        "away_form_ga": away_form["goals_against"],
    }])[FEATURE_NAMES]

    proba = model_bundle["model"].predict_proba(features)[0]
    return dict(zip(model_bundle["classes"], proba.tolist()))
