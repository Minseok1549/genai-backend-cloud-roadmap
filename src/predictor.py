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
    # bundle["classes"]는 predict_proba 결과 순서를 라벨과 맞추는 데 쓰인다. train.py가
    # 저장 시점에 model.classes_에서 그대로 뽑아 쓰므로 정상이라면 항상 일치해야 하는데,
    # bundle이 수동으로 조작되거나 손상되면 확률이 조용히 엉뚱한 라벨에 매핑될 수 있어
    # 로드 시점에 한 번 검증한다.
    if list(bundle["model"].classes_) != bundle["classes"]:
        raise RuntimeError(f"{path}의 model_bundle이 손상됐습니다 — classes가 모델과 불일치")
    return bundle


def _current_season_known_teams(matches: pd.DataFrame) -> set:
    """강등팀은 과거 시즌 로그에 계속 남아있어 known_teams 체크를 그대로 통과하면, 몇 년 전
    기록으로 '최근 폼'이 계산돼버린다. 그래서 팀 존재 여부는 (완료된 경기가 있는) 가장 최근
    시즌 경기로만 판단한다 — 새 시즌에 아직 끝난 경기가 없으면 자연히 직전 시즌이 최근
    시즌으로 선택되므로, 진행 중인 팀은 계속 known으로 인식된다."""
    if matches.empty:
        return set()
    latest_season = matches["season"].max()
    season_matches = matches[matches["season"] == latest_season]
    return set(season_matches["home_team"]) | set(season_matches["away_team"])


def predict_match(home_team: str, away_team: str, matches: pd.DataFrame, model_bundle: dict) -> dict:
    known_teams = _current_season_known_teams(matches)
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
