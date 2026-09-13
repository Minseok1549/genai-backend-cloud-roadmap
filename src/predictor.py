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
    if bundle.get("odds_model") is not None and list(bundle["odds_model"].classes_) != bundle["odds_classes"]:
        raise RuntimeError(f"{path}의 model_bundle이 손상됐습니다 — odds_classes가 odds_model과 불일치")
    return bundle


def _current_season_known_teams(matches: pd.DataFrame) -> set:
    """호출자가 팀 목록을 주지 않았을 때 쓰는 fallback. 강등팀은 과거 시즌 로그에 계속
    남아있어 팀 목록에 그대로 넣으면 몇 년 전 기록으로 '최근 폼'이 계산돼버리므로, 완료된
    경기가 있는 가장 최근 시즌으로 범위를 좁힌다.

    다만 이 방식은 시즌 첫 경기가 막 한 경기 끝난 시점에 그 경기에 나온 두 팀만 인정하게
    되는 한계가 있다 — 그래서 실제 서빙 경로는 일정표 기준 팀 목록(data.load_season_teams)을
    known_teams로 넘긴다."""
    if matches.empty:
        return set()
    latest_season = matches["season"].max()
    season_matches = matches[matches["season"] == latest_season]
    return set(season_matches["home_team"]) | set(season_matches["away_team"])


def predict_match(
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    model_bundle: dict,
    odds: dict | None = None,
    known_teams: set | None = None,
) -> dict:
    # 팀 소속 검증을 두 모델 경로보다 먼저 한다. 이전에는 배당률 경로를 먼저 타서 검증을
    # 건너뛰었는데, 그건 팀 목록을 "완료된 경기"에서만 뽑던 탓에 승격팀이 시즌 첫 경기 전에
    # 막히는 걸 피하려던 우회였다 — 호출자가 일정표 기준 팀 목록을 넘기면 그 우회가 필요
    # 없어지고, 검증이 앞에 있어야 존재하지 않는 팀명이 모델까지 내려가지 않는다.
    # known_teams가 None이거나 빈 집합이면 "일정 캐시가 없어 팀 목록을 만들 수 없었다"는
    # 뜻이다(load_season_teams는 파일이 없으면 빈 집합을 준다). 그때는 완료 경기 기반 판정으로
    # 내려간다 — 일정 정보가 아예 없는 상황에서 쓸 수 있는 유일한 근거다.
    known = known_teams or _current_season_known_teams(matches)
    for team in (home_team, away_team):
        if team not in known:
            raise UnknownTeamError(team)

    # 배당률 모델이 폼 기반 모델보다 정확도가 높다(실험 확인, 55%대 vs 40%대) — 시장이
    # 이미 부상/폼/전술 등 공개정보를 우리 통계모델보다 효율적으로 반영하기 때문이다.
    if odds is not None and model_bundle.get("odds_model") is not None:
        odds_features = pd.DataFrame([odds])[model_bundle["odds_feature_names"]]
        proba = model_bundle["odds_model"].predict_proba(odds_features)[0]
        return dict(zip(model_bundle["odds_classes"], proba.tolist()))

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
