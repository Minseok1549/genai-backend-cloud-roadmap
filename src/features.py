"""팀별 롤링 폼 피처. 각 경기의 피처는 그 경기 '이전'에 끝난 경기만 사용한다 (데이터 누수 방지)."""
import pandas as pd

WINDOW = 5


def _team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """경기 단위 df를 '팀-경기' 단위 long format으로 펼친다. 팀 관점의 득점/실점/승점을 계산."""
    home = matches[["match_id", "date", "home_team", "home_goals", "away_goals", "result"]].copy()
    home = home.rename(columns={"home_team": "team", "home_goals": "goals_for", "away_goals": "goals_against"})
    home["points"] = home["result"].map({"HOME_TEAM": 3, "DRAW": 1, "AWAY_TEAM": 0})

    away = matches[["match_id", "date", "away_team", "away_goals", "home_goals", "result"]].copy()
    away = away.rename(columns={"away_team": "team", "away_goals": "goals_for", "home_goals": "goals_against"})
    away["points"] = away["result"].map({"AWAY_TEAM": 3, "DRAW": 1, "HOME_TEAM": 0})

    log = pd.concat([home, away], ignore_index=True)
    log = log.sort_values(["team", "date"]).reset_index(drop=True)
    return log


def _rolling_form(log: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """팀별로 '이번 경기 전까지' 최근 window경기 평균 승점/득점/실점을 계산한다.

    shift(1) 뒤에 rolling을 적용하므로 현재 행(오늘 경기)의 결과는 절대 자기 자신의
    피처 계산에 들어가지 않는다.
    """
    for col in ["points", "goals_for", "goals_against"]:
        log[f"form_{col}"] = log.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
    return log


def build_features(matches: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """경기 df에 home/away 각각의 폼 피처를 붙인다. 직전 window경기 기록이 없는 경기는 제외한다."""
    log = _rolling_form(_team_match_log(matches), window=window)

    feature_cols = ["match_id", "team", "form_points", "form_goals_for", "form_goals_against"]
    home_feat = log[feature_cols].rename(
        columns={
            "team": "home_team",
            "form_points": "home_form_points",
            "form_goals_for": "home_form_gf",
            "form_goals_against": "home_form_ga",
        }
    )
    away_feat = log[feature_cols].rename(
        columns={
            "team": "away_team",
            "form_points": "away_form_points",
            "form_goals_for": "away_form_gf",
            "form_goals_against": "away_form_ga",
        }
    )

    out = matches.merge(home_feat, on=["match_id", "home_team"], how="left")
    out = out.merge(away_feat, on=["match_id", "away_team"], how="left")

    feature_names = [
        "home_form_points", "home_form_gf", "home_form_ga",
        "away_form_points", "away_form_gf", "away_form_ga",
    ]
    out = out.dropna(subset=feature_names).reset_index(drop=True)
    return out


FEATURE_NAMES = [
    "home_form_points", "home_form_gf", "home_form_ga",
    "away_form_points", "away_form_gf", "away_form_ga",
]


def latest_team_form(matches: pd.DataFrame, team: str, window: int = WINDOW) -> dict | None:
    """아직 열리지 않은 다음 경기를 예측하기 위해, team이 지금까지 치른 마지막 window경기의
    평균 승점/득점/실점을 계산한다. 아직 열리지 않은 경기 자체는 로그에 없으므로 shift 없이
    그대로 tail(window)만 쓰면 된다 (build_features의 shift(1)과 목적이 다름: 그쪽은 '기존
    경기 목록 안의 한 경기'를 예측하고, 이쪽은 '목록 밖의 다음 경기'를 예측한다).

    직전 window경기 기록이 없으면(승격팀 등) None을 반환한다.
    """
    log = _team_match_log(matches)
    team_log = log[log["team"] == team].sort_values("date")
    if len(team_log) < window:
        return None
    recent = team_log.tail(window)
    return {
        "points": recent["points"].mean(),
        "goals_for": recent["goals_for"].mean(),
        "goals_against": recent["goals_against"].mean(),
    }
