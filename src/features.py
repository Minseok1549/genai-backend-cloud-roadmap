"""팀별 롤링 폼 피처. 각 경기의 피처는 그 경기 '이전'에 끝난 경기만 사용한다 (데이터 누수 방지)."""
import pandas as pd

WINDOW = 5

# '최근 폼'으로 인정하는 최대 경과 일수. 이 값을 넘는 기록만 남은 팀은 폼을 계산하지 않는다.
# 경계를 150일로 둔 근거는 실제 데이터다: 지난 시즌부터 계속 리그에 있던 팀은 여름 휴식기를
# 건너뛰어도 5경기 창이 111~119일 안에 들어온다. 반면 한 시즌을 2부에서 보내고 돌아온 팀은
# 마지막 1부 경기가 475일 전이다(2026-27 시즌 개막 시점 Ipswich Town 실측). 즉 이 선은
# "여름을 건너뛴 것"과 "한 시즌을 리그 밖에서 보낸 것"을 가른다.
MAX_FORM_AGE_DAYS = 150


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

    직전 window경기 기록이 없으면(승격팀 등) None을 반환한다. 기록이 window경기 있어도 그게
    너무 오래된 것뿐이면(강등됐다가 돌아온 팀) 역시 None이다 — 15개월 전 1부 경기 성적을
    '최근 폼'이라고 부르면 안 되기 때문이다. 이 판단 기준은 MAX_FORM_AGE_DAYS에 적어뒀다.
    기준 시점은 '지금'으로 삼는다: 이 함수는 아직 열리지 않은 경기를 위해 불리고 그 경기는
    길어도 2주 안에 열리므로, 킥오프 시각을 따로 받아올 만큼의 차이가 생기지 않는다.
    """
    log = _team_match_log(matches)
    team_log = log[log["team"] == team].sort_values("date")
    if len(team_log) < window:
        return None
    recent = team_log.tail(window)
    oldest = recent["date"].min()
    now = pd.Timestamp.now(tz=oldest.tz) if oldest.tzinfo is not None else pd.Timestamp.now()
    if (now - oldest).days > MAX_FORM_AGE_DAYS:
        return None
    return {
        "points": recent["points"].mean(),
        "goals_for": recent["goals_for"].mean(),
        "goals_against": recent["goals_against"].mean(),
    }
