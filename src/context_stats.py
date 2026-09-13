"""대시보드 카드에 붙일 경기 맥락: 순위, 최근 5경기, 홈/원정 경기당 승점.

확률만 있는 카드는 "왜 그 숫자인지"를 볼 수 없다. 3위 팀이 18위 팀을 홈에서 만나는
경기인지, 둘 다 중위권인데 원정팀이 5연승 중인지가 보이면 사용자가 확률을 스스로
가늠할 수 있다.

전부 이미 받아둔 경기 데이터로 계산한다 — 외부 API를 새로 부르지 않는다. 순위표를
제공하는 무료 엔드포인트가 있지만, 끝난 경기 결과가 이미 손에 있으면 순위는 그걸로
계산되는 값이라 호출을 늘릴 이유가 없다.
"""
import pandas as pd

from features import _team_match_log

RECENT_WINDOW = 5


def season_table(matches: pd.DataFrame) -> dict[str, dict]:
    """한 시즌의 끝난 경기로 순위표를 만든다. 팀명 -> {rank, played, points, goal_diff}.

    순위 기준은 EPL 규정 순서를 따른다: 승점 → 골 득실 → 다득점. 같은 값이면 팀명순으로
    안정화한다(표시가 새로고침마다 흔들리지 않게).
    """
    if matches.empty:
        return {}
    log = _team_match_log(matches)
    agg = log.groupby("team").agg(
        played=("match_id", "count"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    agg["goal_diff"] = agg["goals_for"] - agg["goals_against"]
    agg = agg.sort_values(
        ["points", "goal_diff", "goals_for", "team"], ascending=[False, False, False, True]
    )
    return {
        team: {
            "rank": i,
            "played": int(row.played),
            "points": int(row.points),
            "goal_diff": int(row.goal_diff),
        }
        for i, (team, row) in enumerate(agg.iterrows(), start=1)
    }


def recent_form(matches: pd.DataFrame, team: str, window: int = RECENT_WINDOW) -> list[str]:
    """팀의 최근 경기 결과를 오래된 것 → 최신 순으로 W/D/L 리스트로 반환한다.
    왼쪽에서 오른쪽으로 읽는 순서가 시간 순서와 같아야 흐름(연승·연패)이 보인다."""
    log = _team_match_log(matches)
    team_log = log[log["team"] == team].sort_values("date").tail(window)
    return ["W" if p == 3 else "D" if p == 1 else "L" for p in team_log["points"]]


def venue_ppg(matches: pd.DataFrame, team: str, home: bool) -> float | None:
    """홈 경기만(또는 원정 경기만) 추려 경기당 승점을 낸다.

    전체 평균과 나누는 이유: 홈에서 강하지만 원정에서 무너지는 팀이 실제로 흔하고, 이
    경기가 홈 경기인지 원정 경기인지에 따라 참고해야 할 숫자가 다르다. 해당 장소에서
    치른 경기가 없으면(시즌 초) None — 0.0으로 내놓으면 "아주 나쁘다"로 오해된다."""
    side = matches[matches["home_team" if home else "away_team"] == team]
    if side.empty:
        return None
    win, lose = ("HOME_TEAM", "AWAY_TEAM") if home else ("AWAY_TEAM", "HOME_TEAM")
    points = side["result"].map({win: 3, "DRAW": 1, lose: 0}).sum()
    return round(points / len(side), 2)


def match_context(matches: pd.DataFrame, home_team: str, away_team: str, season: int) -> dict:
    """카드 하나에 필요한 맥락을 한 번에 만든다.

    같은 시즌 경기만 쓴다 — 순위와 경기당 승점은 시즌 단위 개념이고, 지난 시즌 성적을
    섞으면 "지금 몇 위인가"가 아닌 값이 된다. 반면 최근 5경기는 시즌 경계와 무관하게
    "직전 5경기"가 자연스러운 해석이라 전체 기록에서 뽑는다(시즌 초에 빈칸이 되는 것보다
    직전 시즌 마지막 경기들을 보여주는 편이 유용하다).
    """
    season_matches = matches[matches["season"] == season] if "season" in matches else matches
    table = season_table(season_matches)
    return {
        "home": {
            "table": table.get(home_team),
            "recent": recent_form(matches, home_team),
            "ppg": venue_ppg(season_matches, home_team, home=True),
        },
        "away": {
            "table": table.get(away_team),
            "recent": recent_form(matches, away_team),
            "ppg": venue_ppg(season_matches, away_team, home=False),
        },
    }
