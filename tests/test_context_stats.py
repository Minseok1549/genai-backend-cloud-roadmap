import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import context_stats  # noqa: E402


def _match(mid, date, home, away, hg, ag, season=2026):
    result = "HOME_TEAM" if hg > ag else "AWAY_TEAM" if ag > hg else "DRAW"
    return {"match_id": mid, "date": pd.Timestamp(date), "season": season,
            "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "result": result}


def _df(rows):
    return pd.DataFrame(rows)


def test_season_table_orders_by_points_then_goal_difference():
    """승점이 같으면 골 득실로 가른다 — EPL 순위 규정 순서다."""
    rows = [
        _match(1, "2026-08-15", "A", "B", 5, 0),  # A 3점 +5
        _match(2, "2026-08-15", "C", "D", 1, 0),  # C 3점 +1
    ]
    table = context_stats.season_table(_df(rows))
    assert table["A"]["rank"] == 1
    assert table["C"]["rank"] == 2
    assert table["A"]["goal_diff"] == 5
    assert table["A"]["points"] == 3
    assert table["B"]["rank"] == 4  # -5로 최하위


def test_season_table_counts_draws_and_played():
    rows = [
        _match(1, "2026-08-15", "A", "B", 1, 1),
        _match(2, "2026-08-22", "B", "A", 2, 2),
    ]
    table = context_stats.season_table(_df(rows))
    assert table["A"] == {"rank": 1, "played": 2, "points": 2, "goal_diff": 0}


def test_season_table_handles_empty_input():
    """시즌 첫 경기 전에는 끝난 경기가 없다 — 여기서 터지면 대시보드 전체가 죽는다."""
    assert context_stats.season_table(_df([])) == {}


def test_recent_form_is_oldest_to_newest():
    """왼쪽에서 오른쪽으로 읽는 순서가 시간 순서와 같아야 연승·연패 흐름이 보인다."""
    rows = [
        _match(1, "2026-08-01", "A", "B", 1, 0),  # A 승
        _match(2, "2026-08-08", "C", "A", 1, 1),  # A 무
        _match(3, "2026-08-15", "A", "D", 0, 2),  # A 패
    ]
    assert context_stats.recent_form(_df(rows), "A") == ["W", "D", "L"]


def test_recent_form_keeps_only_last_five():
    rows = [_match(i, f"2026-08-{i:02d}", "A", "B", 1, 0) for i in range(1, 8)]
    assert context_stats.recent_form(_df(rows), "A") == ["W"] * 5


def test_venue_ppg_separates_home_from_away():
    """홈에서 강하고 원정에서 무너지는 팀이 흔하다 — 전체 평균으로 합치면 그게 안 보인다."""
    rows = [
        _match(1, "2026-08-01", "A", "B", 2, 0),  # A 홈 승
        _match(2, "2026-08-08", "A", "C", 1, 0),  # A 홈 승
        _match(3, "2026-08-15", "D", "A", 3, 0),  # A 원정 패
    ]
    df = _df(rows)
    assert context_stats.venue_ppg(df, "A", home=True) == 3.0
    assert context_stats.venue_ppg(df, "A", home=False) == 0.0


def test_venue_ppg_returns_none_without_matches_at_that_venue():
    """시즌 초에는 아직 원정 경기를 안 치른 팀이 있다. 0.0으로 내놓으면 "아주 나쁘다"로
    오해된다 — 값이 없다는 것과 0점이라는 건 다르다."""
    rows = [_match(1, "2026-08-01", "A", "B", 2, 0)]
    assert context_stats.venue_ppg(_df(rows), "A", home=False) is None


def test_match_context_uses_only_current_season_for_table():
    """순위와 경기당 승점은 시즌 단위 개념이다. 지난 시즌을 섞으면 '지금 몇 위인가'가 아닌
    값이 된다."""
    rows = [
        _match(1, "2025-09-01", "A", "B", 5, 0, season=2025),  # 지난 시즌 — 순위에 섞이면 안 됨
        _match(2, "2026-08-15", "B", "A", 1, 0, season=2026),
    ]
    ctx = context_stats.match_context(_df(rows), "A", "B", season=2026)
    assert ctx["away"]["table"]["points"] == 3   # B는 이번 시즌 1승
    assert ctx["home"]["table"]["points"] == 0   # A는 이번 시즌 1패
    # 최근 5경기는 시즌 경계를 넘어 '직전 5경기'로 읽는 게 자연스럽다.
    assert ctx["home"]["recent"] == ["W", "L"]


def test_match_context_tolerates_team_absent_from_table():
    """승격팀은 시즌 첫 경기를 치르기 전까지 순위표에 없다 — None이 그대로 내려가야 한다."""
    rows = [_match(1, "2026-08-15", "A", "B", 1, 0)]
    ctx = context_stats.match_context(_df(rows), "NewlyPromoted FC", "A", season=2026)
    assert ctx["home"]["table"] is None
    assert ctx["home"]["recent"] == []
