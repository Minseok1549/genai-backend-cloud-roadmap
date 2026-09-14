import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import build_features, latest_team_form, FEATURE_NAMES  # noqa: E402


def _match(match_id, date, home, away, hg, ag):
    result = "HOME_TEAM" if hg > ag else ("AWAY_TEAM" if ag > hg else "DRAW")
    return {
        "match_id": match_id, "date": pd.Timestamp(date), "season": 2023,
        "home_team": home, "away_team": away,
        "home_goals": hg, "away_goals": ag, "result": result,
    }


def _toy_matches(n_per_team=6):
    """A와 B가 번갈아 홈/원정으로 n_per_team번씩 붙는 장난감 데이터."""
    rows = []
    mid = 0
    for i in range(n_per_team):
        home, away = ("A", "B") if i % 2 == 0 else ("B", "A")
        rows.append(_match(mid, f"2024-01-{i+1:02d}", home, away, hg=2, ag=0))
        mid += 1
    return pd.DataFrame(rows)


def test_early_matches_without_full_history_are_dropped():
    matches = _toy_matches(n_per_team=6)
    feats = build_features(matches, window=5)
    # 각 팀의 6번째 경기(index 5)만 직전 5경기 기록이 꽉 차므로 남아야 한다.
    assert len(feats) == 1
    assert feats.iloc[0]["match_id"] == 5


def test_feature_unaffected_by_appending_future_matches():
    """미래 경기를 추가해도 과거 경기의 피처 값은 바뀌면 안 된다 (누수 방지 핵심 검증)."""
    matches = _toy_matches(n_per_team=8)
    feats_full = build_features(matches, window=5)

    matches_truncated = matches.iloc[:-2].reset_index(drop=True)  # 마지막 경기 하나 제거(양 팀 각 1경기치)
    feats_truncated = build_features(matches_truncated, window=5)

    common_ids = set(feats_full["match_id"]) & set(feats_truncated["match_id"])
    assert len(common_ids) > 0
    for mid in common_ids:
        row_full = feats_full[feats_full["match_id"] == mid].iloc[0]
        row_trunc = feats_truncated[feats_truncated["match_id"] == mid].iloc[0]
        for col in FEATURE_NAMES:
            assert row_full[col] == row_trunc[col], f"leakage detected in {col} for match {mid}"


def test_feature_value_matches_hand_computed_average():
    """A는 venue와 무관하게 5경기 연속 2-0 승(승점3, 득점2, 실점0). 6번째 경기 직전 폼은 정확히 이 평균이어야 한다."""
    rows = [
        _match(0, "2024-01-01", "A", "C", hg=2, ag=0),
        _match(1, "2024-01-02", "D", "A", hg=0, ag=2),
        _match(2, "2024-01-03", "A", "E", hg=2, ag=0),
        _match(3, "2024-01-04", "F", "A", hg=0, ag=2),
        _match(4, "2024-01-05", "A", "G", hg=2, ag=0),
        # H도 5경기 기록을 채워야 매치 5가 dropna에서 살아남는다 (내용은 무관, A쪽만 검증)
        _match(6, "2024-01-01", "H", "I", hg=1, ag=1),
        _match(7, "2024-01-02", "J", "H", hg=0, ag=0),
        _match(8, "2024-01-03", "H", "K", hg=1, ag=1),
        _match(9, "2024-01-04", "L", "H", hg=0, ag=0),
        _match(10, "2024-01-05", "H", "M", hg=1, ag=1),
        _match(5, "2024-01-06", "A", "H", hg=1, ag=1),  # 이 경기의 A쪽 피처만 검증 대상
    ]
    matches = pd.DataFrame(rows)
    feats = build_features(matches, window=5)
    row = feats[feats["match_id"] == 5].iloc[0]
    assert row["home_form_points"] == 3.0
    assert row["home_form_gf"] == 2.0
    assert row["home_form_ga"] == 0.0


def _team_matches(team, dates):
    """team이 주어진 날짜마다 홈에서 한 경기씩 치른 기록."""
    return pd.DataFrame([
        _match(i, d, team, f"상대{i}", hg=3, ag=0) for i, d in enumerate(dates)
    ])


def test_form_uses_records_from_across_the_summer_break():
    """여름 휴식기를 건너뛴 기록은 '최근 폼'으로 인정한다 — 지난 시즌부터 계속 리그에 있던
    팀은 시즌 초에 5경기 창이 넉 달 전까지 거슬러 올라가는 게 정상이다."""
    recent = pd.Timestamp.now(tz="UTC")
    dates = [recent - pd.Timedelta(days=d) for d in (119, 117, 10, 5, 2)]
    form = latest_team_form(_team_matches("계속있던팀", dates), "계속있던팀")
    assert form is not None
    assert form["points"] == 3.0


def test_form_is_refused_when_only_long_stale_records_exist():
    """한 시즌을 리그 밖에서 보내고 돌아온 팀은 폼을 계산하지 않아야 한다.

    경기 수만 세면 15개월 전 기록 5경기도 조건을 통과해서, 강등됐다 승격한 팀의 2년 전
    성적이 '최근 폼'으로 예측에 들어간다. 실제로 2026-27 개막 시점 Ipswich Town이 이
    상태였다 — 5경기 창이 475일에 걸쳐 있었다. 기록이 없어서 예측을 못 하는 것과, 아주
    오래된 기록으로 예측을 만들어내는 것 중에는 전자가 정직하다."""
    recent = pd.Timestamp.now(tz="UTC")
    dates = [recent - pd.Timedelta(days=d) for d in (475, 473, 470, 468, 465)]
    assert latest_team_form(_team_matches("돌아온팀", dates), "돌아온팀") is None


def test_form_age_is_measured_on_the_oldest_match_in_the_window():
    """창 안에 최신 경기가 섞여 있어도, 가장 오래된 경기가 기준을 넘으면 거부한다 —
    평균이 오래된 기록에 그만큼 끌려가기 때문이다."""
    recent = pd.Timestamp.now(tz="UTC")
    dates = [recent - pd.Timedelta(days=d) for d in (475, 5, 4, 3, 2)]
    assert latest_team_form(_team_matches("복귀팀", dates), "복귀팀") is None


def test_form_works_on_timezone_naive_dates():
    """tz 정보가 없는 날짜로도 계산이 되어야 한다 — 경과 일수 비교에서 tz-aware와
    tz-naive를 섞으면 TypeError가 난다."""
    recent = pd.Timestamp.now()
    dates = [recent - pd.Timedelta(days=d) for d in (20, 15, 10, 5, 2)]
    assert latest_team_form(_team_matches("naive팀", dates), "naive팀") is not None
