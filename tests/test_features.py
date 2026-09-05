import sys
from pathlib import Path

import pandas as pd


def test_ci_gating_intentional_failure():
    """CI 게이팅 검증용 — 일부러 실패시켜 build/deploy가 스킵되는지 확인 후 되돌린다."""
    assert False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import build_features, FEATURE_NAMES  # noqa: E402


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
