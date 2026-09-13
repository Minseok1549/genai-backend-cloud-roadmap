import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import scorecard  # noqa: E402


def test_log_loss_penalizes_confident_wrong_answer():
    """실제 일어난 결과에 준 확률만 본다. 50%를 줬으면 0.69, 10%만 줬으면 2.30."""
    assert scorecard.log_loss([0.5, 0.3, 0.2], "HOME_TEAM") == abs(math.log(0.5))
    assert scorecard.log_loss([0.1, 0.3, 0.6], "HOME_TEAM") > 2.3
    # 확률 0을 선언한 결과가 일어나도 무한이 되지 않아야 한다(평균 계산이 전부 망가진다).
    assert math.isfinite(scorecard.log_loss([0.0, 0.5, 0.5], "HOME_TEAM"))


def test_rps_treats_draw_guess_as_less_wrong_than_opposite_win():
    """홈-무-원정은 순서가 있는 결과다. 홈 승 경기에서 무승부를 찍은 것은 원정 승을 찍은
    것보다 덜 틀린 것으로 채점돼야 한다 — 정확도로는 둘 다 똑같이 '틀림'이다."""
    assert scorecard.rps([1.0, 0.0, 0.0], "HOME_TEAM") == 0.0
    draw_guess = scorecard.rps([0.0, 1.0, 0.0], "HOME_TEAM")
    away_guess = scorecard.rps([0.0, 0.0, 1.0], "HOME_TEAM")
    assert draw_guess < away_guess
    assert away_guess == 1.0  # 정반대를 확신한 경우가 최악값


def test_score_series_accuracy_uses_most_likely_outcome():
    scored = [
        ([0.6, 0.2, 0.2], "HOME_TEAM"),  # 맞음
        ([0.2, 0.2, 0.6], "HOME_TEAM"),  # 틀림
    ]
    metrics = scorecard.score_series(scored)
    assert metrics["n"] == 2
    assert metrics["accuracy"] == 0.5


def test_calibration_buckets_include_probability_one():
    """확률 1.0이 어느 구간에도 안 들어가면 조용히 표본에서 사라진다."""
    pairs = [(1.0, True), (1.0, False)]
    bins = scorecard._calibration(pairs)
    assert sum(b["n"] for b in bins) == 2
    assert bins[0]["range"] == "90-100%"
    assert bins[0]["actual"] == 0.5


def test_calibration_reports_gap_between_claim_and_reality():
    """70%라고 말한 10경기에서 3번만 일어났다면 과신이다 — 그 격차가 보여야 한다."""
    pairs = [(0.7, True)] * 3 + [(0.7, False)] * 7
    bin_70 = [b for b in scorecard._calibration(pairs) if b["range"] == "70-80%"][0]
    assert bin_70["n"] == 10
    assert bin_70["predicted"] == 0.7
    assert bin_70["actual"] == 0.3


def _entry(match_id, probs=None, genai=None, odds=None):
    e = {"match_id": match_id, "probabilities": probs}
    if genai is not None:
        e["genai_prediction"] = {"probabilities": genai}
    if odds is not None:
        e["odds_snapshot"] = odds
    return e


def _p(h, d, a):
    return {"HOME_TEAM": h, "DRAW": d, "AWAY_TEAM": a}


def test_build_scorecard_skips_unfinished_matches():
    """결과가 없는 경기는 채점 대상이 아니다 — 표본에 섞이면 성적이 조용히 희석된다."""
    entries = [_entry(1, _p(0.6, 0.2, 0.2)), _entry(2, _p(0.6, 0.2, 0.2))]
    card = scorecard.build_scorecard(entries, {1: "HOME_TEAM"}, _p(0.45, 0.25, 0.30))
    assert card["graded_matches"] == 1
    assert card["series"]["market"]["n"] == 1


def test_build_scorecard_scores_each_model_on_its_own_sample():
    """GenAI 예측은 킥오프 24시간 이내 경기에만 있다. 통계 예측과 표본이 다르므로 각자
    자기 표본 수와 함께 나와야 한다 — 표본 수 없이 정확도만 비교하면 오해한다."""
    entries = [
        _entry(1, _p(0.6, 0.2, 0.2), genai=_p(0.5, 0.3, 0.2), odds=_p(0.55, 0.25, 0.20)),
        _entry(2, _p(0.3, 0.3, 0.4)),  # GenAI 없음
    ]
    results = {1: "HOME_TEAM", 2: "AWAY_TEAM"}
    card = scorecard.build_scorecard(entries, results, _p(0.45, 0.25, 0.30))
    assert card["series"]["market"]["n"] == 2
    assert card["series"]["genai"]["n"] == 1
    assert card["series"]["bookmaker"]["n"] == 1


def test_build_scorecard_includes_baselines_on_same_sample():
    """baseline은 채점된 경기와 같은 표본에서 재야 비교가 성립한다."""
    entries = [_entry(1, _p(0.6, 0.2, 0.2)), _entry(2, _p(0.3, 0.3, 0.4))]
    results = {1: "HOME_TEAM", 2: "AWAY_TEAM"}
    card = scorecard.build_scorecard(entries, results, _p(0.45, 0.25, 0.30))
    assert card["series"]["base_rate"]["n"] == 2
    assert card["series"]["always_home"]["n"] == 2
    assert card["series"]["always_home"]["accuracy"] == 0.5  # 2경기 중 홈 승 1
    # 무조건 홈 승은 확률 1.0/0.0 규칙이라 log loss 비교가 의미 없다 — 내보내지 않는다.
    assert "log_loss" not in card["series"]["always_home"]


def test_build_scorecard_ignores_malformed_probabilities():
    """저장된 예측에는 확률이 None인 항목(배당률 미공개)이 섞여 있다. 과거 스키마나
    손상된 값이 지표를 NaN으로 오염시키면 성적표 전체가 못 쓰게 된다."""
    entries = [
        _entry(1, None),
        _entry(2, {"HOME_TEAM": 0.5}),                  # 결과 하나가 빠진 값
        _entry(3, _p(0.0, 0.0, 0.0)),                   # 합이 0
        _entry(4, _p("x", 0.2, 0.2)),                   # 숫자가 아닌 값
        _entry(5, _p(0.6, 0.2, 0.2)),                   # 유일하게 정상
    ]
    results = {i: "HOME_TEAM" for i in range(1, 6)}
    card = scorecard.build_scorecard(entries, results, _p(0.45, 0.25, 0.30))
    assert card["series"]["market"]["n"] == 1
    assert card["graded_matches"] == 1


def test_normalized_rescales_probabilities_that_do_not_sum_to_one():
    """GenAI가 낸 확률은 합이 정확히 1이 아닐 수 있다. 그대로 log loss에 넣으면 합이 작은
    예측이 부당하게 나쁜 점수를 받는다."""
    probs = scorecard._normalized({"HOME_TEAM": 0.4, "DRAW": 0.2, "AWAY_TEAM": 0.2})
    assert abs(sum(probs) - 1.0) < 1e-12
    assert abs(probs[0] - 0.5) < 1e-12


def test_base_rates_falls_back_to_uniform_without_history():
    assert scorecard.base_rates([])["DRAW"] == 1 / 3
    rates = scorecard.base_rates(["HOME_TEAM", "HOME_TEAM", "DRAW", "AWAY_TEAM"])
    assert rates["HOME_TEAM"] == 0.5


def test_normalized_rejects_nan_and_infinity():
    """float()는 "nan"·"inf" 문자열을 예외 없이 통과시킨다. 그 값 하나가 들어오면 평균에
    섞여 성적표의 모든 숫자가 NaN이 되고, 저장되는 JSON도 표준이 아닌 토큰을 담는다 —
    값 하나가 이상한 것과 성적표 전체가 못 쓰게 되는 것은 다른 문제여야 한다."""
    assert scorecard._normalized({"HOME_TEAM": float("nan"), "DRAW": 0.3, "AWAY_TEAM": 0.7}) is None
    assert scorecard._normalized({"HOME_TEAM": float("inf"), "DRAW": 0.3, "AWAY_TEAM": 0.7}) is None
    assert scorecard._normalized({"HOME_TEAM": "nan", "DRAW": 0.3, "AWAY_TEAM": 0.7}) is None
    # 합이 부동소수 상한을 넘겨 overflow되는 경우도 정상값처럼 통과하면 안 된다
    huge = 1e308
    assert scorecard._normalized({"HOME_TEAM": huge, "DRAW": huge, "AWAY_TEAM": huge}) is not None


def test_parse_ts_compares_equivalent_iso_formats_correctly():
    """같은 시각이 Z와 +00:00, 소수초 유무로 다르게 적힐 수 있다. 문자열로 비교하면
    0.5초 더 최신인 값이 더 오래된 값으로 판정된다."""
    older = scorecard.parse_ts("2026-09-13T10:00:00Z")
    newer = scorecard.parse_ts("2026-09-13T10:00:00.500000+00:00")
    assert older < newer
    assert "2026-09-13T10:00:00Z" > "2026-09-13T10:00:00.500000+00:00"  # 사전순은 반대다
    assert scorecard.parse_ts(None) is None
    assert scorecard.parse_ts("어제") is None


def test_entries_computed_after_kickoff_are_not_graded():
    """배치는 경기 하나를 계산하는 동안 뉴스 검색에 수십 초를 쓸 수 있어서, 킥오프 직전에
    검사를 통과한 뒤 경기가 시작된 다음 저장되는 순간이 있다. 그렇게 들어온 값을 "경기 전
    예측"으로 채점하면 성적표가 거짓이 된다."""
    late = {"match_id": 1, "kickoff_utc": "2026-09-12T14:00:00Z",
            "computed_at": "2026-09-12T14:05:00Z",
            "probabilities": {"HOME_TEAM": 0.9, "DRAW": 0.05, "AWAY_TEAM": 0.05}}
    early = dict(late, match_id=2, computed_at="2026-09-12T13:00:00Z")

    card = scorecard.build_scorecard([late, early], {1: "HOME_TEAM", 2: "HOME_TEAM"},
                                     {"HOME_TEAM": 0.43, "DRAW": 0.25, "AWAY_TEAM": 0.32})
    assert card["graded_matches"] == 1
    assert card["series"]["market"]["n"] == 1


def test_entries_without_computed_at_are_still_graded():
    """이 필드가 생기기 전에 저장된 항목은 검증할 방법이 없다. 확인되지 않은 것을 확인되어
    틀린 것처럼 버리면 지금까지 쌓인 기록이 전부 사라진다."""
    old = {"match_id": 1, "kickoff_utc": "2026-09-12T14:00:00Z",
           "probabilities": {"HOME_TEAM": 0.5, "DRAW": 0.25, "AWAY_TEAM": 0.25}}
    card = scorecard.build_scorecard([old], {1: "HOME_TEAM"},
                                    {"HOME_TEAM": 0.43, "DRAW": 0.25, "AWAY_TEAM": 0.32})
    assert card["graded_matches"] == 1


def test_baselines_are_scored_on_the_same_matches_as_the_market_model():
    """GenAI만 있는 경기가 baseline 표본에 섞이면, 표에 나란히 놓인 두 점수의 차이에
    "경기가 달라서 생긴 차이"가 들어가고 그게 모델 차이로 읽힌다."""
    with_market = {"match_id": 1, "probabilities": {"HOME_TEAM": 0.5, "DRAW": 0.25, "AWAY_TEAM": 0.25}}
    genai_only = {"match_id": 2, "genai_prediction": {
        "probabilities": {"HOME_TEAM": 0.4, "DRAW": 0.3, "AWAY_TEAM": 0.3}}}

    card = scorecard.build_scorecard([with_market, genai_only],
                                    {1: "HOME_TEAM", 2: "DRAW"},
                                    {"HOME_TEAM": 0.43, "DRAW": 0.25, "AWAY_TEAM": 0.32})
    assert card["series"]["market"]["n"] == 1
    assert card["series"]["genai"]["n"] == 1
    assert card["series"]["base_rate"]["n"] == 1     # union(2경기)이 아니라 market과 같은 1경기
    assert card["series"]["always_home"]["n"] == 1
