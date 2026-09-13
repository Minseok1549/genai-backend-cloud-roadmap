"""저장된 예측을 실제 결과와 맞춰 채점한다.

왜 정확도만으로는 안 되는가: 우리가 내놓는 건 "홈 승"이 아니라 "홈 승 52%"라는 확률이다.
확률 예측을 정확도로만 재면, 항상 favorite만 고르는 예측기와 확률을 제대로 맞히는 예측기가
같은 점수를 받는다. 그래서 log loss와 RPS를 같이 낸다 — 둘 다 "확률을 얼마나 정직하게
말했는지"를 재는 지표(proper scoring rule)라, 자신 없는 경기에 과한 확신을 실으면 점수가
나빠진다.

비교 대상(baseline)도 같이 낸다. 점수 하나만 보면 그게 좋은지 알 수 없다 — "무조건 홈 승"
같은 무지성 규칙보다 나은지, 북메이커 시장 평균보다 나은지가 실제 판단 기준이다.

여기 있는 함수는 전부 순수 함수다(파일·네트워크 접근 없음). 배치가 데이터를 모아서 넘긴다.
"""
import math
from datetime import datetime, timezone

OUTCOMES = ["HOME_TEAM", "DRAW", "AWAY_TEAM"]  # RPS는 순서가 의미를 가진다(홈-무-원정)
_EPS = 1e-15  # log(0) = -inf 방지. 확률 0을 선언한 결과가 실제로 일어나면 벌점이 무한이 된다


def _normalized(probs: dict) -> list[float] | None:
    """확률 딕셔너리를 [홈, 무, 원정] 순서 리스트로 바꾼다. 세 결과가 다 있고 합이 0보다
    커야 채점할 수 있다 — 아니면 None을 주고 호출자가 그 항목을 건너뛴다.

    NaN·무한대도 걸러낸다. float()는 문자열 "nan"이나 "inf"를 예외 없이 받아들이는데,
    그 값이 통과하면 평균 계산에 섞여 성적표의 모든 숫자가 NaN이 되고, json.dumps는
    표준 JSON이 아닌 NaN 토큰을 그대로 써서 대시보드 쪽 파싱까지 깨진다. 값 하나가
    이상한 것과 성적표 전체가 못 쓰는 것은 다른 문제여야 한다."""
    if not isinstance(probs, dict) or any(o not in probs for o in OUTCOMES):
        return None
    try:
        values = [float(probs[o]) for o in OUTCOMES]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) and v >= 0 for v in values):
        return None
    total = sum(values)
    if not total > 0:
        return None
    return [v / total for v in values]


def parse_ts(value) -> datetime | None:
    """ISO 8601 문자열을 시각으로 바꾼다. 없거나 형식이 아니면 None.

    문자열끼리 비교하지 않고 굳이 파싱하는 이유: 같은 시각이 "…10:00:00Z"와
    "…10:00:00+00:00", 소수초 유무로 다르게 적힐 수 있고, 그때 사전순 비교는 시간순과
    어긋난다. 실제로 0.5초 더 최신인 값이 더 오래된 값으로 판정되는 조합이 있다."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def log_loss(probs: list[float], actual: str) -> float:
    """실제로 일어난 결과에 몇 %를 줬는지만 본다. 50%를 줬으면 0.69, 10%만 줬으면 2.30.
    확률이 낮았던 결과가 일어날수록 벌점이 급격히 커진다."""
    p = probs[OUTCOMES.index(actual)]
    return -math.log(max(p, _EPS))


def rps(probs: list[float], actual: str) -> float:
    """RPS(Ranked Probability Score). 홈-무-원정이 순서를 가진 결과라는 점을 반영한다 —
    홈 승 경기에서 무승부에 확률을 준 것은 원정 승에 준 것보다 덜 틀린 것으로 본다.
    0이 완벽, 1이 최악. 축구 예측 평가에서 관행적으로 쓰인다."""
    actual_vec = [1.0 if o == actual else 0.0 for o in OUTCOMES]
    cum_p = cum_a = 0.0
    total = 0.0
    for i in range(len(OUTCOMES) - 1):  # 마지막 누적합은 항상 1-1=0이라 빼도 같다
        cum_p += probs[i]
        cum_a += actual_vec[i]
        total += (cum_p - cum_a) ** 2
    return total / (len(OUTCOMES) - 1)


CALIBRATION_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                    (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]


def _calibration(pairs: list[tuple[float, bool]]) -> list[dict]:
    """"60%라고 말한 경기에서 실제로 60%쯤 일어났는가"를 구간별로 집계한다.

    세 결과를 한 통에 모아서 본다 — 홈 승 예측만 보면 표본이 1/3로 줄고, 무승부·원정 승
    확률이 과대·과소했는지는 안 보인다. 확률 하나하나가 "이 일이 일어날 가능성"에 대한
    주장이므로 같은 자격으로 채점하는 게 맞다."""
    out = []
    for low, high in CALIBRATION_BINS:
        # 마지막 구간만 상한을 포함한다 — 확률 1.0이 어느 구간에도 안 들어가는 걸 막는다
        in_bin = [(p, hit) for p, hit in pairs
                  if low <= p < high or (high == 1.0 and p == 1.0)]
        if not in_bin:
            continue
        out.append({
            "range": f"{int(low * 100)}-{int(high * 100)}%",
            "n": len(in_bin),
            "predicted": round(sum(p for p, _ in in_bin) / len(in_bin), 4),
            "actual": round(sum(1 for _, hit in in_bin if hit) / len(in_bin), 4),
        })
    return out


def score_series(scored: list[tuple[list[float], str]]) -> dict | None:
    """(확률, 실제 결과) 쌍들을 받아 지표 묶음을 낸다. 표본이 없으면 None."""
    if not scored:
        return None
    n = len(scored)
    hits = sum(1 for probs, actual in scored
               if OUTCOMES[max(range(3), key=lambda i: probs[i])] == actual)
    pairs = [(probs[i], OUTCOMES[i] == actual) for probs, actual in scored for i in range(3)]
    return {
        "n": n,
        "accuracy": round(hits / n, 4),
        "log_loss": round(sum(log_loss(p, a) for p, a in scored) / n, 4),
        "rps": round(sum(rps(p, a) for p, a in scored) / n, 4),
        "calibration": _calibration(pairs),
    }


def base_rates(results: list[str]) -> dict:
    """과거 결과에서 홈 승·무·원정 승 비율을 뽑는다. baseline 확률로 쓴다."""
    n = len(results)
    if n == 0:
        return {o: 1 / 3 for o in OUTCOMES}
    return {o: results.count(o) / n for o in OUTCOMES}


# 예측 항목에서 채점 대상 확률을 꺼내는 방법. 이름이 곧 성적표에 표시되는 모델 구분이다.
_SERIES = {
    "market": lambda e: e.get("probabilities"),
    "genai": lambda e: (e.get("genai_prediction") or {}).get("probabilities"),
    "bookmaker": lambda e: e.get("odds_snapshot"),
}

SERIES_LABELS = {
    "market": "시장 보정 모델",
    "genai": "뉴스 시나리오 (GenAI)",
    "bookmaker": "북메이커 평균 (baseline)",
    "base_rate": "리그 평균 확률 (baseline)",
    "always_home": "무조건 홈 승 (baseline)",
}


def is_pre_kickoff(entry: dict) -> bool:
    """이 예측이 정말 킥오프 전에 계산된 것인지 확인한다.

    성적표의 전제는 "경기 전에 뭐라고 했는지"를 채점한다는 것이다. 그 전제를 만드는 쪽(배치)의
    사전 검사에만 의존할 수는 없다 — 검사하는 시점과 저장하는 시점이 다르기 때문이다. 배치는
    경기 하나를 계산하는 동안 뉴스 검색에 수십 초를 쓸 수 있어서, 킥오프 직전에 검사를 통과한
    뒤 경기가 시작된 다음에 저장하는 순간이 존재한다. 그러면 경기 시작 후 정보가 섞인 값이
    "경기 전 예측"으로 채점된다. 채점하는 쪽에서 한 번 더 본다.

    계산 시각이 기록돼 있지 않은 과거 항목은 검증할 방법이 없으므로 통과시킨다. 확인되지
    않은 것을 확인되어 틀린 것처럼 버리면, 이 필드가 생기기 전에 쌓인 기록이 전부 사라진다."""
    computed_at = parse_ts(entry.get("computed_at"))
    kickoff = parse_ts(entry.get("kickoff_utc"))
    if computed_at is None or kickoff is None:
        return True
    return computed_at < kickoff


def build_scorecard(entries: list[dict], results_by_id: dict[int, str], prior: dict) -> dict:
    """채점표를 만든다.

    entries: 저장된 예측 항목들(여러 날짜를 합친 것). 킥오프 전 마지막 스냅샷이다.
    results_by_id: match_id -> 실제 결과. 여기 없는 경기는 아직 안 끝난 것이라 채점에서 빠진다.
    prior: 리그 평균 확률(baseline). 평가 대상 경기에서 뽑으면 정답을 미리 본 셈이 되므로
           호출자가 채점 대상이 아닌 시즌에서 계산해 넘긴다.
    """
    collected: dict[str, list] = {name: [] for name in _SERIES}
    graded_results = []

    for entry in entries:
        actual = results_by_id.get(entry.get("match_id"))
        if actual not in OUTCOMES:
            continue
        if not is_pre_kickoff(entry):
            continue
        graded = False
        for name, getter in _SERIES.items():
            probs = _normalized(getter(entry))
            if probs is not None:
                collected[name].append((probs, actual))
                graded = True
        if graded:
            graded_results.append(actual)

    series = {}
    for name, scored in collected.items():
        metrics = score_series(scored)
        if metrics:
            series[name] = metrics

    # baseline은 주력 모델(market)이 채점된 경기와 똑같은 표본에서 재야 비교가 성립한다.
    # "어느 series든 하나라도 있으면 포함"으로 모으면, 예컨대 GenAI만 있는 경기가 섞여
    # market은 100경기인데 baseline은 120경기가 된다 — 그러면 표에 나란히 놓인 두 점수의
    # 차이에 "경기가 달라서 생긴 차이"가 섞이고, 그게 모델 차이로 읽힌다.
    baseline_results = [actual for _, actual in collected["market"]]
    if baseline_results:
        prior_vec = _normalized(prior) or [1 / 3] * 3
        series["base_rate"] = score_series([(prior_vec, a) for a in baseline_results])
        # 무조건 홈 승은 확률 1.0/0.0을 선언하는 규칙이라 log loss·RPS가 비교용으로 의미가
        # 없다(틀리면 벌점이 상한에 붙는다). 정확도만 남겨서 "이것보다 나은가"에만 쓴다.
        always_home = score_series([([1.0, 0.0, 0.0], a) for a in baseline_results])
        series["always_home"] = {"n": always_home["n"], "accuracy": always_home["accuracy"]}

    return {
        "series": series,
        "graded_matches": len(graded_results),
        "prior": {k: round(v, 4) for k, v in prior.items()},
    }
