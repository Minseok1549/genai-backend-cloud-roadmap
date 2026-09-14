"""매일 실행되는 배치 작업(Cloud Run Job): 앞으로 MAX_LOOKAHEAD_DAYS일 이내에 열리는 EPL
경기 중 예측 가능한(배당률이 열린) 경기를 찾아 예측하고 결과를 GCS에 날짜별 JSON으로
저장한다. Cloud Scheduler가 하루 한 번 이 Job을 트리거한다.

과거에는 "오늘 날짜" 경기만 대상으로 했다 — 그래서 배당률이 며칠 전부터 열려도 대시보드에는
당일이 될 때까지 예측이 안 보였고, /predict·/dashboard가 그 공백을 메우려고 요청마다
fetch_upcoming_odds()를 직접 불렀다. Cloud Run은 트래픽이 없으면 인스턴스를 죽이는데(scale
to zero), 그 로컬 캐시가 컨테이너 디스크에 있어서 콜드스타트마다 캐시가 사라지고 코드에
박혀 있는 6시간 TTL이 사실상 무의미해진다 — 캐시가 지키는 건 "한 인스턴스가 살아있는 동안의
중복 호출"뿐이고, 대시보드 공유·크롤러·수동 테스트처럼 요청이 뜨문뜨문 오는 상황에서는
요청마다 새 인스턴스가 떠서 매번 API를 다시 태울 수 있다.

그래서 예측을 배치가 미리 계산해 저장하는 쪽으로 소유권을 옮긴다: 배당률이 열리는 즉시
예측해두고, 이후로는 배당률이 실제로 바뀐 경우에만 다시 계산한다(_odds_changed). 매일 모든
예정 경기를 훑다 보면 GenAI 호출까지 매번 다시 태울 수 있어서, 변화가 없으면 기존 값
(GenAI 예측 포함)을 그대로 들고 간다.

/predict(api.py)와 예측 로직(predictor.py)을 공유하므로, API로 직접 호출했을 때와 같은
모델·같은 피처 계산 규칙으로 예측된다 — 결과가 저장소만 다를 뿐 갈라지지 않는다.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from data import load_upcoming_fixtures, load_matches, load_season_teams
from fetch_data import ensure_all_seasons_cached, current_season_cache_age_hours, COMPLETED_SEASONS
from logutil import log_json
from odds import fetch_upcoming_odds
from predictor import predict_match, load_model_bundle, UnknownTeamError, InsufficientFormError
from report import generate_genai_prediction
from scorecard import build_scorecard, base_rates, parse_ts

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
GCS_BUCKET = os.environ.get("PREDICTIONS_BUCKET")

MAX_LOOKAHEAD_DAYS = 14  # 배당률은 보통 이 안쪽에서 열린다 — 그보다 먼 경기는 폼 모델뿐이라 매일 다시 훑을 실익이 적다
ODDS_CHANGE_THRESHOLD = 0.01  # 배당률 내재확률이 이 값(1%p) 미만으로 움직이면 "변화 없음"으로 보고 재계산하지 않는다
GENAI_WINDOW_HOURS = 24  # 킥오프 이 시간 이내(D-1)부터 GenAI 예측을 시도한다


def _odds_probabilities(odds: dict) -> dict:
    return {"HOME_TEAM": odds["odds_p_home"], "DRAW": odds["odds_p_draw"], "AWAY_TEAM": odds["odds_p_away"]}


def _odds_changed(prev: dict | None, current: dict | None) -> bool:
    """이전에 예측에 쓴 배당률과 이번에 받은 배당률을 비교한다. 있다/없다가 바뀐 경우(배당률이
    처음 열렸거나 반대로 API에서 빠진 경우)는 수치 비교가 의미 없으니 무조건 변화로 본다.
    저장된 스냅샷의 키 구성이 지금과 다르면(과거 스키마 등) 값 비교를 신뢰할 수 없으니
    안전하게 변화로 취급한다."""
    if (prev is None) != (current is None):
        return True
    if prev is None:
        return False
    if set(prev) != set(current):
        return True
    return any(abs(prev[k] - current[k]) >= ODDS_CHANGE_THRESHOLD for k in prev)


def _hours_until_kickoff(kickoff_utc: str) -> float:
    kickoff = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
    return (kickoff - datetime.now(timezone.utc)).total_seconds() / 3600


def _within_lookahead(kickoff_utc: str) -> bool:
    hours = _hours_until_kickoff(kickoff_utc)
    return 0 < hours <= MAX_LOOKAHEAD_DAYS * 24


def build_fixture_prediction(
    fixture: dict, matches, bundle: dict, fixture_odds: dict | None, existing: dict | None,
    known_teams: set | None = None,
) -> dict | None:
    """경기 하나의 예측 항목을 만든다. 통계(배당률) 모델은 배당률이 있을 때만 돌린다 —
    배당률 없이 팀 폼 기록만으로 만드는 예측은 정확도가 크게 떨어져 일별 배치에서는 쓰지
    않는다(반면 /predict 엔드포인트는 사용자가 즉시 요청한 경기에 최선의 답을 줘야 해서
    폼 모델 폴백을 그대로 유지한다 — 이 함수와는 별개).

    이번 실행에 배당률이 없으면(API 자체가 실패했든, 이 경기 마켓만 아직 안 열렸든) 이미
    만들어둔 통계 예측이 있어도 새로 계산하지 않고 그대로 얼려서 보존한다. 대신 킥오프
    24시간 이내(D-1)면 GenAI 예측이 통계 모델의 대체재 역할을 한다 — 뉴스·선수단 정보
    기반이라 배당률 없이도 독립적으로 판단할 수 있다.

    UnknownTeamError/InsufficientFormError는 호출자가 잡아 해당 fixture를 건너뛴다.
    통계 예측도 GenAI 예측도 만들 수 없으면(배당률도 없고 아직 D-1도 아님) 아직 보여줄
    게 없다는 뜻이라 None을 반환해 호출자가 이 경기를 건너뛰게 한다."""
    # 킥오프가 지난 경기는 손대지 않는다. 성적표는 "경기 전에 뭐라고 했는지"로 채점해야
    # 의미가 있는데, 킥오프 뒤에 숫자가 한 번이라도 바뀌면 그 채점은 사후 수정된 예측을
    # 채점하는 것이 된다. 정상 경로에서는 끝난 경기가 예정 경기 목록에서 빠져 여기까지
    # 오지 않지만, 외부 API의 상태 갱신이 늦어 킥오프가 지났는데도 SCHEDULED로 남아 있는
    # 경우가 있다 — 원장의 불변성은 그 지연에 의존하지 않아야 한다.
    if _hours_until_kickoff(fixture["kickoff_utc"]) <= 0:
        # 경기 전 기록이 없으면 사전 예측을 새로 만들 수 없다(이미 시작한 경기다) — 건너뛴다.
        return dict(existing) if existing else None

    odds_probs = _odds_probabilities(fixture_odds) if fixture_odds else None
    prev_odds_probs = existing.get("odds_snapshot") if existing else None
    had_stat_prediction = bool(existing and existing.get("probabilities") is not None)

    if odds_probs is not None:
        # 모델 버전이 바뀌었으면 배당률이 그대로여도 다시 계산한다. 이 조건이 없으면
        # train.py로 모델을 다시 학습해 배포해도, 배당률이 1%p 이상 움직이는 경기가 아니면
        # 저장된 예측은 옛 모델이 만든 숫자에 옛 model_version 딱지를 붙인 채로 남는다 —
        # "배포했는데 아무것도 안 바뀌는" 상태이자, 표시된 버전과 실제 산출물이 어긋나는 상태다.
        model_changed = existing is not None and existing.get("model_version") != bundle["model_version"]
        needs_recompute = (
            existing is None
            or not had_stat_prediction
            or model_changed
            or _odds_changed(prev_odds_probs, odds_probs)
        )
    else:
        needs_recompute = False  # 배당률이 없으면 통계 모델은 절대 새로 돌리지 않는다 — 폼 모델로 대체하지 않음

    if needs_recompute:
        probabilities = predict_match(
            fixture["home_team"], fixture["away_team"], matches, bundle,
            odds=fixture_odds, known_teams=known_teams,
        )

        entry = {
            "match_id": fixture["match_id"],
            "kickoff_utc": fixture["kickoff_utc"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "home_crest": fixture.get("home_crest"),
            "away_crest": fixture.get("away_crest"),
            "probabilities": probabilities,
            "model_version": bundle["model_version"],
            "odds_snapshot": odds_probs,
            # 이 확률이 몇 개 북메이커의 가격을 평균한 것인지 같이 남긴다. 3곳 평균과 15곳
            # 평균은 신뢰도가 다른데, 숫자만 보면 구분할 수 없다.
            "bookmaker_count": fixture_odds.get("bookmaker_count"),
        }
        # 배당률이 살짝 움직여 통계 예측을 다시 계산했더라도, 이미 만든 GenAI 예측은
        # 그대로 들고 온다 — 안 그러면 재계산 때마다 GenAI 예측이 통째로 사라진다.
        if existing and existing.get("genai_prediction"):
            entry["genai_prediction"] = existing["genai_prediction"]
    elif had_stat_prediction:
        entry = dict(existing)
    else:
        # 통계 예측이 아직 한 번도 없었다(배당률이 계속 없었음) — GenAI가 유일한 예측이 된다.
        entry = {
            "match_id": fixture["match_id"],
            "kickoff_utc": fixture["kickoff_utc"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "home_crest": fixture.get("home_crest"),
            "away_crest": fixture.get("away_crest"),
            "probabilities": None,
            "model_version": bundle["model_version"],
            "odds_snapshot": None,
        }
        if existing and existing.get("genai_prediction"):
            entry["genai_prediction"] = existing["genai_prediction"]

    # GenAI 예측: 킥오프 24시간 이내(D-1)인데 아직 만든 적 없으면 시도한다. 통계 예측
    # 유무와 무관하게 독립적으로 트리거된다 — 있으면 나란히 병기, 없으면(배당률 미공개)
    # 이게 유일한 예측이 된다. 이번 실행에서 통계 예측을 다시 계산했더라도, 이미 만든
    # GenAI 예측이 있으면 다시 부르지 않는다 — 하루 사이 배당률이 살짝 움직인 정도로
    # 뉴스·선발 정보까지 다시 검색할 필요는 없다.
    if not entry.get("genai_prediction"):
        hours_left = _hours_until_kickoff(fixture["kickoff_utc"])
        if 0 < hours_left <= GENAI_WINDOW_HOURS:
            try:
                genai = generate_genai_prediction(
                    fixture["home_team"], fixture["away_team"], fixture["kickoff_utc"],
                    fixture.get("referee"), entry.get("probabilities"),
                )
                entry["genai_prediction"] = {
                    "probabilities": genai["probabilities"],
                    "headline": genai["headline"],
                    "points": genai["points"],
                    "sources": genai["sources"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                log_json("warning", "genai prediction failed", fixture_id=fixture["match_id"], error=str(e))

    if entry.get("probabilities") is None and not entry.get("genai_prediction"):
        return None  # 통계 예측도 GenAI 예측도 없음 — 아직 보여줄 게 없어 건너뜀

    return entry


def _predictions_blob(target_date: str):
    client = storage.Client()
    return client.bucket(GCS_BUCKET).blob(f"predictions/{target_date}.json")


def fetch_existing_predictions(target_date: str) -> tuple[dict | None, int]:
    """저장된 예측과 함께 GCS 객체의 generation(버전 번호)을 반환한다. 이 번호는 나중에
    upload_to_gcs의 if_generation_match 조건("내가 읽은 버전 이후로 아무도 안 건드렸을 때만
    쓴다")에 쓰여서, 배치가 겹쳐 돌 때 한쪽이 다른 쪽 결과를 조용히 덮어쓰는 걸 막는다.
    객체가 아직 없으면 0을 반환하는데, GCS는 0을 "아직 존재하지 않을 때만 생성 허용"으로
    해석한다."""
    blob = _predictions_blob(target_date)
    try:
        blob.reload()
    except NotFound:
        return None, 0
    return json.loads(blob.download_as_text()), blob.generation


_OLDEST = datetime.min.replace(tzinfo=timezone.utc)


def _computed_at(entry: dict) -> datetime:
    """항목이 마지막으로 계산된 시각. 없으면(과거 스키마) 가장 오래된 것으로 취급한다."""
    return parse_ts(entry.get("computed_at")) or _OLDEST


def merge_predictions(existing: dict | None, new_predictions: list[dict]) -> list[dict]:
    """같은 날 여러 번 실행돼도 먼저 기록된 예측이 사라지지 않게 병합한다.
    이번 실행에서 이미 끝나 fixture 조회에 안 잡히는 경기(오전 예측)는 기존 기록을
    그대로 보존하고, 겹치는 경기는 계산 시각이 더 최근인 쪽을 남긴다.

    시각으로 비교하는 이유: 충돌 재시도에서 그냥 덮어쓰면, 내가 먼저 시작해 오래된 배당률로
    계산한 결과가 그 사이 다른 실행이 더 최신 배당률로 저장해둔 결과를 되돌려버린다. 같은
    경기를 두 실행이 모두 계산한 경우가 그렇다 — 어느 쪽이 나중에 쓰는지가 아니라 어느 쪽이
    나중에 계산됐는지가 기준이어야 한다."""
    by_id = {p["match_id"]: p for p in (existing["predictions"] if existing else [])}
    for p in new_predictions:
        current = by_id.get(p["match_id"])
        if current is None or _computed_at(p) >= _computed_at(current):
            by_id[p["match_id"]] = p
    return list(by_id.values())


def upload_to_gcs(payload: dict, target_date: str, if_generation_match: int) -> str:
    if not GCS_BUCKET:
        raise RuntimeError("PREDICTIONS_BUCKET 환경변수가 없습니다")
    blob = _predictions_blob(target_date)
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, indent=2),
        content_type="application/json",
        if_generation_match=if_generation_match,
    )
    return f"gs://{GCS_BUCKET}/predictions/{target_date}.json"


def build_predictions_by_date() -> dict[str, dict]:
    """예측 가능 창(MAX_LOOKAHEAD_DAYS일 이내) 안의 예정 경기를 훑어 킥오프 날짜별로 예측을
    만들고, 날짜별로 기존 저장분과 병합한 결과를 반환한다."""
    try:
        ensure_all_seasons_cached()
    except Exception as e:
        # error 레벨로 남긴다. 이 실패는 "오늘 경기 일정·결과가 갱신되지 않았다"는 뜻인데
        # 배치는 그래도 계속 진행해 Job은 성공으로 끝난다 — warning으로 묻어두면 토큰 만료
        # 같은 장애가 며칠씩 이어져도 로그에 아무 신호가 남지 않는다. 캐시 나이를 같이 남겨
        # "얼마나 오래된 데이터로 예측하고 있는지"를 로그만 보고 알 수 있게 한다.
        log_json("error", "season cache refresh failed, daily job using existing cache",
                 error=str(e), cache_age_hours=current_season_cache_age_hours())

    fixtures = [f for f in load_upcoming_fixtures() if _within_lookahead(f["kickoff_utc"])]
    matches = load_matches()
    season_teams = load_season_teams()
    bundle = load_model_bundle(MODEL_PATH)

    try:
        upcoming_odds = fetch_upcoming_odds()
    except Exception as e:
        log_json("warning", "live odds fetch failed, using existing predictions where available", error=str(e))
        upcoming_odds = {}

    fixtures_by_date: dict[str, list[dict]] = {}
    for fixture in fixtures:
        fixtures_by_date.setdefault(fixture["kickoff_utc"][:10], []).append(fixture)

    payload_by_date: dict[str, dict] = {}
    for date, date_fixtures in fixtures_by_date.items():
        existing, generation = fetch_existing_predictions(date) if GCS_BUCKET else (None, 0)
        existing_by_id = {p["match_id"]: p for p in (existing["predictions"] if existing else [])}

        predictions = []
        changed = []  # 이번 실행에서 실제로 값이 달라진 항목만 — 재시도 시 overlay로 쓴다
        for fixture in date_fixtures:
            fixture_odds = upcoming_odds.get((fixture["home_team"], fixture["away_team"]))
            if fixture_odds and fixture_odds.get("stale"):
                # odds.py가 API 장애로 TTL 지난 캐시를 대신 돌려준 경우 — 이번 실행엔 "진짜
                # 새 배당률"이 없는 것과 같이 취급해야 한다. 안 그러면 이미 더 최신(정상)
                # 배당률로 저장돼 있던 예측을, 오래된 숫자를 "바뀐 배당률"로 착각해 재계산으로
                # 되돌려버릴 수 있다.
                fixture_odds = None
            existing_entry = existing_by_id.get(fixture["match_id"])
            try:
                entry = build_fixture_prediction(
                    fixture, matches, bundle, fixture_odds, existing_entry, known_teams=season_teams,
                )
            except (UnknownTeamError, InsufficientFormError) as e:
                log_json("warning", "daily prediction skipped for fixture", fixture_id=fixture["match_id"], error=str(e))
                continue
            if entry is None:
                continue
            if entry != existing_entry:
                # 값이 실제로 달라진 항목에만 계산 시각을 새로 찍는다. 재사용된 항목은 저장돼
                # 있던 시각을 그대로 유지해야, 병합할 때 "언제 계산된 값인가"가 보존된다.
                entry["computed_at"] = datetime.now(timezone.utc).isoformat()
                changed.append(entry)
            predictions.append(entry)

        if date_fixtures and not predictions:
            log_json("warning", "all fixtures skipped, no predictions generated", date=date, fixture_count=len(date_fixtures))

        payload_by_date[date] = {
            "date": date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": merge_predictions(existing, predictions),
            # 아래 세 개는 내부용: main()이 업로드 시 동시 쓰기 감지·재병합에만 쓰고 저장 페이로드에서는 뺀다.
            # _new_predictions는 "이번 실행에서 값이 실제로 달라진 항목만" — 재시도할 때 이걸 다시
            # 읽어온 최신 existing 위에 얹어야, 다른 실행이 같은 시점에 갱신한 다른 match_id를 내 쪽의
            # (재시도 시점 기준) 오래된 스냅샷으로 덮어쓰지 않는다. 값이 그대로인 항목까지 넣으면
            # 바로 그 되돌림이 일어난다 — 내가 읽은 버전의 복사본을, 그 뒤에 갱신된 값 위에 다시
            # 쓰는 셈이기 때문이다.
            "_new_predictions": changed,
            "_generation": generation,
            # 저장된 내용과 실제로 달라졌는지 비교용 — 같으면 업로드를 건너뛴다(아래 main()).
            "_existing_predictions": existing["predictions"] if existing else None,
        }

    return payload_by_date


MAX_UPLOAD_RETRIES = 3  # 배치는 하루 한 번만 도는 게 정상이라 충돌은 거의 안 나야 정상 — 재시도는 예외적인 겹침 상황을 안전하게 넘기기 위한 최소한의 안전장치일 뿐, 크게 잡을 이유가 없다


SCORECARD_PATH = "scorecard.json"
HEARTBEAT_PATH = "batch/last_success.json"


def _all_prediction_dates() -> list[str]:
    client = storage.Client()
    blobs = client.list_blobs(GCS_BUCKET, prefix="predictions/")
    return sorted(b.name.removeprefix("predictions/").removesuffix(".json") for b in blobs)


def build_and_upload_scorecard() -> dict | None:
    """저장해둔 모든 예측을 실제 결과와 맞춰 채점하고 결과를 GCS에 올린다.

    매일 전체를 다시 계산한다. 누적 표본을 공개하려면 어차피 전 기간을 봐야 하고, 상태를
    따로 들고 가며 갱신하는 방식은 한 번 틀어지면 조용히 계속 틀린 숫자를 내놓기 때문이다.
    하루 한 번, 날짜별 작은 JSON 파일들을 읽는 정도라 비용도 무시할 수 있다.

    리그 평균 baseline은 채점 대상이 아닌 시즌에서만 계산한다 — 채점 대상 경기에서 뽑으면
    정답 분포를 미리 본 셈이라 baseline이 부당하게 유리해진다.

    날짜 파일 하나라도 못 읽으면 성적표를 만들지 않고 그만둔다. 일부만 모아서 올리면 어제보다
    표본이 줄어든 성적표가 정상본을 덮어쓰는데, 로그를 따로 보지 않으면 그게 "성적이 바뀐 것"과
    구분되지 않는다. 다음 실행에서 다시 시도하면 되는 일이다."""
    dates = _all_prediction_dates()
    entries = []
    for date in dates:
        payload, _ = fetch_existing_predictions(date)
        if payload:
            entries.extend(payload.get("predictions") or [])

    # 같은 경기가 여러 날짜 파일에 들어 있을 수 있다. 일정이 다른 UTC 날짜로 변경되면 새 날짜에
    # 항목이 생기고 옛 날짜 파일은 남기 때문이다. 그대로 펼치면 결과 하나가 두 번 채점돼
    # 표본 수와 평균이 다 틀어진다 — 경기당 하나, 마지막에 계산된 것만 남긴다.
    latest_by_id: dict[int, dict] = {}
    for entry in entries:
        match_id = entry.get("match_id")
        current = latest_by_id.get(match_id)
        if current is None or _computed_at(entry) >= _computed_at(current):
            latest_by_id[match_id] = entry
    entries = list(latest_by_id.values())

    finished = load_matches()
    results_by_id = dict(zip(finished["match_id"], finished["result"]))

    # 채점 대상 경기가 속한 시즌은 prior에서 뺀다. 지금은 예측이 진행 중 시즌에만 있어서
    # COMPLETED_SEASONS와 겹치지 않지만, 시즌이 끝나고도 누적 기록을 계속 보여주는 설계이므로
    # 시즌이 넘어가는 순간 채점 대상 경기의 결과가 prior에 섞이게 된다.
    graded_seasons = set(finished.loc[finished["match_id"].isin(latest_by_id), "season"])
    prior_seasons = [s for s in COMPLETED_SEASONS if s not in graded_seasons]
    prior_results = load_matches(seasons=prior_seasons)["result"].tolist() if prior_seasons else []

    card = build_scorecard(entries, results_by_id, base_rates(prior_results))
    card["generated_at"] = datetime.now(timezone.utc).isoformat()
    card["date_range"] = [dates[0], dates[-1]] if dates else None

    blob = storage.Client().bucket(GCS_BUCKET).blob(SCORECARD_PATH)
    blob.upload_from_string(
        json.dumps(card, ensure_ascii=False, indent=2), content_type="application/json",
    )
    log_json("info", "scorecard uploaded", graded_matches=card["graded_matches"])
    return card


def write_heartbeat() -> None:
    """배치가 끝까지 성공했다는 사실과 그 시각을 GCS에 남긴다.

    이 파일은 "배치가 실패했다"가 아니라 "배치가 아예 돌지 않았다"를 알아차리기 위한 것이다.
    스케줄러가 멈추거나 삭제되면 Job은 실패조차 하지 않으므로 실패 알림에 걸리지 않고, 예측이
    며칠씩 비는 걸 아무도 모르게 된다. Cloud Monitoring의 '지표 부재(metric absence)' 조건으로
    잡으려 했지만 그 조건은 최대 23시간 30분까지만 기다릴 수 있어, 24시간마다 도는 배치에는
    정상인 날에도 매일 알림이 뜬다 — 그래서 부재 감시 대신 이 하트비트를 /health/batch가
    읽고 uptime check가 그 엔드포인트를 보는 구조로 돌렸다.

    예측 파일(predictions/YYYY-MM-DD.json)의 최신 날짜로 대신 판단하지 않는 이유: 예정 경기가
    없거나 배당률이 거의 안 움직인 날은 올릴 게 없어서 파일을 새로 쓰지 않는다. 그건 정상
    동작인데 "배치가 안 돌았다"와 구별되지 않는다. 그래서 성공 사실만 따로 기록한다."""
    payload = {"completed_at": datetime.now(timezone.utc).isoformat()}
    blob = storage.Client().bucket(GCS_BUCKET).blob(HEARTBEAT_PATH)
    blob.upload_from_string(json.dumps(payload), content_type="application/json")
    log_json("info", "batch heartbeat written", completed_at=payload["completed_at"])


def main() -> None:
    # 버킷 설정은 아무것도 하기 전에 확인한다. 전에는 upload_to_gcs()에서야 검사해서,
    # 설정이 빠진 상태로 돌면 배당률 API와 Gemini 검색 호출을 전부 소비한 뒤 마지막
    # 업로드 단계에서 실패했다 — 결과물은 하나도 안 남고 비용만 나가는 실행이 된다.
    if not GCS_BUCKET:
        raise RuntimeError("PREDICTIONS_BUCKET 환경변수가 없습니다")

    payload_by_date = build_predictions_by_date()

    if not payload_by_date:
        log_json("info", "no upcoming fixtures within lookahead window, nothing to upload")

    any_failed = False
    for date, payload in payload_by_date.items():
        if not payload["predictions"]:
            log_json("info", "no predictions for date, skipping upload", date=date)
            continue

        generation = payload.pop("_generation")
        new_predictions = payload.pop("_new_predictions")
        existing_predictions = payload.pop("_existing_predictions")

        # 예측 내용이 저장된 것과 완전히 같으면 올리지 않는다. 배치는 매일 도는데 배당률이
        # 1%p 미만으로만 움직인 날은 모든 항목이 그대로라, 올려도 바뀌는 건 generated_at
        # 하나뿐이다 — 그러면 대시보드는 "방금 생성됨"으로 보이지만 실제 숫자는 며칠 전
        # 것이라 신선도를 오해하게 만든다. 쓰기를 건너뛰면 표시되는 시각이 실제로 계산한
        # 시각을 가리키게 된다.
        if existing_predictions is not None and payload["predictions"] == existing_predictions:
            log_json("info", "predictions unchanged since last run, skipping upload", date=date)
            continue

        uri = None
        for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
            try:
                uri = upload_to_gcs(payload, date, if_generation_match=generation)
                break
            except PreconditionFailed:
                # 이 파일을 읽은 뒤로 다른 실행이 먼저 써버렸다는 뜻 — 최신 상태를 다시 읽어
                # "이번 실행이 새로 만든 항목만" 그 위에 다시 얹고 재시도한다(동시 쓰기 레이스 대응).
                # payload["predictions"](기존 병합 스냅샷 전체)를 overlay로 쓰면 다른 실행이 그 사이
                # 갱신한 match_id를 내가 들고 있던 낡은 버전으로 되돌려버리는 2차 레이스가 생긴다.
                log_json("warning", "concurrent write detected, re-merging and retrying", date=date, attempt=attempt)
                existing, generation = fetch_existing_predictions(date)
                payload["predictions"] = merge_predictions(existing, new_predictions)
        if uri is None:
            log_json("error", "upload failed after retries due to repeated concurrent writes", date=date)
            any_failed = True
            continue
        log_json("info", "daily predictions uploaded", date=date, uri=uri, prediction_count=len(payload["predictions"]))

    # 채점은 예정 경기가 없는 날에도 돌려야 한다 — 결과는 계속 들어오므로, 어제 끝난 경기가
    # 오늘 성적표에 반영돼야 한다. 실패해도 Job은 실패로 만들지 않는다: 그날의 주 산출물은
    # 예측이고, 성적표는 이미 저장된 데이터로 언제든 다시 만들 수 있는 파생물이다.
    try:
        build_and_upload_scorecard()
    except Exception as e:
        log_json("error", "scorecard build failed", error=str(e))

    if any_failed:
        # Cloud Run Job이 실패를 인지하고 재시도하도록 non-zero로 종료한다 — 로그만 남기고
        # 정상 종료하면 그날 예측이 저장 안 됐는데도 Job은 "성공"으로 기록돼 아무도 모르게 된다.
        sys.exit(1)

    write_heartbeat()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_json("error", "daily prediction job failed", error=str(e))
        sys.exit(1)
