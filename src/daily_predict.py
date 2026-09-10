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
예정 경기를 훑다 보면 안 그래도 Gemini 리포트까지 매번 다시 생성하게 되어 무료 티어를
예전보다 더 태울 수 있어서, 변화가 없으면 기존 값(리포트·GenAI 예측 포함)을 그대로 들고 간다.

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

from data import load_upcoming_fixtures, load_matches
from fetch_data import ensure_all_seasons_cached
from logutil import log_json
from odds import fetch_upcoming_odds
from predictor import predict_match, load_model_bundle, UnknownTeamError, InsufficientFormError
from report import generate_match_report, generate_genai_prediction

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
    odds_probs = _odds_probabilities(fixture_odds) if fixture_odds else None
    prev_odds_probs = existing.get("odds_snapshot") if existing else None
    had_stat_prediction = bool(existing and existing.get("probabilities") is not None)

    if odds_probs is not None:
        needs_recompute = existing is None or not had_stat_prediction or _odds_changed(prev_odds_probs, odds_probs)
    else:
        needs_recompute = False  # 배당률이 없으면 통계 모델은 절대 새로 돌리지 않는다 — 폼 모델로 대체하지 않음

    if needs_recompute:
        probabilities = predict_match(fixture["home_team"], fixture["away_team"], matches, bundle, odds=fixture_odds)

        try:
            report = generate_match_report(
                fixture["home_team"], fixture["away_team"], fixture["kickoff_utc"],
                fixture.get("referee"), probabilities,
            )
        except Exception as e:
            log_json("warning", "match report generation failed", fixture_id=fixture["match_id"], error=str(e))
            report = None

        entry = {
            "match_id": fixture["match_id"],
            "kickoff_utc": fixture["kickoff_utc"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "home_crest": fixture.get("home_crest"),
            "away_crest": fixture.get("away_crest"),
            "probabilities": probabilities,
            "model_version": bundle["model_version"],
            "report": report,
            "odds_snapshot": odds_probs,
        }
        # 배당률이 살짝 움직여 통계 예측을 다시 계산했더라도, 이미 만든 GenAI 예측은
        # 그대로 들고 온다 — 안 그러면 재계산 때마다 GenAI 예측이 통째로 사라진다.
        if existing and existing.get("genai_prediction"):
            entry["genai_prediction"] = existing["genai_prediction"]
    elif had_stat_prediction:
        entry = dict(existing)
        if entry.get("report") is None:
            try:
                entry["report"] = generate_match_report(
                    fixture["home_team"], fixture["away_team"], fixture["kickoff_utc"],
                    fixture.get("referee"), entry["probabilities"],
                )
            except Exception as e:
                log_json("warning", "match report generation failed", fixture_id=fixture["match_id"], error=str(e))
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
            "report": None,
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


def merge_predictions(existing: dict | None, new_predictions: list[dict]) -> list[dict]:
    """같은 날 여러 번 실행돼도 먼저 기록된 예측이 사라지지 않게 병합한다.
    이번 실행에서 이미 끝나 fixture 조회에 안 잡히는 경기(오전 예측)는 기존 기록을
    그대로 보존하고, 겹치는 경기는 이번 실행 결과로 덮어쓴다."""
    by_id = {p["match_id"]: p for p in (existing["predictions"] if existing else [])}
    for p in new_predictions:
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
        log_json("warning", "season cache refresh failed, daily job using existing cache", error=str(e))

    fixtures = [f for f in load_upcoming_fixtures() if _within_lookahead(f["kickoff_utc"])]
    matches = load_matches()
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
        for fixture in date_fixtures:
            fixture_odds = upcoming_odds.get((fixture["home_team"], fixture["away_team"]))
            if fixture_odds and fixture_odds.get("stale"):
                # odds.py가 API 장애로 TTL 지난 캐시를 대신 돌려준 경우 — 이번 실행엔 "진짜
                # 새 배당률"이 없는 것과 같이 취급해야 한다. 안 그러면 이미 더 최신(정상)
                # 배당률로 저장돼 있던 예측을, 오래된 숫자를 "바뀐 배당률"로 착각해 재계산으로
                # 되돌려버릴 수 있다.
                fixture_odds = None
            try:
                entry = build_fixture_prediction(
                    fixture, matches, bundle, fixture_odds, existing_by_id.get(fixture["match_id"]),
                )
            except (UnknownTeamError, InsufficientFormError) as e:
                log_json("warning", "daily prediction skipped for fixture", fixture_id=fixture["match_id"], error=str(e))
                continue
            if entry is None:
                continue
            predictions.append(entry)

        if date_fixtures and not predictions:
            log_json("warning", "all fixtures skipped, no predictions generated", date=date, fixture_count=len(date_fixtures))

        payload_by_date[date] = {
            "date": date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": merge_predictions(existing, predictions),
            # 아래 두 개는 내부용: main()이 업로드 시 동시 쓰기 감지·재병합에만 쓰고 저장 페이로드에서는 뺀다.
            # _new_predictions는 "이번 실행이 이번에 만든 항목만"(기존 스냅샷 제외) — 재시도할 때 이걸 다시
            # 읽어온 최신 existing 위에 얹어야, 다른 실행이 같은 시점에 갱신한 다른 match_id를 내 쪽의
            # (재시도 시점 기준) 오래된 스냅샷으로 덮어쓰지 않는다.
            "_new_predictions": predictions,
            "_generation": generation,
        }

    return payload_by_date


MAX_UPLOAD_RETRIES = 3  # 배치는 하루 한 번만 도는 게 정상이라 충돌은 거의 안 나야 정상 — 재시도는 예외적인 겹침 상황을 안전하게 넘기기 위한 최소한의 안전장치일 뿐, 크게 잡을 이유가 없다


def main() -> None:
    payload_by_date = build_predictions_by_date()

    if not payload_by_date:
        log_json("info", "no upcoming fixtures within lookahead window, nothing to upload")
        return

    any_failed = False
    for date, payload in payload_by_date.items():
        if not payload["predictions"]:
            log_json("info", "no predictions for date, skipping upload", date=date)
            continue

        generation = payload.pop("_generation")
        new_predictions = payload.pop("_new_predictions")
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

    if any_failed:
        # Cloud Run Job이 실패를 인지하고 재시도하도록 non-zero로 종료한다 — 로그만 남기고
        # 정상 종료하면 그날 예측이 저장 안 됐는데도 Job은 "성공"으로 기록돼 아무도 모르게 된다.
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_json("error", "daily prediction job failed", error=str(e))
        sys.exit(1)
