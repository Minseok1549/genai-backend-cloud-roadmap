"""매일 실행되는 배치 작업(Cloud Run Job): 그날 UTC 날짜에 예정된 EPL 경기를 찾아 예측하고
결과를 GCS에 JSON으로 저장한다. Cloud Scheduler가 하루 한 번 이 Job을 트리거한다.

/predict(api.py)와 예측 로직(predictor.py)을 공유하므로, API로 직접 호출했을 때와 같은
모델·같은 피처 계산 규칙으로 예측된다 — 결과가 저장소만 다를 뿐 갈라지지 않는다.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from data import load_fixtures_on_date, load_matches
from fetch_data import ensure_all_seasons_cached
from logutil import log_json
from predictor import predict_match, load_model_bundle, UnknownTeamError, InsufficientFormError

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
GCS_BUCKET = os.environ.get("PREDICTIONS_BUCKET")


def build_daily_predictions(target_date: str) -> dict:
    ensure_all_seasons_cached()
    fixtures = load_fixtures_on_date(target_date)
    matches = load_matches()
    bundle = load_model_bundle(MODEL_PATH)

    predictions = []
    for fixture in fixtures:
        try:
            probabilities = predict_match(fixture["home_team"], fixture["away_team"], matches, bundle)
        except (UnknownTeamError, InsufficientFormError) as e:
            # 승격팀 등 폼 기록이 부족한 경기는 건너뛰고 계속 진행한다 — 경기 하나를
            # 예측할 수 없다고 그날 전체 배치가 실패하면 안 된다.
            log_json("warning", "daily prediction skipped for fixture", fixture_id=fixture["match_id"], error=str(e))
            continue
        predictions.append(
            {
                "match_id": fixture["match_id"],
                "kickoff_utc": fixture["kickoff_utc"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "probabilities": probabilities,
                "model_version": bundle["model_version"],
            }
        )

    return {
        "date": target_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
    }


def _predictions_blob(target_date: str):
    client = storage.Client()
    return client.bucket(GCS_BUCKET).blob(f"predictions/{target_date}.json")


def fetch_existing_predictions(target_date: str) -> dict | None:
    blob = _predictions_blob(target_date)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def merge_predictions(existing: dict | None, new_predictions: list[dict]) -> list[dict]:
    """같은 날 여러 번 실행돼도 먼저 기록된 예측이 사라지지 않게 병합한다.
    이번 실행에서 이미 끝나 fixture 조회에 안 잡히는 경기(오전 예측)는 기존 기록을
    그대로 보존하고, 겹치는 경기는 이번 실행 결과로 덮어쓴다."""
    by_id = {p["match_id"]: p for p in (existing["predictions"] if existing else [])}
    for p in new_predictions:
        by_id[p["match_id"]] = p
    return list(by_id.values())


def upload_to_gcs(payload: dict, target_date: str) -> str:
    if not GCS_BUCKET:
        raise RuntimeError("PREDICTIONS_BUCKET 환경변수가 없습니다")
    blob = _predictions_blob(target_date)
    blob.upload_from_string(json.dumps(payload, ensure_ascii=False, indent=2), content_type="application/json")
    return f"gs://{GCS_BUCKET}/predictions/{target_date}.json"


def main() -> None:
    target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = build_daily_predictions(target_date)
    log_json("info", "daily predictions built", date=target_date, fixture_count=len(payload["predictions"]))

    existing = fetch_existing_predictions(target_date) if GCS_BUCKET else None
    payload["predictions"] = merge_predictions(existing, payload["predictions"])

    if not payload["predictions"]:
        log_json("info", "no fixtures scheduled today, nothing to upload", date=target_date)
        return

    uri = upload_to_gcs(payload, target_date)
    log_json("info", "daily predictions uploaded", date=target_date, uri=uri, prediction_count=len(payload["predictions"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_json("error", "daily prediction job failed", error=str(e))
        sys.exit(1)
