import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from google.api_core.exceptions import PreconditionFailed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data as data_module  # noqa: E402
from data import load_fixtures_on_date  # noqa: E402
import daily_predict  # noqa: E402
from predictor import UnknownTeamError  # noqa: E402


def _iso(hours_from_now: float) -> str:
    dt = datetime.now(timezone.utc) + timedelta(hours=hours_from_now)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture(autouse=True)
def _no_real_scorecard_upload(monkeypatch):
    """main()은 마지막에 성적표를 만들어 GCS에 올린다. 테스트에서 막지 않으면 실제 인증·네트워크
    호출을 시도하고, main()이 그 예외를 삼키기 때문에 실패가 아니라 "조용히 느려짐"으로만
    나타난다. 채점 로직 자체는 test_scorecard.py에서 본다."""
    calls = []
    monkeypatch.setattr(daily_predict, "build_and_upload_scorecard", lambda: calls.append(1))
    return calls


def test_load_fixtures_on_date_filters_status_and_date(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "RAW_DIR", tmp_path)
    payload = {
        "matches": [
            {"id": 1, "status": "TIMED", "utcDate": "2026-09-05T11:30:00Z",
             "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
             "referees": [{"name": "John Referee", "type": "REFEREE"}]},
            {"id": 2, "status": "SCHEDULED", "utcDate": "2026-09-06T11:30:00Z",
             "homeTeam": {"name": "C"}, "awayTeam": {"name": "D"}},
            {"id": 3, "status": "FINISHED", "utcDate": "2026-09-05T09:00:00Z",
             "homeTeam": {"name": "E"}, "awayTeam": {"name": "F"},
             "score": {"fullTime": {"home": 1, "away": 0}, "winner": "HOME_TEAM"}},
            {"id": 4, "status": "POSTPONED", "utcDate": "2026-09-05T15:00:00Z",
             "homeTeam": {"name": "G"}, "awayTeam": {"name": "H"}},
        ]
    }
    (tmp_path / "matches_2026.json").write_text(json.dumps(payload))

    fixtures = load_fixtures_on_date("2026-09-05", season=2026)
    assert len(fixtures) == 1
    assert fixtures[0]["match_id"] == 1
    assert fixtures[0]["referee"] == "John Referee"  # referees 목록의 첫 번째 이름을 씀


def test_load_fixtures_on_date_referee_none_when_not_assigned(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "RAW_DIR", tmp_path)
    payload = {
        "matches": [
            {"id": 1, "status": "TIMED", "utcDate": "2026-09-05T11:30:00Z",
             "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}},
        ]
    }
    (tmp_path / "matches_2026.json").write_text(json.dumps(payload))

    fixtures = load_fixtures_on_date("2026-09-05", season=2026)
    assert fixtures[0]["referee"] is None


def test_load_upcoming_fixtures_sorted_by_kickoff(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "RAW_DIR", tmp_path)
    payload = {
        "matches": [
            {"id": 2, "status": "SCHEDULED", "utcDate": "2026-09-10T11:30:00Z",
             "homeTeam": {"name": "C"}, "awayTeam": {"name": "D"}},
            {"id": 1, "status": "TIMED", "utcDate": "2026-09-05T11:30:00Z",
             "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}},
            {"id": 3, "status": "FINISHED", "utcDate": "2026-09-01T09:00:00Z",
             "homeTeam": {"name": "E"}, "awayTeam": {"name": "F"},
             "score": {"fullTime": {"home": 1, "away": 0}, "winner": "HOME_TEAM"}},
        ]
    }
    (tmp_path / "matches_2026.json").write_text(json.dumps(payload))

    fixtures = data_module.load_upcoming_fixtures(season=2026)
    assert [f["match_id"] for f in fixtures] == [1, 2]


def test_load_matchday_info_returns_current_round_and_its_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "RAW_DIR", tmp_path)
    payload = {
        "matches": [
            {"id": 1, "status": "FINISHED", "utcDate": "2026-09-04T19:00:00Z", "matchday": 3,
             "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
             "score": {"fullTime": {"home": 2, "away": 1}, "winner": "HOME_TEAM"}},
            {"id": 2, "status": "TIMED", "utcDate": "2026-09-06T13:00:00Z", "matchday": 3,
             "homeTeam": {"name": "C"}, "awayTeam": {"name": "D"}},
            {"id": 3, "status": "SCHEDULED", "utcDate": "2026-09-13T13:00:00Z", "matchday": 4,
             "homeTeam": {"name": "E"}, "awayTeam": {"name": "F"}},
        ]
    }
    (tmp_path / "matches_2026.json").write_text(json.dumps(payload))

    info = data_module.load_matchday_info(season=2026)
    assert info["matchday"] == 3
    assert info["dates"] == ["2026-09-04", "2026-09-06"]
    assert [f["match_id"] for f in info["fixtures"]] == [1, 2]
    assert info["fixtures"][0]["status"] == "FINISHED"
    assert info["fixtures"][0]["score"] == {"home": 2, "away": 1}
    assert "score" not in info["fixtures"][1]  # 아직 안 끝난 경기는 score 필드 자체가 없음


ODDS = {"odds_p_home": 0.5, "odds_p_draw": 0.3, "odds_p_away": 0.2}
ODDS_MOVED = {"odds_p_home": 0.6, "odds_p_draw": 0.25, "odds_p_away": 0.15}
ODDS_BARELY_MOVED = {"odds_p_home": 0.505, "odds_p_draw": 0.298, "odds_p_away": 0.197}
STAT_PROBS = {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2}


def _fixture(hours_to_kickoff: float = 200, match_id: int = 1) -> dict:
    return {
        "match_id": match_id, "kickoff_utc": _iso(hours_to_kickoff),
        "home_team": "A", "away_team": "B", "home_crest": None, "away_crest": None, "referee": None,
    }


def test_build_fixture_prediction_computes_fresh_when_no_existing(monkeypatch):
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))

    entry = daily_predict.build_fixture_prediction(_fixture(200), pd.DataFrame(), {"model_version": "v1"}, ODDS, None)

    assert entry["probabilities"] == STAT_PROBS
    assert entry["odds_snapshot"] == {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2}
    assert "genai_prediction" not in entry  # 킥오프 24시간 밖이라 GenAI는 시도하지 않음


def test_build_fixture_prediction_reuses_existing_when_odds_unchanged(monkeypatch):
    def fail_predict(*a, **kw):
        raise AssertionError("배당률이 안 바뀌었으면 재계산하면 안 된다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)
    existing = {
        "match_id": 1, "probabilities": STAT_PROBS,
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    }

    entry = daily_predict.build_fixture_prediction(_fixture(200), pd.DataFrame(), {"model_version": "v1"}, ODDS, existing)

    assert entry["probabilities"] == STAT_PROBS


def test_build_fixture_prediction_reuses_existing_when_odds_move_below_threshold(monkeypatch):
    def fail_predict(*a, **kw):
        raise AssertionError("1%p 미만 변화는 재계산하면 안 된다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)
    existing = {
        "match_id": 1, "probabilities": STAT_PROBS,
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    }

    entry = daily_predict.build_fixture_prediction(
        _fixture(200), pd.DataFrame(), {"model_version": "v1"}, ODDS_BARELY_MOVED, existing
    )

    assert entry["probabilities"] == STAT_PROBS


def test_build_fixture_prediction_recomputes_when_odds_move_beyond_threshold(monkeypatch):
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15})
    existing = {
        "match_id": 1, "probabilities": STAT_PROBS,
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    }

    entry = daily_predict.build_fixture_prediction(
        _fixture(200), pd.DataFrame(), {"model_version": "v1"}, ODDS_MOVED, existing
    )

    assert entry["probabilities"] == {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15}


def test_build_fixture_prediction_recomputes_when_odds_first_appear(monkeypatch):
    """폼 모델로만 예측해둔 경기에 배당률이 처음 열리면(있다/없다가 바뀜) 수치 비교와
    무관하게 무조건 다시 계산해야 한다 — 모델 자체가 바뀌기 때문."""
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))
    existing = {"match_id": 1, "probabilities": {"HOME_TEAM": 0.4, "DRAW": 0.3, "AWAY_TEAM": 0.3}, "model_version": "v1", "odds_snapshot": None}

    entry = daily_predict.build_fixture_prediction(_fixture(200), pd.DataFrame(), {"model_version": "v1"}, ODDS, existing)

    assert entry["probabilities"] == STAT_PROBS


def test_build_fixture_prediction_triggers_genai_within_window(monkeypatch):
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))
    captured = {}

    def fake_genai(home, away, kickoff, referee, stat_probs):
        captured["stat_probs"] = stat_probs
        return {"probabilities": {"HOME_TEAM": 0.55, "DRAW": 0.25, "AWAY_TEAM": 0.2}, "headline": "근거", "points": [], "sources": []}

    monkeypatch.setattr(daily_predict, "generate_genai_prediction", fake_genai)

    entry = daily_predict.build_fixture_prediction(_fixture(10), pd.DataFrame(), {"model_version": "v1"}, ODDS, None)

    assert entry["genai_prediction"]["probabilities"] == {"HOME_TEAM": 0.55, "DRAW": 0.25, "AWAY_TEAM": 0.2}
    assert entry["genai_prediction"]["headline"] == "근거"
    assert captured["stat_probs"] == STAT_PROBS


def test_build_fixture_prediction_skips_genai_outside_window(monkeypatch):
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))

    def fail_genai(*a, **kw):
        raise AssertionError("24시간 밖이면 GenAI 예측을 시도하면 안 된다")

    monkeypatch.setattr(daily_predict, "generate_genai_prediction", fail_genai)

    entry = daily_predict.build_fixture_prediction(_fixture(48), pd.DataFrame(), {"model_version": "v1"}, ODDS, None)
    assert "genai_prediction" not in entry


def test_build_fixture_prediction_uses_genai_as_fallback_without_odds(monkeypatch):
    """배당률이 없는 경기는 통계 모델(폼 모델 포함)을 절대 돌리지 않는다. 대신 킥오프
    24시간 이내(D-1)면 GenAI 예측이 fallback으로 대신 만들어져야 한다 — 폼 모델로
    대체하지 않는다는 게 이번 설계의 핵심."""
    def fail_predict(*a, **kw):
        raise AssertionError("배당률 없는 경기는 통계 모델(폼 모델 포함)을 돌리면 안 된다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)
    captured = {}

    def fake_genai(home, away, kickoff, referee, stat_probs):
        captured["stat_probs"] = stat_probs
        return {"probabilities": {"HOME_TEAM": 0.55, "DRAW": 0.25, "AWAY_TEAM": 0.2}, "headline": "뉴스 기반 예측", "points": [], "sources": []}

    monkeypatch.setattr(daily_predict, "generate_genai_prediction", fake_genai)

    entry = daily_predict.build_fixture_prediction(_fixture(10), pd.DataFrame(), {"model_version": "v1"}, None, None)

    assert entry is not None
    assert entry["probabilities"] is None  # 통계 모델 예측은 만들지 않음
    assert entry["genai_prediction"]["headline"] == "뉴스 기반 예측"
    assert captured["stat_probs"] is None  # 참고할 통계 확률 자체가 없음


def test_build_fixture_prediction_returns_none_when_no_odds_and_outside_genai_window(monkeypatch):
    """배당률도 없고 킥오프도 아직 멀었으면(D-1 밖) 통계 모델도 GenAI도 돌릴 수 없다 —
    아직 보여줄 예측이 없다는 뜻이라 None을 반환해 호출자가 이 경기를 건너뛰게 한다."""
    def fail_predict(*a, **kw):
        raise AssertionError("배당률 없는 경기는 통계 모델을 돌리면 안 된다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)

    def fail_genai(*a, **kw):
        raise AssertionError("D-1 밖이면 GenAI도 시도하면 안 된다")

    monkeypatch.setattr(daily_predict, "generate_genai_prediction", fail_genai)

    entry = daily_predict.build_fixture_prediction(_fixture(200), pd.DataFrame(), {"model_version": "v1"}, None, None)
    assert entry is None


def test_build_fixture_prediction_freezes_existing_stat_prediction_when_odds_absent(monkeypatch):
    """전에는 배당률이 있어 통계 예측을 만들어뒀는데 이번 실행엔 배당률이 없는 경우
    (API 장애든, 이 경기 마켓만 아직 안 열렸든, 킥오프 임박해 마켓이 닫혔든) — 폼 모델로
    되돌리지 않고 마지막으로 만든 통계 예측을 그대로 얼려서 유지한다."""
    def fail_predict(*a, **kw):
        raise AssertionError("배당률이 없다고 폼 모델로 재계산하면 안 된다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)
    existing = {
        "match_id": 1, "probabilities": STAT_PROBS,
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    }

    entry = daily_predict.build_fixture_prediction(_fixture(200), pd.DataFrame(), {"model_version": "v1"}, None, existing)

    assert entry["probabilities"] == STAT_PROBS


def test_build_fixture_prediction_skips_genai_when_already_present(monkeypatch):
    def fail_predict(*a, **kw):
        raise AssertionError("배당률 불변이면 통계 예측도 재계산하지 않아야 한다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)

    def fail_genai(*a, **kw):
        raise AssertionError("이미 GenAI 예측이 있으면 다시 부르면 안 된다")

    monkeypatch.setattr(daily_predict, "generate_genai_prediction", fail_genai)
    existing = {
        "match_id": 1, "probabilities": STAT_PROBS,
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        "genai_prediction": {"probabilities": STAT_PROBS, "reasoning": "old", "sources": []},
    }

    entry = daily_predict.build_fixture_prediction(_fixture(10), pd.DataFrame(), {"model_version": "v1"}, ODDS, existing)
    assert entry["genai_prediction"]["reasoning"] == "old"


def test_build_fixture_prediction_genai_failure_does_not_drop_stat_prediction(monkeypatch):
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))

    def fake_genai(*a, **kw):
        raise RuntimeError("GEMINI_API_KEY not found in environment or .env")

    monkeypatch.setattr(daily_predict, "generate_genai_prediction", fake_genai)

    entry = daily_predict.build_fixture_prediction(_fixture(10), pd.DataFrame(), {"model_version": "v1"}, ODDS, None)
    assert entry["probabilities"] == STAT_PROBS
    assert "genai_prediction" not in entry


def test_build_fixture_prediction_preserves_genai_on_recompute(monkeypatch):
    """배당률이 살짝 움직여 통계 예측을 다시 계산하더라도, 이미 만들어둔 GenAI 예측은
    그대로 유지돼야 한다 — 재계산 때마다 GenAI 블록이 사라지면 안 된다."""
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15})

    def fail_genai(*a, **kw):
        raise AssertionError("기존 GenAI 예측이 있으면 재계산 중에도 다시 부르면 안 된다")

    monkeypatch.setattr(daily_predict, "generate_genai_prediction", fail_genai)
    existing = {
        "match_id": 1, "probabilities": STAT_PROBS,
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        "genai_prediction": {"probabilities": STAT_PROBS, "reasoning": "old genai", "sources": []},
    }

    entry = daily_predict.build_fixture_prediction(
        _fixture(10), pd.DataFrame(), {"model_version": "v1"}, ODDS_MOVED, existing
    )

    assert entry["probabilities"] == {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15}  # 통계 예측은 새로 계산됨
    assert entry["genai_prediction"]["reasoning"] == "old genai"  # GenAI는 보존됨


def test_build_fixture_prediction_retries_genai_when_field_explicitly_null(monkeypatch):
    """레거시 데이터에 genai_prediction 키가 null로 남아있는 경우(과거 실패)도
    "genai_prediction" not in entry 방식이면 영원히 재시도되지 않는다 — 값 자체로 판단해야 한다."""
    def fail_predict(*a, **kw):
        raise AssertionError("배당률 불변이면 확률은 재계산하면 안 된다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)
    captured = {}

    def fake_genai(home, away, kickoff, referee, stat_probs):
        captured["called"] = True
        return {"probabilities": {"HOME_TEAM": 0.55, "DRAW": 0.25, "AWAY_TEAM": 0.2}, "headline": "recovered", "points": [], "sources": []}

    monkeypatch.setattr(daily_predict, "generate_genai_prediction", fake_genai)
    existing = {
        "match_id": 1, "probabilities": STAT_PROBS,
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        "genai_prediction": None,
    }

    entry = daily_predict.build_fixture_prediction(_fixture(10), pd.DataFrame(), {"model_version": "v1"}, ODDS, existing)

    assert captured.get("called") is True
    assert entry["genai_prediction"]["headline"] == "recovered"


def test_build_fixture_prediction_treats_malformed_odds_snapshot_as_changed(monkeypatch):
    """저장된 odds_snapshot의 키 구성이 지금(HOME_TEAM/DRAW/AWAY_TEAM)과 다르면(과거
    스키마 등) 수치 비교를 신뢰할 수 없으니 안전하게 변화로 보고 재계산해야 한다."""
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))
    existing = {
        "match_id": 1, "probabilities": {"HOME_TEAM": 0.4, "DRAW": 0.3, "AWAY_TEAM": 0.3},
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3},  # AWAY_TEAM 키가 빠진 손상된 스냅샷
    }

    entry = daily_predict.build_fixture_prediction(_fixture(200), pd.DataFrame(), {"model_version": "v1"}, ODDS, existing)

    assert entry["probabilities"] == STAT_PROBS


def _patch_batch_common(monkeypatch, fixtures, odds=None, existing_by_date=None):
    monkeypatch.setattr(daily_predict, "ensure_all_seasons_cached", lambda: None)
    monkeypatch.setattr(daily_predict, "load_upcoming_fixtures", lambda: fixtures)
    monkeypatch.setattr(daily_predict, "load_matches", lambda: pd.DataFrame())
    # 팀 목록도 fixtures에서 만든다 — 안 그러면 실제 data/raw/ 캐시를 읽어 테스트가 로컬
    # 데이터 상태에 따라 흔들린다. 이 테스트들은 팀 검증이 아니라 배치 흐름을 본다.
    monkeypatch.setattr(daily_predict, "load_season_teams",
                        lambda: {t for f in fixtures for t in (f["home_team"], f["away_team"])})
    monkeypatch.setattr(daily_predict, "load_model_bundle", lambda path: {"model_version": "v1"})
    monkeypatch.setattr(daily_predict, "fetch_upcoming_odds", lambda: odds or {})
    # GenAI는 통계 예측 유무와 무관하게 D-1 안이면 독립적으로 트리거되므로, 배치 레벨
    # 테스트가 실제로 Gemini를 호출하지 않도록 기본값을 안전하게 목킹해둔다 — 아래 테스트들은
    # genai_prediction 내용 자체를 검증하지 않는다.
    monkeypatch.setattr(daily_predict, "generate_genai_prediction", lambda *a, **kw: {
        "probabilities": dict(STAT_PROBS), "headline": "genai", "points": [], "sources": [],
    })
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", "test-bucket")  # fetch_existing_predictions 호출 경로를 실제로 태우기 위함
    existing_by_date = existing_by_date or {}
    monkeypatch.setattr(daily_predict, "fetch_existing_predictions", lambda date: (existing_by_date.get(date), 1))


def test_build_predictions_by_date_groups_fixtures_by_kickoff_date(monkeypatch):
    fixtures = [_fixture(hours_to_kickoff=10, match_id=1), _fixture(hours_to_kickoff=250, match_id=2)]
    _patch_batch_common(monkeypatch, fixtures, odds={("A", "B"): ODDS})
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))

    payload_by_date = daily_predict.build_predictions_by_date()

    assert len(payload_by_date) == 2
    all_ids = {p["match_id"] for payload in payload_by_date.values() for p in payload["predictions"]}
    assert all_ids == {1, 2}


def test_build_predictions_by_date_filters_outside_lookahead_window(monkeypatch):
    fixtures = [_fixture(hours_to_kickoff=10, match_id=1), _fixture(hours_to_kickoff=24 * 20, match_id=2)]
    _patch_batch_common(monkeypatch, fixtures, odds={("A", "B"): ODDS})
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))

    payload_by_date = daily_predict.build_predictions_by_date()

    all_ids = {p["match_id"] for payload in payload_by_date.values() for p in payload["predictions"]}
    assert all_ids == {1}  # 20일 뒤 경기는 MAX_LOOKAHEAD_DAYS(14일) 밖이라 제외됨


def test_build_predictions_by_date_skips_failed_fixture_and_continues(monkeypatch):
    fixtures = [
        {"match_id": 1, "kickoff_utc": _iso(10), "home_team": "Unknown FC", "away_team": "B", "home_crest": None, "away_crest": None, "referee": None},
        _fixture(hours_to_kickoff=10, match_id=2),
    ]
    _patch_batch_common(monkeypatch, fixtures, odds={("Unknown FC", "B"): ODDS, ("A", "B"): ODDS})

    def fake_predict_match(home, away, matches, bundle, odds=None, known_teams=None):
        if home == "Unknown FC":
            raise UnknownTeamError(home)
        return dict(STAT_PROBS)

    monkeypatch.setattr(daily_predict, "predict_match", fake_predict_match)

    payload_by_date = daily_predict.build_predictions_by_date()
    predictions = next(iter(payload_by_date.values()))["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["match_id"] == 2
    assert predictions[0]["probabilities"] == STAT_PROBS


def test_build_predictions_by_date_continues_when_season_cache_refresh_fails(monkeypatch):
    fixtures = [_fixture(hours_to_kickoff=10, match_id=1)]
    _patch_batch_common(monkeypatch, fixtures, odds={("A", "B"): ODDS})

    def fake_ensure_cached():
        raise RuntimeError("football-data.org unreachable")

    monkeypatch.setattr(daily_predict, "ensure_all_seasons_cached", fake_ensure_cached)
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))

    payload_by_date = daily_predict.build_predictions_by_date()
    predictions = next(iter(payload_by_date.values()))["predictions"]
    assert len(predictions) == 1


def test_build_predictions_by_date_preserves_existing_when_odds_fetch_fails(monkeypatch):
    """fetch_upcoming_odds() 전체가 실패하면(API 장애) 이미 배당률 기반으로 만들어둔
    예측이 폼 모델 예측으로 덮어써지면 안 된다 — 이번 실행 결과를 그냥 버려야 한다."""
    fixture = _fixture(hours_to_kickoff=10, match_id=1)
    date = fixture["kickoff_utc"][:10]
    existing_payload = {
        "predictions": [{
            "match_id": 1, "kickoff_utc": fixture["kickoff_utc"], "home_team": "A", "away_team": "B",
            "probabilities": STAT_PROBS,
            "model_version": "v1",
            "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        }]
    }
    _patch_batch_common(monkeypatch, [fixture], existing_by_date={date: existing_payload})

    def fail_fetch_odds():
        raise RuntimeError("odds API down")

    monkeypatch.setattr(daily_predict, "fetch_upcoming_odds", fail_fetch_odds)

    def fail_predict(*a, **kw):
        raise AssertionError("배당률 API 장애 상황에서는 재계산하면 안 된다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)

    payload_by_date = daily_predict.build_predictions_by_date()
    predictions = payload_by_date[date]["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["probabilities"] == STAT_PROBS


def test_build_predictions_by_date_logs_warning_when_all_fixtures_skipped(monkeypatch):
    fixtures = [{"match_id": 1, "kickoff_utc": _iso(10), "home_team": "Unknown FC", "away_team": "B", "home_crest": None, "away_crest": None, "referee": None}]
    _patch_batch_common(monkeypatch, fixtures, odds={("Unknown FC", "B"): ODDS})

    def fake_predict_match(home, away, matches, bundle, odds=None, known_teams=None):
        raise UnknownTeamError(home)

    monkeypatch.setattr(daily_predict, "predict_match", fake_predict_match)

    logged = []
    monkeypatch.setattr(daily_predict, "log_json", lambda level, message, **fields: logged.append((level, message, fields)))

    payload_by_date = daily_predict.build_predictions_by_date()
    predictions = next(iter(payload_by_date.values()))["predictions"]
    assert predictions == []
    assert any(msg == "all fixtures skipped, no predictions generated" for _, msg, _ in logged)


def test_build_predictions_by_date_reuses_existing_via_gcs_lookup(monkeypatch):
    """build_predictions_by_date가 실제로 GCS에서 기존 예측을 읽어와 변화 감지에 쓰는지
    확인한다(build_fixture_prediction 단위 테스트는 existing을 직접 주입해서 검증하지만,
    여기서는 그 배선 자체가 맞는지를 본다)."""
    fixture = _fixture(hours_to_kickoff=10, match_id=1)
    date = fixture["kickoff_utc"][:10]
    existing_payload = {
        "predictions": [{
            "match_id": 1, "probabilities": STAT_PROBS,
            "model_version": "v1",
            "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        }]
    }
    _patch_batch_common(monkeypatch, [fixture], odds={("A", "B"): ODDS}, existing_by_date={date: existing_payload})

    def fail_predict(*a, **kw):
        raise AssertionError("배당률이 안 바뀌었으면 GCS의 기존 예측을 재사용해야 한다")

    monkeypatch.setattr(daily_predict, "predict_match", fail_predict)

    payload_by_date = daily_predict.build_predictions_by_date()
    predictions = payload_by_date[date]["predictions"]
    assert predictions[0]["probabilities"] == STAT_PROBS


def test_build_predictions_by_date_empty_when_no_fixtures(monkeypatch):
    _patch_batch_common(monkeypatch, [])
    payload_by_date = daily_predict.build_predictions_by_date()
    assert payload_by_date == {}


def test_merge_predictions_preserves_matches_missing_from_new_run():
    """오전 실행에서 기록된 예측이, 오후 재실행 시(해당 경기가 이미 끝나 fixture 조회에
    안 잡혀도) 사라지지 않고 그대로 남아야 한다."""
    existing = {
        "predictions": [
            {"match_id": 1, "home_team": "A", "away_team": "B", "probabilities": {}},
            {"match_id": 2, "home_team": "C", "away_team": "D", "probabilities": {}},
        ]
    }
    new_predictions = [
        {"match_id": 2, "home_team": "C", "away_team": "D", "probabilities": {"updated": True}},
        {"match_id": 3, "home_team": "E", "away_team": "F", "probabilities": {}},
    ]

    merged = daily_predict.merge_predictions(existing, new_predictions)
    by_id = {p["match_id"]: p for p in merged}
    assert set(by_id) == {1, 2, 3}
    assert by_id[2]["probabilities"] == {"updated": True}  # 겹치는 경기는 새 결과로 덮어씀


def test_merge_predictions_with_no_existing_file_returns_new_only():
    merged = daily_predict.merge_predictions(None, [{"match_id": 1, "probabilities": {}}])
    assert len(merged) == 1


class _FakeBlob:
    """storage.Blob을 흉내내는 테스트용 더미. GCS 동시 쓰기 레이스 방지 로직
    (fetch_existing_predictions/upload_to_gcs)이 generation 번호를 올바르게 읽고
    쓰는지를 실제 GCS 없이 검증하기 위함."""

    def __init__(self, exists=True, generation=1, content=None):
        self._exists = exists
        self.generation = generation
        self._content = content or {"predictions": []}
        self.uploaded = None
        self.upload_kwargs = None

    def reload(self):
        if not self._exists:
            from google.api_core.exceptions import NotFound
            raise NotFound("no such object")

    def download_as_text(self):
        return json.dumps(self._content)

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        self.uploaded = json.loads(data)
        self.upload_kwargs = {"content_type": content_type, "if_generation_match": if_generation_match}


def test_fetch_existing_predictions_returns_none_and_zero_generation_when_blob_missing(monkeypatch):
    monkeypatch.setattr(daily_predict, "_predictions_blob", lambda date: _FakeBlob(exists=False))
    existing, generation = daily_predict.fetch_existing_predictions("2026-09-06")
    assert existing is None
    assert generation == 0


def test_fetch_existing_predictions_returns_payload_and_generation_when_blob_exists(monkeypatch):
    fake = _FakeBlob(generation=7, content={"predictions": [{"match_id": 1}]})
    monkeypatch.setattr(daily_predict, "_predictions_blob", lambda date: fake)
    existing, generation = daily_predict.fetch_existing_predictions("2026-09-06")
    assert existing == {"predictions": [{"match_id": 1}]}
    assert generation == 7


def test_upload_to_gcs_passes_generation_precondition(monkeypatch):
    fake = _FakeBlob()
    monkeypatch.setattr(daily_predict, "_predictions_blob", lambda date: fake)
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", "test-bucket")
    daily_predict.upload_to_gcs({"predictions": []}, "2026-09-06", if_generation_match=3)
    assert fake.upload_kwargs["if_generation_match"] == 3


def test_main_retries_and_remerges_on_concurrent_write_conflict(monkeypatch):
    """다른 실행이 먼저 써서 generation이 바뀐 상황(PreconditionFailed)을 시뮬레이션한다.
    재시도 시 최신 상태(다른 실행이 추가한 match_id 99)를 다시 읽어와 이번 실행 결과와
    재병합한 뒤 새 generation으로 다시 써야 한다 — 동시 쓰기 레이스로 데이터가 조용히
    사라지지 않는지 확인."""
    payload_by_date = {
        "2026-09-06": {
            "date": "2026-09-06",
            "generated_at": "2026-09-06T00:00:00Z",
            "predictions": [{"match_id": 1, "probabilities": {}}],
            "_new_predictions": [{"match_id": 1, "probabilities": {}}],
            "_generation": 5,
            # 저장본과 내용이 달라야 업로드 경로를 타므로 빈 목록을 준다(변경 없음 스킵 조건 회피).
            "_existing_predictions": [],
        }
    }
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(daily_predict, "build_predictions_by_date", lambda: payload_by_date)

    calls = []

    def fake_upload(payload, date, if_generation_match):
        calls.append((dict(payload), if_generation_match))
        if len(calls) == 1:
            raise PreconditionFailed("conflict")
        return f"gs://bucket/predictions/{date}.json"

    monkeypatch.setattr(daily_predict, "upload_to_gcs", fake_upload)
    monkeypatch.setattr(
        daily_predict, "fetch_existing_predictions",
        lambda date: ({"predictions": [{"match_id": 99, "probabilities": {}}]}, 6),
    )
    logged = []
    monkeypatch.setattr(daily_predict, "log_json", lambda level, message, **fields: logged.append((level, message, fields)))

    daily_predict.main()

    assert len(calls) == 2
    assert calls[0][1] == 5  # 첫 시도는 최초 fetch 시점의 generation
    assert calls[1][1] == 6  # 재시도는 재조회한 새 generation
    merged_ids = {p["match_id"] for p in calls[1][0]["predictions"]}
    assert merged_ids == {1, 99}  # 다른 실행이 추가한 match_id 99가 사라지지 않고 병합됨
    assert any(msg == "concurrent write detected, re-merging and retrying" for _, msg, _ in logged)


def test_main_retry_does_not_revert_concurrent_update_to_untouched_match(monkeypatch):
    """Codex 리뷰에서 지적된 버그 재현: 재시도 시 overlay로 "이번 실행이 이번에 만든
    항목만"이 아니라 최초 병합 스냅샷 전체를 쓰면, 다른 실행이 그 사이 match_id 2를
    갱신한 내용을 내가 들고 있던 낡은 버전으로 되돌려버린다. 이번 실행은 match_id 1만
    건드렸으므로, 재시도 후 match_id 2는 반드시 "새로 읽은 최신 값"이어야 한다."""
    payload_by_date = {
        "2026-09-06": {
            "date": "2026-09-06",
            "generated_at": "2026-09-06T00:00:00Z",
            # 첫 병합 스냅샷: 이번 실행이 만든 match_id 1 + 최초 fetch 시점의 match_id 2(old report)
            "predictions": [
                {"match_id": 1, "report": "this run"},
                {"match_id": 2, "report": "stale, from initial fetch"},
            ],
            "_new_predictions": [{"match_id": 1, "report": "this run"}],
            "_generation": 5,
            # 저장본과 내용이 달라야 업로드 경로를 타므로 빈 목록을 준다(변경 없음 스킵 조건 회피).
            "_existing_predictions": [],
        }
    }
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(daily_predict, "build_predictions_by_date", lambda: payload_by_date)

    calls = []

    def fake_upload(payload, date, if_generation_match):
        calls.append(dict(payload))
        if len(calls) == 1:
            raise PreconditionFailed("conflict")
        return f"gs://bucket/predictions/{date}.json"

    monkeypatch.setattr(daily_predict, "upload_to_gcs", fake_upload)
    monkeypatch.setattr(
        daily_predict, "fetch_existing_predictions",
        # 다른 실행이 그 사이 match_id 2를 새 report로 갱신하고 match_id 99를 추가함
        lambda date: (
            {"predictions": [
                {"match_id": 2, "report": "updated by concurrent run"},
                {"match_id": 99, "report": "added by concurrent run"},
            ]},
            6,
        ),
    )
    monkeypatch.setattr(daily_predict, "log_json", lambda *a, **kw: None)

    daily_predict.main()

    by_id = {p["match_id"]: p for p in calls[1]["predictions"]}
    assert by_id[1]["report"] == "this run"  # 이번 실행 결과는 유지
    assert by_id[2]["report"] == "updated by concurrent run"  # 다른 실행의 최신 갱신이 되돌려지지 않음
    assert by_id[99]["report"] == "added by concurrent run"  # 다른 실행이 추가한 항목도 보존


def test_main_gives_up_after_max_retries_on_persistent_conflict(monkeypatch):
    payload_by_date = {
        "2026-09-06": {
            "date": "2026-09-06",
            "generated_at": "2026-09-06T00:00:00Z",
            "predictions": [{"match_id": 1, "probabilities": {}}],
            "_new_predictions": [{"match_id": 1, "probabilities": {}}],
            "_generation": 5,
            # 저장본과 내용이 달라야 업로드 경로를 타므로 빈 목록을 준다(변경 없음 스킵 조건 회피).
            "_existing_predictions": [],
        }
    }
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(daily_predict, "build_predictions_by_date", lambda: payload_by_date)

    calls = []

    def always_fail(payload, date, if_generation_match):
        calls.append(if_generation_match)
        raise PreconditionFailed("conflict")

    monkeypatch.setattr(daily_predict, "upload_to_gcs", always_fail)
    monkeypatch.setattr(daily_predict, "fetch_existing_predictions", lambda date: (None, 0))
    logged = []
    monkeypatch.setattr(daily_predict, "log_json", lambda level, message, **fields: logged.append((level, message, fields)))

    # 재시도를 다 써도 그날 업로드가 안 됐다는 걸 Cloud Run Job이 실패로 인식해야
    # 한다 — 로그만 남기고 조용히 성공 종료하면 아무도 모르게 그날 예측이 안 올라간다.
    with pytest.raises(SystemExit) as exc_info:
        daily_predict.main()
    assert exc_info.value.code == 1

    assert len(calls) == daily_predict.MAX_UPLOAD_RETRIES
    assert any(msg == "upload failed after retries due to repeated concurrent writes" for _, msg, _ in logged)


def test_fetch_upcoming_odds_marks_stale_cache_fallback(tmp_path, monkeypatch):
    """API 호출이 실패했는데 TTL 지난 캐시로 대신 응답하는 경우, 호출자가 "이건 방금
    받은 신선한 배당률이 아니다"를 구분할 수 있어야 한다 — 안 그러면 daily_predict가
    이미 최신 배당률로 저장해둔 예측을, 오래된 숫자를 "바뀐 배당률"로 착각해 되돌린다."""
    import odds as odds_module

    cache_path = tmp_path / "odds_live_cache.json"
    cache_path.write_text(json.dumps([
        {"home_team": "A", "away_team": "B", "odds_p_home": 0.5, "odds_p_draw": 0.3, "odds_p_away": 0.2, "bookmaker_count": 3}
    ]))
    old_mtime = time.time() - odds_module.LIVE_CACHE_TTL_SECONDS - 3600
    os.utime(cache_path, (old_mtime, old_mtime))

    monkeypatch.setattr(odds_module, "ODDS_LIVE_CACHE_PATH", cache_path)

    def fail_api(api_key):
        raise RuntimeError("odds API down")

    monkeypatch.setattr(odds_module, "_fetch_live_odds_from_api", fail_api)
    monkeypatch.setattr(odds_module, "load_odds_api_key", lambda: "key")

    result = odds_module.fetch_upcoming_odds()
    assert result[("A", "B")]["stale"] is True


def test_build_fixture_prediction_recomputes_when_model_version_changes(monkeypatch):
    """배당률이 그대로여도 모델 버전이 바뀌면 다시 계산해야 한다. 이 조건이 없으면 재학습한
    모델을 배포해도 배당률이 1%p 이상 움직이는 경기 말고는 저장된 예측이 옛 모델 결과로
    남고, 표시되는 model_version과 실제 산출물이 어긋난다."""
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: {"HOME_TEAM": 0.7, "DRAW": 0.2, "AWAY_TEAM": 0.1})
    existing = {
        "match_id": 1, "probabilities": STAT_PROBS,
        "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    }

    entry = daily_predict.build_fixture_prediction(
        _fixture(200), pd.DataFrame(), {"model_version": "v2"}, ODDS, existing
    )

    assert entry["probabilities"] == {"HOME_TEAM": 0.7, "DRAW": 0.2, "AWAY_TEAM": 0.1}
    assert entry["model_version"] == "v2"


def test_main_skips_upload_when_predictions_unchanged(monkeypatch):
    """저장된 내용과 완전히 같으면 올리지 않는다 — 올려도 바뀌는 건 generated_at 하나뿐이라,
    며칠 전 숫자가 "방금 생성됨"으로 보이게 만들기만 한다."""
    predictions = [{"match_id": 1, "probabilities": {}}]
    payload_by_date = {
        "2026-09-06": {
            "date": "2026-09-06",
            "generated_at": "2026-09-06T00:00:00Z",
            "predictions": list(predictions),
            "_new_predictions": list(predictions),
            "_generation": 5,
            "_existing_predictions": list(predictions),
        }
    }
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(daily_predict, "build_predictions_by_date", lambda: payload_by_date)

    def fail_upload(*a, **kw):
        raise AssertionError("내용이 안 바뀌었으면 업로드하면 안 된다")

    monkeypatch.setattr(daily_predict, "upload_to_gcs", fail_upload)
    monkeypatch.setattr(daily_predict, "log_json", lambda *a, **kw: None)

    daily_predict.main()  # 예외 없이 끝나야 한다


def test_main_fails_fast_when_bucket_missing(monkeypatch):
    """버킷 설정이 없으면 배당률·Gemini 호출을 시작하기 전에 끝나야 한다 — 전에는 비용을
    전부 쓴 뒤 업로드 단계에서 실패해 결과물 없이 돈만 나갔다."""
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", None)

    def fail_build():
        raise AssertionError("버킷 검사 전에 예측을 만들면 안 된다")

    monkeypatch.setattr(daily_predict, "build_predictions_by_date", fail_build)

    with pytest.raises(RuntimeError, match="PREDICTIONS_BUCKET"):
        daily_predict.main()


def test_load_season_teams_includes_teams_without_finished_matches(tmp_path, monkeypatch):
    """팀 소속 판정은 일정표로 한다. 완료된 경기로만 판단하면 시즌 첫 경기가 한 경기 끝난
    시점에 그 경기에 나온 두 팀만 인정돼, 나머지 팀이 모두 "알 수 없는 팀"이 된다."""
    monkeypatch.setattr(data_module, "RAW_DIR", tmp_path)
    payload = {
        "matches": [
            {"id": 1, "status": "FINISHED", "utcDate": "2026-08-15T19:00:00Z",
             "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
             "score": {"fullTime": {"home": 1, "away": 0}, "winner": "HOME_TEAM"}},
            {"id": 2, "status": "SCHEDULED", "utcDate": "2026-08-22T14:00:00Z",
             "homeTeam": {"name": "C"}, "awayTeam": {"name": "D"}},
        ]
    }
    (tmp_path / "matches_2026.json").write_text(json.dumps(payload))

    assert data_module.load_season_teams(season=2026) == {"A", "B", "C", "D"}


def test_build_predictions_by_date_excludes_reused_entries_from_retry_overlay(monkeypatch):
    """재시도 overlay(_new_predictions)에는 이번 실행에서 값이 실제로 달라진 항목만 들어가야
    한다. 값이 그대로인 항목까지 넣으면, 충돌 재시도가 "내가 읽은 버전의 복사본"을 그 뒤에
    다른 실행이 갱신한 값 위에 다시 써서 최신 예측을 과거 값으로 되돌린다."""
    fixtures = [_fixture(hours_to_kickoff=200, match_id=1), _fixture(hours_to_kickoff=200, match_id=2)]
    date = fixtures[0]["kickoff_utc"][:10]
    reused = {
        "match_id": 2, "kickoff_utc": fixtures[1]["kickoff_utc"],
        "home_team": "A", "away_team": "B", "home_crest": None, "away_crest": None,
        "probabilities": dict(STAT_PROBS), "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    }
    existing_by_date = {date: {"predictions": [reused]}}
    _patch_batch_common(monkeypatch, fixtures, odds={("A", "B"): ODDS}, existing_by_date=existing_by_date)
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))

    payload = daily_predict.build_predictions_by_date()[date]

    # 경기 2는 배당률이 그대로여서 저장본을 재사용했다 — overlay에 들어가면 안 된다.
    assert {p["match_id"] for p in payload["_new_predictions"]} == {1}
    # 저장 대상 목록에는 둘 다 있어야 한다.
    assert {p["match_id"] for p in payload["predictions"]} == {1, 2}


def test_kickoff_passed_prediction_is_frozen(monkeypatch):
    """킥오프가 지난 경기는 다시 계산하지 않는다. 성적표는 "경기 전에 뭐라고 했는지"로
    채점해야 의미가 있는데, 킥오프 뒤에 숫자가 바뀌면 사후 수정된 예측을 채점하게 된다.
    정상 경로에선 끝난 경기가 예정 목록에서 빠지지만, 외부 API 상태 갱신이 늦으면 킥오프가
    지났는데도 SCHEDULED로 남는다 — 불변성이 그 지연에 의존해선 안 된다."""
    fixture = _fixture(hours_to_kickoff=-1, match_id=1)  # 한 시간 전에 시작했어야 하는 경기
    existing = {
        "match_id": 1, "kickoff_utc": fixture["kickoff_utc"],
        "home_team": "A", "away_team": "B", "home_crest": None, "away_crest": None,
        "probabilities": dict(STAT_PROBS), "model_version": "v1",
        "odds_snapshot": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    }
    monkeypatch.setattr(daily_predict, "predict_match",
                        lambda *a, **kw: pytest.fail("킥오프 뒤에 예측을 다시 계산했다"))
    monkeypatch.setattr(daily_predict, "generate_genai_prediction",
                        lambda *a, **kw: pytest.fail("킥오프 뒤에 GenAI를 호출했다"))

    # 배당률이 크게 움직였더라도(재계산 조건 충족) 손대지 않아야 한다.
    moved_odds = {"odds_p_home": 0.9, "odds_p_draw": 0.05, "odds_p_away": 0.05}
    entry = daily_predict.build_fixture_prediction(
        fixture, None, {"model_version": "v2"}, moved_odds, existing,
    )
    assert entry == existing

    # 경기 전 기록이 아예 없으면 사전 예측을 새로 만들 수 없다 — 건너뛴다.
    assert daily_predict.build_fixture_prediction(
        fixture, None, {"model_version": "v2"}, moved_odds, None,
    ) is None


def test_main_builds_scorecard_even_without_upcoming_fixtures(monkeypatch, _no_real_scorecard_upload):
    """예정 경기가 없는 날에도 채점은 돌아야 한다 — 결과는 계속 들어오므로 어제 끝난 경기가
    오늘 성적표에 반영돼야 한다. 전에는 여기서 바로 return해서 성적표가 갱신되지 않았다."""
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(daily_predict, "build_predictions_by_date", lambda: {})

    daily_predict.main()

    assert _no_real_scorecard_upload == [1]


def test_main_survives_scorecard_failure(monkeypatch):
    """성적표는 이미 저장된 데이터로 언제든 다시 만들 수 있는 파생물이다. 채점이 실패해도
    그날의 주 산출물(예측)이 올라갔다면 Job을 실패로 만들지 않는다."""
    monkeypatch.setattr(daily_predict, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(daily_predict, "build_predictions_by_date", lambda: {})

    def boom():
        raise RuntimeError("GCS 장애")

    monkeypatch.setattr(daily_predict, "build_and_upload_scorecard", boom)
    logged = []
    monkeypatch.setattr(daily_predict, "log_json", lambda level, message, **f: logged.append((level, message)))

    daily_predict.main()  # SystemExit이 나면 안 된다

    assert ("error", "scorecard build failed") in logged


def test_merge_keeps_newer_computation_for_same_match(monkeypatch):
    """같은 경기를 두 실행이 모두 계산한 경우, 나중에 쓰는 쪽이 아니라 나중에 계산된 쪽이
    남아야 한다. 충돌 재시도는 내가 먼저(오래된 배당률로) 계산한 결과를 들고 다시 쓰는데,
    그냥 match_id로 덮으면 그 사이 다른 실행이 최신 배당률로 저장해둔 값을 되돌린다."""
    def entry(computed_at, p_home):
        return {"match_id": 1, "computed_at": computed_at,
                "probabilities": {"HOME_TEAM": p_home, "DRAW": 0.2, "AWAY_TEAM": 0.8 - p_home}}

    older = entry("2026-09-13T10:00:00+00:00", 0.50)  # 내 실행이 먼저 계산한 값
    newer = entry("2026-09-13T10:05:00+00:00", 0.70)  # 그 사이 다른 실행이 저장한 값

    merged = daily_predict.merge_predictions({"predictions": [newer]}, [older])
    assert merged == [newer], "오래된 계산이 최신 계산을 덮었다"

    # 반대 방향은 정상 갱신이므로 새 값이 이겨야 한다.
    merged = daily_predict.merge_predictions({"predictions": [older]}, [newer])
    assert merged == [newer]


def test_cache_age_helper_returns_none_when_stat_fails(monkeypatch):
    """캐시 나이 계산은 갱신 실패를 기록하는 except 블록의 로그 인자로 불린다. 여기서 예외가
    새어나가면 원래 실패 원인이 가려지고, 기존 캐시로 계속 진행해야 할 배치가 중단된다."""
    import fetch_data

    def raising_stat(self, **kwargs):
        raise OSError("I/O error")

    monkeypatch.setattr(Path, "stat", raising_stat)
    assert fetch_data.current_season_cache_age_hours() is None


def test_build_predictions_by_date_logs_error_when_refresh_fails(monkeypatch):
    """일정 갱신 실패는 error로 남겨야 한다 — 배치는 그래도 성공으로 끝나므로, warning이면
    토큰 만료 같은 장애가 며칠 이어져도 로그에 신호가 남지 않는다."""
    fixtures = [_fixture(hours_to_kickoff=200, match_id=1)]
    _patch_batch_common(monkeypatch, fixtures, odds={("A", "B"): ODDS})
    monkeypatch.setattr(daily_predict, "predict_match", lambda *a, **kw: dict(STAT_PROBS))

    def failing_refresh():
        raise RuntimeError("Your API token is invalid.")

    monkeypatch.setattr(daily_predict, "ensure_all_seasons_cached", failing_refresh)
    logged = []
    monkeypatch.setattr(daily_predict, "log_json", lambda level, message, **f: logged.append((level, message, f)))

    daily_predict.build_predictions_by_date()

    refresh_logs = [entry for entry in logged if "season cache refresh failed" in entry[1]]
    assert refresh_logs, "갱신 실패 로그가 남지 않았다"
    assert refresh_logs[0][0] == "error"
    assert "cache_age_hours" in refresh_logs[0][2]


# autouse 픽스처가 모듈 속성을 스텁으로 바꾸기 때문에, 실제 함수 자체를 검사하려면 import
# 시점(픽스처가 돌기 전)에 원본을 붙잡아 둬야 한다.
_REAL_BUILD_SCORECARD = daily_predict.build_and_upload_scorecard


def _scorecard_env(monkeypatch, files: dict, matches, *, fail_dates=()):
    """성적표 생성에 필요한 GCS·데이터 접근을 모두 스텁으로 바꾸고, build_scorecard에
    실제로 넘어간 인자를 돌려준다."""
    captured = {}
    uploaded = []

    def fake_fetch(date):
        if date in fail_dates:
            raise RuntimeError("GCS 읽기 실패")
        return files.get(date), 1

    monkeypatch.setattr(daily_predict, "_all_prediction_dates", lambda: sorted(files))
    monkeypatch.setattr(daily_predict, "fetch_existing_predictions", fake_fetch)
    monkeypatch.setattr(daily_predict, "load_matches",
                        lambda seasons=None: matches if seasons is None
                        else matches[matches["season"].isin(seasons)])

    def fake_build(entries, results_by_id, prior):
        captured["entries"] = entries
        captured["prior"] = prior
        return {"series": {}, "graded_matches": len(entries), "prior": prior}

    monkeypatch.setattr(daily_predict, "build_scorecard", fake_build)

    class FakeBlob:
        def upload_from_string(self, data, content_type=None):
            uploaded.append(data)

    class FakeBucket:
        def blob(self, name):
            return FakeBlob()

    class FakeClient:
        def bucket(self, name):
            return FakeBucket()

    monkeypatch.setattr(daily_predict.storage, "Client", FakeClient)
    captured["uploaded"] = uploaded
    return captured


def _pred(match_id, computed_at, probs=(0.5, 0.25, 0.25)):
    return {"match_id": match_id, "kickoff_utc": "2026-09-12T14:00:00Z",
            "computed_at": computed_at,
            "probabilities": dict(zip(["HOME_TEAM", "DRAW", "AWAY_TEAM"], probs))}


def _matches_df(rows):
    return pd.DataFrame([{"match_id": m, "result": r, "season": s} for m, r, s in rows])


def test_scorecard_counts_a_rescheduled_match_only_once(monkeypatch):
    """경기 일정이 다른 UTC 날짜로 옮겨지면 새 날짜 파일에 항목이 생기고 옛 날짜 파일은
    그대로 남는다. 두 파일을 그냥 펼치면 결과 하나가 두 번 채점돼 표본 수와 평균이 다 틀어진다."""
    files = {
        "2026-09-12": {"predictions": [_pred(1, "2026-09-11T06:00:00Z")]},
        "2026-09-13": {"predictions": [_pred(1, "2026-09-12T06:00:00Z")]},
    }
    captured = _scorecard_env(monkeypatch, files, _matches_df([(1, "HOME_TEAM", 2026)]))

    _REAL_BUILD_SCORECARD()

    assert len(captured["entries"]) == 1
    assert captured["entries"][0]["computed_at"] == "2026-09-12T06:00:00Z"  # 나중에 계산된 쪽


def test_scorecard_is_not_uploaded_when_a_date_cannot_be_read(monkeypatch):
    """일부만 모아서 올리면 표본이 줄어든 성적표가 정상본을 덮어쓴다. 로그를 따로 보지
    않으면 그게 "성적이 바뀐 것"과 구분되지 않는다 — 다음 실행에서 다시 시도하면 된다."""
    files = {
        "2026-09-12": {"predictions": [_pred(1, "2026-09-11T06:00:00Z")]},
        "2026-09-13": {"predictions": [_pred(2, "2026-09-12T06:00:00Z")]},
    }
    captured = _scorecard_env(monkeypatch, files, _matches_df([(1, "HOME_TEAM", 2026)]),
                              fail_dates=("2026-09-13",))

    with pytest.raises(RuntimeError):
        _REAL_BUILD_SCORECARD()
    assert captured["uploaded"] == []


def test_prior_excludes_seasons_that_contain_graded_matches(monkeypatch):
    """리그 평균 baseline을 채점 대상 경기에서 뽑으면 정답 분포를 미리 본 셈이 된다.
    지금은 예측이 진행 중 시즌에만 있어 겹치지 않지만, 시즌이 끝나고도 누적 기록을 계속
    보여주는 설계라 시즌이 넘어가는 순간 겹친다."""
    files = {"2026-09-12": {"predictions": [_pred(10, "2026-09-11T06:00:00Z")]}}
    matches = _matches_df([(10, "HOME_TEAM", 2025), (11, "AWAY_TEAM", 2025),
                           (12, "DRAW", 2024), (13, "DRAW", 2024)])
    monkeypatch.setattr(daily_predict, "COMPLETED_SEASONS", [2024, 2025])
    captured = _scorecard_env(monkeypatch, files, matches)

    _REAL_BUILD_SCORECARD()

    # 채점 대상 경기(10)가 2025 시즌이므로 prior는 2024 시즌만 — 전부 무승부다
    assert captured["prior"]["DRAW"] == 1.0


def test_merge_compares_times_not_strings(monkeypatch):
    """같은 시각이 Z와 +00:00, 소수초 유무로 다르게 적힐 수 있다. 문자열로 비교하면 0.5초
    더 최신인 예측이 더 오래된 것으로 판정돼 버려진다."""
    existing = {"predictions": [_pred(1, "2026-09-13T10:00:00Z", probs=(0.2, 0.3, 0.5))]}
    newer = _pred(1, "2026-09-13T10:00:00.500000+00:00", probs=(0.7, 0.2, 0.1))

    merged = daily_predict.merge_predictions(existing, [newer])

    assert merged[0]["probabilities"]["HOME_TEAM"] == 0.7
