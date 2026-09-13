import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fastapi.testclient import TestClient
import api  # noqa: E402
from api import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _card(n=120):
    """비교가 의미 있는 규모(기본 120경기)의 성적표. n을 낮추면 소표본 경로를 볼 수 있다."""
    return {
        "generated_at": "2026-09-13T06:00:00Z",
        "date_range": ["2026-08-15", "2026-09-12"],
        "graded_matches": n,
        "series": {
            "market": {"n": n, "accuracy": 0.55, "log_loss": 0.961, "rps": 0.19,
                       "calibration": [{"range": "60-70%", "n": 12, "predicted": 0.65, "actual": 0.5}]},
            "genai": {"n": 12, "accuracy": 0.5, "log_loss": 1.05, "rps": 0.21, "calibration": []},
            "bookmaker": {"n": n, "accuracy": 0.55, "log_loss": 0.97, "rps": 0.195, "calibration": []},
            "base_rate": {"n": n, "accuracy": 0.45, "log_loss": 1.06, "rps": 0.22, "calibration": []},
            "always_home": {"n": n, "accuracy": 0.45},
        },
    }


def test_scorecard_page_shows_metrics_and_sample_sizes(client, monkeypatch):
    """표본 수 없이 적중률만 보여주면 12경기짜리 성적과 40경기짜리 성적을 같은 무게로
    읽게 된다 — 모델마다 채점 표본이 다르므로 반드시 같이 나와야 한다."""
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: _card())

    resp = client.get("/scorecard")
    assert resp.status_code == 200
    assert "0.961" in resp.text          # log loss
    assert "채점 경기 120경기" in resp.text
    assert "시장 보정 모델" in resp.text
    assert ">12<" in resp.text           # GenAI 표본 수가 별도로 표시된다


def test_scorecard_page_shows_baselines_for_comparison(client, monkeypatch):
    """점수 하나만 보면 그게 좋은지 알 수 없다 — 북메이커 평균과 단순 규칙이 같이 있어야
    '이것보다 나은가'로 읽힌다."""
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: _card())

    resp = client.get("/scorecard")
    assert "북메이커 평균 (baseline)" in resp.text
    assert "무조건 홈 승 (baseline)" in resp.text
    assert "리그 평균 확률 (baseline)" in resp.text


def test_scorecard_page_omits_log_loss_for_always_home_baseline(client, monkeypatch):
    """'무조건 홈 승'은 확률 1.0/0.0을 선언하는 규칙이라 log loss 비교가 의미 없다.
    빈 칸으로 남아야 하고, 여기서 KeyError로 페이지가 죽어서도 안 된다."""
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: _card())

    resp = client.get("/scorecard")
    assert resp.status_code == 200


def test_scorecard_page_shows_calibration_gap(client, monkeypatch):
    """65%라고 말한 12경기에서 50%만 일어났다면 과신이다 — 그 격차가 보여야 한다."""
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: _card())

    resp = client.get("/scorecard")
    assert "65.0%" in resp.text
    assert "50.0%" in resp.text
    assert "-15.0%p" in resp.text


def test_scorecard_page_handles_missing_data(client, monkeypatch):
    """배치가 아직 성적표를 만들지 않았거나 채점할 경기가 없는 상태 — 500이 아니라
    설명이 나와야 한다."""
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: None)

    resp = client.get("/scorecard")
    assert resp.status_code == 200
    assert "아직 채점된 예측이 없습니다" in resp.text


def _patch_dashboard(monkeypatch, prediction_extra=None, fixture_extra=None):
    fx = {
        "match_id": 1, "kickoff_utc": "2026-09-06T15:30:00Z",
        "home_team": "Arsenal FC", "away_team": "Chelsea FC",
        "home_crest": None, "away_crest": None, "status": "SCHEDULED",
    }
    fx.update(fixture_extra or {})
    pred = {"match_id": 1, "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15}}
    pred.update(prediction_extra or {})
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(api, "load_all_fixtures_on_date", lambda date_str: [fx])
    monkeypatch.setattr(api, "_fetch_daily_predictions",
                        lambda date_str: {"generated_at": "2026-09-06T06:00:00Z", "predictions": [pred]})
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: None)


def test_dashboard_shows_match_context(client, monkeypatch):
    """확률만 있는 카드는 '왜 그 숫자인지'를 볼 수 없다. 순위·최근 5경기·홈/원정 경기당
    승점이 카드에 있어야 사용자가 확률을 스스로 가늠할 수 있다."""
    _patch_dashboard(monkeypatch)
    monkeypatch.setattr(api, "match_context", lambda matches, home, away, season: {
        "home": {"table": {"rank": 3, "played": 4, "points": 9, "goal_diff": 5},
                 "recent": ["W", "W", "D"], "ppg": 2.5},
        "away": {"table": {"rank": 11, "played": 4, "points": 4, "goal_diff": -2},
                 "recent": ["L", "D"], "ppg": 0.5},
    })

    resp = client.get("/dashboard?date=2026-09-06")
    assert "3위" in resp.text and "11위" in resp.text
    assert "form-w" in resp.text and "form-l" in resp.text
    assert "2.50" in resp.text and "0.50" in resp.text


def test_dashboard_context_failure_does_not_break_page(client, monkeypatch):
    """맥락은 부가 정보다. 순위 계산이 실패해도 예측 확률은 계속 보여야 한다."""
    _patch_dashboard(monkeypatch)

    def boom(matches, home, away, season):
        raise ValueError("깨진 데이터")

    monkeypatch.setattr(api, "match_context", boom)

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert "60.0%" in resp.text


def test_dashboard_shows_probability_provenance(client, monkeypatch):
    """같은 크기의 막대로 보이는 확률이 실제로는 근거와 시점이 다르다. 북메이커 3곳 평균과
    15곳 평균은 신뢰도가 다르고, 개별 경기가 며칠 전에 계산된 값일 수도 있다."""
    _patch_dashboard(monkeypatch, prediction_extra={
        "bookmaker_count": 12, "model_version": "v3", "computed_at": "2026-09-05T09:00:00Z",
    })

    resp = client.get("/dashboard?date=2026-09-06")
    assert "북메이커 12곳 평균" in resp.text
    assert "모델 v3" in resp.text
    assert "계산" in resp.text


def test_dashboard_provenance_omitted_when_metadata_absent(client, monkeypatch):
    """과거에 저장된 예측에는 이 정보가 없다 — 빈 구분점만 남는 줄이 생기면 안 된다."""
    _patch_dashboard(monkeypatch)

    resp = client.get("/dashboard?date=2026-09-06")
    assert 'class="provenance"' not in resp.text


def test_dashboard_date_chips_label_today_and_tomorrow(client, monkeypatch):
    """ISO 날짜만 있으면 어느 게 오늘인지 세어봐야 안다."""
    today = datetime.now(api.KST).date()
    dates = [(today + timedelta(days=d)).isoformat() for d in (1, 0, -1)]
    _patch_dashboard(monkeypatch)
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: dates)

    resp = client.get("/dashboard?date=2026-09-06")
    assert ">오늘<" in resp.text
    assert ">내일<" in resp.text
    assert ">어제<" in resp.text


def test_dashboard_offers_link_back_to_current_round(client, monkeypatch):
    """날짜를 고른 상태에서 현재 라운드로 돌아갈 방법이 없으면 주소를 직접 지워야 한다."""
    _patch_dashboard(monkeypatch)

    resp = client.get("/dashboard?date=2026-09-06")
    assert 'href="/dashboard">현재 라운드</a>' in resp.text


def test_dashboard_surfaces_cumulative_accuracy(client, monkeypatch):
    """성적표를 따로 들어가야만 볼 수 있으면, 확률을 믿을 근거가 있는지 모르는 채로 숫자만
    읽게 된다 — 누적 성적이 첫 화면에 있어야 한다."""
    _patch_dashboard(monkeypatch)
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: _card())

    resp = client.get("/dashboard?date=2026-09-06")
    assert "적중률 <b>55%</b>" in resp.text
    assert "120경기 채점" in resp.text


def test_scorecard_page_warns_and_drops_highlighting_on_small_sample(client, monkeypatch):
    """14경기짜리 성적에서 모델 간 점수 차이는 대부분 운이다. 최고값을 초록색으로 칠하면
    그 우연을 실력으로 읽게 되므로, 표본이 쌓이기 전까지는 강조하지 않고 경고를 띄운다."""
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: _card(n=14))

    resp = client.get("/scorecard")
    assert "아직 14경기만 채점됐습니다" in resp.text
    assert 'class="best"' not in resp.text
    assert "0.961" in resp.text  # 숫자 자체는 그대로 기록으로 남는다


def test_dashboard_chip_hides_accuracy_on_small_sample(client, monkeypatch):
    """첫 화면의 "적중률 14%"는 모델이 그 정도라는 뜻이 아니라 아직 알 수 없다는 뜻인데,
    칩으로 보면 전자로 읽힌다 — 표본이 적으면 숫자를 걸지 않는다."""
    _patch_dashboard(monkeypatch)
    monkeypatch.setattr(api, "_fetch_scorecard", lambda: _card(n=14))

    resp = client.get("/dashboard?date=2026-09-06")
    assert "표본 부족" in resp.text
    assert "적중률" not in resp.text
