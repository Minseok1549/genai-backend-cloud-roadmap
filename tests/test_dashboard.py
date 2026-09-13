import sys
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


def _fx(match_id, home, away, kickoff, status="SCHEDULED", **extra):
    """date= 뷰는 이제 실제 fixture 목록(load_all_fixtures_on_date)과 저장된 예측 기록을
    match_id로 병합해서 보여준다 — 팀명/크레스트/상태/스코어는 fixture에서, 확률/리포트/
    GenAI 예측은 저장된 기록에서 온다. 테스트용 fixture 스텁을 만드는 헬퍼."""
    fx = {
        "match_id": match_id, "kickoff_utc": kickoff, "home_team": home, "away_team": away,
        "home_crest": None, "away_crest": None, "status": status,
    }
    fx.update(extra)
    return fx


def test_dashboard_renders_predictions_table(client, monkeypatch):
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06", "2026-09-05"])
    monkeypatch.setattr(
        api, "load_all_fixtures_on_date",
        lambda date_str: [_fx(1, "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z")],
    )
    monkeypatch.setattr(
        api,
        "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {
                    "match_id": 1,
                    "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
                }
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert "Arsenal FC" in resp.text
    assert "60.0%" in resp.text


def test_dashboard_renders_crest_when_present_and_placeholder_when_absent(client, monkeypatch):
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(
        api, "load_all_fixtures_on_date",
        lambda date_str: [_fx(
            1, "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z",
            home_crest="https://crests.football-data.org/57.png", away_crest=None,
        )],
    )
    monkeypatch.setattr(
        api,
        "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {
                    "match_id": 1,
                    "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
                }
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert '<img class="crest" src="https://crests.football-data.org/57.png"' in resp.text
    assert '<span class="crest crest-empty">' in resp.text


def test_dashboard_renders_genai_prediction_alongside_stat_model(client, monkeypatch):
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(
        api, "load_all_fixtures_on_date",
        lambda date_str: [_fx(1, "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z")],
    )
    monkeypatch.setattr(
        api,
        "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {
                    "match_id": 1,
                    "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
                    "genai_prediction": {
                        "probabilities": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
                        "reasoning": "주전 공격수 부상으로 홈팀 우세가 다소 줄어듦",
                        "sources": [{"title": "<b>속보</b>", "url": "https://example.com/a"}],
                    },
                }
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert "배당률 기반 예측" in resp.text
    assert "뉴스 시나리오 (위 확률을 뉴스로 조정)" in resp.text
    assert "60.0%" in resp.text and "50.0%" in resp.text  # 두 확률이 나란히 보임
    assert "주전 공격수 부상으로 홈팀 우세가 다소 줄어듦" in resp.text
    assert "&lt;b&gt;속보&lt;/b&gt;" in resp.text  # 출처 제목 escape


def test_dashboard_renders_genai_reasoning_as_headline_and_tagged_points(client, monkeypatch):
    """GenAI 예측 근거도 통계 모델 프리뷰와 동일하게 한 줄 총평 + 태그 리스트로
    렌더링돼야 한다."""
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(
        api, "load_all_fixtures_on_date",
        lambda date_str: [_fx(1, "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z")],
    )
    monkeypatch.setattr(
        api,
        "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {
                    "match_id": 1,
                    "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
                    "genai_prediction": {
                        "probabilities": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
                        "headline": "주전 공격수 부상으로 홈팀 우세 축소",
                        "points": [{"tag": "부상", "text": "홈팀 주전 공격수 결장"}],
                        "sources": [],
                    },
                }
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert '<p class="ai-headline">주전 공격수 부상으로 홈팀 우세 축소</p>' in resp.text
    assert 'class="ai-point ai-point-injury"' in resp.text
    assert "홈팀 주전 공격수 결장" in resp.text


def test_dashboard_keeps_genai_block_for_finished_match(client, monkeypatch):
    """종료된 경기도 스코어와 함께 "당시 예측 근거"가 유지돼야 한다 — 특히 배당률이 끝까지
    안 열려 GenAI가 유일한 예측이었던 경기는, 생략하면 예측 기록 자체가 사라져버린다."""
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(
        api, "load_all_fixtures_on_date",
        lambda date_str: [_fx(
            1, "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z",
            status="FINISHED", score={"home": 2, "away": 1},
        )],
    )
    monkeypatch.setattr(
        api,
        "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {
                    "match_id": 1,
                    "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
                    "genai_prediction": {
                        "probabilities": {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
                        "reasoning": "경기 전 뉴스 기반 근거",
                        "sources": [],
                    },
                }
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert "경기 전 뉴스 시나리오" in resp.text
    assert "경기 전 뉴스 기반 근거" in resp.text
    assert "2 : 1" in resp.text  # 실제 스코어는 fixture에서 채워짐


def test_dashboard_keeps_genai_only_block_for_finished_match_when_odds_never_opened(client, monkeypatch):
    """Codex 리뷰가 지적한 최악의 케이스: 배당률이 끝까지 공개되지 않아 통계 모델 예측이
    아예 없던 경기(probabilities=None)가 종료된 경우. GenAI 예측이 그 경기의 유일한
    사전 예측 기록이므로, 스코어만 남고 근거가 통째로 사라지면 안 된다."""
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(
        api, "load_all_fixtures_on_date",
        lambda date_str: [_fx(
            1, "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z",
            status="FINISHED", score={"home": 1, "away": 1},
        )],
    )
    monkeypatch.setattr(
        api, "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {
                    "match_id": 1,
                    "probabilities": None,
                    "genai_prediction": {
                        "probabilities": {"HOME_TEAM": 0.4, "DRAW": 0.35, "AWAY_TEAM": 0.25},
                        "reasoning": "배당률 공개 전, 뉴스 기반 근거",
                        "sources": [],
                    },
                }
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert "1 : 1" in resp.text
    assert "경기 전 뉴스 시나리오" in resp.text
    assert "배당률 공개 전, 뉴스 기반 근거" in resp.text
    assert "40.0%" in resp.text


def test_dashboard_date_view_shows_score_for_finished_match_and_keeps_prior_prediction(client, monkeypatch):
    """날짜별 뷰(?date=)도 라운드 뷰처럼 종료된 경기는 실제 스코어를 보여줘야 한다
    (과거 버그: date= 라우트는 저장된 예측 기록만 그대로 보여줘서 종료된 경기도 계속
    'vs' 카드로 남아 있었음). 경기 전에 만들어둔 예측 기록(확률)은 결과와 함께 유지된다."""
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(
        api, "load_all_fixtures_on_date",
        lambda date_str: [_fx(
            1, "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z",
            status="FINISHED", score={"home": 3, "away": 1},
        )],
    )
    monkeypatch.setattr(
        api, "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {"match_id": 1, "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15}}
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert "3 : 1" in resp.text
    assert "종료" in resp.text
    assert "60.0%" in resp.text  # 경기 전 예측 기록은 결과와 함께 유지됨


def test_dashboard_renders_genai_only_card_when_odds_not_open_yet(client, monkeypatch):
    """배당률이 아직 공개되지 않은 경기는 통계 모델 예측 없이 GenAI 예측만 있을 수
    있다(D-1 fallback) — '예측 없음'으로 빠지지 않고 GenAI 예측이 유일한 예측으로 나와야
    한다."""
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(
        api, "load_all_fixtures_on_date",
        lambda date_str: [_fx(1, "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z")],
    )
    monkeypatch.setattr(
        api, "_fetch_daily_predictions",
        lambda date_str: {
            "generated_at": "2026-09-06T06:00:00Z",
            "predictions": [
                {
                    "match_id": 1,
                    "probabilities": None,
                    "genai_prediction": {
                        "probabilities": {"HOME_TEAM": 0.55, "DRAW": 0.25, "AWAY_TEAM": 0.2},
                        "reasoning": "뉴스 기반 예측",
                        "sources": [],
                    },
                }
            ],
        },
    )

    resp = client.get("/dashboard?date=2026-09-06")
    assert resp.status_code == 200
    assert "뉴스 시나리오 (배당률 공개 전, 뉴스만 근거)" in resp.text
    assert "55.0%" in resp.text
    assert "사전 예측 없음" not in resp.text


def test_dashboard_escapes_date_query_param(client, monkeypatch):
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06"])
    monkeypatch.setattr(api, "_fetch_daily_predictions", lambda date_str: None)

    resp = client.get("/dashboard?date=%3Cb%3Ehi%3C%2Fb%3E")
    assert resp.status_code == 200
    assert "<b>hi</b>" not in resp.text  # 반사형 XSS 방지 — escape돼야 함
    assert "&lt;b&gt;hi&lt;/b&gt;" in resp.text


def test_dashboard_shows_empty_state_when_no_predictions(client, monkeypatch):
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: [])
    monkeypatch.setattr(api, "_fetch_daily_predictions", lambda date_str: None)

    resp = client.get("/dashboard?date=2099-01-01")
    assert resp.status_code == 200
    assert "예측 기록이 없습니다" in resp.text


def test_dashboard_default_view_aggregates_full_matchday_across_dates(client, monkeypatch):
    """하나의 라운드가 여러 날짜에 걸쳐 저장돼도, 날짜 파라미터 없이 들어오면
    라운드 전체 경기가 한 화면에 모여야 한다(과거 버그: 한 날짜 파일만 보여줌).
    예측이 저장 안 된 경기(폼 데이터 부족으로 스킵됨 등)도 라운드에서 빠지지 않고
    이유와 함께 나와야 한다(과거 버그: 예측 있는 경기만 보여서 라운드 일부가 통째로 빠짐)."""
    monkeypatch.setattr(api, "_list_available_dates", lambda limit=14: ["2026-09-06", "2026-09-05"])
    monkeypatch.setattr(
        api,
        "load_matchday_info",
        lambda: {
            "matchday": 3,
            "dates": ["2026-09-05", "2026-09-06"],
            "fixtures": [
                {
                    "match_id": 1,
                    "kickoff_utc": "2026-09-05T14:00:00Z",
                    "home_team": "Fulham FC",
                    "away_team": "Crystal Palace FC",
                    "status": "FINISHED",
                },
                {
                    "match_id": 99,
                    "kickoff_utc": "2026-09-05T14:00:00Z",
                    "home_team": "Manchester City FC",
                    "away_team": "Coventry City FC",
                    "status": "FINISHED",
                    "score": {"home": 3, "away": 0},
                },
                {
                    "match_id": 2,
                    "kickoff_utc": "2026-09-06T15:30:00Z",
                    "home_team": "Arsenal FC",
                    "away_team": "Chelsea FC",
                    "status": "TIMED",
                },
            ],
        },
    )

    def fake_fetch(date_str):
        if date_str == "2026-09-05":
            return {
                "generated_at": "2026-09-05T06:00:00Z",
                "predictions": [
                    {
                        "match_id": 1,
                        "kickoff_utc": "2026-09-05T14:00:00Z",
                        "home_team": "Fulham FC",
                        "away_team": "Crystal Palace FC",
                        "probabilities": {"HOME_TEAM": 0.4, "DRAW": 0.3, "AWAY_TEAM": 0.3},
                    }
                ],
            }
        if date_str == "2026-09-06":
            return {
                "generated_at": "2026-09-06T06:00:00Z",
                "predictions": [
                    {
                        "match_id": 2,
                        "kickoff_utc": "2026-09-06T15:30:00Z",
                        "home_team": "Arsenal FC",
                        "away_team": "Chelsea FC",
                        "probabilities": {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
                    }
                ],
            }
        return None

    monkeypatch.setattr(api, "_fetch_daily_predictions", fake_fetch)

    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "3라운드" in resp.text
    assert "Fulham FC" in resp.text
    assert "Arsenal FC" in resp.text
    assert "Coventry City FC" in resp.text
    assert "3 : 0" in resp.text  # 예측 없는 종료 경기도 실제 스코어가 나와야 함
    assert "사전 예측 없음" in resp.text
