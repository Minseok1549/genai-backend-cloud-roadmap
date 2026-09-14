import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import report  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_normalize_points_returns_empty_when_not_a_list():
    """Gemini가 지시를 어기고 points를 dict/숫자/문자열로 주면, 여기서 예외가 터져
    이미 파싱된 확률까지 호출부에서 함께 버려지면 안 된다."""
    assert report._normalize_points({"tag": "부상", "text": "..."}) == []
    assert report._normalize_points(7) == []
    assert report._normalize_points("부상") == []
    assert report._normalize_points(None) == []


def test_normalize_points_drops_whitespace_only_text():
    assert report._normalize_points([{"tag": "부상", "text": "   "}]) == []


def test_normalize_points_keeps_valid_points_after_invalid_ones_within_limit():
    """앞쪽에 무효한 항목이 섞여 있어도, 유효성 검사를 먼저 끝내고 나서 4개로 잘라야
    뒤쪽의 유효한 항목이 밀려나지 않는다."""
    raw = [
        {"tag": "부상", "text": ""},
        "invalid-item",
        {"tag": "부상", "text": "a"},
        {"tag": "폼", "text": "b"},
        {"tag": "전술", "text": "c"},
        {"tag": "주심", "text": "d"},
        {"tag": "기타", "text": "e (버려져야 함, 5번째 유효 항목)"},
    ]
    result = report._normalize_points(raw)
    assert [p["text"] for p in result] == ["a", "b", "c", "d"]


def _fake_prediction_response(text: str) -> _FakeResponse:
    return _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


def test_generate_genai_prediction_parses_probabilities_headline_and_points(monkeypatch):
    """승부 확률뿐 아니라 한 줄 총평(headline)과 태그 달린 핵심 포인트(points)도
    같은 PREDICTION_JSON 블록에서 함께 뽑는다 — 긴 프리뷰 문단은 더 이상 쓰지 않는다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    captured = {}
    response_text = "PREDICTION_JSON: " + json.dumps({
        "home_win": 0.55, "draw": 0.25, "away_win": 0.2,
        "headline": "부상 이슈로 홈팀이 약간 더 유리",
        "points": [{"tag": "부상", "text": "원정팀 주전 결장"}],
    }, ensure_ascii=False)

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _fake_prediction_response(response_text)

    monkeypatch.setattr(report.requests, "post", fake_post)

    result = report.generate_genai_prediction(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", "John Referee",
        {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    )

    assert result["probabilities"] == {"HOME_TEAM": 0.55, "DRAW": 0.25, "AWAY_TEAM": 0.2}
    assert result["headline"] == "부상 이슈로 홈팀이 약간 더 유리"
    assert result["points"] == [{"tag": "부상", "text": "원정팀 주전 결장"}]
    assert captured["json"]["tools"] == [{"google_search": {}}]


def test_generate_genai_prediction_normalizes_when_probabilities_dont_sum_to_one(monkeypatch):
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = '요약.\nPREDICTION_JSON: {"home_win": 0.5, "draw": 0.3, "away_win": 0.3}'
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response(text))

    result = report.generate_genai_prediction(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
        {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    )

    total = sum(result["probabilities"].values())
    assert total == pytest.approx(1.0)


def test_generate_genai_prediction_raises_when_json_line_missing(monkeypatch):
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response("그냥 설명만 있음"))

    with pytest.raises(ValueError):
        report.generate_genai_prediction(
            "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
            {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        )


def test_generate_genai_prediction_parses_multiline_pretty_printed_json(monkeypatch):
    """Gemini가 지시(한 줄로 출력)와 달리 JSON을 예쁘게 여러 줄로 출력하는 경우가
    실제로 있다 — re.DOTALL 없이는 이 형태를 놓친다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = (
        "요약 설명입니다.\n"
        "PREDICTION_JSON: {\n"
        '  "home_win": 0.5,\n'
        '  "draw": 0.3,\n'
        '  "away_win": 0.2\n'
        "}"
    )
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response(text))

    result = report.generate_genai_prediction(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
        {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    )

    assert result["probabilities"] == {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2}


def test_generate_genai_prediction_uses_last_json_occurrence(monkeypatch):
    """본문 설명 중에 형식을 언급하다 우연히 PREDICTION_JSON: {...} 형태가 먼저
    등장할 수 있다 — 실제 값은 항상 마지막 줄의 것을 써야 한다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = (
        '설명 중에 PREDICTION_JSON: {"home_win": 0.1, "draw": 0.1, "away_win": 0.1} 형식을 언급함.\n'
        'PREDICTION_JSON: {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}'
    )
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response(text))

    result = report.generate_genai_prediction(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
        {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    )

    assert result["probabilities"] == {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2}


def test_generate_genai_prediction_raises_when_probability_negative(monkeypatch):
    """개별 확률이 음수여도 합만 1 근처면 통과하던 버그 — 값 하나하나도 0~1 범위인지
    검증해야 한다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = 'PREDICTION_JSON: {"home_win": -0.1, "draw": 0.5, "away_win": 0.6}'
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response(text))

    with pytest.raises(ValueError):
        report.generate_genai_prediction(
            "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
            {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        )


def test_generate_genai_prediction_accepts_probability_sum_at_float_lower_bound(monkeypatch):
    """0.3 + 0.3 + 0.3은 부동소수점 오차로 0.8999999999999999가 되어, 경계값 0.9
    inclusive인데도 부동소수점 비교로 거부되면 안 된다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = 'PREDICTION_JSON: {"home_win": 0.3, "draw": 0.3, "away_win": 0.3}'
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response(text))

    result = report.generate_genai_prediction(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
        {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
    )

    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_generate_genai_prediction_raises_when_probability_sum_way_off(monkeypatch):
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = 'PREDICTION_JSON: {"home_win": 0.1, "draw": 0.1, "away_win": 0.1}'
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response(text))

    with pytest.raises(ValueError):
        report.generate_genai_prediction(
            "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
            {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        )


def test_genai_prediction_does_not_adopt_an_unrelated_json_block(monkeypatch):
    """PREDICTION_JSON 값이 null인데 응답 뒤에 다른 JSON 객체가 붙어 있으면, 그걸 예측으로
    집어오는 대신 오류로 처리해야 한다.

    마커 뒤에서 무조건 첫 '{'를 찾으면 값이 비었을 때 한참 아래의 무관한 객체를 읽어온다 —
    엉뚱한 숫자가 승부 확률로 대시보드에 실린다. 값 자리에 JSON 객체가 없으면 '예측 없음'이
    맞는 결과다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = (
        "PREDICTION_JSON: null\n"
        'DEBUG_JSON: {"home_win": 0.9, "draw": 0.05, "away_win": 0.05}'
    )
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response(text))

    with pytest.raises(ValueError):
        report.generate_genai_prediction(
            "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
            {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        )


def test_genai_prediction_rejects_boolean_probabilities(monkeypatch):
    """true/false를 확률로 받아들이면 안 된다. 파이썬에서 bool은 int의 하위 타입이라
    float(True)가 1.0으로 조용히 통과하고, 그러면 "home_win": true 한 줄이 '홈 승리 100%'
    예측이 돼 범위·합계 검사까지 전부 지나쳐 그대로 저장된다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = 'PREDICTION_JSON: {"home_win": true, "draw": false, "away_win": false}'
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _fake_prediction_response(text))

    with pytest.raises(ValueError):
        report.generate_genai_prediction(
            "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
            {"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        )
