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


def _report_json_text(headline="리포트 총평", points=None):
    points = points if points is not None else [{"tag": "부상", "text": "주전 공격수 부상 결장"}]
    return f'REPORT_JSON: {json.dumps({"headline": headline, "points": points}, ensure_ascii=False)}'


def test_generate_match_report_parses_headline_and_points_and_sends_grounding_tool(monkeypatch):
    """기존 승부 예측 플랫폼처럼 긴 문단 대신 한 줄 총평(headline) + 태그 달린 핵심
    포인트(points) 목록으로 응답을 구조화해서 받는다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["headers"] = headers
        captured["json"] = json
        text = _report_json_text(
            headline="홈팀이 근소하게 유리",
            points=[
                {"tag": "부상", "text": "원정팀 주전 수비수 결장 유력"},
                {"tag": "폼", "text": "홈팀 최근 5경기 4승"},
            ],
        )
        return _FakeResponse(200, {
            "candidates": [{
                "content": {"parts": [{"text": text}]},
                "groundingMetadata": {"groundingChunks": [
                    {"web": {"uri": "https://example.com/a", "title": "A 기사"}},
                    {"web": {"uri": "https://example.com/a", "title": "A 기사 중복"}},
                    {"web": {"uri": "https://example.com/b", "title": "B 기사"}},
                ]},
            }]
        })

    monkeypatch.setattr(report.requests, "post", fake_post)

    result = report.generate_match_report(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", "John Referee",
        {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
    )

    assert result["headline"] == "홈팀이 근소하게 유리"
    assert result["points"] == [
        {"tag": "부상", "text": "원정팀 주전 수비수 결장 유력"},
        {"tag": "폼", "text": "홈팀 최근 5경기 4승"},
    ]
    assert result["sources"] == [
        {"title": "A 기사", "url": "https://example.com/a"},
        {"title": "B 기사", "url": "https://example.com/b"},
    ]  # 중복 URL은 한 번만
    assert captured["headers"]["x-goog-api-key"] == "fake-key"
    assert captured["json"]["tools"] == [{"google_search": {}}]
    prompt = captured["json"]["contents"][0]["parts"][0]["text"]
    assert "Arsenal FC" in prompt and "Chelsea FC" in prompt
    assert "John Referee" in prompt


def test_generate_match_report_drops_points_with_invalid_tag_to_other(monkeypatch):
    """Gemini가 지정된 태그(부상/폼/전술/주심/기타) 외의 값을 쓰면 UI 렌더링이 깨지지
    않도록 '기타'로 정규화한다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = _report_json_text(points=[{"tag": "날씨", "text": "폭우 예보"}])
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]}))

    result = report.generate_match_report(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
        {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
    )

    assert result["points"] == [{"tag": "기타", "text": "폭우 예보"}]


def test_generate_match_report_parses_when_point_text_contains_braces(monkeypatch):
    """point.text 안에 '{'나 '}'가 섞여 있으면 괄호 depth를 문자 단위로 세는 방식은
    JSON을 조기 종료하거나 못 닫힌 것으로 오판한다 — JSONDecoder.raw_decode는 문자열
    문법을 알기 때문에 안전해야 한다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = _report_json_text(
        headline="폼 { 반전",
        points=[{"tag": "폼", "text": "홈팀 최근 폼 } 반전 조짐"}],
    )
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]}))

    result = report.generate_match_report(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
        {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
    )

    assert result["headline"] == "폼 { 반전"
    assert result["points"] == [{"tag": "폼", "text": "홈팀 최근 폼 } 반전 조짐"}]


def test_generate_match_report_ignores_prefix_quoted_inside_point_text(monkeypatch):
    """point.text가 'REPORT_JSON:' 문자열을 그대로 인용하면, 아무 데서나 찾는 방식은
    실제 trailer 대신 문자열 내부의 가짜 prefix를 골라 파싱을 망가뜨린다. 실제 trailer는
    프롬프트 지시대로 항상 줄 맨 앞에서 시작하므로 그것만 인정해야 한다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    text = _report_json_text(
        headline="총평",
        points=[{"tag": "기타", "text": "형식은 REPORT_JSON: {...} 이었음"}],
    )
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]}))

    result = report.generate_match_report(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
        {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
    )

    assert result["headline"] == "총평"
    assert result["points"] == [{"tag": "기타", "text": "형식은 REPORT_JSON: {...} 이었음"}]


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


def test_generate_match_report_prompt_notes_missing_referee(monkeypatch):
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": _report_json_text()}]}}]})

    monkeypatch.setattr(report.requests, "post", fake_post)

    report.generate_match_report(
        "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
        {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
    )

    prompt = captured["json"]["contents"][0]["parts"][0]["text"]
    assert "배정된 주심 정보 없음" in prompt


def test_generate_match_report_raises_when_report_json_missing(monkeypatch):
    """다른 설명 문장 없이 REPORT_JSON 한 줄만 응답하라고 지시했는데도 형식을 지키지
    않으면, 파싱 실패를 조용히 삼키지 말고 예외로 올려야 호출자가 재시도/폴백을 판단할
    수 있다."""
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "그냥 설명만 있음"}]}}]}))

    with pytest.raises(ValueError):
        report.generate_match_report(
            "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
            {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
        )


def test_generate_match_report_raises_on_empty_text(monkeypatch):
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    monkeypatch.setattr(
        report.requests, "post",
        lambda *a, **kw: _FakeResponse(200, {"candidates": [{"content": {"parts": []}}]}),
    )

    with pytest.raises(RuntimeError):
        report.generate_match_report(
            "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
            {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
        )


def test_generate_match_report_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(report, "load_gemini_api_key", lambda: "fake-key")
    monkeypatch.setattr(report.requests, "post", lambda *a, **kw: _FakeResponse(429, {}))

    with pytest.raises(RuntimeError):
        report.generate_match_report(
            "Arsenal FC", "Chelsea FC", "2026-09-06T15:30:00Z", None,
            {"HOME_TEAM": 0.6, "DRAW": 0.25, "AWAY_TEAM": 0.15},
        )


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
