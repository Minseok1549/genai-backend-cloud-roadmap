import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data as data_module  # noqa: E402


def _match(match_id, matchday, kickoff, status, home="홈팀", away="원정팀"):
    return {
        "id": match_id, "matchday": matchday, "utcDate": kickoff, "status": status,
        "homeTeam": {"name": home}, "awayTeam": {"name": away},
        "score": {"fullTime": {"home": 1, "away": 0}, "winner": "HOME_TEAM"},
    }


def _write_season(tmp_path, monkeypatch, matches, season=2026):
    monkeypatch.setattr(data_module, "RAW_DIR", tmp_path)
    (tmp_path / f"matches_{season}.json").write_text(json.dumps({"matches": matches}))


def test_matchday_follows_the_earliest_kickoff_not_the_lowest_round_number(tmp_path, monkeypatch):
    """대시보드가 고를 라운드는 '가장 먼저 열리는 경기가 속한 라운드'여야 한다.

    연기된 경기가 나중 날짜로 재편성되면 라운드 번호는 원래 값을 그대로 유지한다. 라운드
    번호의 최솟값을 고르면 그 한 경기 때문에 이미 다 끝난 과거 라운드에 대시보드가 묶인다 —
    3라운드 경기 하나가 12월로 밀리면 9월부터 12월까지 계속 8월 경기들을 보여준다.
    """
    matches = [_match(100 + i, 3, "2026-08-22T14:00:00Z", "FINISHED") for i in range(9)]
    matches.append(_match(199, 3, "2026-12-16T19:45:00Z", "TIMED"))  # 12월로 재편성된 3라운드 경기
    matches += [_match(300 + i, 12, "2026-11-07T15:00:00Z", "TIMED") for i in range(10)]
    _write_season(tmp_path, monkeypatch, matches)

    info = data_module.load_matchday_info(2026)

    assert info["matchday"] == 12
    assert info["dates"] == ["2026-11-07"]
    assert len(info["fixtures"]) == 10


def test_matchday_ignores_postponed_matches(tmp_path, monkeypatch):
    """POSTPONED는 '예정'이 아니다 — 날짜가 정해지지 않은 경기가 라운드 선택을 끌고 가면
    안 된다."""
    matches = [_match(1, 3, "2026-08-22T14:00:00Z", "POSTPONED")]
    matches += [_match(300 + i, 12, "2026-11-07T15:00:00Z", "TIMED") for i in range(10)]
    _write_season(tmp_path, monkeypatch, matches)

    assert data_module.load_matchday_info(2026)["matchday"] == 12


def test_matchday_falls_back_to_the_last_finished_round(tmp_path, monkeypatch):
    """시즌이 끝나 남은 경기가 없으면 마지막으로 치른 라운드를 보여준다."""
    matches = [_match(1, 37, "2026-05-17T14:00:00Z", "FINISHED"),
               _match(2, 38, "2026-05-24T14:00:00Z", "FINISHED")]
    _write_season(tmp_path, monkeypatch, matches)

    assert data_module.load_matchday_info(2026)["matchday"] == 38


def test_matchday_info_is_empty_without_a_cached_season(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "RAW_DIR", tmp_path)

    assert data_module.load_matchday_info(2026) == {"matchday": None, "dates": [], "fixtures": []}


def test_matchday_stays_on_the_round_that_is_being_played(tmp_path, monkeypatch):
    """한 라운드의 마지막 경기가 진행 중이면 대시보드는 그 라운드에 남아 있어야 한다.

    진행 중(IN_PLAY/PAUSED)인 경기를 "안 끝난 경기"로 세지 않으면, 마지막 경기가 킥오프하는
    순간 그 라운드에 남은 게 없어져서 경기가 진행되는 동안 화면이 다음 라운드로 넘어간다 —
    사람들이 결과를 보려고 들어오는 바로 그 시간에 보고 싶은 라운드가 사라진다."""
    matches = [_match(100 + i, 4, "2026-09-12T14:00:00Z", "FINISHED") for i in range(9)]
    matches.append(_match(199, 4, "2026-09-13T13:00:00Z", "IN_PLAY"))
    matches += [_match(300 + i, 5, "2026-09-20T14:00:00Z", "TIMED") for i in range(10)]
    _write_season(tmp_path, monkeypatch, matches)

    info = data_module.load_matchday_info(2026)

    assert info["matchday"] == 4
    assert len(info["fixtures"]) == 10


def test_matchday_stays_on_a_paused_round(tmp_path, monkeypatch):
    """하프타임(PAUSED)도 진행 중인 경기다 — 15분 사이에 라운드가 넘어가면 안 된다."""
    matches = [_match(1, 4, "2026-09-13T13:00:00Z", "PAUSED")]
    matches += [_match(300 + i, 5, "2026-09-20T14:00:00Z", "TIMED") for i in range(10)]
    _write_season(tmp_path, monkeypatch, matches)

    assert data_module.load_matchday_info(2026)["matchday"] == 4
