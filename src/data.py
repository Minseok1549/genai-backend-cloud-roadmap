"""캐시된 raw JSON을 읽어 날짜순으로 정렬된 경기 DataFrame으로 변환한다."""
import json
from pathlib import Path

import pandas as pd

from fetch_data import COMPLETED_SEASONS, CURRENT_SEASON

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ALL_SEASONS = COMPLETED_SEASONS + [CURRENT_SEASON]


def load_matches(seasons: list[int] | None = None) -> pd.DataFrame:
    """seasons를 지정하지 않으면 완결 시즌 + 진행 중 시즌을 모두 읽는다(캐시 파일이 있는 것만).

    학습(train.py)은 완결 시즌만 넘겨서 진행 중 시즌의 적은 표본이 섞이지 않게 하고,
    실시간 폼 계산(api.py)은 기본값을 그대로 써서 오늘 시점까지 끝난 경기를 전부 반영한다.
    """
    seasons = seasons if seasons is not None else ALL_SEASONS
    rows = []
    for season in seasons:
        path = RAW_DIR / f"matches_{season}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for m in data["matches"]:
            if m["status"] != "FINISHED":
                continue
            rows.append(
                {
                    "match_id": m["id"],
                    "date": pd.Timestamp(m["utcDate"]),
                    "season": season,
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                    "home_goals": m["score"]["fullTime"]["home"],
                    "away_goals": m["score"]["fullTime"]["away"],
                    "result": m["score"]["winner"],  # HOME_TEAM / AWAY_TEAM / DRAW
                }
            )
    columns = ["match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals", "result"]
    df = pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)
    return df


UPCOMING_STATUSES = {"SCHEDULED", "TIMED"}
LIVE_STATUSES = {"IN_PLAY", "PAUSED"}


def load_upcoming_fixtures(season: int = CURRENT_SEASON) -> list[dict]:
    """상태가 SCHEDULED/TIMED인 경기 전체를 킥오프 시각순으로 반환한다 —
    POSTPONED/CANCELLED/SUSPENDED는 "예정"이 아니라 이미 무산됐거나 불확실한 경기라
    예측 대상에서 제외하고, IN_PLAY/PAUSED는 이미 시작해서 사전 예측의 의미가 없으므로
    제외한다.

    주의: 여기서 걸러지는 건 상태값뿐이고 킥오프 시각은 보지 않는다. 이 목록의 출처는 최대
    6시간 묵을 수 있는 시즌 캐시라, 실제로는 이미 시작한 경기가 캐시에서 아직 TIMED로 남아
    통과할 수 있다 — "시작 전"을 보장하는 건 이 함수가 아니라 호출자다. 유일한 호출자인
    daily_predict.py가 _within_lookahead와 build_fixture_prediction에서 킥오프까지 남은
    시간이 0보다 큰지 두 번 확인하므로, 여기에 같은 검사를 또 넣지 않는다."""
    path = RAW_DIR / f"matches_{season}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    fixtures = []
    for m in data["matches"]:
        if m["status"] not in UPCOMING_STATUSES:
            continue
        referees = m.get("referees") or []
        fixtures.append(
            {
                "match_id": m["id"],
                "kickoff_utc": m["utcDate"],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_crest": m["homeTeam"].get("crest"),
                "away_crest": m["awayTeam"].get("crest"),
                "referee": referees[0]["name"] if referees else None,
            }
        )
    return sorted(fixtures, key=lambda f: f["kickoff_utc"])


def load_fixtures_on_date(target_date, season: int = CURRENT_SEASON) -> list[dict]:
    """target_date(같은 UTC 날짜)에 예정된 경기 목록을 반환한다."""
    target = pd.Timestamp(target_date).date()
    return [f for f in load_upcoming_fixtures(season) if pd.Timestamp(f["kickoff_utc"]).date() == target]


def _fixture_with_status(m: dict) -> dict:
    """raw 경기 JSON 하나를 상태(그리고 FINISHED면 스코어)까지 포함한 fixture dict로
    변환한다. load_matchday_info와 load_all_fixtures_on_date가 공유한다."""
    fx = {
        "match_id": m["id"],
        "kickoff_utc": m["utcDate"],
        "home_team": m["homeTeam"]["name"],
        "away_team": m["awayTeam"]["name"],
        "home_crest": m["homeTeam"].get("crest"),
        "away_crest": m["awayTeam"].get("crest"),
        "status": m["status"],
    }
    if m["status"] == "FINISHED":
        full_time = m.get("score", {}).get("fullTime", {})
        fx["score"] = {"home": full_time.get("home"), "away": full_time.get("away")}
    return fx


def load_all_fixtures_on_date(target_date, season: int = CURRENT_SEASON) -> list[dict]:
    """target_date(같은 UTC 날짜)에 열리는 경기를 상태 무관(FINISHED 포함, 스코어 포함)으로
    전부 반환한다. load_fixtures_on_date는 SCHEDULED/TIMED만 보여줘서 이미 끝난 경기가
    날짜별 대시보드에서 사라지는 문제가 있었다 — 종료된 경기는 예측이 아니라 실제 결과를
    보여줘야 하므로 이 함수를 쓴다."""
    path = RAW_DIR / f"matches_{season}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    target = pd.Timestamp(target_date).date()
    matches = [m for m in data["matches"] if pd.Timestamp(m["utcDate"]).date() == target]
    return sorted((_fixture_with_status(m) for m in matches), key=lambda f: f["kickoff_utc"])


def load_season_teams(season: int = CURRENT_SEASON) -> set[str]:
    """이번 시즌 일정에 이름이 올라 있는 팀 전체를 반환한다(경기 상태 무관).

    "이 팀이 지금 이 리그에 있는가"를 완료된 경기로 판단하면 두 방향으로 틀린다: 시즌 첫
    경기 전에는 지난 시즌 강등팀이 통과하고, 반대로 첫 경기가 한 경기만 끝난 시점에는 그
    경기에 나온 두 팀만 인정돼 나머지 18팀이 "알 수 없는 팀"이 된다. 일정표는 개막 전부터
    20팀 전부를 담고 있으므로 소속 판정은 이쪽이 맞다."""
    path = RAW_DIR / f"matches_{season}.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    teams = set()
    for m in data["matches"]:
        teams.add(m["homeTeam"]["name"])
        teams.add(m["awayTeam"]["name"])
    return teams


def load_matchday_info(season: int = CURRENT_SEASON) -> dict:
    """가장 임박한(아직 안 끝난 경기가 있는) matchday 번호, 그 라운드가 걸쳐 있는 UTC 날짜
    목록, 그리고 라운드에 속한 전체 경기 목록(상태 무관, FINISHED 포함)을 반환한다.

    EPL 한 라운드는 보통 목~월 여러 날짜에 걸쳐 열린다 — 대시보드가 날짜 하나만 보면
    같은 라운드의 나머지 경기가 안 보이는 문제가 생겨서, 날짜 대신 matchday로 묶는다.
    fixtures를 상태와 무관하게 전부 담는 이유: 예측이 저장 안 된 경기(폼 데이터 부족으로
    모델이 건너뛴 경기, 배치가 아직 안 돌았던 경기 등)도 "이 라운드에 있었다"는 사실은
    대시보드에서 보여줘야 라운드 전체를 빠짐없이 확인할 수 있다."""
    path = RAW_DIR / f"matches_{season}.json"
    if not path.exists():
        return {"matchday": None, "dates": [], "fixtures": []}
    data = json.loads(path.read_text())
    unfinished = [m for m in data["matches"] if m["status"] in UPCOMING_STATUSES | LIVE_STATUSES]
    if unfinished:
        # 라운드 번호의 최솟값이 아니라, 아직 안 끝난 경기 중 가장 먼저 시작한(또는 시작할)
        # 경기가 속한 라운드를 고른다.
        # 최솟값을 쓰면 안 되는 이유: 연기된 경기가 나중 날짜로 재편성되면 라운드 번호는 원래
        # 것을 그대로 유지하므로, 그 한 경기 때문에 이미 다 끝난 과거 라운드에 대시보드가
        # 묶인다 (3라운드 경기 하나가 12월로 밀리면 9월부터 12월까지 계속 3라운드를 보여준다).
        # 진행 중(IN_PLAY/PAUSED)인 경기도 "안 끝난 경기"로 세는 이유: 이걸 빼면 한 라운드의
        # 마지막 경기가 킥오프하는 순간 그 라운드에 남은 게 없어져서, 경기가 진행되는 동안
        # 대시보드가 다음 라운드로 넘어가버린다 — 사람들이 결과를 확인하려고 들어오는 바로
        # 그 시간에 보고 싶은 라운드가 화면에서 사라진다.
        matchday = min(unfinished, key=lambda m: m["utcDate"])["matchday"]
    else:
        finished = [m for m in data["matches"] if m.get("matchday") is not None]
        matchday = max((m["matchday"] for m in finished), default=None)
    if matchday is None:
        return {"matchday": None, "dates": [], "fixtures": []}
    round_matches = sorted(
        (m for m in data["matches"] if m["matchday"] == matchday), key=lambda m: m["utcDate"]
    )
    dates = sorted({m["utcDate"][:10] for m in round_matches})
    fixtures = [_fixture_with_status(m) for m in round_matches]
    return {"matchday": matchday, "dates": dates, "fixtures": fixtures}
