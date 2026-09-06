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


def load_fixtures_on_date(target_date, season: int = CURRENT_SEASON) -> list[dict]:
    """target_date(같은 UTC 날짜)에 예정된 경기 목록을 반환한다.
    아직 시작 전(SCHEDULED/TIMED)인 경기만 남긴다 — POSTPONED/CANCELLED/SUSPENDED는
    "예정"이 아니라 이미 무산됐거나 불확실한 경기라 예측 대상에서 제외하고,
    IN_PLAY/PAUSED는 이미 시작해서 사전 예측의 의미가 없으므로 제외한다."""
    path = RAW_DIR / f"matches_{season}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    target = pd.Timestamp(target_date).date()
    fixtures = []
    for m in data["matches"]:
        if m["status"] not in UPCOMING_STATUSES:
            continue
        kickoff = pd.Timestamp(m["utcDate"])
        if kickoff.date() != target:
            continue
        fixtures.append(
            {
                "match_id": m["id"],
                "kickoff_utc": m["utcDate"],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
            }
        )
    return fixtures


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
    upcoming = [m for m in data["matches"] if m["status"] in UPCOMING_STATUSES]
    if upcoming:
        matchday = min(m["matchday"] for m in upcoming)
    else:
        finished = [m for m in data["matches"] if m.get("matchday") is not None]
        matchday = max((m["matchday"] for m in finished), default=None)
    if matchday is None:
        return {"matchday": None, "dates": [], "fixtures": []}
    round_matches = sorted(
        (m for m in data["matches"] if m["matchday"] == matchday), key=lambda m: m["utcDate"]
    )
    dates = sorted({m["utcDate"][:10] for m in round_matches})
    fixtures = []
    for m in round_matches:
        fx = {
            "match_id": m["id"],
            "kickoff_utc": m["utcDate"],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "status": m["status"],
        }
        if m["status"] == "FINISHED":
            full_time = m.get("score", {}).get("fullTime", {})
            fx["score"] = {"home": full_time.get("home"), "away": full_time.get("away")}
        fixtures.append(fx)
    return {"matchday": matchday, "dates": dates, "fixtures": fixtures}
