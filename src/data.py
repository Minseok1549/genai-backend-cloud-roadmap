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
