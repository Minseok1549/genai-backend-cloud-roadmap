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
