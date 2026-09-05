import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import db  # noqa: E402


@pytest.fixture
def pool():
    p = db.create_pool()
    db.init_schema(p)
    yield p
    p.closeall()


def _fetch(pool, prediction_id):
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT home_team, away_team, model_version, data_version, schema_version "
                "FROM predictions WHERE id = %s",
                (prediction_id,),
            )
            return cur.fetchone()
    finally:
        pool.putconn(conn)


def test_save_prediction_persists_versions(pool):
    row_id = db.save_prediction(
        pool,
        home_team="Arsenal FC",
        away_team="Aston Villa FC",
        probabilities={"HOME_TEAM": 0.5, "DRAW": 0.3, "AWAY_TEAM": 0.2},
        model_version="v-test-1",
        data_version="2026-01-01",
    )
    home_team, away_team, model_version, data_version, schema_version = _fetch(pool, row_id)
    assert (home_team, away_team) == ("Arsenal FC", "Aston Villa FC")
    assert model_version == "v-test-1"
    assert data_version == "2026-01-01"
    assert schema_version == db.SCHEMA_VERSION


def test_retraining_does_not_overwrite_past_prediction_versions(pool):
    """블록 3 검증 기준: 모델을 재학습해 model_version이 바뀌어도 이미 저장된
    과거 예측 기록의 model_version은 그대로여야 한다 (save_prediction은 INSERT-only)."""
    old_id = db.save_prediction(
        pool,
        home_team="Arsenal FC",
        away_team="Chelsea FC",
        probabilities={"HOME_TEAM": 0.4, "DRAW": 0.3, "AWAY_TEAM": 0.3},
        model_version="v-old",
        data_version="2026-01-01",
    )

    # "재학습" 시뮬레이션 — 실제로는 train.py 재실행마다 model_version이 새로 발급된다.
    new_id = db.save_prediction(
        pool,
        home_team="Arsenal FC",
        away_team="Chelsea FC",
        probabilities={"HOME_TEAM": 0.4, "DRAW": 0.3, "AWAY_TEAM": 0.3},
        model_version="v-new",
        data_version="2026-01-02",
    )

    assert old_id != new_id
    assert _fetch(pool, old_id)[2] == "v-old"
    assert _fetch(pool, new_id)[2] == "v-new"
