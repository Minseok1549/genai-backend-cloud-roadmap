"""예측 기록을 Postgres에 저장한다.

Alembic 같은 마이그레이션 도구는 이 프로젝트 규모에서 과한 스코프라 안 쓴다 —
predictions 테이블은 CREATE TABLE IF NOT EXISTS로 충분하고, 구조가 바뀌면
SCHEMA_VERSION을 올리고 각 행에 그 값을 같이 저장해 나중에 "이 행이 어떤
스키마로 쓰였는지"를 구분할 수 있게 한다.
"""
import os
from pathlib import Path

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
POOL_SIZE = 5  # FastAPI의 sync 핸들러는 스레드풀에서 동시 실행되므로 스레드 세이프한 풀이 필요


def load_database_url() -> str:
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("DATABASE_URL not found in environment or .env")


def create_pool() -> ThreadedConnectionPool:
    # SimpleConnectionPool은 스레드 세이프하지 않다(psycopg2 공식 문서에 명시) — 요청마다
    # 스레드풀에서 동시 호출되는 이 API에는 부적합해서 ThreadedConnectionPool을 쓴다.
    # min=max로 둬서(요청 부담이 적은 이 서비스 규모에서) 유휴 커넥션이 매번 닫혔다 새로
    # 열리는 것도 피한다. connect_timeout 없이는 DB가 응답 없을 때 OS 기본 TCP 타임아웃까지
    # (수십 초~분 단위) 그냥 멈춰있을 수 있어 명시적으로 짧게 잡는다.
    return ThreadedConnectionPool(POOL_SIZE, POOL_SIZE, dsn=load_database_url(), connect_timeout=5)


def init_schema(pool: ThreadedConnectionPool) -> None:
    """CREATE TABLE IF NOT EXISTS는 두 프로세스가 완전히 동시에 실행하면(예: 클라우드에서
    replica 여러 개가 동시에 기동) 존재 확인과 생성 사이의 레이스가 난다. 실제로 여러
    스레드로 재현해보면 테이블 자체가 아니라 SERIAL이 암묵적으로 만드는 시퀀스 이름
    충돌로 UniqueViolation이 뜬다 — 특정 예외 타입 하나만 잡는 건 신뢰할 수 없어서,
    실패하면 일단 rollback하고 "테이블이 실제로 존재하는가"를 다시 확인해 그렇다면
    (다른 인스턴스가 이미 만든 것이니) 성공으로 취급한다."""
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        home_team TEXT NOT NULL,
                        away_team TEXT NOT NULL,
                        prob_home DOUBLE PRECISION NOT NULL,
                        prob_draw DOUBLE PRECISION NOT NULL,
                        prob_away DOUBLE PRECISION NOT NULL,
                        model_version TEXT NOT NULL,
                        data_version TEXT NOT NULL,
                        schema_version INTEGER NOT NULL
                    )
                    """
                )
                conn.commit()
            except psycopg2.Error:
                conn.rollback()
                cur.execute("SELECT to_regclass('predictions')")
                if cur.fetchone()[0] is None:
                    raise  # 테이블이 정말 없다 — 레이스가 아니라 진짜 실패
                conn.commit()
    finally:
        pool.putconn(conn)


def save_prediction(
    pool: ThreadedConnectionPool,
    *,
    home_team: str,
    away_team: str,
    probabilities: dict,
    model_version: str,
    data_version: str,
) -> int:
    """예측 한 건을 기록하고 새로 생긴 행의 id를 반환한다. model_version/data_version은
    호출 시점의 모델·데이터를 가리키는 스냅샷이므로, 이후 재학습·재수집이 일어나도
    이미 저장된 행의 값은 절대 갱신하지 않는다(UPDATE 없음, INSERT만 함)."""
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (home_team, away_team, prob_home, prob_draw, prob_away,
                     model_version, data_version, schema_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    home_team,
                    away_team,
                    probabilities["HOME_TEAM"],
                    probabilities["DRAW"],
                    probabilities["AWAY_TEAM"],
                    model_version,
                    data_version,
                    SCHEMA_VERSION,
                ),
            )
            return cur.fetchone()[0]
    finally:
        pool.putconn(conn)
