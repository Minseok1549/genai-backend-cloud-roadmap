"""표준출력에 한 줄짜리 JSON 로그를 찍는다 — Cloud Run/CloudWatch/Azure Monitor 등 대부분의
클라우드 로깅은 컨테이너 stdout을 그대로 수집하므로, 별도 로깅 에이전트나 SDK 없이 이 형태만으로
"구조화된 검색 가능한 로그"가 된다. api.py(상시 서비스)와 daily_predict.py(배치 Job)가 공유한다."""
import json
import logging
import sys

logger = logging.getLogger("epl_predictor")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False


def log_json(level: str, message: str, **fields) -> None:
    logger.log(getattr(logging, level.upper()), json.dumps({"level": level, "message": message, **fields}, ensure_ascii=False))
