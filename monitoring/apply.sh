#!/usr/bin/env bash
# Cloud Monitoring 알림 설정을 이 디렉터리의 정의 파일대로 맞춘다.
#
# 왜 스크립트인가: 콘솔에서 클릭으로 만든 알림은 어디에 무엇이 설정됐는지 코드에 남지 않아,
# 시간이 지나면 "지금 어떤 상황에 메일이 오는가"를 아무도 확인할 수 없게 된다. 정의를 파일로
# 두고 이 스크립트로 적용하면 변경 이력이 git에 남고, 프로젝트를 다시 만들어도 복구된다.
#
# 왜 gcloud 대신 REST API를 직접 부르는가: 알림 정책 생성은 gcloud의 alpha 컴포넌트에만
# 있어서, 이 스크립트를 돌리는 사람마다 SDK에 컴포넌트를 추가로 깔아야 한다. 정책 정의가
# 어차피 JSON이므로 API를 그대로 부르는 쪽이 의존성이 적다.
#
# 같은 이름(displayName)의 채널/정책이 이미 있으면 새로 만들지 않고 내용을 덮어쓴다 —
# 여러 번 돌려도 중복이 생기지 않는다.
set -euo pipefail

PROJECT="genai-backend-cloud-roadmap"
HOST="epl-predictor-187506981041.asia-northeast3.run.app"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API="https://monitoring.googleapis.com/v3/projects/${PROJECT}"

token() { gcloud auth print-access-token; }

api() { # api <METHOD> <URL> [BODY]
  local method="$1" url="$2" body="${3:-}"
  if [[ -n "$body" ]]; then
    curl -sS -X "$method" -H "Authorization: Bearer $(token)" \
      -H "Content-Type: application/json" -d "$body" "$url"
  else
    curl -sS -X "$method" -H "Authorization: Bearer $(token)" "$url"
  fi
}

# --- 1. 메일 알림 채널 ---------------------------------------------------------
channel_body="$(cat "${HERE}/notification-channel-email.json")"
channel_display="$(jq -r .displayName <<<"$channel_body")"
CHANNEL="$(api GET "${API}/notificationChannels" \
  | jq -r --arg d "$channel_display" '.notificationChannels // [] | map(select(.displayName==$d)) | .[0].name // empty')"

if [[ -z "$CHANNEL" ]]; then
  CHANNEL="$(api POST "${API}/notificationChannels" "$channel_body" | jq -r .name)"
  echo "채널 생성: $CHANNEL"
else
  api PATCH "https://monitoring.googleapis.com/v3/${CHANNEL}?updateMask=labels,description,enabled" "$channel_body" >/dev/null
  echo "채널 갱신: $CHANNEL"
fi

# --- 2. uptime check 두 개 -----------------------------------------------------
# /health은 웹 서비스가 살아있는지, /health/batch는 매일 배치가 최근에 성공했는지 본다.
# 배치는 별개의 Cloud Run Job이라 그게 며칠째 멈춰도 /health는 계속 200이고 대시보드는 예전
# 예측을 그대로 보여주므로, 두 개를 따로 감시해야 한다.
#
# 5분 주기: 1분 주기로 두면 여러 리전에서 동시에 두드려 Cloud Run 인스턴스가 사실상 계속
# 깨어 있게 된다. 이 서비스는 요청이 없으면 인스턴스를 0으로 줄이는 구성이므로, 감시가
# 그 절약을 상쇄하지 않는 선에서 가장 촘촘한 간격으로 잡았다.
ensure_uptime() { # ensure_uptime <표시 이름> <경로> → check_id를 표준출력으로
  local display="$1" path="$2" id
  id="$(gcloud monitoring uptime list-configs --project="$PROJECT" --format=json \
    | jq -r --arg d "$display" 'map(select(.displayName==$d)) | .[0].name // empty' | awk -F/ '{print $NF}')"
  if [[ -z "$id" ]]; then
    # 본문 문자열까지 확인한다 — 200이지만 내용이 이상한 경우(엉뚱한 서비스가 같은 URL을
    # 차지한 경우 등)를 상태 코드만으로는 구분할 수 없다.
    id="$(gcloud monitoring uptime create "$display" \
      --project="$PROJECT" \
      --resource-type=uptime-url \
      --resource-labels="host=${HOST},project_id=${PROJECT}" \
      --protocol=https --port=443 --path="$path" \
      --period=5 --timeout=10 \
      --matcher-type=contains-string --matcher-content='"status":"ok"' \
      --format='value(name)' | awk -F/ '{print $NF}')"
  fi
  echo "$id"
}

CHECK_ID_HEALTH="$(ensure_uptime "epl-predictor /health" /health)"
CHECK_ID_BATCH="$(ensure_uptime "epl-predictor /health/batch" /health/batch)"
echo "uptime check: health=${CHECK_ID_HEALTH} batch=${CHECK_ID_BATCH}"

# --- 3. 알림 정책 --------------------------------------------------------------
# 정의 파일의 CHANNEL / CHECK_ID 자리표시자를 실제 값으로 바꿔 적용한다.
existing="$(api GET "${API}/alertPolicies")"

for f in policy-service-down.json policy-service-errors.json \
         policy-job-failed.json policy-batch-stale.json policy-scheduler-errors.json; do
  body="$(sed -e "s|CHANNEL|${CHANNEL}|" \
              -e "s|CHECK_ID_HEALTH|${CHECK_ID_HEALTH}|" \
              -e "s|CHECK_ID_BATCH|${CHECK_ID_BATCH}|" "${HERE}/${f}")"
  display="$(jq -r .displayName <<<"$body")"
  name="$(jq -r --arg d "$display" '.alertPolicies // [] | map(select(.displayName==$d)) | .[0].name // empty' <<<"$existing")"

  if [[ -z "$name" ]]; then
    created="$(api POST "${API}/alertPolicies" "$body")"
    echo "정책 생성: $(jq -r '.displayName // .error.message' <<<"$created")"
  else
    # 갱신할 때는 displayName을 그대로 다시 보내도 되지만, 서버가 채우는 필드(name, 조건별
    # name 등)를 우리가 보내면 거부되므로 우리가 관리하는 필드만 마스크로 지정한다.
    updated="$(api PATCH "https://monitoring.googleapis.com/v3/${name}?updateMask=conditions,documentation,notificationChannels,alertStrategy,combiner,enabled" "$body")"
    echo "정책 갱신: $(jq -r '.displayName // .error.message' <<<"$updated")"
  fi
done
