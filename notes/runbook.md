# 운영 런북 — 관측 & 롤백

이 문서 하나만 보고, 이 프로젝트를 처음 보는 사람도 "장애를 알아차리고 → 이전
버전으로 되돌린다"를 끝까지 따라갈 수 있어야 한다. 아래 명령어는 이 프로젝트의
실제 값(프로젝트명, 서비스명, 리전)이 그대로 박혀 있어 복붙으로 실행 가능하다.

- 프로젝트: `genai-backend-cloud-roadmap`
- 서비스: `epl-predictor`
- 리전: `asia-northeast3`
- 배포된 URL: `https://epl-predictor-187506981041.asia-northeast3.run.app`

---

## 1. 헬스체크 실패를 어떻게 알아차리는가

**지금 이 스프린트 스코프에서는 자동 알림(alerting)이 설정돼 있지 않다.** 실제
프로덕션이라면 Cloud Monitoring의 Uptime Check + Alerting Policy(또는 AWS의
Route 53 Health Check + CloudWatch Alarm, Azure의 Application Insights 가용성
테스트 — 이름은 다르지만 "주기적으로 외부에서 두드려보고, 실패하면 알림"이라는
원리는 동일)를 붙이는 게 맞다. 이건 이번 스코프 밖이라 아래는 수동 확인 절차다.

**수동으로 지금 살아있는지 확인:**
```bash
curl -sS -w "\nstatus: %{http_code}\n" https://epl-predictor-187506981041.asia-northeast3.run.app/health
```
`{"status":"ok"}`와 `status: 200`이 아니면 장애다.

**최근 리비전들의 상태 확인:**
```bash
gcloud run revisions list \
  --service=epl-predictor --region=asia-northeast3 \
  --project=genai-backend-cloud-roadmap
```
`ACTIVE` 컬럼과 `READY` 상태를 본다. 새 리비전이 `False`(준비 안 됨)로 나오면
배포 자체가 실패한 것 — 이 경우 Cloud Run은 기본적으로 트래픽을 그 리비전으로
넘기지 않으므로 서비스는 이전 리비전으로 계속 살아있다(자동 안전장치). 문제는
"리비전은 준비됐지만(READY=True) 동작이 잘못된 경우"다 — 이땐 트래픽이 이미
넘어가 있으므로 아래 2번 롤백 절차가 필요하다.

**최근 요청 로그를 구조화 로그로 훑기(요청 ID 포함):**
```bash
gcloud logging read \
  'resource.type=cloud_run_revision AND resource.labels.service_name=epl-predictor AND jsonPayload.level="error"' \
  --project=genai-backend-cloud-roadmap --limit=20 \
  --format="table(timestamp, jsonPayload.request_id, jsonPayload.message, jsonPayload.error)"
```
특정 요청 하나를 끝까지 추적하고 싶으면 `jsonPayload.request_id="<값>"` 조건을
추가한다 — API 응답 헤더 `X-Request-ID`에 그 요청의 ID가 그대로 실려 있으므로,
사용자가 신고한 요청의 ID만 있으면 그 요청과 관련된 로그 줄을 전부 찾을 수 있다.

---

## 2. 롤백 절차

**2-1. 정상이었던 리비전 이름을 확인한다:**
```bash
gcloud run revisions list \
  --service=epl-predictor --region=asia-northeast3 \
  --project=genai-backend-cloud-roadmap \
  --format="table(name, active, status.conditions[0].status, creationTimestamp)"
```
`active=True`가 지금 트래픽을 받는 리비전이다. 그 바로 이전, 정상 작동이 확인됐던
리비전 이름(예: `epl-predictor-00003-4jp`)을 적어둔다.

**2-2. 트래픽을 그 리비전으로 100% 되돌린다:**
```bash
gcloud run services update-traffic epl-predictor \
  --region=asia-northeast3 --project=genai-backend-cloud-roadmap \
  --to-revisions=epl-predictor-00003-4jp=100
```
(`epl-predictor-00003-4jp` 자리에 2-1에서 확인한 실제 리비전 이름을 넣는다.)

**2-3. 되돌아갔는지 확인한다:**
```bash
curl -sS -w "\nstatus: %{http_code}\n" https://epl-predictor-187506981041.asia-northeast3.run.app/health
```
`{"status":"ok"}` / `status: 200`이면 롤백 완료.

**주의할 것**: 이 절차는 트래픽만 되돌릴 뿐, 문제가 된 리비전 자체를 지우지는
않는다. 원인 분석이 끝난 뒤에는 새 코드로 정상 리비전을 다시 배포해서 문제
리비전을 대체하는 게 정석 — 트래픽 롤백은 "일단 지혈"이지 "치료"가 아니다.

---

## 3. 매일 자동 예측 Job(`epl-daily-predict`) 상태 확인

Cloud Scheduler(`epl-daily-predict-trigger`)가 매일 06:00 UTC에 Cloud Run Job
(`epl-daily-predict`)을 트리거하고, 결과는 `gs://epl-predictor-daily-genai-backend/predictions/YYYY-MM-DD.json`에 쌓인다.

**오늘 파일이 생겼는지 확인:**
```bash
gcloud storage cat gs://epl-predictor-daily-genai-backend/predictions/$(date -u +%Y-%m-%d).json
```
그날 EPL 경기가 없으면 파일 자체가 없는 게 정상(Job이 "오늘 경기 없음"으로
로그만 남기고 종료).

**최근 실행 이력과 성공/실패 확인:**
```bash
gcloud run jobs executions list --job=epl-daily-predict \
  --region=asia-northeast3 --project=genai-backend-cloud-roadmap --limit=5
```
`FAILED_COUNT`가 1이면 실패 — 아래 로그로 원인 확인:
```bash
gcloud logging read \
  'resource.type=cloud_run_job AND resource.labels.job_name=epl-daily-predict AND jsonPayload.level="error"' \
  --project=genai-backend-cloud-roadmap --limit=20 \
  --format="table(timestamp, jsonPayload.message, jsonPayload.error)"
```

**스케줄러 자체가 Job을 트리거했는지(403/401 등 인증 오류) 확인:**
```bash
gcloud logging read 'logName:"cloudscheduler.googleapis.com"' \
  --project=genai-backend-cloud-roadmap --freshness=1d \
  --format="table(timestamp, httpRequest.status, jsonPayload.@type)"
```
`httpRequest.status`가 200이 아니면 스케줄러→Job 트리거 자체가 실패한 것(Job
안쪽 코드 문제가 아님) — 대상 URI가 `run.googleapis.com/v2/...:run`(v2 API)인지,
`scheduler-invoker` 서비스 계정에 대상 Job의 `roles/run.invoker`가 붙어있는지
확인한다.
