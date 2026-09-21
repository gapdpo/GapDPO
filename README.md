# Multi-Agent Deepvoice Phishing Dataset Generator

딥보이스 기반 보이스피싱 탐지와 방어 연구를 위한 **한국어 합성 대화 데이터셋 생성기**입니다.

이 프로젝트는 실제 공격을 돕기 위한 도구가 아니라, 딥보이스 사칭 상황에서 피해자가 어떤 압박을 받고 어떻게 방어하거나 망설이는지를 구조화된 데이터로 만들기 위한 연구용 파이프라인입니다. 생성되는 모든 대화는 민감정보를 placeholder로 마스킹하며, 공격 수행에 필요한 실제 URL, 인증정보, 악성 인프라, 우회 절차는 포함하지 않도록 설계되어 있습니다.

## 프로젝트 의도

딥보이스 보이스피싱은 단순한 텍스트 피싱보다 더 복잡합니다. 공격자는 가족, 기관, 회사, 병원, 학교, 플랫폼 담당자처럼 보이는 역할을 수행하고, 피해자는 자신의 상황과 취약점에 따라 다르게 반응합니다.

이 프로젝트의 목표는 그런 상호작용을 다음과 같은 형태로 데이터화하는 것입니다.

- 공격자와 피해자를 각각 별도 agent/persona로 구성한다.
- 피해자의 보호 자산과 취약 맥락을 명시한다.
- 공격자의 목표와 압박 방식을 통제한다.
- 대화 turn마다 intent, risk signal, masked item을 기록한다.
- 생성 결과를 schema, safety, QA signal로 검증한다.

현재 기본 파이프라인은 **attack-only** 데이터셋 생성기입니다. 즉 생성된 record의 `label`은 모두 `attack`입니다. benign 또는 borderline 샘플 생성은 아직 이 파이프라인에 포함되어 있지 않습니다.

## 데이터셋이 만들어지는 방식

생성 파이프라인은 크게 네 단계로 동작합니다.

1. `data/personas/victims.json`에서 피해자 페르소나를 읽는다.
2. `data/personas/attackers.json`에서 공격자 페르소나를 읽는다.
3. `configs/default.json`의 scenario, risk label, turn 수, 모델 설정을 적용한다.
4. `AttackerAgent`와 `VictimAgent`가 번갈아 발화하고, 각 turn을 마스킹/라벨링/검증한다.

`default_limit`은 기본 생성 record 수입니다. 현재 값 12는 smoke run에서 기존 기준 pair를 한 번씩 확인하기 위한 생성량이며, 페르소나 카탈로그의 최대 크기를 뜻하지 않습니다. 카탈로그 용량 상한은 별도 설정인 `persona_catalog_limit`이 담당하며, 현재 기본값은 24입니다. `default_limit`은 `persona_catalog_limit` 이하이어야 하며, 예전 설정처럼 `persona_catalog_limit`이 생략되면 `default_limit`과 같은 값으로 해석됩니다.

현재 카탈로그는 피해자 12개, 공격자 12개이고 설정상 남은 용량은 각각 12칸입니다. 따라서 `12 / 12`는 현재 카탈로그 수를 뜻할 수는 있지만 기본 설정의 확장 상한은 아닙니다. 새 페르소나를 추가할 때는 `persona_catalog_limit` 이하인지와 별개로, 기존 pair와 중복되지 않는 방어적 연구 목적, 안전한 placeholder 사용, scenario 정합성, 테스트 가능성이 확인되어야 합니다.

Cycle note나 human 지시가 `12 / 12`처럼 더 엄격한 임시 확장 gate를 둘 수 있습니다. 이 gate는 현재 작업 승인 조건이며, `configs/*.json`의 `persona_catalog_limit` 값이나 생성 record 수 제한을 바꾸지 않습니다.

출력은 JSONL입니다. 한 줄이 하나의 대화 record입니다.

각 record에는 다음 정보가 들어갑니다.

- `victim_persona`, `attacker_persona`
- `scenario_type`
- `protected_assets`
- `attacker_goal`
- `dialogue`
- `risk_labels`
- `safety_masks`
- `outcome`
- `quality_checks`
- `gen_metadata`

`gen_metadata.environmental_context`에는 각 합성 통화의 비운영 환경 축도 기록됩니다. 현재 필드는 `channel`, `timing`, `trigger_event`, `claimed_authority`, `victim_constraint`, `verification_path`이며, 실제 연락처나 링크 대신 추상 category만 사용합니다.

일부 저장소 내 과거 generated artifact는 이 metadata가 도입되기 전에 생성되어 `schema_contract_validation`은 통과하더라도 `environmental_context_coverage.missing_environmental_context`에 잡힐 수 있습니다. 따라서 환경 축 coverage는 schema pass와 별도로 확인해야 합니다.

대화는 기본적으로 공격자가 먼저 시작하고 피해자가 응답하는 구조입니다. `max_turns`는 현재 2에서 10 사이를 지원합니다.

## 피해자 페르소나

현재 피해자 페르소나는 12개입니다.

| id | 설명 |
| --- | --- |
| `elderly_parent` | 가족 긴급 상황과 정서적 압박에 민감한 고령층 |
| `office_worker` | 상급자, 거래처, 결재 요청에 빠르게 반응하는 직장인 |
| `student_or_young_adult` | 모바일 계정 보안 알림과 지인 사칭에 노출된 청년층 |
| `small_business_owner` | 정산, 납품, 예약, 고객 응대를 직접 처리하는 소상공인 |
| `remote_job_seeker` | 비대면 채용, 계약, 온보딩 안내를 기다리는 구직자 |
| `telehealth_patient` | 비대면 진료, 검사 결과, 보험 청구 안내를 기다리는 의료 이용자 |
| `renter_or_tenant` | 임대차, 관리비, 수리 일정 안내를 받는 주거 임차인 |
| `utility_account_holder` | 전기, 수도, 통신, 인터넷 계정과 고지서를 관리하는 이용자 |
| `public_benefit_recipient` | 지원금, 장학금, 보조금, 공공 서비스 안내를 기다리는 이용자 |
| `freelance_creator` | 콘텐츠 수익화, 후원 정산, 협찬 계약을 처리하는 창작자 |
| `caregiver_or_guardian` | 학교, 돌봄, 보호자 동의 안내를 받는 보호자 |
| `language_access_service_user` | 번역 지원, 행정 서류, 예약 안내를 받는 언어 접근 지원 이용자 |

각 피해자 페르소나는 `vulnerabilities`, `resistance_style`, `protected_assets`를 갖습니다. 예를 들어 `elderly_parent`는 가족 긴급 상황에 민감하지만, 후반에는 가족 확인과 은행 확인을 요구하도록 설정되어 있습니다.

## 공격자 페르소나

현재 공격자 페르소나도 12개입니다.

| id | 설명 |
| --- | --- |
| `family_impersonator` | 가족 또는 지인 목소리를 흉내 내는 사칭자 |
| `institution_impersonator` | 은행, 수사기관, 공공기관 사칭자 |
| `corporate_impersonator` | 임원, 거래처, 내부 보안팀 사칭자 |
| `service_provider_impersonator` | 결제, 배송, 예약, 유지보수 담당자 사칭자 |
| `recruiter_impersonator` | 채용 담당자, 계약 관리자, 온보딩 담당자 사칭자 |
| `healthcare_support_impersonator` | 진료 예약, 검사 결과, 보험 청구 담당자 사칭자 |
| `housing_admin_impersonator` | 임대 관리, 건물 관리, 수리 접수 담당자 사칭자 |
| `utility_support_impersonator` | 전기, 수도, 통신, 인터넷 지원 담당자 사칭자 |
| `benefits_caseworker_impersonator` | 지원금, 장학금, 보조금 담당자 사칭자 |
| `platform_policy_impersonator` | 플랫폼 정책 심사, 수익화, 정산 담당자 사칭자 |
| `education_admin_impersonator` | 학교, 돌봄, 활동 운영 담당자 사칭자 |
| `document_support_impersonator` | 행정 서류, 번역 지원, 접수 보완 담당자 사칭자 |

공격자 페르소나는 `default_goals`와 `pressure_methods`를 갖습니다. 예를 들어 `family_impersonator`는 송금 요청이나 통화 유지 요청을 목표로 하고, `institution_impersonator`는 인증정보나 신원정보 요청을 목표로 합니다.

## 시나리오와 risk label

현재 지원하는 `scenario_type`은 다음 8개입니다.

- `family_emergency`
- `bank_fraud_alert`
- `workplace_authority`
- `essential_service_notice`
- `document_verification_notice`
- `public_benefit_notice`
- `platform_policy_notice`
- `education_admin_notice`

각 시나리오는 기본 risk context를 갖습니다. 예를 들어 가족 긴급 상황은 `voice_impersonation`, `urgency`, `emotional_pressure`와 연결되고, 공공지원 안내는 `authority_impersonation`, `identity_information_request`, `urgency`와 연결됩니다.

현재 risk label set은 다음과 같습니다.

- `voice_impersonation`
- `urgency`
- `authority_impersonation`
- `emotional_pressure`
- `money_transfer_request`
- `credential_request`
- `identity_information_request`
- `callback_or_voice_sample_request`

## 안전 마스킹

생성된 발화는 민감값을 직접 포함하지 않고 placeholder를 사용합니다.

표준 placeholder는 다음과 같습니다.

| placeholder | 의미 |
| --- | --- |
| `[MASKED_URL]` | URL 또는 링크 |
| `[MASKED_RRN]` | 주민등록번호 형태 |
| `[MASKED_OTP]` | OTP, 인증번호, 보안코드 |
| `[MASKED_PHONE]` | 전화번호 |
| `[MASKED_EMAIL]` | 이메일 주소 |
| `[MASKED_ACCOUNT]` | 계좌, 송금, 정산, 입금 대상 |
| `[MASKED_CREDENTIAL]` | 비밀번호, PIN 등 인증정보 |

QA는 비표준 `[MASKED_*]`도 별도로 집계합니다. 예를 들어 모델이 `[MASKED_NAME]`, `[MASKED_COMPANY]`, `[MASKED_ID_IMAGE]` 같은 토큰을 만들면 데이터 실패로 처리하지는 않지만 `unknown_masked_placeholder_count`와 예시로 리포트합니다.

현재 비표준 placeholder는 표준 mask로 자동 승격하지 않고, QA의 `unknown_masked_placeholder_policy`에서 검토 상태를 함께 표시합니다.

| decision | placeholder | 처리 |
| --- | --- | --- |
| `allowlist_candidate` | `[MASKED_NAME]`, `[MASKED_COMPANY]`, `[MASKED_CASE_REFERENCE]`, `[MASKED_ID_IMAGE]` | 범용 synthetic context로 유용하지만 아직 표준 placeholder는 아니므로 warning 집계 유지 |
| `normalize_to_existing` | `[MASKED_RECRUITER_NAME]`, `[MASKED_CREATOR_NAME]`, `[MASKED_COMPANY_EMAIL]` | 후속 정규화 시 이름 계열은 `[MASKED_NAME]`, 이메일 계열은 `[MASKED_EMAIL]`로 접을 후보 |
| `warning_keep_unstandardized` | `[MASKED_JOB_TITLE]`, `[MASKED_LANGUAGE_SUPPORT_PORTAL]` | 현재 masking 표준과 의미가 맞지 않아 별도 설계 전까지 warning 유지 |

## outcome과 labeling

현재 outcome은 다음 값 중 하나입니다.

- `defended_success`: 피해자가 공식 확인, 거절, 직접 확인 등으로 방어에 성공
- `compromised`: 피해자가 placeholder 기반으로라도 순응 또는 자산 제공 의사를 보임
- `inconclusive`: 방어와 부분 순응이 섞여 명확히 성공/실패로 보기 어려움
- `aborted`: 대화가 중단됨

예를 들어 “공식 확인 전에는 제공할 수 없습니다”는 `defended_success`로 볼 수 있습니다. 반면 “`[MASKED_ACCOUNT]`로 제출할 수 있습니다”는 부분 순응 신호이며, “공식 확인 절차를 거친 후에만 제출할 수 있습니다”처럼 조건부 방어가 붙으면 `inconclusive`로 분리합니다.

## Quick Start

기본 conda 환경을 만듭니다.

```bash
conda env create -f environment.yml
conda activate multi-agent-db
```

모델 서버 없이 deterministic mock 데이터 3건을 생성합니다.

```bash
PYTHONPATH=src python -m madb.cli \
  --config configs/default.json \
  --output data/generated/sample.jsonl \
  --limit 3 \
  --mock
```

QA 리포트까지 생성하려면 다음처럼 실행합니다.

```bash
PYTHONPATH=src python -m madb.cli \
  --config configs/default.json \
  --output data/generated/run.jsonl \
  --limit 12 \
  --mock \
  --qa-report data/generated/qa.md \
  --qa-report-json data/generated/qa.json
```

중단된 run을 이어서 만들고 싶으면 `--append` 또는 `--resume`을 사용합니다. `--append`는 기존 output의 record id를 건너뛰면서 `--limit` 개수만큼 새 record를 덧붙이고, `--resume`은 기존 record 수를 포함해 총 `--limit` 개수가 되도록 부족분만 생성합니다.

```bash
PYTHONPATH=src python -m madb.cli \
  --config configs/default.json \
  --output data/generated/run.jsonl \
  --limit 100 \
  --mock \
  --append

PYTHONPATH=src python -m madb.cli \
  --config configs/default.json \
  --output data/generated/run.jsonl \
  --limit 100 \
  --mock \
  --resume
```

## Blackwell/vLLM 실행

Blackwell GPU 환경에서는 `environment-blackwell.yml`과 로컬 Qwen3 모델을 사용합니다.

```bash
conda env create -f environment-blackwell.yml
conda activate multi-db-blackwell
```

로컬 모델이 없다면 먼저 다운로드합니다.

```bash
bash ./scripts/download_qwen3_30b_a3b.sh
```

vLLM 서버를 띄우고 12건을 생성하려면 다음 스크립트를 사용합니다.

```bash
bash ./scripts/run_local_qwen3_30b_a3b_generation.sh
```

출력 경로를 지정하려면 환경변수를 넘깁니다.

```bash
LIMIT=12 \
OUTPUT_PATH=data/generated/run_quality_v3.jsonl \
QA_REPORT_PATH=data/generated/qa_quality_v3.md \
QA_REPORT_JSON_PATH=data/generated/qa_quality_v3.json \
bash ./scripts/run_local_qwen3_30b_a3b_generation.sh
```

기본 로컬 모델 설정은 다음과 같습니다.

- model: `./models/Qwen3-30B-A3B-Instruct-2507`
- served model name: `local-qwen3-30b-a3b-instruct-2507`
- default GPU ids: `3,4`
- tensor parallel size: visible GPU 개수
- max model length: `8192`
- MoE backend: `triton`
- FlashInfer sampler: disabled via `VLLM_USE_FLASHINFER_SAMPLER=0`

## QA 리포트 읽는 법

QA 리포트는 `data/generated/*.md`와 `*.json`으로 생성됩니다.

중요한 항목은 다음과 같습니다.

- `schema_contract_validation`: config 기준 schema 검증 결과
- `quality_checks`: record 내부의 기본 품질 검사
- `stage_direction_count`: 발화 안에 무대 지시문이 남았는지
- `mask_usage_anomaly_count`: OTP를 계좌처럼 쓰는 등 mask 의미 오류가 있는지
- `unknown_masked_placeholder_count`: 표준이 아닌 `[MASKED_*]` 사용 수
- `partial_compliance_signal_count`: 피해자의 부분 순응 신호 수
- `scenario_persona_mismatch_count`: pair별 시나리오 매핑 오류 수
- `attacker_pressure_cue_diversity`: 공격자 발화의 압박 cue가 urgency에 과도하게 치우치는지 보는 warning-only category 분포
- `outcome_distribution`: 방어 성공, 부분 실패, 애매함 등 outcome 분포
- `environmental_context_coverage`: `gen_metadata.environmental_context`의 구조화 환경 필드 누락과 분포

현재 outcome 다양성은 blocking gate가 아니라 warning-only signal입니다. 작은 12건 smoke run에서는 모든 outcome을 강제로 다양하게 만드는 것보다 schema, safety, scenario 정합성, 마스킹 품질을 우선 확인합니다.

## 테스트

전체 테스트는 다음 명령으로 실행합니다.

```bash
PYTHONPATH=src python -m pytest -q
```

## 주요 파일

| 경로 | 역할 |
| --- | --- |
| `configs/default.json` | 기본 생성 설정, 시나리오, risk label set |
| `configs/local-qwen3-30b-a3b.json` | 로컬 Qwen3/vLLM 실행용 설정 |
| `data/personas/victims.json` | 피해자 페르소나 카탈로그 |
| `data/personas/attackers.json` | 공격자 페르소나 카탈로그 |
| `src/madb/orchestrator.py` | 페르소나 매칭, 시나리오 선택, record 생성 |
| `src/madb/prompts.py` | 공격자/피해자 prompt 정책 |
| `src/madb/safety.py` | 민감값 마스킹 |
| `src/madb/labeling.py` | turn intent, risk signal, outcome 추론 |
| `src/madb/qa.py` | QA summary와 Markdown/JSON 리포트 |
| `scripts/run_local_qwen3_30b_a3b_generation.sh` | 로컬 vLLM 서버 실행 및 데이터 생성 |
| `ETHICS.md` | 윤리 및 안전 경계 |
| `DATASHEET.md` | 데이터셋 공개/검토용 datasheet |

## 안전 경계

이 프로젝트는 탐지, 평가, 교육, 방어 연구를 위한 합성 데이터 생성기입니다.

다음은 포함하지 않아야 합니다.

- 실제 인증정보, 실제 OTP, 실제 계좌번호
- 유효한 피싱 URL 또는 운영 가능한 공격 인프라
- 악성코드 설치, 인증 우회, 보안 우회 절차
- 특정 개인이나 기관을 대상으로 한 실제 공격 지시

생성 데이터를 외부에 공유하거나 논문/보고서/벤치마크에 사용할 경우 `ETHICS.md`와 `DATASHEET.md`를 먼저 확인해야 합니다.
