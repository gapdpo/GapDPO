# Multi-Agent Deepvoice Phishing Dataset Generator

A **Korean synthetic dialogue dataset generator** for research on detecting and defending against deepvoice-based voice phishing.

This project is not a tool for assisting real attacks. It is a research pipeline that turns deepvoice impersonation situations into structured data, capturing what kind of pressure victims face and how they defend themselves or hesitate. All generated dialogues mask sensitive information with placeholders, and the pipeline is designed so that no real URLs, credentials, malicious infrastructure, or bypass procedures needed to carry out an attack are included.

## Project Motivation

Deepvoice voice phishing is more complex than simple text-based phishing. Attackers take on roles that appear to be family members, institutions, companies, hospitals, schools, or platform staff, and victims respond differently depending on their circumstances and vulnerabilities.

The goal of this project is to turn such interactions into data of the following form.

- Configure the attacker and the victim as separate agents/personas.
- Specify the victim's protected assets and vulnerability context.
- Control the attacker's goals and pressure methods.
- Record the intent, risk signals, and masked items for every dialogue turn.
- Validate generated outputs with schema, safety, and QA signals.

The current default pipeline is an **attack-only** dataset generator. That is, the `label` of every generated record is `attack`. Generation of benign or borderline samples is not yet included in this pipeline.

## How the Dataset Is Generated

The generation pipeline operates in four main stages.

1. Load victim personas from `data/personas/victims.json`.
2. Load attacker personas from `data/personas/attackers.json`.
3. Apply the scenarios, risk labels, number of turns, and model settings from `configs/default.json`.
4. `AttackerAgent` and `VictimAgent` take turns speaking, and each turn is masked, labeled, and validated.

`default_limit` is the default number of records to generate. The current value of 12 is the generation size used in smoke runs to check each baseline pair once; it does not represent the maximum size of the persona catalog. The catalog capacity limit is controlled by a separate setting, `persona_catalog_limit`, whose current default is 24. `default_limit` must be less than or equal to `persona_catalog_limit`, and if `persona_catalog_limit` is omitted, as in older configurations, it is interpreted as equal to `default_limit`.

The current catalog contains 12 victims and 12 attackers, leaving 12 remaining slots for each under the configuration. Therefore, `12 / 12` may refer to the current catalog size, but it is not the expansion limit of the default configuration. When adding a new persona, apart from staying within `persona_catalog_limit`, the following must also be confirmed: a defensive research purpose that does not duplicate existing pairs, safe placeholder usage, scenario consistency, and testability.

A cycle note or human instruction may impose a stricter temporary expansion gate such as `12 / 12`. Such a gate is an approval condition for the current task and does not change the `persona_catalog_limit` value in `configs/*.json` or the limit on the number of generated records.

The output is JSONL, with one dialogue record per line.

Each record contains the following information.

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

`gen_metadata.environmental_context` also records the non-operational environmental dimensions of each synthetic call. The current fields are `channel`, `timing`, `trigger_event`, `claimed_authority`, `victim_constraint`, and `verification_path`, and only abstract categories are used instead of real contact information or links.

Some past generated artifacts in the repository were produced before this metadata was introduced, so they may pass `schema_contract_validation` but still be flagged under `environmental_context_coverage.missing_environmental_context`. Coverage of the environmental dimensions should therefore be checked separately from the schema pass.

By default, dialogues are structured so that the attacker speaks first and the victim responds. `max_turns` currently supports values between 2 and 10.

## Victim Personas

There are currently 12 victim personas.

| id | Description |
| --- | --- |
| `elderly_parent` | An older adult sensitive to family emergencies and emotional pressure |
| `office_worker` | An office worker who responds quickly to requests from superiors, business partners, and approval workflows |
| `student_or_young_adult` | A young adult exposed to mobile account security alerts and acquaintance impersonation |
| `small_business_owner` | A small business owner who personally handles settlements, deliveries, reservations, and customer service |
| `remote_job_seeker` | A job seeker awaiting remote hiring, contract, and onboarding notices |
| `telehealth_patient` | A healthcare user awaiting telehealth consultation, test result, and insurance claim notices |
| `renter_or_tenant` | A residential tenant receiving notices about leases, maintenance fees, and repair schedules |
| `utility_account_holder` | A user who manages electricity, water, telecom, and internet accounts and bills |
| `public_benefit_recipient` | A user awaiting notices about subsidies, scholarships, grants, and public services |
| `freelance_creator` | A creator who handles content monetization, donation settlements, and sponsorship contracts |
| `caregiver_or_guardian` | A guardian receiving notices about school, childcare, and guardian consent |
| `language_access_service_user` | A language access service user receiving translation support, administrative document, and reservation notices |

Each victim persona has `vulnerabilities`, `resistance_style`, and `protected_assets`. For example, `elderly_parent` is sensitive to family emergencies but is configured to request confirmation from family members and the bank in the later part of the dialogue.

## Attacker Personas

There are also currently 12 attacker personas.

| id | Description |
| --- | --- |
| `family_impersonator` | An impersonator mimicking the voice of a family member or acquaintance |
| `institution_impersonator` | An impersonator of a bank, investigative agency, or public institution |
| `corporate_impersonator` | An impersonator of an executive, business partner, or internal security team |
| `service_provider_impersonator` | An impersonator of payment, delivery, reservation, or maintenance staff |
| `recruiter_impersonator` | An impersonator of a recruiter, contract manager, or onboarding coordinator |
| `healthcare_support_impersonator` | An impersonator of staff handling appointments, test results, or insurance claims |
| `housing_admin_impersonator` | An impersonator of lease management, building management, or repair intake staff |
| `utility_support_impersonator` | An impersonator of electricity, water, telecom, or internet support staff |
| `benefits_caseworker_impersonator` | An impersonator of staff handling subsidies, scholarships, or grants |
| `platform_policy_impersonator` | An impersonator of staff handling platform policy review, monetization, or settlements |
| `education_admin_impersonator` | An impersonator of school, childcare, or activity program staff |
| `document_support_impersonator` | An impersonator of staff handling administrative documents, translation support, or application supplements |

Attacker personas have `default_goals` and `pressure_methods`. For example, `family_impersonator` aims to request a money transfer or to keep the call going, while `institution_impersonator` aims to obtain credentials or identity information.

## Scenarios and Risk Labels

The following 8 `scenario_type` values are currently supported.

- `family_emergency`
- `bank_fraud_alert`
- `workplace_authority`
- `essential_service_notice`
- `document_verification_notice`
- `public_benefit_notice`
- `platform_policy_notice`
- `education_admin_notice`

Each scenario has a default risk context. For example, a family emergency is associated with `voice_impersonation`, `urgency`, and `emotional_pressure`, while a public benefit notice is associated with `authority_impersonation`, `identity_information_request`, and `urgency`.

The current risk label set is as follows.

- `voice_impersonation`
- `urgency`
- `authority_impersonation`
- `emotional_pressure`
- `money_transfer_request`
- `credential_request`
- `identity_information_request`
- `callback_or_voice_sample_request`

## Safety Masking

Generated utterances do not contain sensitive values directly and use placeholders instead.

The standard placeholders are as follows.

| placeholder | Meaning |
| --- | --- |
| `[MASKED_URL]` | URL or link |
| `[MASKED_RRN]` | Resident registration number format |
| `[MASKED_OTP]` | OTP, verification code, or security code |
| `[MASKED_PHONE]` | Phone number |
| `[MASKED_EMAIL]` | Email address |
| `[MASKED_ACCOUNT]` | Account, transfer, settlement, or deposit target |
| `[MASKED_CREDENTIAL]` | Password, PIN, or other credentials |

QA also counts non-standard `[MASKED_*]` tokens separately. For example, if the model produces tokens such as `[MASKED_NAME]`, `[MASKED_COMPANY]`, or `[MASKED_ID_IMAGE]`, they are not treated as data failures but are reported in `unknown_masked_placeholder_count` along with examples.

Non-standard placeholders are currently not automatically promoted to standard masks; instead, their review status is shown in QA's `unknown_masked_placeholder_policy`.

| decision | placeholder | Handling |
| --- | --- | --- |
| `allowlist_candidate` | `[MASKED_NAME]`, `[MASKED_COMPANY]`, `[MASKED_CASE_REFERENCE]`, `[MASKED_ID_IMAGE]` | Useful as general synthetic context but not yet standard placeholders, so warning counts are retained |
| `normalize_to_existing` | `[MASKED_RECRUITER_NAME]`, `[MASKED_CREATOR_NAME]`, `[MASKED_COMPANY_EMAIL]` | Candidates to be folded during later normalization: name-type tokens into `[MASKED_NAME]` and email-type tokens into `[MASKED_EMAIL]` |
| `warning_keep_unstandardized` | `[MASKED_JOB_TITLE]`, `[MASKED_LANGUAGE_SUPPORT_PORTAL]` | Do not fit the meaning of the current masking standard, so warnings are retained until a separate design is made |

## Outcome and Labeling

The outcome is currently one of the following values.

- `defended_success`: The victim successfully defends through official verification, refusal, direct confirmation, or similar means
- `compromised`: The victim shows compliance or willingness to provide assets, even if only on a placeholder basis
- `inconclusive`: Defense and partial compliance are mixed, making it difficult to classify clearly as success or failure
- `aborted`: The dialogue was interrupted

For example, "I cannot provide it before official verification" can be regarded as `defended_success`. In contrast, "I can submit it to `[MASKED_ACCOUNT]`" is a partial compliance signal, and when a conditional defense is attached, as in "I can submit it only after going through an official verification procedure," it is classified separately as `inconclusive`.

## Quick Start

Create the default conda environment.

```bash
conda env create -f environment.yml
conda activate multi-agent-db
```

Generate 3 deterministic mock records without a model server.

```bash
PYTHONPATH=src python -m madb.cli \
  --config configs/default.json \
  --output data/generated/sample.jsonl \
  --limit 3 \
  --mock
```

To also generate a QA report, run the following.

```bash
PYTHONPATH=src python -m madb.cli \
  --config configs/default.json \
  --output data/generated/run.jsonl \
  --limit 12 \
  --mock \
  --qa-report data/generated/qa.md \
  --qa-report-json data/generated/qa.json
```

To continue an interrupted run, use `--append` or `--resume`. `--append` skips record ids already in the existing output and appends `--limit` new records, while `--resume` generates only the shortfall so that the total, including existing records, reaches `--limit`.

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

## Running on Blackwell/vLLM

In a Blackwell GPU environment, use `environment-blackwell.yml` and a local Qwen3 model.

```bash
conda env create -f environment-blackwell.yml
conda activate multi-db-blackwell
```

If the local model is not available, download it first.

```bash
bash ./scripts/download_qwen3_30b_a3b.sh
```

To launch a vLLM server and generate 12 records, use the following script.

```bash
bash ./scripts/run_local_qwen3_30b_a3b_generation.sh
```

To specify output paths, pass environment variables.

```bash
LIMIT=12 \
OUTPUT_PATH=data/generated/run_quality_v3.jsonl \
QA_REPORT_PATH=data/generated/qa_quality_v3.md \
QA_REPORT_JSON_PATH=data/generated/qa_quality_v3.json \
bash ./scripts/run_local_qwen3_30b_a3b_generation.sh
```

The default local model settings are as follows.

- model: `./models/Qwen3-30B-A3B-Instruct-2507`
- served model name: `local-qwen3-30b-a3b-instruct-2507`
- default GPU ids: `3,4`
- tensor parallel size: number of visible GPUs
- max model length: `8192`
- MoE backend: `triton`
- FlashInfer sampler: disabled via `VLLM_USE_FLASHINFER_SAMPLER=0`

## Reading the QA Report

QA reports are generated as `data/generated/*.md` and `*.json`.

The key items are as follows.

- `schema_contract_validation`: Schema validation results against the config
- `quality_checks`: Basic quality checks within each record
- `stage_direction_count`: Whether stage directions remain in utterances
- `mask_usage_anomaly_count`: Whether there are mask semantic errors, such as using an OTP mask as an account
- `unknown_masked_placeholder_count`: Number of non-standard `[MASKED_*]` usages
- `partial_compliance_signal_count`: Number of partial compliance signals from the victim
- `scenario_persona_mismatch_count`: Number of scenario mapping errors per pair
- `attacker_pressure_cue_diversity`: A warning-only category distribution showing whether pressure cues in attacker utterances are overly skewed toward urgency
- `outcome_distribution`: Distribution of outcomes such as successful defense, partial failure, and ambiguity
- `environmental_context_coverage`: Missing fields and distribution of the structured environmental fields in `gen_metadata.environmental_context`

Outcome diversity is currently a warning-only signal, not a blocking gate. In small 12-record smoke runs, checking schema, safety, scenario consistency, and masking quality takes priority over forcing all outcomes to be diverse.

## Testing

Run the full test suite with the following command.

```bash
PYTHONPATH=src python -m pytest -q
```

## Key Files

| Path | Role |
| --- | --- |
| `configs/default.json` | Default generation settings, scenarios, and risk label set |
| `configs/local-qwen3-30b-a3b.json` | Settings for running local Qwen3/vLLM |
| `data/personas/victims.json` | Victim persona catalog |
| `data/personas/attackers.json` | Attacker persona catalog |
| `src/madb/orchestrator.py` | Persona matching, scenario selection, and record generation |
| `src/madb/prompts.py` | Attacker/victim prompt policies |
| `src/madb/safety.py` | Sensitive value masking |
| `src/madb/labeling.py` | Turn intent, risk signal, and outcome inference |
| `src/madb/qa.py` | QA summary and Markdown/JSON reports |
| `scripts/run_local_qwen3_30b_a3b_generation.sh` | Launching the local vLLM server and generating data |
| `ETHICS.md` | Ethics and safety boundaries |
| `DATASHEET.md` | Datasheet for dataset release and review |

## Safety Boundaries

This project is a synthetic data generator for detection, evaluation, education, and defense research.

The following must not be included.

- Real credentials, real OTPs, or real account numbers
- Valid phishing URLs or operational attack infrastructure
- Procedures for installing malware, bypassing authentication, or circumventing security
- Real attack instructions targeting specific individuals or institutions

If you share the generated data externally or use it in papers, reports, or benchmarks, please review `ETHICS.md` and `DATASHEET.md` first.
