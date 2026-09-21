# Datasheet

## Dataset Motivation

The dataset is intended to support defensive research on synthetic deepvoice phishing dialogues. It focuses on detection, evaluation, safety review, and education rather than operational fraud simulation.

## Composition

Each JSONL record contains one synthetic phone-dialogue scenario with:

- scenario metadata and train/validation/test split
- compact victim and attacker personas
- protected asset categories
- attacker goal and outcome labels
- turn-level phase, speaker, utterance, intent label, and risk signal
- safety mask metadata without original sensitive values
- generation metadata, non-operational environmental context categories, and quality checks

The current default generator is attack-only and produces `attack`-labeled samples. `configs/default.json` does not declare a target label distribution because `benign` and `borderline` generation paths are not implemented in this pipeline. The schema still accepts `benign` and `borderline` for future compatible datasets, but those labels are outside the current default generation contract.

The local persona catalogs currently include 12 victim personas and 12 attacker personas. The default generation limit is 12 records so the baseline smoke schedule covers the current reference pairs once in insertion order, but this generation limit is not the persona catalog capacity. Catalog capacity is controlled separately by `persona_catalog_limit`, which is currently 24, leaving 12 configured slots for each catalog. The default limit must be less than or equal to this catalog capacity; legacy configs that omit `persona_catalog_limit` fall back to `default_limit`. A generation run with a manual limit below the catalog size produces only the deterministic prefix and should not be interpreted as full persona coverage. Recent coverage pairs include `freelance_creator` matched to `platform_policy_impersonator` under `platform_policy_notice`, `caregiver_or_guardian` matched to `education_admin_impersonator` under `education_admin_notice`, and `language_access_service_user` matched to `document_support_impersonator` under `document_verification_notice`. These scenarios remain non-operational and use masked placeholders for any sensitive value.

Persona expansion is not automatic just because configured capacity remains. Any added victim or attacker persona should be reviewed as synthetic defensive coverage, avoid real targets or operational instructions, use only masked sensitive values, and include focused tests for catalog capacity and schedule/scenario behavior.

Cycle-level instructions may impose a stricter temporary expansion gate, such as treating the current 12 victim and 12 attacker personas as the active cycle maximum. That gate is an approval condition for the current work, not a replacement for the repository-level `persona_catalog_limit` contract or a cap on generated record count.

## Collection and Generation

Records are generated from local persona/config files and either deterministic mock clients or OpenAI-compatible vLLM endpoints. The generation process is synthetic and should not include real calls, real audio, real victims, real credentials, or live infrastructure. Each generated record stores `gen_metadata.environmental_context` with abstract categories for channel, timing, trigger event, claimed authority, victim constraint, and verification path; these are not real contact details or operational instructions.
Some checked-in generated artifacts predate this metadata field. They may pass schema-contract validation for backward compatibility while still showing missing environmental context in QA coverage, so environment coverage should be reviewed separately from schema pass/fail.
The current generator supports 2-10 dialogue turns per record. Configurations above 10 turns are rejected because the implemented phase schedule is capped at that length.

## Preprocessing and Safety

Generated utterances pass through model-output cleanup and masking. The masking layer replaces likely URLs, OTPs, phone numbers, account-like numbers, credentials, and Korean resident-registration-like patterns with placeholders. Exported mask entries include label and placeholder only.

## Quality Signals

Records include per-record checks for role order, safety status, duplicate turns, label consistency, language consistency, refusal leaks, residual PII, and schema validity. Schema validation for the default generation path also rejects record-level `risk_labels` outside the configured `risk_label_set`. QA reports summarize label, outcome, scenario, split, persona, mask, duplicate, turn-count, quality-check, schema-contract validation, environmental context coverage, attacker pressure cue diversity, and corpus residual-pattern statistics. When the CLI writes a QA report and the fresh schema contract validation fails, it preserves the report and exits with a non-zero status. QA reports also include an attack-only policy check that flags any non-`attack` label under the current default dataset policy. Duplicate-utterance and attacker pressure cue diversity metrics are descriptive statistics; deterministic mock runs can overstate repetition because they exercise a small fixed template space.

## Recommended Uses

- defensive classifier or detector evaluation
- red-team/blue-team education with synthetic examples
- annotation workflow prototyping
- dataset schema and QA pipeline testing

## Out-of-Scope Uses

- generating scripts for real social engineering
- collecting or replaying real victim audio
- storing real sensitive values in generated records
- publishing records with failed safety checks without review

## Known Limitations

Turn labels currently rely on heuristics rather than a dedicated annotator model. Outcomes are inferred from dialogue signals and do not yet model fine-grained state transitions. The default generator does not yet produce benign or borderline samples. Duplicate-utterance rates from deterministic mock runs are not a substitute for reviewing real vLLM output diversity. vLLM behavior depends on serving configuration, chat templates, available GPU memory, and model choice.

## Maintenance

For each generated release, keep the config, code revision, model names, generation command, QA reports, and any `.errors.jsonl` file. Re-run tests and QA after prompt, safety, schema, or model changes.
