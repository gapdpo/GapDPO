# Ethics and Governance

## Intended Use

This project creates synthetic multi-agent dialogue records for defensive deepvoice phishing research, detector evaluation, analyst training, and safety education. Generated records should remain clearly labeled as synthetic data.

## Prohibited Use

Do not use this project or its outputs to plan, automate, rehearse, or optimize social-engineering activity against real people or organizations. Do not add real credentials, OTPs, account numbers, phone numbers, identity documents, live URLs, operational infrastructure, or instructions for bypassing security controls.

## Data Safety Boundary

The generator masks common sensitive value patterns and omits original values from exported `safety_masks`. This is a guardrail, not a guarantee. Before release, run corpus-level QA and review any records with failed `safety_passed`, `residual_pii_clean`, `no_refusal_leak`, or `schema_valid` checks.

## Release Guidance

Public releases should include only synthetic, masked, non-operational records. Release notes should state that the dataset contains no real voice recordings, no real victims, no real credentials, and no valid operational phishing infrastructure.

## Review Requirements

Before sharing generated data outside the project team:

- Run the unit test suite.
- Generate Markdown and JSON QA reports.
- Inspect failed quality checks and error logs.
- Confirm that examples remain defensive and non-operational.
- Document the model endpoints, prompt version, config, and generation date used for the run.
