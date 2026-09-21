from __future__ import annotations

import json
from typing import Any


SYSTEM_POLICY = """You generate synthetic dialogue for defensive deepvoice phishing detection research.
Do not provide real credentials, valid URLs, operational fraud steps, malware, bypass instructions, or real targeted personal data.
Use placeholders such as [MASKED_OTP], [MASKED_URL], [MASKED_ACCOUNT], [MASKED_PHONE], [MASKED_EMAIL], and [MASKED_CREDENTIAL] for sensitive values.
Use [MASKED_ACCOUNT] for account, transfer, deposit, payout, or settlement destinations. Use [MASKED_OTP] only for OTP, verification code, or security code values.
Do not include stage directions, bracketed voice descriptions, narrator notes, markdown, or role labels in the utterance.
Keep the utterance realistic enough for risk-labeling but not operationally useful."""

PRESSURE_STYLE_INSTRUCTIONS = {
    "urgency": "Use limited immediacy language for timing pressure, without making every turn urgent.",
    "authority": "Emphasize claimed role, procedure, review status, or institutional authority. Avoid urgent words such as 지금, 바로, 즉시, 긴급 unless the scenario itself requires mild timing.",
    "consequence": "Emphasize non-operational consequences such as review delay, hold, cancellation, or service interruption. Avoid urgent words such as 지금, 바로, 즉시, 긴급.",
    "relationship_or_emotional_pressure": "Emphasize relationship, care responsibility, worry, or personal obligation. Avoid generic time pressure unless the relationship context directly requires it.",
    "process_control": "Emphasize staying within the caller's process, call continuity, verification sequence, or submission flow. Avoid urgent words such as 지금, 바로, 즉시, 긴급.",
}

TARGET_OUTCOME_INSTRUCTIONS = {
    "defended_success": "The victim should ultimately refuse disclosure and redirect to official or trusted verification.",
    "inconclusive": "The victim should remain unresolved through hesitation, confusion, or delay only. Do not include masked placeholder disclosure, firm refusal, or official/trusted verification.",
    "masked_compromise": "The victim may disclose only masked synthetic placeholders; never include real values or operational destinations.",
}


def attacker_messages(
    victim: dict[str, Any],
    attacker: dict[str, Any],
    scenario_type: str,
    attacker_goal: str,
    protected_assets: list[str],
    history: list[dict[str, Any]],
    phase: str,
    turn_instruction: str,
    environmental_context: dict[str, str] | None = None,
    target_outcome_mode: str = "defended_success",
    pressure_style: str = "urgency",
) -> list[dict[str, str]]:
    environmental_context_json = json.dumps(environmental_context or {}, ensure_ascii=False, sort_keys=True)
    pressure_instruction = PRESSURE_STYLE_INSTRUCTIONS.get(pressure_style, PRESSURE_STYLE_INSTRUCTIONS["urgency"])
    outcome_instruction = TARGET_OUTCOME_INSTRUCTIONS.get(
        target_outcome_mode,
        TARGET_OUTCOME_INSTRUCTIONS["defended_success"],
    )
    return [
        {"role": "system", "content": SYSTEM_POLICY},
        {
            "role": "user",
            "content": (
                "ROLE=attacker\n"
                f"scenario_type={scenario_type}\n"
                f"environmental_context={environmental_context_json}\n"
                f"attacker_persona={attacker['id']}: {attacker['description']}\n"
                f"victim_persona={victim['id']}: {victim['description']}\n"
                f"attacker_goal={attacker_goal}\n"
                f"protected_assets={protected_assets}\n"
                f"target_outcome_mode={target_outcome_mode}\n"
                f"pressure_style={pressure_style}\n"
                f"phase={phase}\n"
                f"turn_instruction={turn_instruction}\n"
                f"pressure_instruction={pressure_instruction}\n"
                f"target_outcome_instruction={outcome_instruction}\n"
                f"dialogue_history={history}\n"
                "Write only the next single Korean utterance. Adapt to the previous victim response, avoid repetition, and keep it masked and non-operational. "
                "Do not praise official verification or tell the victim that official channels are safest; keep pressure without giving safety coaching. "
                "Follow pressure_style directly; do not add urgency wording unless pressure_style=urgency or the scenario context requires mild timing language. "
                "If target_outcome_mode=inconclusive, avoid forcing a binary comply/refuse decision; leave room for hesitation without adding safety advice. "
                "If scenario_type=family_emergency and victim_persona=elderly_parent, speak as the victim's child or younger relative; do not call the victim 아빠 from the victim side."
            ),
        },
    ]


def victim_messages(
    victim: dict[str, Any],
    scenario_type: str,
    protected_assets: list[str],
    history: list[dict[str, Any]],
    phase: str,
    turn_instruction: str,
    response_mode: str = "defensive",
    environmental_context: dict[str, str] | None = None,
    target_outcome_mode: str = "defended_success",
) -> list[dict[str, str]]:
    environmental_context_json = json.dumps(environmental_context or {}, ensure_ascii=False, sort_keys=True)
    outcome_instruction = TARGET_OUTCOME_INSTRUCTIONS.get(
        target_outcome_mode,
        TARGET_OUTCOME_INSTRUCTIONS["defended_success"],
    )
    return [
        {"role": "system", "content": SYSTEM_POLICY},
        {
            "role": "user",
            "content": (
                "ROLE=victim\n"
                f"scenario_type={scenario_type}\n"
                f"environmental_context={environmental_context_json}\n"
                f"victim_persona={victim['id']}: {victim['description']}\n"
                f"resistance_style={victim.get('resistance_style', '')}\n"
                f"protected_assets={protected_assets}\n"
                f"phase={phase}\n"
                f"target_outcome_mode={target_outcome_mode}\n"
                f"victim_response_mode={response_mode}\n"
                f"turn_instruction={turn_instruction}\n"
                f"target_outcome_instruction={outcome_instruction}\n"
                f"dialogue_history={history}\n"
                "Write only the next single Korean utterance. React to the prior attacker utterance. "
                "If victim_response_mode=defensive, verify through official or trusted channels without disclosure. "
                "If victim_response_mode=uncertain, show hesitation and unresolved concern while avoiding full disclosure. "
                "If target_outcome_mode=inconclusive, do not use official/trusted verification, direct checking, refusal, denial, reporting, or any masked placeholder. "
                "For target_outcome_mode=inconclusive, answer with only uncertainty, delay, confusion, or difficulty deciding. "
                "If victim_response_mode=partial_disclosure, allow only masked placeholder-based partial compliance and never reveal real protected assets. "
                "If target_outcome_mode=masked_compromise, any compliance must use masked synthetic placeholders only. "
                "Do not include stage directions, role labels, or bracketed voice descriptions."
            ),
        },
    ]
