from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_test.benchmark_lib import MASKED_PLACEHOLDER_RE, write_jsonl
from benchmark_stream_v1.constants import EVENT_TO_ACTION, EVENT_TO_POLICY_RULE_IDS
from madb.config import load_json_list, validate_personas
from madb.labeling import collect_risk_labels, infer_record_outcome
from madb.orchestrator import (
    SCENARIO_RISK_LABELS,
    choose_attacker_goal,
    choose_scenario_type,
    compact_persona,
    context_for_personas,
    default_assets_for_scenario,
    difficulty_for_index,
    environmental_context_for_scenario,
    label_consistent_for_record,
    language_consistent_for_record,
    make_record_id,
    no_duplicate_turns,
    no_refusal_leak,
    role_order_valid,
    split_for_index,
)
from madb.postprocess import normalized_for_duplicate_check
from madb.safety import SafetyMasker
from madb.schema import validate_record

STREAM_PROFILES = (
    "safe_containment",
    "caller_directed_path",
    "partial_disclosure_recovery",
    "benign_thin_context",
    "pressure_escalation",
)

TRAJECTORY_VARIED_MODE = "event_balanced_300_trajectory_varied"
TRAJECTORY_VARIANT_COUNT = 6

MODE_DEFAULTS = {
    "event_balanced_300": 300,
    "persona_balanced_288": 288,
    TRAJECTORY_VARIED_MODE: 300,
}

PHASE_BY_TURN = {
    1: "opening",
    2: "opening",
    3: "establish_context",
    4: "establish_context",
    5: "make_request",
    6: "handle_resistance",
    7: "handle_resistance",
    8: "close",
}

DIRECT_DISCLOSURE_VERBS = (
    "전달하겠습니다",
    "제공하겠습니다",
    "말씀드리겠습니다",
    "입력하겠습니다",
    "공유하겠습니다",
    "알려드리겠습니다",
    "알려드릴게요",
)

CALLER_REQUEST_VOICE_MARKERS = (
    "확인해 주시면",
    "확인해주시면",
    "확인 부탁",
    "부탁드립니다",
    "주시기 바랍니다",
    "제공해 주시면",
    "입력해 주시면",
    "제출해 주시면",
    "필요합니다",
    "요구드립니다",
    "조치 부탁",
)

PRESSURE_MARKERS = (
    "긴급",
    "즉시",
    "지금",
    "오늘",
    "마감",
    "지연",
    "보류",
    "취소",
    "제한",
    "손실",
    "차질",
    "불가능",
)

CONTAINMENT_MARKERS = (
    "제공하지",
    "입력하지",
    "사용하지",
    "접속하지",
    "진행하지",
    "공식",
    "저장된",
    "직접",
    "확인하겠습니다",
)

BENIGN_STRONG_RESPONSE_MARKERS = (
    "신고",
    "차단",
    "고발",
)

INSTRUCTION_RESIDUE_MARKERS = (
    "보류하세요",
    "답하세요",
    "설명하세요",
    "요구하세요",
    "마무리하세요",
    "선택하세요",
    "제시하세요",
    "말하세요",
    "망설이세요",
    "끝내세요",
)

PATH_LANGUAGE_MARKERS = (
    "링크",
    "접속",
    "경로",
    "포털",
    "주소",
    "열어",
    "들어가",
)

VERY_SHORT_UTTERANCES = (
    "기다리겠습니다",
    "보류하겠습니다",
    "설명만 듣겠습니다",
)


@dataclass(frozen=True)
class PlannedTurn:
    speaker: str
    phase: str
    intent_label: str
    risk_signal: str
    event: str
    placeholders: tuple[str, ...]
    instruction: str
    fallback: str
    evidence_turn_ids: tuple[int, ...]
    action: str | None = None
    policy_rule_ids: tuple[str, ...] = ()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate persona-grounded stream-ready dialogue DB records.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--mode", choices=sorted(MODE_DEFAULTS), default="event_balanced_300")
    parser.add_argument("--records", type=int, help="Override record count.")
    parser.add_argument("--dataset-name", default="deepvoice_phishing_stream_v2")
    parser.add_argument("--victims", default="data/personas/victims.json")
    parser.add_argument("--attackers", default="data/personas/attackers.json")
    parser.add_argument("--config", default="configs/local-qwen3-30b-a3b.json", help="Scenario list source.")
    parser.add_argument("--vllm-endpoint", help="OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--vllm-model", default="local-qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    records = generate_records(
        mode=args.mode,
        record_count=args.records or MODE_DEFAULTS[args.mode],
        dataset_name=args.dataset_name,
        victims_path=args.victims,
        attackers_path=args.attackers,
        config_path=args.config,
        vllm_endpoint=args.vllm_endpoint,
        vllm_model=args.vllm_model,
        max_retries=args.max_retries,
    )
    write_jsonl(args.out, records)
    print(f"Wrote {len(records)} persona stream records to {args.out}")
    print(f"profile_distribution={dict(Counter(r['gen_metadata']['stream_profile'] for r in records))}")
    print(f"persona_pair_count={len(set((r['victim_persona']['id'], r['attacker_persona']['id']) for r in records))}")
    return 0


def generate_records(
    *,
    mode: str,
    record_count: int,
    dataset_name: str,
    victims_path: str | Path = "data/personas/victims.json",
    attackers_path: str | Path = "data/personas/attackers.json",
    config_path: str | Path = "configs/local-qwen3-30b-a3b.json",
    vllm_endpoint: str | None = None,
    vllm_model: str = "local-qwen3-30b-a3b-instruct-2507",
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    if mode not in MODE_DEFAULTS:
        raise ValueError(f"Unsupported mode: {mode}")
    if record_count < 1:
        raise ValueError("record_count must be positive")
    victims = load_json_list(victims_path)
    attackers = load_json_list(attackers_path)
    validate_personas(victims, kind="victim")
    validate_personas(attackers, kind="attacker")
    configured_scenarios = configured_scenario_types(config_path)
    safety = SafetyMasker()
    return [
        build_record(
            index=index,
            mode=mode,
            dataset_name=dataset_name,
            victims=victims,
            attackers=attackers,
            configured_scenarios=configured_scenarios,
            safety=safety,
            vllm_endpoint=vllm_endpoint,
            vllm_model=vllm_model,
            max_retries=max_retries,
        )
        for index in range(record_count)
    ]


def configured_scenario_types(config_path: str | Path) -> list[str]:
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    scenarios = data.get("scenario_types")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"Missing scenario_types in {config_path}")
    return [str(item) for item in scenarios]


def build_record(
    *,
    index: int,
    mode: str,
    dataset_name: str,
    victims: list[dict[str, Any]],
    attackers: list[dict[str, Any]],
    configured_scenarios: list[str],
    safety: SafetyMasker,
    vllm_endpoint: str | None,
    vllm_model: str,
    max_retries: int,
) -> dict[str, Any]:
    pair_index = index % min(len(victims), len(attackers))
    victim = victims[pair_index]
    attacker = attackers[pair_index]
    profile = profile_for_index(index, mode)
    variant_index = index // min(len(victims), len(attackers))
    scenario_type = choose_scenario_type(victim, attacker, configured_scenarios, index)
    protected_assets = list(victim.get("protected_assets") or default_assets_for_scenario(scenario_type))
    attacker_goal = choose_attacker_goal(attacker, protected_assets)
    environmental_context = environmental_context_for_scenario(scenario_type, variant_index=variant_index)
    trajectory_variant = trajectory_variant_for_index(index, mode)
    plan = plan_for_profile(
        profile,
        scenario_type,
        protected_assets,
        attacker_goal,
        environmental_context,
        trajectory_variant=trajectory_variant,
    )
    dialogue, fallback_turn_ids = render_dialogue(
        plan=plan,
        victim=victim,
        attacker=attacker,
        scenario_type=scenario_type,
        context=context_for_personas(victim, attacker, scenario_type),
        protected_assets=protected_assets,
        environmental_context=environmental_context,
        profile=profile,
        safety=safety,
        vllm_endpoint=vllm_endpoint,
        vllm_model=vllm_model,
        max_retries=max_retries,
        trajectory_variant=trajectory_variant,
        record_seed=index,
    )
    semantic_repair_turn_ids = repair_dialogue_quality(
        dialogue=dialogue,
        plan=plan,
        victim=victim,
        scenario_type=scenario_type,
        protected_assets=protected_assets,
        environmental_context=environmental_context,
        profile=profile,
        safety=safety,
        trajectory_variant=trajectory_variant,
        record_seed=index,
    )
    outcome, compromised_assets = expected_outcome(profile, dialogue)
    risk_labels = collect_risk_labels(dialogue, SCENARIO_RISK_LABELS.get(scenario_type, ["voice_impersonation"]))
    quality_flags = stream_quality_flags(profile, dialogue)
    record = {
        "id": make_record_id(dataset_name, victim["id"], attacker["id"], scenario_type, index),
        "dataset_name": dataset_name,
        "label": "borderline" if profile == "benign_thin_context" else "attack",
        "scenario_type": scenario_type,
        "victim_persona": compact_persona(victim),
        "attacker_persona": compact_persona(attacker),
        "protected_assets": protected_assets,
        "attacker_goal": attacker_goal,
        "outcome": outcome,
        "compromised_assets": compromised_assets,
        "difficulty": difficulty_for_index(index),
        "channel": environmental_context.get("channel", "phone_call"),
        "locale": "ko-KR",
        "split": split_for_index(index),
        "context": context_for_personas(victim, attacker, scenario_type),
        "dialogue": dialogue,
        "risk_labels": risk_labels,
        "safety_masks": [mask for turn in dialogue for mask in turn.get("masked_items", [])],
        "gen_metadata": {
            "generator": "benchmark_final.generate_persona_stream_corpus",
            "prompt_version": "stream_v2_persona_grounded",
            "stream_profile": profile,
            "stream_mode": mode,
            "trajectory_variant": trajectory_variant if trajectory_variant is not None else "canonical",
            "canonical_pair_index": pair_index,
            "variant_index": variant_index,
            "environmental_context": environmental_context,
            "vllm_refinement": {"model": vllm_model, "enabled": bool(vllm_endpoint)},
            "fallback_turn_ids": fallback_turn_ids,
            "semantic_repair_turn_ids": semantic_repair_turn_ids,
            "seed": index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "quality_checks": {
            "turn_count": len(dialogue),
            "role_order_valid": role_order_valid(dialogue),
            "safety_passed": not any(safety.contains_forbidden_operational_detail(str(t.get("utterance", ""))) for t in dialogue),
            "no_duplicate_turns": no_duplicate_turns(dialogue),
            "label_consistent": True,
            "language_consistent": language_consistent_for_record(dialogue, "ko-KR"),
            "no_refusal_leak": no_refusal_leak(dialogue),
            "residual_pii_clean": safety.is_residual_pii_clean("\n".join(str(t.get("utterance", "")) for t in dialogue)),
            "schema_valid": True,
            **quality_flags,
        },
    }
    record["quality_checks"]["label_consistent"] = label_consistent_for_record(record["label"], dialogue, risk_labels)
    validate_record(record, max_turns=10)
    return record


def profile_for_index(index: int, mode: str) -> str:
    # Both supported modes cycle profiles globally. 300 records gives exact
    # event balance; 288 records gives exact persona-pair balance.
    return STREAM_PROFILES[index % len(STREAM_PROFILES)]


def trajectory_variant_for_index(index: int, mode: str) -> int | None:
    if mode != TRAJECTORY_VARIED_MODE:
        return None
    return (index // len(STREAM_PROFILES)) % TRAJECTORY_VARIANT_COUNT


def render_dialogue(
    *,
    plan: list[PlannedTurn],
    victim: dict[str, Any],
    attacker: dict[str, Any],
    scenario_type: str,
    context: str,
    protected_assets: list[str],
    environmental_context: dict[str, str],
    profile: str,
    safety: SafetyMasker,
    vllm_endpoint: str | None,
    vllm_model: str,
    max_retries: int,
    trajectory_variant: int | None,
    record_seed: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    dialogue: list[dict[str, Any]] = []
    fallback_turn_ids: list[int] = []
    for turn_id, planned in enumerate(plan, start=1):
        utterance = semantic_fallback_utterance(
            planned=planned,
            turn_id=turn_id,
            history=dialogue,
            victim=victim,
            scenario_type=scenario_type,
            protected_assets=protected_assets,
            environmental_context=environmental_context,
            profile=profile,
            trajectory_variant=trajectory_variant,
            record_seed=record_seed,
        )
        used_fallback = True
        if vllm_endpoint:
            candidate = generate_turn_with_vllm(
                endpoint=vllm_endpoint,
                model=vllm_model,
                planned=planned,
                turn_id=turn_id,
                history=dialogue,
                victim=victim,
                attacker=attacker,
                scenario_type=scenario_type,
                context=context,
                protected_assets=protected_assets,
                environmental_context=environmental_context,
                profile=profile,
                max_retries=max_retries,
                record_seed=record_seed,
            )
            if candidate is not None:
                utterance = candidate
                used_fallback = False
        if used_fallback:
            fallback_turn_ids.append(turn_id)
        utterance = dedupe_repeated_nouns(utterance)
        safe_text = safety.mask_text(utterance)
        turn = {
            "turn_id": turn_id,
            "speaker": planned.speaker,
            "phase": planned.phase,
            "intent_label": planned.intent_label,
            "risk_signal": planned.risk_signal,
            "utterance": safe_text.text,
            "masked_items": safety.masks_to_dicts(safe_text.masked_items),
            "stream_event": planned.event,
            "stream_action": planned.action or EVENT_TO_ACTION[planned.event],
            "stream_policy_rule_ids": list(planned.policy_rule_ids or tuple(EVENT_TO_POLICY_RULE_IDS[planned.event])),
            "stream_evidence_turn_ids": list(planned.evidence_turn_ids),
        }
        dialogue.append(turn)
    return dialogue, fallback_turn_ids


def generate_turn_with_vllm(
    *,
    endpoint: str,
    model: str,
    planned: PlannedTurn,
    turn_id: int,
    history: list[dict[str, Any]],
    victim: dict[str, Any],
    attacker: dict[str, Any],
    scenario_type: str,
    context: str,
    protected_assets: list[str],
    environmental_context: dict[str, str],
    profile: str,
    max_retries: int,
    record_seed: int,
) -> str | None:
    required = sorted(planned.placeholders)
    last_failure = ""
    for attempt in range(max_retries + 1):
        prompt = turn_prompt(
            planned=planned,
            turn_id=turn_id,
            history=history,
            victim=victim,
            attacker=attacker,
            scenario_type=scenario_type,
            context=context,
            protected_assets=protected_assets,
            environmental_context=environmental_context,
            profile=profile,
            attempt=attempt,
            last_failure=last_failure,
        )
        raw = call_chat_completion(endpoint, model, prompt)
        if raw is None:
            last_failure = "model_call_failed"
            continue
        candidate = clean_utterance(raw)
        if not valid_generated_utterance(candidate, required):
            last_failure = "placeholder_or_format_validation_failed"
            continue
        if is_duplicate_text(candidate, history):
            last_failure = "duplicate_or_repetitive_with_dialogue_history"
            continue
        semantic_error = turn_semantic_error(candidate, planned, profile)
        if semantic_error:
            last_failure = semantic_error
            continue
        candidate = diversify_generated_utterance(
            candidate,
            planned=planned,
            turn_id=turn_id,
            scenario_type=scenario_type,
            environmental_context=environmental_context,
            victim=victim,
            profile=profile,
            record_seed=record_seed,
        )
        semantic_error = turn_semantic_error(candidate, planned, profile)
        if semantic_error:
            last_failure = semantic_error
            continue
        return candidate
    return None


def turn_prompt(
    *,
    planned: PlannedTurn,
    turn_id: int,
    history: list[dict[str, Any]],
    victim: dict[str, Any],
    attacker: dict[str, Any],
    scenario_type: str,
    context: str,
    protected_assets: list[str],
    environmental_context: dict[str, str],
    profile: str,
    attempt: int,
    last_failure: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You write one Korean utterance for a synthetic defensive deepvoice-phishing benchmark. "
                "Keep content masked, non-operational, and safe. Do not include role labels, markdown, "
                "stage directions, real URLs, real phone numbers, real credentials, or procedural fraud guidance."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "speaker": planned.speaker,
                    "turn_id": turn_id,
                    "phase": planned.phase,
                    "scenario_type": scenario_type,
                    "context": context,
                    "stream_profile": profile,
                    "target_event": planned.event,
                    "target_action": planned.action or EVENT_TO_ACTION[planned.event],
                    "policy_rule_ids": list(planned.policy_rule_ids or tuple(EVENT_TO_POLICY_RULE_IDS[planned.event])),
                    "turn_instruction": planned.instruction,
                    "action_visible_requirement": action_visible_requirement(planned),
                    "victim_persona": {
                        "id": victim.get("id"),
                        "description": victim.get("description"),
                        "resistance_style": victim.get("resistance_style"),
                        "vulnerabilities": victim.get("vulnerabilities"),
                    },
                    "attacker_persona": {
                        "id": attacker.get("id"),
                        "description": attacker.get("description"),
                        "pressure_methods": attacker.get("pressure_methods"),
                    },
                    "protected_assets": protected_assets,
                    "environmental_context": environmental_context,
                    "dialogue_history": [
                        {"speaker": row.get("speaker"), "utterance": row.get("utterance")}
                        for row in history
                    ],
                    "required_placeholders_exactly": list(planned.placeholders),
                    "semantic_requirements": semantic_requirements(planned, profile),
                    "previous_failure_to_fix": last_failure,
                    "retry_hint": retry_hint_for_failure(last_failure, planned),
                    "attempt": attempt,
                    "constraints": [
                        "Write only the next utterance in Korean.",
                        "Write one complete sentence with substantive Korean text outside any placeholder.",
                        "Use exactly the listed [MASKED_*] placeholders and no other bracketed text.",
                        "Do not output only a placeholder.",
                        "For [MASKED_URL], include link/path wording outside the placeholder.",
                        "If no placeholder is listed, do not use any [MASKED_*] placeholder.",
                        "Preserve the target event and stream profile.",
                        "Do not copy the turn_instruction text literally.",
                        "Do not use meta-instruction imperatives such as 보류하세요, 답하세요, 설명하세요, 요구하세요, 마무리하세요, or 망설이세요.",
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def action_visible_requirement(planned: PlannedTurn) -> str:
    action = planned.action or EVENT_TO_ACTION[planned.event]
    if action == "no_alert":
        return "Make it visible that there is no new request or no immediate defensive action yet."
    if action == "monitor_only":
        return "Make it visible that the listener should keep observing because context remains incomplete."
    if action == "pause_or_verify":
        return "Make it visible that the listener should pause and verify through a trusted channel."
    if action == "refuse_or_pause":
        return "Make it visible that the listener should refuse or stop before sharing anything."
    if action == "preserve_and_report":
        return "Make it visible that prior exposure requires preserving the record and reporting or recovery."
    return "Make the recommended defensive stance visible in natural dialogue."


def retry_hint_for_failure(last_failure: str, planned: PlannedTurn) -> str:
    if not last_failure:
        return ""
    if last_failure == "placeholder_only_utterance":
        return "Add Korean context around the placeholder; never return only [MASKED_*]."
    if last_failure == "very_short_utterance":
        return "Use a fuller sentence that explains the speaker's hesitation, request, or verification stance."
    if last_failure == "instruction_residue_in_utterance":
        return "Rewrite as natural dialogue, not a command copied from the planning instruction."
    if last_failure == "new_caller_directed_path_missing_path_language":
        return "Mention a link, path, portal, address, opening the link, or entering the path outside [MASKED_URL]."
    if last_failure == "recipient_disclosure_missing_first_person_disclosure_verb":
        return "Use first-person disclosure wording such as 전달하겠습니다, 말씀드리겠습니다, 입력하겠습니다, or 공유하겠습니다."
    if last_failure == "recipient_containment_missing_safe_path_or_refusal":
        return "Include explicit refusal, pausing, or official/saved/direct verification language."
    if last_failure.endswith("_missing_pressure_signal"):
        return "Add urgency or consequence language such as 오늘, 마감, 지연, 보류, 제한, or 차질."
    if planned.placeholders:
        return "Keep the exact required placeholder set and repair only the failed semantic requirement."
    return "Repair the failed semantic requirement without adding any masked placeholder."


def diversify_generated_utterance(
    text: str,
    *,
    planned: PlannedTurn,
    turn_id: int,
    scenario_type: str,
    environmental_context: dict[str, str],
    victim: dict[str, Any],
    profile: str,
    record_seed: int,
) -> str:
    if planned.event != "needs_more_context" or planned.placeholders:
        return text
    if not needs_generated_context_suffix(text, turn_id):
        return text
    subject = subject_for_scenario(scenario_type, environmental_context)
    channel = context_label(environmental_context.get("channel", ""))
    timing = context_label(environmental_context.get("timing", ""))
    verification_path = readable_path(environmental_context.get("verification_path", "공식 경로"))
    persona_hint = str(victim.get("description", "")).split(" ")[0] or "상황"
    suffixes = [
        f"{channel} 안내라 먼저 배경만 확인하는 단계입니다.",
        f"{timing} 상황이라 세부 조치는 아직 안내하지 않겠습니다.",
        f"{verification_path}에서도 확인할 수 있는 배경 설명부터 드리겠습니다.",
        f"{persona_hint} 맥락을 고려해 지금은 상황 설명만 하겠습니다.",
        f"{subject} 접수 흐름을 먼저 정리하려는 연락입니다.",
        "민감한 입력을 요구하기 전의 일반 안내입니다.",
        "바로 진행할 단계가 아니라 사실관계만 설명하겠습니다.",
        "결정은 안내를 들은 뒤에 하셔도 됩니다.",
        f"{channel}에서 확인된 알림 배경만 짧게 말씀드리겠습니다.",
        f"{timing} 기준으로 남은 안내가 있는지만 확인하겠습니다.",
        f"{verification_path} 확인 전에 통화에서는 개요만 말씀드립니다.",
        f"{subject} 관련 세부 요청은 아직 하지 않겠습니다.",
    ]
    suffix = suffixes[deterministic_index(profile, scenario_type, turn_id, victim.get("id"), record_seed, text, len(suffixes))]
    return join_fallback_context(text, suffix)


def needs_generated_context_suffix(text: str, turn_id: int) -> bool:
    if turn_id == 1:
        return True
    normalized = normalized_instruction_text(text)
    generic_markers = (
        "관련안내를드리기위해연락드렸습니다",
        "안내를드리기위해연락드렸습니다",
        "상황을설명드리려고연락했습니다",
    )
    return substantive_korean_length(text) < 22 or any(marker in normalized for marker in generic_markers)


def call_chat_completion(endpoint: str, model: str, messages: list[dict[str, str]]) -> str | None:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.65,
        "max_tokens": 220,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return None


def clean_utterance(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith(("\"", "'")) and cleaned.endswith(("\"", "'")) and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"^(attacker|victim|caller|recipient|공격자|피해자)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    return dedupe_repeated_nouns(" ".join(cleaned.split()))


def dedupe_repeated_nouns(text: str) -> str:
    for noun in ("알림", "안내", "확인", "처리", "경로", "절차", "요청"):
        text = re.sub(rf"({noun})(\s+\1)+", noun, text)
    return text


def valid_generated_utterance(text: str, required_placeholders: list[str]) -> bool:
    if not text or "\n" in text:
        return False
    if re.search(r"\[(?!MASKED_[A-Z0-9_]+\])[^\]]+\]", text):
        return False
    if sorted(MASKED_PLACEHOLDER_RE.findall(text)) != required_placeholders:
        return False
    if re.search(r"https?://|www\.|\b\d{2,3}-\d{3,4}-\d{4}\b", text):
        return False
    if has_any(text, INSTRUCTION_RESIDUE_MARKERS):
        return False
    if utterance_artifact_error(text) is not None:
        return False
    return True


def strip_masked_placeholders(text: str) -> str:
    return MASKED_PLACEHOLDER_RE.sub(" ", text)


def substantive_korean_length(text: str) -> int:
    stripped = strip_masked_placeholders(text)
    return len(re.findall(r"[가-힣]", stripped))


def normalized_instruction_text(text: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", strip_masked_placeholders(text))


def is_placeholder_only(text: str) -> bool:
    without_placeholders = strip_masked_placeholders(text)
    return not re.search(r"[가-힣A-Za-z0-9]", without_placeholders)


def utterance_artifact_error(text: str, planned: PlannedTurn | None = None) -> str | None:
    if is_placeholder_only(text):
        return "placeholder_only_utterance"
    normalized = normalized_instruction_text(text)
    if has_any(text, INSTRUCTION_RESIDUE_MARKERS):
        return "instruction_residue_in_utterance"
    if any(marker in text for marker in VERY_SHORT_UTTERANCES) or substantive_korean_length(text) < 9:
        return "very_short_utterance"
    if planned is not None and planned.instruction:
        planned_normalized = normalized_instruction_text(planned.instruction)
        if planned_normalized and normalized == planned_normalized:
            return "instruction_residue_in_utterance"
    return None


def semantic_requirements(planned: PlannedTurn, profile: str) -> list[str]:
    event = planned.event
    requirements = [f"speaker must remain {planned.speaker}", f"target event is {event}"]
    if profile == "benign_thin_context":
        requirements.append("Do not include sensitive placeholders or hard refusal/reporting language.")
    if event == "recipient_partial_or_direct_disclosure":
        requirements.extend(
            [
                "The victim must speak in first person and directly provide the masked value.",
                "Use wording like 전달하겠습니다, 제공하겠습니다, 말씀드리겠습니다, 입력하겠습니다, or 공유하겠습니다.",
                "Do not write caller/request phrasing such as 확인해 주시면, 부탁드립니다, 필요합니다.",
                "Do not include official-channel verification, refusal, reporting, or recovery language.",
            ]
        )
    elif event == "recipient_containment":
        requirements.append("The victim must refuse/pause or redirect to an official, saved, or directly verified path.")
    elif event == "new_sensitive_request":
        requirements.append("The caller must request the exact masked sensitive value for the first time.")
    elif event == "new_caller_directed_path":
        requirements.append("The caller must introduce the exact masked URL/link/path.")
    elif event in {"pressure_escalation", "pressure_only_monitor"}:
        requirements.append("Use pressure/urgency/consequence wording without adding a new sensitive value.")
    elif event == "needs_more_context":
        requirements.append("Keep the utterance contextual only; do not request or disclose sensitive values.")
    elif event == "recovery_followup_due":
        if planned.speaker == "victim":
            requirements.append("The victim must stop, preserve/report, or move to official recovery after prior disclosure.")
        else:
            requirements.append("The caller may continue after prior disclosure, preserving only the exact requested placeholder.")
    return requirements


def turn_semantic_error(text: str, planned: PlannedTurn, profile: str) -> str | None:
    placeholders = MASKED_PLACEHOLDER_RE.findall(text)
    if sorted(placeholders) != sorted(planned.placeholders):
        return "semantic_placeholder_mismatch"
    artifact_error = utterance_artifact_error(text, planned)
    if artifact_error:
        return artifact_error
    if profile == "benign_thin_context" and placeholders:
        return "benign_thin_context_must_not_use_sensitive_placeholder"
    if profile == "benign_thin_context" and any(marker in text for marker in BENIGN_STRONG_RESPONSE_MARKERS):
        return "benign_thin_context_uses_overstrong_response_language"

    event = planned.event
    if event == "needs_more_context":
        if placeholders:
            return "needs_more_context_must_not_use_sensitive_placeholder"
        return None
    if event == "pressure_only_monitor":
        if placeholders:
            return "pressure_only_monitor_must_not_use_sensitive_placeholder"
        if not has_any(text, PRESSURE_MARKERS):
            return "pressure_only_monitor_needs_pressure_signal"
        return None
    if event == "new_sensitive_request":
        if planned.speaker != "attacker":
            return "new_sensitive_request_must_be_attacker"
        if not placeholders:
            return "new_sensitive_request_missing_placeholder"
        if not has_any(text, ("확인", "말씀", "제공", "입력", "보내", "필요", "제출")):
            return "new_sensitive_request_missing_request_language"
        return None
    if event == "new_caller_directed_path":
        if planned.speaker != "attacker":
            return "new_caller_directed_path_must_be_attacker"
        if "[MASKED_URL]" not in placeholders:
            return "new_caller_directed_path_missing_masked_url"
        if not has_any(strip_masked_placeholders(text), PATH_LANGUAGE_MARKERS):
            return "new_caller_directed_path_missing_path_language"
        return None
    if event == "pressure_escalation":
        if planned.speaker != "attacker":
            return "pressure_escalation_must_be_attacker"
        if placeholders:
            return "pressure_escalation_must_not_add_placeholder"
        if not has_any(text, PRESSURE_MARKERS):
            return "pressure_escalation_missing_pressure_signal"
        return None
    if event == "recipient_containment":
        if planned.speaker != "victim":
            return "recipient_containment_must_be_victim"
        if has_direct_disclosure(text) and not has_any(text, ("제공하지", "입력하지", "사용하지", "접속하지")):
            return "recipient_containment_contains_direct_disclosure"
        if not has_any(text, CONTAINMENT_MARKERS):
            return "recipient_containment_missing_safe_path_or_refusal"
        return None
    if event == "recipient_partial_or_direct_disclosure":
        if planned.speaker != "victim":
            return "recipient_disclosure_must_be_victim"
        if not placeholders:
            return "recipient_disclosure_missing_placeholder"
        if not has_direct_disclosure(text):
            return "recipient_disclosure_missing_first_person_disclosure_verb"
        if has_any(text, CALLER_REQUEST_VOICE_MARKERS):
            return "recipient_disclosure_uses_caller_request_voice"
        if has_any(text, ("공식", "신고", "중단", "보존", "제공하지", "입력하지")):
            return "recipient_disclosure_contains_recovery_or_refusal_language"
        return None
    if event == "recovery_followup_due":
        if planned.speaker == "victim":
            if placeholders:
                return "victim_recovery_followup_must_not_add_placeholder"
            if not has_any(text, ("중단", "보존", "신고", "공식", "확인", "기록")):
                return "victim_recovery_followup_missing_recovery_language"
        elif planned.speaker == "attacker" and planned.placeholders and not has_any(text, ("추가", "이어", "확인", "같은 방식")):
            return "caller_recovery_followup_missing_continuation_language"
        return None
    if planned.speaker == "victim" and has_direct_disclosure(text) and event not in {"recipient_partial_or_direct_disclosure"}:
        return "victim_direct_disclosure_on_non_disclosure_event"
    return None


def has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def has_direct_disclosure(text: str) -> bool:
    return "[MASKED_" in text and has_any(text, DIRECT_DISCLOSURE_VERBS)


def is_duplicate_text(text: str, history: list[dict[str, Any]]) -> bool:
    current = normalized_for_duplicate_check(text)
    return any(current == normalized_for_duplicate_check(str(turn.get("utterance", ""))) for turn in history)


def repair_dialogue_quality(
    *,
    dialogue: list[dict[str, Any]],
    plan: list[PlannedTurn],
    victim: dict[str, Any],
    scenario_type: str,
    protected_assets: list[str],
    environmental_context: dict[str, str],
    profile: str,
    safety: SafetyMasker,
    trajectory_variant: int | None,
    record_seed: int,
) -> list[int]:
    repaired: list[dict[str, Any]] = []
    repaired_turn_ids: list[int] = []
    for index, turn in enumerate(dialogue):
        planned_turn = plan[index]
        text = str(turn.get("utterance", ""))
        semantic_error = turn_semantic_error(text, planned_turn, profile)
        duplicate = is_duplicate_text(text, repaired)
        if semantic_error or duplicate:
            replacement = semantic_fallback_utterance(
                planned=planned_turn,
                turn_id=index + 1,
                history=repaired,
                victim=victim,
                scenario_type=scenario_type,
                protected_assets=protected_assets,
                environmental_context=environmental_context,
                profile=profile,
                trajectory_variant=trajectory_variant,
                record_seed=record_seed,
            )
            replacement = dedupe_repeated_nouns(replacement)
            safe_text = safety.mask_text(replacement)
            turn["utterance"] = safe_text.text
            turn["masked_items"] = safety.masks_to_dicts(safe_text.masked_items)
            repaired_turn_ids.append(index + 1)
        repaired.append(turn)
    return repaired_turn_ids


def stream_quality_flags(profile: str, dialogue: list[dict[str, Any]]) -> dict[str, bool]:
    planned_turns = [planned_from_turn(turn) for turn in dialogue]
    semantic_errors = [
        turn_semantic_error(str(turn.get("utterance", "")), planned_turn, profile)
        for turn, planned_turn in zip(dialogue, planned_turns)
    ]
    partial_turns = [
        turn
        for turn in dialogue
        if turn.get("stream_event") == "recipient_partial_or_direct_disclosure"
    ]
    partial_valid = True
    if profile == "partial_disclosure_recovery":
        partial_valid = bool(partial_turns) and all(
            turn_semantic_error(str(turn.get("utterance", "")), planned_from_turn(turn), profile) is None
            for turn in partial_turns
        )
    benign_valid = True
    if profile == "benign_thin_context":
        text = "\n".join(str(turn.get("utterance", "")) for turn in dialogue)
        benign_valid = "[MASKED_" not in text and not has_any(text, BENIGN_STRONG_RESPONSE_MARKERS)
    speaker_perspective_valid = not any(
        turn.get("speaker") == "victim"
        and turn.get("stream_event") == "recipient_partial_or_direct_disclosure"
        and has_any(str(turn.get("utterance", "")), CALLER_REQUEST_VOICE_MARKERS)
        for turn in dialogue
    )
    duplicate_clean = no_duplicate_turns(dialogue)
    return {
        "stream_semantic_valid": not any(semantic_errors),
        "partial_disclosure_semantic_valid": partial_valid,
        "speaker_perspective_valid": speaker_perspective_valid,
        "duplicate_turns_clean": duplicate_clean,
        "benign_thin_context_valid": benign_valid,
    }


def planned_from_turn(turn: dict[str, Any] | None) -> PlannedTurn:
    if not turn:
        return PlannedTurn("", "", "", "", "no_new_actionable_event", (), "", "", ())
    return PlannedTurn(
        speaker=str(turn.get("speaker", "")),
        phase=str(turn.get("phase", "")),
        intent_label=str(turn.get("intent_label", "")),
        risk_signal=str(turn.get("risk_signal", "")),
        event=str(turn.get("stream_event", "no_new_actionable_event")),
        placeholders=tuple(MASKED_PLACEHOLDER_RE.findall(str(turn.get("utterance", "")))),
        instruction="",
        fallback="",
        evidence_turn_ids=tuple(int(eid) for eid in turn.get("stream_evidence_turn_ids", []) if isinstance(eid, int)),
        action=str(turn.get("stream_action", "")) or None,
        policy_rule_ids=tuple(str(rule_id) for rule_id in turn.get("stream_policy_rule_ids", [])),
    )


def semantic_fallback_utterance(
    *,
    planned: PlannedTurn,
    turn_id: int,
    history: list[dict[str, Any]],
    victim: dict[str, Any],
    scenario_type: str,
    protected_assets: list[str],
    environmental_context: dict[str, str],
    profile: str,
    trajectory_variant: int | None,
    record_seed: int,
) -> str:
    subject = subject_for_scenario(scenario_type, environmental_context)
    verification_path = readable_path(environmental_context.get("verification_path", "공식 경로"))
    candidates = fallback_candidates(
        planned=planned,
        subject=subject,
        verification_path=verification_path,
        victim=victim,
        protected_assets=protected_assets,
        profile=profile,
    )
    candidates = contextual_fallback_candidates(
        candidates,
        planned=planned,
        subject=subject,
        verification_path=verification_path,
        victim=victim,
        environmental_context=environmental_context,
        turn_id=turn_id,
        trajectory_variant=trajectory_variant,
        record_seed=record_seed,
    )
    start = deterministic_index(
        profile,
        scenario_type,
        victim.get("id"),
        turn_id,
        len(history),
        planned.event,
        planned.intent_label,
        planned.risk_signal,
        environmental_context.get("trigger_event"),
        environmental_context.get("timing"),
        "|".join(protected_assets),
        trajectory_variant if trajectory_variant is not None else "canonical",
        record_seed,
        len(candidates),
    )
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        if not is_duplicate_text(candidate, history) and turn_semantic_error(candidate, planned, profile) is None:
            return candidate
    return planned.fallback


def contextual_fallback_candidates(
    candidates: list[str],
    *,
    planned: PlannedTurn,
    subject: str,
    verification_path: str,
    victim: dict[str, Any],
    environmental_context: dict[str, str],
    turn_id: int,
    trajectory_variant: int | None,
    record_seed: int,
) -> list[str]:
    persona_hint = str(victim.get("description", "")).split(" ")[0] or "상황"
    channel = context_label(environmental_context.get("channel", ""))
    timing = context_label(environmental_context.get("timing", ""))
    constraint = constraint_label(environmental_context.get("victim_constraint", ""))
    variant_text = f"{subject} {channel} {timing} {constraint} {verification_path} {persona_hint} {turn_id} {trajectory_variant} {record_seed}"
    event_suffixes = suffixes_for_event(
        planned=planned,
        subject=subject,
        verification_path=verification_path,
        persona_hint=persona_hint,
        channel=channel,
        timing=timing,
        constraint=constraint,
    )
    start = deterministic_index(variant_text, len(event_suffixes))
    ordered_suffixes = event_suffixes[start:] + event_suffixes[:start]
    expanded: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for suffix in ordered_suffixes:
            note = fallback_variation_note(
                planned=planned,
                subject=subject,
                channel=channel,
                timing=timing,
                constraint=constraint,
                candidate=candidate,
                suffix=suffix,
                turn_id=turn_id,
                record_seed=record_seed,
            )
            text = join_fallback_context(join_fallback_context(candidate, suffix), note)
            if text not in seen:
                expanded.append(text)
                seen.add(text)
    return expanded or candidates


def fallback_variation_note(
    *,
    planned: PlannedTurn,
    subject: str,
    channel: str,
    timing: str,
    constraint: str,
    candidate: str,
    suffix: str,
    turn_id: int,
    record_seed: int,
) -> str:
    if planned.event == "recipient_partial_or_direct_disclosure":
        starts = [
            "급한 안내로 이해하고 그대로 진행하겠습니다.",
            "확인이 끝난다고 하셔서 더 미루지 않겠습니다.",
            "말씀하신 순서에 맞춰 이어가겠습니다.",
            f"{timing} 상황이라 바로 처리하겠습니다.",
            "지금 단계에서는 안내를 믿고 따르겠습니다.",
            f"{subject} 처리가 끝난다는 설명으로 이해하겠습니다.",
        ]
        details = [
            "처리 지연을 피하려고 안내에 맞추겠습니다.",
            "확인 단계가 남았다는 설명을 따르겠습니다.",
            "말씀하신 절차가 끝난다고 보고 진행하겠습니다.",
            "통화 흐름이 끊기지 않게 이어가겠습니다.",
            "지금 안내된 순서대로 하겠습니다.",
            "더 묻지 않고 확인을 마치겠습니다.",
            f"{channel} 안내라 짧게 답하겠습니다.",
            f"{constraint} 상황이라 빠르게 처리하겠습니다.",
        ]
        tails = [
            "안내받은 순서대로 마무리하겠습니다.",
            "지금은 그대로 이어가겠습니다.",
            "확인이 끝난다는 설명을 따르겠습니다.",
            "절차가 남았다는 말에 맞추겠습니다.",
            "더 지연되지 않게 하겠습니다.",
            "말씀하신 확인을 우선하겠습니다.",
        ]
    elif planned.speaker == "attacker":
        starts = [
            f"{subject} 접수 기준으로 말씀드립니다.",
            f"{channel} 안내 흐름에서 확인되는 내용입니다.",
            f"{timing} 처리 순서에 맞춰 안내드립니다.",
            "현재 기록상 남은 안내를 짧게 말씀드립니다.",
            "접수 흐름이 끊기지 않도록 안내를 이어가겠습니다.",
            "후속 처리가 밀리지 않도록 지금 설명드리는 단계입니다.",
            f"{constraint} 상황을 고려해 빠르게 안내드리겠습니다.",
            "담당 안내 기준으로는 지금 확인이 필요합니다.",
        ]
        details = [
            "현재 단계의 확인만 마치면 다음 안내로 넘어갈 수 있습니다.",
            "접수 순서가 밀리지 않도록 안내를 계속하겠습니다.",
            "처리 지연을 줄이기 위한 확인이라고 보시면 됩니다.",
            "지금 안내를 놓치면 후속 확인이 늦어질 수 있습니다.",
            f"{subject} 관련 남은 절차를 정리하는 중입니다.",
            f"{channel}에서 확인 가능한 흐름만 말씀드립니다.",
            f"{timing} 기준으로 안내가 지연되지 않게 하겠습니다.",
            f"{constraint} 점 때문에 짧게 확인하고 넘어가겠습니다.",
            "담당 기록과 현재 안내를 맞추려는 단계입니다.",
            "이 안내가 끝나야 후속 상태를 정리할 수 있습니다.",
        ]
        tails = [
            "남은 안내를 계속하겠습니다.",
            "현재 단계만 짧게 확인하겠습니다.",
            "후속 처리를 위해 이어가겠습니다.",
            "지금 확인이 늦어지지 않게 하겠습니다.",
            "접수 흐름을 유지하겠습니다.",
            "안내가 끊기지 않도록 하겠습니다.",
            "상태 정리를 마치려는 단계입니다.",
            "담당 기록 기준으로 이어가겠습니다.",
        ]
    else:
        starts = [
            f"{subject} 연락이라 바로 결론 내리지 않겠습니다.",
            f"{channel}을 통해 받은 안내라 내용을 더 따져보겠습니다.",
            f"{timing} 상황이라 잠시 정리하고 판단하겠습니다.",
            f"{constraint} 상태라 급하게 움직이지 않겠습니다.",
            "갑작스러운 연락이라 한 번 더 생각하겠습니다.",
            "지금 들은 내용만으로는 바로 결정하지 않겠습니다.",
            "제 쪽에서 확인할 여지를 남겨두겠습니다.",
            "설명은 들었지만 다음 행동은 신중하게 정하겠습니다.",
        ]
        details = [
            "안내 흐름을 메모해 두고 다시 보겠습니다.",
            "지금은 통화 내용만 정리하겠습니다.",
            "새 행동을 정하기 전 상황을 더 보겠습니다.",
            "조급하게 움직이지 않고 판단을 미루겠습니다.",
            "들어본 내용을 제 기준으로 다시 확인하겠습니다.",
            f"{channel} 안내라 한 번 더 살피겠습니다.",
            f"{timing} 상황이라 즉시 결정하지 않겠습니다.",
            f"{constraint} 점도 있어 차분히 판단하겠습니다.",
            f"{subject} 관련 설명을 더 확인해 보겠습니다.",
            "다음 행동은 확실해진 뒤에 정하겠습니다.",
        ]
        tails = [
            "잠시 더 확인하겠습니다.",
            "바로 움직이지 않겠습니다.",
            "메모만 남겨두겠습니다.",
            "결정은 뒤로 미루겠습니다.",
            "제 쪽 확인을 먼저 하겠습니다.",
            "안내 흐름만 더 보겠습니다.",
            "추가 행동은 보류하겠습니다.",
            "상황을 다시 정리하겠습니다.",
        ]
    start_index = deterministic_index("start", candidate, suffix, turn_id, record_seed, len(starts))
    detail_index = deterministic_index("detail", candidate, suffix, turn_id, record_seed, len(details))
    tail_index = deterministic_index("tail", candidate, suffix, turn_id, record_seed, len(tails))
    return f"{starts[start_index]} {details[detail_index]} {tails[tail_index]}"


def suffixes_for_event(
    *,
    planned: PlannedTurn,
    subject: str,
    verification_path: str,
    persona_hint: str,
    channel: str,
    timing: str,
    constraint: str,
) -> list[str]:
    channel_note = f"{channel} 상황이라 표현을 짧게 정리하겠습니다."
    timing_note = f"{timing} 기준으로 판단하겠습니다."
    constraint_note = f"{constraint} 점도 고려하겠습니다."
    subject_case = subject if subject.endswith("건") else f"{subject} 건"
    if planned.event == "recipient_partial_or_direct_disclosure":
        return [
            "말씀하신 절차로 이해하고 진행하겠습니다.",
            f"{persona_hint} 상황이라 더 지체하지 않겠습니다.",
            "지금 안내를 믿고 이어가겠습니다.",
            "확인이 끝난다고 하셔서 그대로 하겠습니다.",
        ]
    if planned.event == "recipient_containment":
        return [
            f"{verification_path}에서 직접 확인하겠습니다.",
            "통화 중 결정보다는 제가 아는 경로를 먼저 보겠습니다.",
            f"{constraint} 점이 있어도 새 경로는 쓰지 않겠습니다.",
            "추가 안내는 저장된 공식 채널에서만 확인하겠습니다.",
        ]
    if planned.event == "no_new_actionable_event":
        if planned.speaker == "attacker":
            return [
                f"{subject} 안내 흐름만 유지하겠습니다.",
                "새 입력 단계가 아니라는 점을 다시 말씀드립니다.",
                f"{channel}으로 남긴 안내만 확인하시면 됩니다.",
                f"{timing} 기준으로 추가 제출 없이 현재 설명만 이어가겠습니다.",
            ]
        return [
            f"{constraint} 점이 있어 바로 움직이지 않겠습니다.",
            f"{timing} 상황이라 우선 내용을 더 정리하겠습니다.",
            f"{subject_case}에서 새 행동은 정하지 않고 설명만 더 듣겠습니다.",
            f"{verification_path}에서 나중에 직접 확인할 여지는 남겨두겠습니다.",
        ]
    if planned.event in {"pressure_escalation", "pressure_only_monitor"}:
        return [
            f"{timing} 안에 확인이 늦으면 {subject} 일정이 지연될 수 있습니다.",
            f"{channel} 안내를 놓치면 후속 처리가 보류될 수 있습니다.",
            "지금 판단이 늦어지면 처리 순서에 차질이 생길 수 있습니다.",
            f"{subject} 마감이 남아 있어 지연 가능성을 먼저 알려드립니다.",
        ]
    if planned.event == "new_caller_directed_path":
        return [
            f"{channel} 안내 중이라 해당 링크 경로로 바로 확인하시면 됩니다.",
            f"{timing} 처리 지연을 줄이려는 별도 접속 경로입니다.",
            "문자 대신 이 경로를 열어 상태 확인을 이어가시면 됩니다.",
            f"{subject} 확인용 포털 경로라고 보시면 됩니다.",
        ]
    if planned.event == "new_sensitive_request":
        return [
            f"{subject} 상태를 맞추기 위한 확인이라고 안내드립니다.",
            f"{channel} 중이라 짧게 확인하고 넘어가겠습니다.",
            f"{timing} 처리 기준에 맞추려는 단계입니다.",
            "다음 안내로 넘어가기 전에 필요한 확인입니다.",
        ]
    if planned.event == "recovery_followup_due":
        if planned.speaker == "attacker":
            return [
                f"{subject} 접수 흐름상 이어지는 확인이라고 안내드립니다.",
                "방금 단계와 같은 접수 건으로 계속 확인하겠습니다.",
                f"{channel} 안내가 끊기지 않도록 이어가겠습니다.",
                "남은 확인까지 끝나야 처리를 마무리할 수 있습니다.",
            ]
        return [
            f"{verification_path}에서 복구 여부를 다시 보겠습니다.",
            "대화 기록을 남기고 공식 창구에서 확인하겠습니다.",
            f"{constraint} 점이 있어도 추가 제공은 멈추겠습니다.",
            "방금 내용을 보존하고 정식 지원 경로를 이용하겠습니다.",
        ]
    return [channel_note, timing_note, constraint_note, f"{subject} 내용을 더 확인하겠습니다."]


def join_fallback_context(candidate: str, suffix: str) -> str:
    if not suffix:
        return candidate
    if candidate.endswith((".", "?", "!")):
        return f"{candidate} {suffix}"
    return f"{candidate}. {suffix}"


def fallback_candidates(
    *,
    planned: PlannedTurn,
    subject: str,
    verification_path: str,
    victim: dict[str, Any],
    protected_assets: list[str],
    profile: str,
) -> list[str]:
    placeholder = planned.placeholders[0] if planned.placeholders else ""
    persona_hint = str(victim.get("description", "")).split(" ")[0] or "상황"
    if profile == "benign_thin_context":
        return benign_fallback_candidates(planned.event, planned.speaker, subject)
    if planned.event == "needs_more_context":
        if planned.speaker == "attacker":
            return [
                f"안녕하세요. {subject} 관련 안내가 접수되어 간단히 상황을 설명드리려고 연락했습니다.",
                f"{subject} 알림이 있어 연락드렸습니다. 지금은 배경 설명만 드리겠습니다.",
                f"최근 {subject} 안내가 남아 있어 확인 차 연락했습니다. 민감한 입력을 요구하는 단계는 아닙니다.",
                f"{subject}와 관련해 문의가 들어와 사실관계만 먼저 안내드리겠습니다.",
                "먼저 상황 설명부터 드리겠습니다. 지금 단계에서 입력하실 값은 없습니다.",
                f"{subject} 건으로 안내가 남아 있어 연락드렸고, 세부 조치는 아직 말씀드리지 않겠습니다.",
                "접수된 안내 내용을 설명드리려는 연락입니다. 결정은 설명을 들으신 뒤에 하셔도 됩니다.",
                f"{subject} 관련 기록을 확인하는 중이라 배경만 짧게 말씀드리겠습니다.",
            ]
        return [
            "아직 상황을 정확히 모르겠습니다. 먼저 어떤 안내인지 설명을 듣고 판단하겠습니다.",
            "갑작스러운 연락이라 바로 결정하기 어렵습니다. 우선 배경을 더 듣겠습니다.",
            f"{persona_hint} 관련 연락이라 신중하게 듣고 있습니다. 아직 조치 여부는 판단하지 않겠습니다.",
            "무슨 내용인지 먼저 이해하고 싶습니다. 지금은 설명만 듣겠습니다.",
            "바로 뭔가 하기는 어렵습니다. 어떤 상황인지부터 확인하겠습니다.",
            "일단 내용을 듣고 제 쪽에서 다시 확인해 보겠습니다.",
            "처음 듣는 이야기라서, 안내의 배경을 더 들어봐야겠습니다.",
            "지금은 판단하지 않고 설명만 더 확인하겠습니다.",
        ]
    if planned.event == "pressure_only_monitor":
        return [
            f"{subject} 일정이 지연될 수 있어 안내 흐름만 놓치지 않도록 확인 부탁드립니다.",
            "오늘 안에 확인이 늦어지면 처리가 뒤로 밀릴 수 있어 상황을 계속 안내드리겠습니다.",
            "처리 시간이 제한되어 있어 지연 가능성이 있다는 점만 먼저 알려드립니다.",
            "마감 전에 안내를 확인하지 않으면 이후 일정이 지연될 수 있습니다.",
            "지금 대응이 늦어지면 접수 순서가 뒤로 밀릴 수 있습니다.",
            "오늘 중 확인이 없으면 보류 상태가 길어질 수 있어 먼저 알려드립니다.",
            "시간 제한이 있어 안내를 놓치면 처리 차질이 생길 수 있습니다.",
            f"{subject} 확인이 늦으면 후속 일정이 제한될 수 있습니다.",
            "지연이 생기면 다시 접수해야 할 수 있어 안내를 이어가겠습니다.",
        ]
    if planned.event == "new_sensitive_request":
        return [
            f"{subject} 처리를 위해 {placeholder} 확인이 필요합니다.",
            f"현재 상태 확인을 이어가려면 {placeholder}를 이 통화에서 확인해야 합니다.",
            f"안내 절차상 {placeholder} 확인을 먼저 진행해 주셔야 합니다.",
            f"접수 상태를 맞추려면 {placeholder} 확인이 필요합니다.",
            f"다음 안내로 넘어가기 전에 {placeholder}를 확인하겠습니다.",
            f"기록을 조회하려면 {placeholder}가 있어야 합니다.",
            f"본인 확인 단계라 {placeholder}를 말씀해 주셔야 합니다.",
            f"처리 내역을 맞추려면 {placeholder}를 확인해야 합니다.",
            f"상태 조회를 위해 {placeholder} 입력이 필요합니다.",
            f"확인을 계속하려면 {placeholder}를 먼저 알려주셔야 합니다.",
            f"담당 기록과 대조하려면 {placeholder} 확인이 필요합니다.",
            f"접수된 안내를 열람하려면 {placeholder} 제출이 필요합니다.",
        ]
    if planned.event == "recipient_containment":
        if placeholder:
            return [
                f"통화 중에는 {placeholder}를 제공하지 않겠습니다. {verification_path}로 직접 확인하겠습니다.",
                f"{placeholder}는 이 통화에서 입력하지 않겠습니다. 저장된 공식 경로로 확인하겠습니다.",
                f"지금은 {placeholder}를 공유하지 않고, 제가 직접 공식 채널에서 확인하겠습니다.",
                f"{placeholder}는 전화로 말하지 않겠습니다. 제가 알고 있는 경로에서 확인하겠습니다.",
                f"요청하신 {placeholder}는 제공하지 않고 공식 안내를 직접 확인하겠습니다.",
                f"이 통화에서는 {placeholder} 입력을 하지 않겠습니다. 저장된 채널로 확인하겠습니다.",
                f"{placeholder} 관련 요청은 멈추고 제가 직접 확인하겠습니다.",
                f"먼저 공식 경로를 확인하겠습니다. {placeholder}는 여기서 제공하지 않겠습니다.",
            ]
        return [
            f"그 경로는 사용하지 않고 {verification_path}에서 직접 확인하겠습니다.",
            "압박이 있어도 통화 중에는 진행하지 않겠습니다. 공식 경로에서 확인하겠습니다.",
            "지금 안내만으로는 진행하지 않고 저장된 연락처로 직접 확인하겠습니다.",
            "새로 받은 경로는 사용하지 않고 제가 알고 있는 채널에서 확인하겠습니다.",
            "이 통화에서 바로 진행하지 않겠습니다. 직접 확인한 뒤 판단하겠습니다.",
            "안내는 들었지만 지금은 멈추고 공식 채널을 확인하겠습니다.",
            "통화 중 안내만으로는 움직이지 않겠습니다. 저장된 연락처를 이용하겠습니다.",
            "일단 중단하고 제가 직접 공식 앱에서 확인하겠습니다.",
        ]
    if planned.event == "no_new_actionable_event":
        if planned.speaker == "attacker" and placeholder:
            return [
                f"앞서 말씀드린 같은 {placeholder} 확인만 다시 요청드립니다. 새 자료를 추가로 요구하는 것은 아닙니다.",
                f"현재 필요한 것은 같은 {placeholder} 확인입니다. 다른 값은 요청하지 않겠습니다.",
                f"절차는 동일합니다. 같은 {placeholder} 확인이 끝나야 다음 안내가 가능합니다.",
            ]
        if planned.speaker == "attacker":
            return [
                "현재 단계에서는 추가 자료를 요구하지 않습니다. 안내가 오면 확인해 주세요.",
                "새로 입력하실 내용은 없습니다. 기존 안내만 확인해 주시면 됩니다.",
                "추가 요청 없이 안내 상태만 유지하겠습니다.",
                "지금은 새로 제출하실 내용 없이 안내만 이어가겠습니다.",
                "별도 입력 단계는 아니니 현재 안내를 참고하시면 됩니다.",
                "추가로 필요한 값은 없습니다. 상황만 계속 확인하겠습니다.",
                "새 요청은 없고 접수 흐름만 설명드리는 단계입니다.",
                "현재 안내 외에 더 요구드릴 자료는 없습니다.",
            ]
        return [
            "아직은 별도 조치를 하지 않고 다음 안내를 기다리겠습니다.",
            "지금은 판단을 보류하고 추가 설명을 듣겠습니다.",
            "바로 움직이지 않고 안내 내용을 더 확인해 보겠습니다.",
            "지금 당장 할 일은 없어 보여서 일단 지켜보겠습니다.",
            "조금 더 확인한 뒤에 움직이겠습니다.",
            "바로 결정하지 않고 내용을 정리해 보겠습니다.",
            "일단은 기다리면서 상황을 더 보겠습니다.",
            "아직 확신이 없으니 추가 설명을 듣겠습니다.",
        ]
    if planned.event == "new_caller_directed_path":
        return [
            "공식 앱이 지연될 수 있으니 임시 확인 링크 [MASKED_URL]로 접속해 진행해 주세요.",
            "처리가 늦어지지 않게 별도 확인 경로 [MASKED_URL]에서 바로 이어가시면 됩니다.",
            "현재 통화 중 안내된 링크 [MASKED_URL]를 열어 확인 절차를 완료해 주세요.",
            "대기 시간이 길어져 [MASKED_URL] 경로에서 확인을 이어가시면 됩니다.",
            "앱 접속이 어려우면 [MASKED_URL] 링크로 들어가 확인하시면 됩니다.",
            "빠른 확인을 위해 별도 포털 [MASKED_URL]에서 상태를 보시면 됩니다.",
            "안내 문자 대신 [MASKED_URL] 경로를 열어 절차를 진행하시면 됩니다.",
            "현재 접수는 [MASKED_URL] 링크에서 바로 확인 가능합니다.",
        ]
    if planned.event == "recipient_partial_or_direct_disclosure":
        return [
            "안내하신 절차에 맞춰 [MASKED_PHONE]를 전달하겠습니다.",
            "지금 확인을 위해 [MASKED_PHONE]를 말씀드리겠습니다.",
            "요청하신 대로 [MASKED_PHONE]를 제공하겠습니다.",
            "절차를 이어가려고 [MASKED_PHONE]를 공유하겠습니다.",
            "[MASKED_PHONE] 번호를 입력하겠습니다.",
            "확인을 위해 [MASKED_PHONE]를 알려드리겠습니다.",
            "말씀하신 대로 [MASKED_PHONE]를 전달하겠습니다.",
            "지금은 [MASKED_PHONE]를 제공하겠습니다.",
            "급하다고 하시니 [MASKED_PHONE]를 말씀드리겠습니다.",
            "일단 [MASKED_PHONE]를 공유하겠습니다.",
            "확인이 필요하다면 [MASKED_PHONE]를 입력하겠습니다.",
            "제가 알고 있는 [MASKED_PHONE]를 알려드리겠습니다.",
            "안내를 믿고 [MASKED_PHONE]를 제공하겠습니다.",
            "진행을 위해 [MASKED_PHONE]를 전달하겠습니다.",
        ]
    if planned.event == "recovery_followup_due":
        if planned.speaker == "attacker":
            return [
                f"확인을 이어가려면 {placeholder}도 같은 방식으로 확인해야 합니다.",
                f"방금 확인한 내용에 이어 추가로 {placeholder} 확인이 필요합니다.",
                f"절차가 남아 있어 {placeholder}까지 확인하면 처리를 이어갈 수 있습니다.",
                f"다음 단계로 넘어가려면 {placeholder} 확인도 이어서 필요합니다.",
                f"기록을 마무리하려면 {placeholder}까지 확인해야 합니다.",
                f"이어지는 확인 단계라 {placeholder}도 추가로 확인하겠습니다.",
                f"방금 절차와 같은 방식으로 {placeholder} 확인을 이어가야 합니다.",
                f"추가 검증 항목으로 {placeholder} 확인이 남아 있습니다.",
                f"처리를 계속하려면 이어서 {placeholder} 확인이 필요합니다.",
                f"같은 접수 건이라 {placeholder}까지 확인해야 마무리됩니다.",
            ]
        return [
            "방금 공유한 내용은 중단하고 보존하겠습니다. 공식 채널에 확인하고 신고하겠습니다.",
            "지금 대화는 멈추고 기록을 남기겠습니다. 공식 경로로 복구 절차를 확인하겠습니다.",
            "추가 제공은 하지 않겠습니다. 방금 내용을 보존하고 공식 창구에 신고하겠습니다.",
            "더 진행하지 않겠습니다. 지금까지의 내용을 기록하고 공식 경로로 확인하겠습니다.",
            "방금 말한 내용은 멈추고 보존하겠습니다. 정식 창구에서 복구 절차를 확인하겠습니다.",
            "잠시 중단하겠습니다. 대화 내용을 기록하고 공식 채널로 확인하겠습니다.",
            "여기서 멈추고 방금 내용을 보존하겠습니다. 공식 창구에 신고하겠습니다.",
            "더 말하지 않겠습니다. 기록을 남긴 뒤 공식 경로에서 확인하겠습니다.",
            "추가 요청에는 응하지 않고 지금 내용을 보존해 신고하겠습니다.",
            "통화는 중단하고 공식 지원 창구에서 복구 여부를 확인하겠습니다.",
        ]
    if planned.event == "pressure_escalation":
        return [
            "이 절차를 중단하면 이후 일정이 보류되고 복구가 지연될 수 있습니다.",
            "오늘 안에 확인이 끝나지 않으면 처리가 제한되어 일정에 차질이 생길 수 있습니다.",
            "담당 권한상 지금 결정이 늦어지면 후속 처리가 불가능해질 수 있습니다.",
            "지금 멈추면 접수가 보류되어 오늘 안에 처리하기 어려울 수 있습니다.",
            "확인이 지연되면 다음 단계가 제한될 수 있어 바로 판단이 필요합니다.",
            "마감 전 결론이 없으면 일정이 뒤로 밀릴 가능성이 큽니다.",
            "지금 대응하지 않으면 이후 복구 시간이 더 길어질 수 있습니다.",
            "처리 순서가 밀리면 오늘 안에 다시 열기 어려울 수 있습니다.",
        ]
    return [planned.fallback]


def benign_fallback_candidates(event: str, speaker: str, subject: str) -> list[str]:
    rows = {
        ("needs_more_context", "attacker"): [
            f"{subject} 관련 일반 안내를 확인하려고 연락드렸습니다. 지금은 알림 여부만 설명드리겠습니다.",
            f"{subject} 알림 수신 여부만 확인하면 됩니다. 별도 입력은 필요하지 않습니다.",
            "현재 단계에서는 일반 안내만 드리며 추가 자료를 요구하지 않습니다.",
            "민감한 확인 단계가 아니라 안내 수신 여부만 설명드리는 연락입니다.",
            f"{subject} 관련 공지가 남아 있어 사실관계만 안내드리겠습니다.",
            "입력하실 정보는 없고, 일반 안내 내용을 짧게 확인하려고 합니다.",
        ],
        ("needs_more_context", "victim"): [
            "어떤 안내인지 아직 모르겠습니다. 민감한 내용이 아니라면 설명을 들어보겠습니다.",
            "알림 수신 여부 정도라면 이해했습니다. 자세한 내용은 나중에 제가 확인하겠습니다.",
            "지금은 안내 내용을 듣고 나중에 천천히 확인하겠습니다.",
            "일반 안내라면 우선 듣겠습니다. 바로 조치하지는 않겠습니다.",
            "내용만 확인하고 필요하면 나중에 직접 찾아보겠습니다.",
            "민감한 요청이 아니라면 설명만 듣겠습니다.",
        ],
        ("pressure_only_monitor", "attacker"): [
            "확인 일정이 늦어질 수 있으니 안내를 놓치지 않도록 주의해 주세요.",
            "오늘 안내를 확인하지 않으면 처리 일정이 뒤로 밀릴 수 있습니다.",
            "마감 전 안내를 보셔야 일정 지연을 줄일 수 있습니다.",
            "안내 확인이 늦어지면 일정이 보류될 수 있어 알려드립니다.",
            "오늘 중 알림을 확인하지 않으면 다음 안내가 지연될 수 있습니다.",
            "처리 차질을 줄이려면 마감 전 안내만 확인하시면 됩니다.",
        ],
        ("no_new_actionable_event", "attacker"): [
            "현재 단계에서는 추가 자료를 요구하지 않습니다. 안내가 오면 확인해 주세요.",
            "새로 입력하실 내용은 없습니다. 기존 안내만 확인해 주시면 됩니다.",
            "추가 요청 없이 안내 상태만 유지하겠습니다.",
            "별도 제출 없이 안내만 확인하시면 됩니다.",
            "지금은 새 조치가 필요한 단계가 아닙니다.",
            "추가 입력 없이 알림 상태만 참고하시면 됩니다.",
        ],
        ("no_new_actionable_event", "victim"): [
            "아직은 별도 조치를 하지 않고 공식 알림을 기다리겠습니다.",
            "지금은 추가로 할 일 없이 안내만 더 살펴보겠습니다.",
            "새로 제출할 내용은 없어 보이니 일단 기다리겠습니다.",
            "지금은 따로 움직이지 않고 나중에 확인하겠습니다.",
            "추가 행동은 하지 않고 안내만 기록해 두겠습니다.",
            "별도 조치 없이 상황을 더 지켜보겠습니다.",
        ],
        ("recipient_containment", "victim"): [
            "추가 요청이 생기면 등록된 채널에서만 확인하겠습니다.",
            "새로운 조치가 필요해지면 제가 알고 있는 경로로 확인하겠습니다.",
            "지금은 통화에서 더 진행하지 않고 추후 안내를 확인하겠습니다.",
            "필요하면 공식 앱에서 직접 확인하겠습니다.",
            "지금 통화에서는 진행하지 않고 나중에 확인하겠습니다.",
            "추가 안내는 제가 알고 있는 채널에서 확인하겠습니다.",
        ],
    }
    return rows.get((event, speaker), [f"{subject} 안내는 더 확인한 뒤 판단하겠습니다."])


def readable_path(path: str) -> str:
    return path.replace("_", " ").replace("or", "또는")


def context_label(value: str) -> str:
    labels = {
        "phone_call": "전화 통화",
        "messenger_call": "메신저 통화",
        "support_callback": "지원 콜백",
        "voice_message": "음성 메시지",
        "late_evening": "늦은 저녁",
        "during_commute": "이동 중",
        "before_deadline": "마감 전",
        "after_business_hours": "업무시간 이후",
        "lunch_break": "점심시간",
        "unspecified": "현재",
    }
    return labels.get(value, value.replace("_", " ") or "현재")


def constraint_label(value: str) -> str:
    if "cannot" in value or "difficult" in value:
        return "바로 확인하기 어려운"
    if "deadline" in value or "pending" in value or "time_sensitive" in value:
        return "일정 압박이 있는"
    if "cashflow" in value or "continuity" in value:
        return "생활상 부담이 있는"
    if "child" in value or "care" in value:
        return "돌봄 부담이 있는"
    if "language" in value:
        return "언어 지원이 필요한"
    if value:
        return value.replace("_", " ")
    return "개인 사정이 있는"


def deterministic_index(*parts: Any) -> int:
    modulo = int(parts[-1])
    if modulo <= 0:
        return 0
    key = "\u241f".join(str(part) for part in parts[:-1])
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % modulo


def expected_outcome(profile: str, dialogue: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if profile == "partial_disclosure_recovery":
        return "compromised", ["unknown_masked_asset"]
    if profile in {"benign_thin_context", "pressure_escalation"}:
        return "inconclusive", []
    observed, compromised = infer_record_outcome(dialogue)
    if observed in {"defended_success", "inconclusive"}:
        return observed, compromised
    return "defended_success", []


ACTION_TO_POLICY_RULE_IDS = {
    "no_alert": ("P_MONITOR",),
    "monitor_only": ("P_MONITOR",),
    "pause_or_verify": ("P_VERIFY",),
    "refuse_or_pause": ("P_NO_SECRET", "P_REFUSE"),
    "preserve_and_report": ("P_RECOVER",),
}

VISIBLE_ACTION_CHOICES = {
    "no_new_actionable_event": ("no_alert", "monitor_only"),
    "needs_more_context": ("monitor_only", "no_alert"),
    "pressure_only_monitor": ("monitor_only", "pause_or_verify"),
    "new_sensitive_request": ("refuse_or_pause", "pause_or_verify"),
    "new_caller_directed_path": ("refuse_or_pause", "pause_or_verify"),
    "pressure_escalation": ("pause_or_verify", "monitor_only"),
    "recipient_containment": ("no_alert", "monitor_only"),
    "recipient_partial_or_direct_disclosure": ("preserve_and_report", "pause_or_verify"),
    "recovery_followup_due": ("preserve_and_report", "refuse_or_pause"),
}


def with_visible_policy_actions(plan: list[PlannedTurn]) -> list[PlannedTurn]:
    result: list[PlannedTurn] = []
    for turn_id, row in enumerate(plan, start=1):
        action = visible_policy_action(row, turn_id)
        result.append(
            PlannedTurn(
                speaker=row.speaker,
                phase=row.phase,
                intent_label=row.intent_label,
                risk_signal=row.risk_signal,
                event=row.event,
                placeholders=row.placeholders,
                instruction=row.instruction,
                fallback=row.fallback,
                evidence_turn_ids=row.evidence_turn_ids,
                action=action,
                policy_rule_ids=ACTION_TO_POLICY_RULE_IDS[action],
            )
        )
    return result


def visible_policy_action(row: PlannedTurn, turn_id: int) -> str:
    text = f"{row.instruction} {row.fallback} {row.risk_signal}"
    has_url = "[MASKED_URL]" in row.placeholders or "URL" in text or "링크" in text or "경로" in text
    has_secret_placeholder = any(item in row.placeholders for item in ("[MASKED_CREDENTIAL]", "[MASKED_ACCOUNT]"))
    has_phone_placeholder = "[MASKED_PHONE]" in row.placeholders
    has_pressure = has_any(text, PRESSURE_MARKERS) or row.risk_signal in {"urgency", "money_transfer_request"}
    mentions_official_or_refusal = has_any(text, ("공식", "저장된", "제공하지", "입력하지", "열지", "사용하지"))
    early_turn = turn_id <= 4

    if row.event == "needs_more_context":
        return "no_alert" if row.speaker == "attacker" else "monitor_only"
    if row.event == "no_new_actionable_event":
        if row.speaker == "attacker":
            return "no_alert"
        return "monitor_only" if early_turn or not mentions_official_or_refusal else "no_alert"
    if row.event == "pressure_only_monitor":
        return "pause_or_verify" if has_pressure and turn_id >= 5 else "monitor_only"
    if row.event == "new_sensitive_request":
        if has_secret_placeholder or turn_id >= 5:
            return "refuse_or_pause"
        return "pause_or_verify"
    if row.event == "new_caller_directed_path":
        if has_url and turn_id in {5, 8}:
            return "refuse_or_pause"
        return "pause_or_verify"
    if row.event == "pressure_escalation":
        return "pause_or_verify" if has_pressure and turn_id >= 7 else "monitor_only"
    if row.event == "recipient_containment":
        if turn_id >= 8 and not row.placeholders:
            return "no_alert"
        if row.placeholders or mentions_official_or_refusal:
            return "monitor_only"
        return "no_alert"
    if row.event == "recipient_partial_or_direct_disclosure":
        return "pause_or_verify" if early_turn and has_phone_placeholder else "preserve_and_report"
    if row.event == "recovery_followup_due":
        return "preserve_and_report" if row.speaker == "victim" else "refuse_or_pause"
    return EVENT_TO_ACTION[row.event]


def plan_for_profile_trajectory_varied(
    profile: str,
    scenario_type: str,
    protected_assets: list[str],
    attacker_goal: str,
    environmental_context: dict[str, str],
    trajectory_variant: int,
) -> list[PlannedTurn]:
    variant = trajectory_variant % TRAJECTORY_VARIANT_COUNT
    subject = subject_for_scenario(scenario_type, environmental_context)
    request_placeholder = placeholder_for_goal(attacker_goal, protected_assets)

    def t(
        turn_id: int,
        speaker: str,
        intent_label: str,
        risk_signal: str,
        event: str,
        placeholders: tuple[str, ...],
        instruction: str,
        fallback: str,
        evidence_turn_ids: tuple[int, ...],
    ) -> PlannedTurn:
        return planned(turn_id, speaker, intent_label, risk_signal, event, placeholders, instruction, fallback, evidence_turn_ids)

    common_open = [
        t(1, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), f"{subject}에 관해 연락한 이유만 설명하고 민감값은 요구하지 마세요.", f"안녕하세요. {subject} 때문에 연락드렸습니다. 먼저 상황만 간단히 설명드리겠습니다.", (1,)),
        t(2, "victim", "uncertain_response", "victim_uncertainty", "needs_more_context", (), "혼란스럽지만 설명은 듣겠다고 답하세요.", "갑작스러운 연락이라 조금 당황스럽습니다. 우선 어떤 내용인지 듣고 판단하겠습니다.", (2,)),
    ]

    if profile == "safe_containment":
        variants = [
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "처리 지연 압박만 주고 민감값은 요구하지 마세요.", f"오늘 확인이 늦어지면 {subject} 처리가 밀릴 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "아직 추가 조치 없이 상황만 더 듣겠다고 답하세요.", "아직 어떤 조치를 해야 하는지는 모르겠습니다. 설명을 조금 더 듣겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "처음으로 masked 민감값을 요구하세요.", f"상태 확인을 이어가려면 {request_placeholder} 확인이 필요합니다.", (5,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "통화 중에는 masked 값을 제공하지 않겠다고 답하세요.", f"통화 중에는 {request_placeholder}를 제공하지 않겠습니다. 제가 알고 있는 공식 경로로 확인하겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "no_new_actionable_event", (request_placeholder,), "같은 masked 값을 반복 요청하되 새 요구는 추가하지 마세요.", f"새로운 값은 아닙니다. 같은 {request_placeholder} 확인이 끝나야 다음 안내가 가능합니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "반복 요청에도 공식 경로 확인을 선택하세요.", "반복해서 말씀하셔도 이 통화에서는 진행하지 않겠습니다. 공식 경로에서 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "초반에 masked 민감값을 요구하세요.", f"{subject} 확인을 위해 {request_placeholder}를 먼저 확인해야 합니다.", (3,)),
                t(4, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "즉시 제공하지 않고 검증 경로를 말하세요.", f"{request_placeholder}는 전화로 말하지 않겠습니다. 저장된 연락처로 직접 확인하겠습니다.", (4,)),
                t(5, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "새 민감값 없이 지연 압박만 강화하세요.", "지금 멈추면 오늘 안에 처리가 어려워질 수 있습니다.", (5,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "압박에도 검증을 유지하세요.", "급하다고 하셔도 통화 중에는 진행하지 않겠습니다. 공식 채널에서 확인하겠습니다.", (6,)),
                t(7, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "새 요청 없이 안내만 유지하세요.", "추가 자료를 요구하는 것은 아니고 안내 상태만 남겨두겠습니다.", (7,)),
                t(8, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "추가 행동 없이 종료하세요.", "일단 더 진행하지 않고 나중에 직접 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "일정 압박만 말하세요.", "확인이 늦어지면 처리 일정에 차질이 생길 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "아직 보류한다고 답하세요.", "바로 판단하기 어렵습니다. 아직은 지켜보겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "caller-directed masked URL을 제시하세요.", "앱이 느릴 수 있으니 확인 경로 [MASKED_URL]에서 이어가시면 됩니다.", (5,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "링크를 쓰지 않겠다고 답하세요.", "보내주신 링크는 사용하지 않겠습니다. 제가 직접 공식 경로로 확인하겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "다른 masked 민감값을 요구하세요.", f"그러면 확인을 위해 {request_placeholder}만 말씀해 주세요.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "masked 값을 제공하지 않고 종료하세요.", f"{request_placeholder}도 통화 중에는 제공하지 않겠습니다. 공식 창구에서 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "낮은 위험 설명만 이어가세요.", f"{subject}의 접수 상태만 설명드리는 단계입니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "아직 조치하지 않는다고 답하세요.", "설명은 이해했지만 지금 당장 할 일은 없어 보입니다.", (4,)),
                t(5, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "민감값 없이 마감 압박만 주세요.", "오늘 안에 확인하지 않으면 순서가 뒤로 밀릴 수 있습니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류한다고 답하세요.", "그래도 바로 움직이기보다는 확인 경로를 먼저 찾아보겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "늦은 턴에서 처음으로 masked 민감값을 요구하세요.", f"마지막 확인으로 {request_placeholder}가 필요합니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "늦은 요청을 거절하고 검증하세요.", f"{request_placeholder}는 제공하지 않겠습니다. 공식 경로에서 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "초반 masked 민감값 요구를 하세요.", f"처리 전 {request_placeholder} 확인이 필요합니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "즉시 거절보다 혼란과 보류를 표현하세요.", "왜 필요한지 아직 이해가 되지 않습니다. 조금 더 설명해 주세요.", (4,)),
                t(5, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "caller-directed masked URL을 제시하세요.", "설명이 길어지니 [MASKED_URL]에서 상태를 확인하시면 됩니다.", (5,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "링크 사용을 거절하세요.", "그 링크는 열지 않겠습니다. 제가 알고 있는 경로로 확인하겠습니다.", (6,)),
                t(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "새 민감값 없이 압박하세요.", "지금 확인하지 않으면 이후 처리가 제한될 수 있습니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "압박에도 중단하세요.", "제한이 있다고 해도 이 통화에서는 진행하지 않겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "초기 압박만 주세요.", f"{subject} 일정이 지연될 수 있어 빠르게 확인이 필요합니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "혼란과 유보를 표현하세요.", "급한 건 알겠지만 아직 무엇을 해야 할지 모르겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "masked 민감값을 요구하세요.", f"본인 확인을 위해 {request_placeholder}를 확인해야 합니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "완전 거절 전 망설임을 표현하세요.", "그 정보를 바로 말하기는 조심스럽습니다. 다른 방법은 없나요?", (6,)),
                t(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "새 민감값 없이 압박을 강화하세요.", "다른 방법은 시간이 오래 걸리고 오늘 처리가 어려울 수 있습니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "최종적으로 공식 확인을 선택하세요.", "그래도 통화 중에는 진행하지 않고 공식 경로에서 확인하겠습니다.", (8,)),
            ],
        ]
        return with_visible_policy_actions(variants[variant])

    if profile == "caller_directed_path":
        variants = [
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "지연 압박만 하세요.", "확인이 늦어지면 처리 순서가 밀릴 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하며 묻는 답변을 하세요.", "그렇다면 공식 경로에서도 확인할 수 있는지 궁금합니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "masked 민감값을 요구하세요.", f"현재 상태 확인에는 {request_placeholder}가 필요합니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "부담스럽지만 아직 결론을 내리지 마세요.", "요청이 부담스럽습니다. 다른 확인 방법이 있나요?", (6,)),
                t(7, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "caller-directed masked URL을 제시하세요.", "그러면 별도 확인 경로 [MASKED_URL]에서 이어가시면 됩니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "caller가 준 링크를 사용하지 마세요.", "그 링크는 사용하지 않고 저장된 공식 경로로 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "초반에 caller-directed masked URL을 제시하세요.", "접수 상태는 [MASKED_URL]에서 바로 확인하시면 됩니다.", (3,)),
                t(4, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "링크를 열지 않겠다고 답하세요.", "처음 받은 링크는 열지 않겠습니다. 기존 앱에서 확인하겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "masked 민감값도 요구하세요.", f"앱이 어렵다면 {request_placeholder}를 통화에서 확인할 수 있습니다.", (5,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "masked 값 제공도 거절하세요.", f"{request_placeholder}는 통화 중 제공하지 않겠습니다.", (6,)),
                t(7, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "새 요청 없이 안내만 유지하세요.", "새로운 요청은 아니고 안내 절차가 남아 있다는 점만 말씀드립니다.", (7,)),
                t(8, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "추가 행동 없이 종료하세요.", "일단 추가 행동은 하지 않고 직접 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "압박만 주세요.", "오늘 확인이 늦어지면 처리가 보류될 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "망설이세요.", "바로 진행하기에는 조심스럽습니다.", (4,)),
                t(5, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "중간 턴에서 URL을 제시하세요.", "공식 접수와 별도로 [MASKED_URL]에서 빠르게 확인할 수 있습니다.", (5,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "링크 사용을 거절하세요.", "별도 링크는 사용하지 않겠습니다. 제가 직접 확인하겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "늦은 턴에서 masked 민감값을 요구하세요.", f"그렇다면 통화에서 {request_placeholder}를 확인해 주세요.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "값 제공을 거절하세요.", f"{request_placeholder}도 제공하지 않고 공식 경로로 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "배경 설명만 이어가세요.", "현재는 접수 상태 설명만 드리는 단계입니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "아직 듣기만 하세요.", "설명은 듣겠지만 바로 조치하지는 않겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "민감값 요구를 하세요.", f"확인을 위해 {request_placeholder}가 필요합니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "대체 경로를 물어보세요.", "그 정보 대신 공식 경로에서 확인하는 방법은 없나요?", (6,)),
                t(7, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "URL을 늦게 제시하세요.", "그럼 [MASKED_URL]에서 직접 입력하시면 됩니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "URL 사용을 거절하세요.", "그 경로는 사용하지 않고 저장된 공식 경로로 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "초반에 민감값을 요구하세요.", f"{request_placeholder} 확인을 먼저 진행해야 합니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "혼란과 보류를 표현하세요.", "갑자기 그런 정보를 요청하셔서 당황스럽습니다.", (4,)),
                t(5, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "새 민감값 없이 압박하세요.", "지금 확인하지 않으면 처리가 지연될 수 있습니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "확정 거절 전 대체 경로를 물으세요.", "그럼 공식 앱에서 확인할 수 있는지 알려주세요.", (6,)),
                t(7, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "URL을 제시하세요.", "앱 대신 [MASKED_URL] 경로로 들어가시면 빠릅니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "URL을 쓰지 않으세요.", "그 링크는 사용하지 않겠습니다. 직접 공식 앱을 열어 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "압박만 주세요.", "마감 전에 확인해야 처리 지연을 줄일 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "혼란스럽다고 답하세요.", "무슨 절차인지 아직 잘 모르겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "민감값을 요구하세요.", f"확인을 위해 {request_placeholder}를 알려주셔야 합니다.", (5,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "제공하지 않는다고 답하세요.", f"{request_placeholder}는 전화로 제공하지 않겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "그래도 URL을 제시하세요.", "그럼 [MASKED_URL]에서 본인이 직접 확인하시면 됩니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "URL도 거절하세요.", "그 경로도 사용하지 않고 제가 아는 공식 경로로 확인하겠습니다.", (8,)),
            ],
        ]
        return with_visible_policy_actions(variants[variant])

    if profile == "partial_disclosure_recovery":
        variants = [
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "민감값 없는 압박만 하세요.", "확인이 늦어지면 처리가 밀릴 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "아직 보류하세요.", "조금 불안하지만 설명을 더 듣겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", "identity_information_request", "new_sensitive_request", ("[MASKED_PHONE]",), "masked phone을 요구하세요.", f"{subject} 확인을 위해 등록된 연락처 [MASKED_PHONE]가 필요합니다.", (5,)),
                t(6, "victim", "disclose_or_comply", "asset_disclosure", "recipient_partial_or_direct_disclosure", ("[MASKED_PHONE]",), "위험하게 masked phone을 제공하세요.", "[MASKED_PHONE] 번호를 말씀드리겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "recovery_followup_due", (request_placeholder,), "추가 masked 값을 요구하세요.", f"확인을 이어가려면 {request_placeholder}도 같은 방식으로 확인해야 합니다.", (6,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recovery_followup_due", (), "중단/보존/공식 복구를 말하세요.", "방금 공유한 내용은 중단하고 기록하겠습니다. 공식 창구에 확인하겠습니다.", (6,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", "identity_information_request", "new_sensitive_request", ("[MASKED_PHONE]",), "초반에 masked phone을 요구하세요.", "등록 여부 확인을 위해 [MASKED_PHONE]를 먼저 확인해 주세요.", (3,)),
                t(4, "victim", "disclose_or_comply", "asset_disclosure", "recipient_partial_or_direct_disclosure", ("[MASKED_PHONE]",), "초반에 부분 노출하세요.", "확인을 위해 [MASKED_PHONE]를 입력하겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "recovery_followup_due", (request_placeholder,), "추가 masked 값을 요구하세요.", f"방금 확인한 내용에 이어 {request_placeholder}도 확인해야 합니다.", (4,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recovery_followup_due", (), "회복 조치를 말하세요.", "잠깐 멈추겠습니다. 방금 내용을 기록하고 공식 경로로 확인하겠습니다.", (4,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "recovery_followup_due", (request_placeholder,), "같은 추가 요구를 이어가세요.", f"절차가 남아 있어 {request_placeholder} 확인이 계속 필요합니다.", (4,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recovery_followup_due", (), "더 이상 제공하지 않고 신고/복구하세요.", "더 제공하지 않겠습니다. 대화 기록을 보존하고 공식 창구에 신고하겠습니다.", (4,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "압박만 하세요.", "오늘 확인하지 않으면 일정이 늦어질 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "망설이세요.", "어떤 확인인지 아직 헷갈립니다.", (4,)),
                t(5, "attacker", "request_identity_information", "identity_information_request", "new_sensitive_request", ("[MASKED_PHONE]",), "masked phone을 요구하세요.", "등록 연락처 확인을 위해 [MASKED_PHONE]를 알려주세요.", (5,)),
                t(6, "victim", "disclose_or_comply", "asset_disclosure", "recipient_partial_or_direct_disclosure", ("[MASKED_PHONE]",), "부분 노출하세요.", "요청하신 대로 [MASKED_PHONE]를 제공하겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "recovery_followup_due", (request_placeholder,), "추가 masked 값을 요구하세요.", f"이제 {request_placeholder}까지 확인하면 처리가 이어집니다.", (6,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recovery_followup_due", (), "회복 조치를 말하세요.", "추가 제공은 중단하겠습니다. 공식 경로에서 복구 절차를 확인하겠습니다.", (6,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", "identity_information_request", "new_sensitive_request", ("[MASKED_PHONE]",), "masked phone을 요구하세요.", "본인 확인을 위해 [MASKED_PHONE]를 먼저 확인하겠습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "망설이며 더 묻습니다.", "그 번호를 왜 지금 확인해야 하는지 설명해 주세요.", (4,)),
                t(5, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "새 민감값 없이 압박하세요.", "지금 확인하지 않으면 오늘 처리가 어렵습니다.", (5,)),
                t(6, "victim", "disclose_or_comply", "asset_disclosure", "recipient_partial_or_direct_disclosure", ("[MASKED_PHONE]",), "압박에 밀려 부분 노출하세요.", "그렇다면 [MASKED_PHONE]를 말씀드리겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "recovery_followup_due", (request_placeholder,), "추가 masked 값을 요구하세요.", f"이어지는 절차로 {request_placeholder}도 확인해야 합니다.", (6,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recovery_followup_due", (), "회복 조치를 말하세요.", "방금 공유한 부분은 멈추고 기록하겠습니다. 공식 채널로 확인하겠습니다.", (6,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "초기 압박만 주세요.", "확인이 늦으면 순서가 뒤로 밀릴 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하세요.", "일단 설명은 이해했지만 바로 말하기는 어렵습니다.", (4,)),
                t(5, "attacker", "request_identity_information", "identity_information_request", "new_sensitive_request", ("[MASKED_PHONE]",), "masked phone을 요구하세요.", "등록 확인을 위해 [MASKED_PHONE]가 필요합니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "아직 제공하지 않고 흔들리는 답변을 하세요.", "잠시만요, 제가 이걸 말해도 되는지 모르겠습니다.", (6,)),
                t(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "압박을 강화하세요.", "지금 늦어지면 접수가 보류될 수 있습니다.", (7,)),
                t(8, "victim", "disclose_or_comply", "asset_disclosure", "recipient_partial_or_direct_disclosure", ("[MASKED_PHONE]",), "마지막 턴에서 부분 노출하세요.", "급하다고 하시니 [MASKED_PHONE]를 공유하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", "identity_information_request", "new_sensitive_request", ("[MASKED_PHONE]",), "초반 masked phone을 요구하세요.", "[MASKED_PHONE] 확인이 먼저 필요합니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "망설이세요.", "그 정보를 지금 말하는 건 조금 걱정됩니다.", (4,)),
                t(5, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "masked URL을 제시하세요.", "그럼 [MASKED_URL]에 직접 입력하시면 됩니다.", (5,)),
                t(6, "victim", "disclose_or_comply", "asset_disclosure", "recipient_partial_or_direct_disclosure", ("[MASKED_PHONE]",), "URL 대신 통화에서 일부 노출하세요.", "링크는 어렵고, 여기서 [MASKED_PHONE]를 말씀드리겠습니다.", (6,)),
                t(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "recovery_followup_due", (request_placeholder,), "추가 masked 값을 요구하세요.", f"이어서 {request_placeholder}도 같은 방식으로 확인하겠습니다.", (6,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recovery_followup_due", (), "회복 조치를 말하세요.", "더 진행하지 않겠습니다. 방금 대화는 보존하고 공식 경로로 확인하겠습니다.", (6,)),
            ],
        ]
        return with_visible_policy_actions(variants[variant])

    if profile == "pressure_escalation":
        variants = [
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "초기 압박만 주세요.", "확인이 늦어지면 처리 순서가 밀릴 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하세요.", "급하다는 말은 이해했지만 바로 판단하기 어렵습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "masked 민감값을 요구하세요.", f"본인 확인을 위해 {request_placeholder}가 필요합니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "근거를 더 요구하세요.", "왜 그 정보가 필요한지 더 설명해 주세요.", (6,)),
                t(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "새 민감값 없이 압박을 강화하세요.", "지금 멈추면 오늘 안에 처리가 어려워질 수 있습니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "공식 확인으로 마무리하세요.", "그래도 이 통화에서는 진행하지 않고 공식 경로에서 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "초반 masked 민감값을 요구하세요.", f"{request_placeholder} 확인을 먼저 해야 합니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "망설이며 거절 전 상태로 답하세요.", "바로 말하기에는 조심스럽습니다.", (4,)),
                t(5, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "중간 턴에서 압박하세요.", "지금 확인하지 않으면 접수가 지연될 수 있습니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "아직 보류하세요.", "그렇다고 해도 먼저 확인할 시간이 필요합니다.", (6,)),
                t(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "다시 압박하세요.", "시간이 지나면 이 경로로는 처리가 지연될 수 있습니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "결국 containment를 선택하세요.", "압박이 있어도 전화로는 진행하지 않겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "배경 설명만 하세요.", f"{subject} 접수 상황을 먼저 설명드리겠습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "설명을 더 듣겠다고 답하세요.", "아직은 설명만 듣겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "중간 턴에서 masked 민감값을 요구하세요.", f"다음 단계로 {request_placeholder} 확인이 필요합니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하세요.", "그 정보는 바로 말하기 어렵습니다.", (6,)),
                t(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "늦은 턴에서 압박하세요.", "오늘 안에 확인하지 않으면 일정에 차질이 생길 수 있습니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "공식 확인으로 끝내세요.", "먼저 공식 채널에서 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "민감값 없는 압박만 하세요.", "처리 시간이 제한되어 있어 빠른 확인이 필요합니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "혼란스러운 답변을 하세요.", "어떤 확인인지 아직 정확히 모르겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "민감값을 요구하세요.", f"확인을 위해 {request_placeholder}를 알려주셔야 합니다.", (5,)),
                t(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "masked 값 제공을 거절하세요.", f"{request_placeholder}는 통화 중 제공하지 않겠습니다.", (6,)),
                t(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "거절 이후 압박하세요.", "그렇게 멈추면 처리가 보류될 수 있습니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "공식 확인을 유지하세요.", "보류된다고 해도 공식 경로에서 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "초반 민감값을 요구하세요.", f"먼저 {request_placeholder} 확인이 필요합니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "망설이세요.", "제가 바로 말해도 되는지 모르겠습니다.", (4,)),
                t(5, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "URL을 제시하세요.", "그럼 [MASKED_URL] 경로에서 빠르게 확인하시면 됩니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "링크도 즉시 거절하지 않고 당황하세요.", "처음 보는 경로라 조금 불안합니다.", (6,)),
                t(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "URL 이후 압박하세요.", "지금 처리하지 않으면 이후 접근이 제한될 수 있습니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "공식 확인을 선택하세요.", "그 경로는 사용하지 않고 공식 앱에서 확인하겠습니다.", (8,)),
            ],
            common_open + [
                t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "초기 압박만 주세요.", "확인이 늦어지면 후속 일정이 밀릴 수 있습니다.", (3,)),
                t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하세요.", "아직은 제가 확인할 시간이 필요합니다.", (4,)),
                t(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "민감값을 요구하세요.", f"처리 확인에는 {request_placeholder}가 필요합니다.", (5,)),
                t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "흔들리지만 제공하지 마세요.", "말해야 할지 모르겠어서 잠시만 생각하겠습니다.", (6,)),
                t(7, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "새 요구 없이 안내만 유지하세요.", "추가 요청은 아니고 시간이 많지 않다는 점만 알려드립니다.", (7,)),
                t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "공식 확인으로 마무리하세요.", "통화는 여기서 멈추고 공식 경로로 확인하겠습니다.", (8,)),
            ],
        ]
        return with_visible_policy_actions(variants[variant])

    variants = [
        common_open + [
            t(3, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "낮은 위험 안내만 하세요.", f"{subject} 접수 여부만 설명드리겠습니다.", (3,)),
            t(4, "victim", "uncertain_response", "victim_uncertainty", "needs_more_context", (), "설명만 듣겠다고 답하세요.", "민감한 내용이 아니라면 설명은 듣겠습니다.", (4,)),
            t(5, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "일정 지연 압박만 하세요.", "오늘 확인이 늦어지면 일정이 뒤로 밀릴 수 있습니다.", (5,)),
            t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "별도 조치 없이 기다리겠다고 답하세요.", "아직은 별도 조치를 하지 않고 기다리겠습니다.", (6,)),
            t(7, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "추가 자료 요구 없이 안내만 유지하세요.", "현재 단계에서는 추가 자료를 요구하지 않습니다.", (7,)),
            t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "추가 요청은 공식 경로에서 확인하겠다고 답하세요.", "추가 요청이 생기면 공식 채널에서만 확인하겠습니다.", (8,)),
        ],
        common_open + [
            t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "초반 압박만 하세요.", "오늘 안내를 확인하지 않으면 처리 일정이 늦어질 수 있습니다.", (3,)),
            t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하세요.", "일정이 있다는 점은 알겠습니다. 나중에 확인하겠습니다.", (4,)),
            t(5, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "일반 안내만 하세요.", "지금은 안내 상태만 설명드리는 단계입니다.", (5,)),
            t(6, "victim", "uncertain_response", "victim_uncertainty", "needs_more_context", (), "설명만 듣습니다.", "그 정도 안내라면 듣고 나중에 확인하겠습니다.", (6,)),
            t(7, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "새 요청 없이 마무리하세요.", "추가로 입력하실 내용은 없습니다.", (7,)),
            t(8, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "별도 행동 없이 종료하세요.", "알겠습니다. 별도 조치는 하지 않겠습니다.", (8,)),
        ],
        common_open + [
            t(3, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "일반 배경만 설명하세요.", "접수된 안내의 배경만 말씀드리겠습니다.", (3,)),
            t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "아직 보류하세요.", "바로 할 일은 없어 보입니다.", (4,)),
            t(5, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "추가 배경만 말하세요.", "현재는 대상 여부만 확인 중입니다.", (5,)),
            t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "기다리겠다고 답하세요.", "그럼 우선 기다리겠습니다.", (6,)),
            t(7, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "늦은 턴에서 압박만 하세요.", "마감 전 안내를 확인해야 일정 지연을 줄일 수 있습니다.", (7,)),
            t(8, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "별도 행동 없이 끝내세요.", "나중에 직접 확인해 보겠습니다.", (8,)),
        ],
        common_open + [
            t(3, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "압박만 하세요.", "확인 일정이 늦어질 수 있으니 안내를 놓치지 마세요.", (3,)),
            t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하세요.", "알겠습니다. 지금은 더 진행하지 않겠습니다.", (4,)),
            t(5, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "다른 압박 표현을 하세요.", "마감 전에 확인해야 처리 지연을 줄일 수 있습니다.", (5,)),
            t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "별도 조치하지 마세요.", "그래도 추가 조치는 하지 않겠습니다.", (6,)),
            t(7, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "새 요청 없이 안내만 유지하세요.", "새로 제출하실 자료는 없습니다.", (7,)),
            t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "공식 확인을 말하되 신고는 말하지 마세요.", "필요하면 공식 채널에서만 다시 확인하겠습니다.", (8,)),
        ],
        common_open + [
            t(3, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "추가 요구 없이 설명하세요.", "현재는 별도로 입력하실 내용이 없습니다.", (3,)),
            t(4, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "추가 행동 없이 답하세요.", "그럼 지금은 특별히 할 일이 없겠네요.", (4,)),
            t(5, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "압박만 하세요.", "안내 확인이 늦어지면 일정이 뒤로 밀릴 수 있습니다.", (5,)),
            t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하세요.", "나중에 제가 직접 확인하겠습니다.", (6,)),
            t(7, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "늦은 압박을 하세요.", "오늘 안에 확인하셔야 지연을 줄일 수 있습니다.", (7,)),
            t(8, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "별도 행동 없이 마무리하세요.", "알겠습니다. 지금은 더 진행하지 않겠습니다.", (8,)),
        ],
        common_open + [
            t(3, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "배경 설명만 하세요.", "현재는 안내 배경만 전달드리고 있습니다.", (3,)),
            t(4, "victim", "uncertain_response", "victim_uncertainty", "needs_more_context", (), "설명만 듣겠다고 답하세요.", "설명은 듣겠지만 지금 결정하지는 않겠습니다.", (4,)),
            t(5, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "새 요청 없이 안내하세요.", "지금 단계에서 새 자료는 필요하지 않습니다.", (5,)),
            t(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "보류하세요.", "알겠습니다. 일단 기다리겠습니다.", (6,)),
            t(7, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "압박만 하세요.", "확인이 늦어지면 일정이 미뤄질 수 있습니다.", (7,)),
            t(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "필요 시 공식 확인한다고 답하세요.", "필요해지면 공식 채널에서만 확인하겠습니다.", (8,)),
        ],
    ]
    return with_visible_policy_actions(variants[variant])


def plan_for_profile(
    profile: str,
    scenario_type: str,
    protected_assets: list[str],
    attacker_goal: str,
    environmental_context: dict[str, str],
    trajectory_variant: int | None = None,
) -> list[PlannedTurn]:
    if trajectory_variant is not None:
        return plan_for_profile_trajectory_varied(
            profile,
            scenario_type,
            protected_assets,
            attacker_goal,
            environmental_context,
            trajectory_variant,
        )
    subject = subject_for_scenario(scenario_type, environmental_context)
    request_placeholder = placeholder_for_goal(attacker_goal, protected_assets)
    common = [
        planned(
            1,
            "attacker",
            "social_engineering_probe",
            "authority_impersonation",
            "needs_more_context",
            (),
            f"{subject}와 관련해 왜 연락했는지 설명하되 민감값 요청은 하지 마세요.",
            f"안녕하세요. {subject} 안내 확인 차 연락드렸습니다. 최근 알림을 보셨는지 여쭙고 싶습니다.",
            (1,),
        ),
        planned(
            2,
            "victim",
            "uncertain_response",
            "victim_uncertainty",
            "needs_more_context",
            (),
            "상황을 이해하려 하지만 아직 민감한 행동은 하지 않겠다는 태도로 답하세요.",
            "아직 정확히 어떤 내용인지 모르겠습니다. 먼저 설명을 듣고 판단하겠습니다.",
            (2,),
        ),
        planned(
            3,
            "attacker",
            "social_engineering_probe",
            "urgency",
            "pressure_only_monitor",
            (),
            "일정 지연이나 처리 지연 압박만 주고 민감값 요청은 하지 마세요.",
            f"{subject} 처리 일정이 지연될 수 있어서 안내 흐름만 계속 확인해 주시면 됩니다.",
            (3,),
        ),
        planned(
            4,
            "victim",
            "uncertain_response",
            "victim_uncertainty",
            "no_new_actionable_event",
            (),
            "아직 판단이 어렵고 추가 설명을 듣겠다는 식으로 답하세요.",
            "일정이 지연될 수 있다는 점은 이해했지만 아직 필요한 조치가 무엇인지 분명하지 않습니다.",
            (4,),
        ),
    ]
    if profile == "safe_containment":
        return common + [
            planned(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "처음으로 민감한 masked 값을 요구하세요.", f"{subject} 처리를 위해 {request_placeholder} 확인이 필요합니다.", (5,)),
            planned(6, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (request_placeholder,), "해당 masked 값을 제공하지 않고 persona의 검증 습관에 맞게 공식/신뢰 경로 확인을 말하세요.", f"통화 중에는 {request_placeholder}를 제공할 수 없습니다. 공식 경로로 직접 확인하겠습니다.", (6,)),
            planned(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "no_new_actionable_event", (request_placeholder,), "같은 민감값 요청을 반복하되 새로운 압박 유형이나 새 민감값은 추가하지 마세요.", f"앞서 말씀드린 같은 {request_placeholder} 확인이 필요합니다. 같은 확인을 다시 요청드립니다.", (7,)),
            planned(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "반복 요청에도 공식 경로 확인으로 마무리하세요.", "반복 요청이어도 이 통화에서는 진행하지 않겠습니다. 공식 경로에서 확인하겠습니다.", (8,)),
        ]
    if profile == "caller_directed_path":
        return common + [
            planned(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "처음으로 민감한 masked 값을 요구하세요.", f"{subject} 상태 확인을 위해 {request_placeholder} 확인이 필요합니다.", (5,)),
            planned(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "요청이 부담스럽지만 아직 확정적으로 거절하지 않고 공식 경로 가능성을 물으세요.", "요청은 부담스럽습니다. 공식 경로에서 볼 수 있는지 알고 싶습니다.", (6,)),
            planned(7, "attacker", "request_identity_information", "identity_information_request", "new_caller_directed_path", ("[MASKED_URL]",), "새로운 caller-directed masked URL 또는 링크 경로를 제시하세요.", "공식 앱이 지연될 수 있으니 임시 확인 링크 [MASKED_URL]로 접속해 진행해 주세요.", (7,)),
            planned(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "caller가 준 링크를 쓰지 않고 저장된 공식 경로로 확인한다고 답하세요.", "링크로는 접속하지 않겠습니다. 저장된 공식 번호로 확인하겠습니다.", (8,)),
        ]
    if profile == "partial_disclosure_recovery":
        return common + [
            planned(5, "attacker", "request_identity_information", "identity_information_request", "new_sensitive_request", ("[MASKED_PHONE]",), "등록 연락처처럼 masked phone 값을 요구하세요.", f"{subject} 확인을 위해 등록된 연락처 [MASKED_PHONE]를 말씀해 주세요.", (5,)),
            planned(6, "victim", "disclose_or_comply", "asset_disclosure", "recipient_partial_or_direct_disclosure", ("[MASKED_PHONE]",), "위험하게 masked phone 값을 제공하는 문장으로 답하세요.", "안내하신 절차에 맞춰 [MASKED_PHONE]를 전달하겠습니다.", (6,)),
            planned(7, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "recovery_followup_due", (request_placeholder,), "이미 일부 공유가 있었으므로 추가 masked 값을 요구하세요.", f"확인을 이어가려면 {request_placeholder}도 같은 방식으로 확인해야 합니다.", (6,)),
            planned(8, "victim", "resist_or_verify_request", "victim_resistance", "recovery_followup_due", (), "방금 공유한 내용을 중단/보존/신고하고 공식 경로로 복구한다고 답하세요.", "방금 공유한 내용은 중단하고 보존하겠습니다. 공식 채널에 확인하고 신고하겠습니다.", (6,)),
        ]
    if profile == "pressure_escalation":
        return common + [
            planned(5, "attacker", "request_identity_information", risk_for_goal(attacker_goal), "new_sensitive_request", (request_placeholder,), "처음으로 민감한 masked 값을 요구하세요.", f"{subject} 상태 확인을 위해 {request_placeholder} 확인이 필요합니다.", (5,)),
            planned(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "바로 제공하기 어렵고 근거를 더 듣겠다고 답하세요.", "바로 제공하기는 어렵습니다. 어떤 근거로 필요한지 더 듣겠습니다.", (6,)),
            planned(7, "attacker", "escalate_urgency", "urgency", "pressure_escalation", (), "새로운 압박 유형을 추가해 지연/손실 또는 권한 압박을 강화하되 새 민감값은 요구하지 마세요.", "센터 담당자로서 안내드립니다. 이 절차를 중단하면 이후 일정이 보류되고 복구가 지연될 수 있습니다.", (7,)),
            planned(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "압박에도 공식 경로 확인을 선택한다고 답하세요.", "그렇더라도 먼저 공식 앱에서 확인하겠습니다. 통화 중 제공은 하지 않겠습니다.", (8,)),
        ]
    return [
        planned(1, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), f"{subject} 일반 안내를 설명하고 민감값 요청은 하지 마세요.", f"안녕하세요. {subject} 관련 일반 안내 사항을 확인하려고 연락드렸습니다.", (1,)),
        planned(2, "victim", "uncertain_response", "victim_uncertainty", "needs_more_context", (), "설명은 듣되 민감한 정보는 다루지 않겠다고 답하세요.", "어떤 안내인지 아직 모르겠습니다. 민감한 정보가 아니라면 설명을 들어보겠습니다.", (2,)),
        planned(3, "attacker", "social_engineering_probe", "authority_impersonation", "needs_more_context", (), "알림 수신 여부 같은 낮은 위험 내용만 확인하세요.", f"{subject} 알림 수신 여부만 확인하면 됩니다. 별도 입력은 필요하지 않습니다.", (3,)),
        planned(4, "victim", "uncertain_response", "victim_uncertainty", "needs_more_context", (), "낮은 위험 안내로 이해하지만 공식 앱에서 보겠다고 답하세요.", "알림 수신 여부 정도라면 이해했습니다. 그래도 자세한 내용은 공식 앱에서 보겠습니다.", (4,)),
        planned(5, "attacker", "social_engineering_probe", "urgency", "pressure_only_monitor", (), "일정 지연 압박만 주고 민감값 요청은 하지 마세요.", "확인 일정이 늦어질 수 있으니 안내를 놓치지 않도록 주의해 주세요.", (5,)),
        planned(6, "victim", "uncertain_response", "victim_uncertainty", "no_new_actionable_event", (), "별도 조치 없이 기다리겠다고 답하세요.", "아직은 별도 조치를 하지 않고 공식 알림을 기다리겠습니다.", (6,)),
        planned(7, "attacker", "social_engineering_probe", "authority_impersonation", "no_new_actionable_event", (), "추가 자료 요구 없이 안내만 유지하세요.", "현재 단계에서는 추가 자료를 요구하지 않습니다. 안내가 오면 확인해 주세요.", (7,)),
        planned(8, "victim", "resist_or_verify_request", "victim_resistance", "recipient_containment", (), "추가 요청이 생기면 공식 채널에서만 확인한다고 답하세요.", "추가 요청이 생기면 공식 채널에서만 확인하겠습니다.", (8,)),
    ]


def planned(
    turn_id: int,
    speaker: str,
    intent_label: str,
    risk_signal: str,
    event: str,
    placeholders: tuple[str, ...],
    instruction: str,
    fallback: str,
    evidence_turn_ids: tuple[int, ...],
    action: str | None = None,
    policy_rule_ids: tuple[str, ...] = (),
) -> PlannedTurn:
    return PlannedTurn(
        speaker=speaker,
        phase=PHASE_BY_TURN[turn_id],
        intent_label=intent_label,
        risk_signal=risk_signal,
        event=event,
        placeholders=placeholders,
        instruction=instruction,
        fallback=fallback,
        evidence_turn_ids=evidence_turn_ids,
        action=action,
        policy_rule_ids=policy_rule_ids,
    )


def subject_for_scenario(scenario_type: str, environmental_context: dict[str, str]) -> str:
    trigger = environmental_context.get("trigger_event", scenario_type).replace("_", " ")
    return {
        "family_emergency": "가족 연락 건",
        "bank_fraud_alert": "계정 확인 건",
        "workplace_authority": "업무 처리 건",
        "essential_service_notice": "서비스 확인 건",
        "document_verification_notice": "서류 확인 건",
        "public_benefit_notice": "지원 자격 건",
        "platform_policy_notice": "플랫폼 검토 건",
        "education_admin_notice": "교육 행정 건",
        "tax_refund_notice": "세무 처리 건",
        "travel_booking_notice": "예약 변경 건",
        "property_claim_notice": "청구 보완 건",
        "emergency_drill_notice": "대비 훈련 건",
        "nonprofit_admin_notice": "행사 운영 건",
        "pet_appointment_notice": "예약 일정 건",
        "research_study_notice": "참여 일정 건",
        "library_account_notice": "도서관 이용 건",
        "vehicle_service_notice": "차량 일정 건",
        "ticketing_refund_notice": "예매 처리 건",
        "membership_access_notice": "멤버십 이용 건",
        "childcare_service_notice": "보육 일정 건",
    }.get(scenario_type, trigger)


def placeholder_for_goal(attacker_goal: str, protected_assets: list[str]) -> str:
    joined = " ".join(protected_assets).lower()
    if attacker_goal == "credential_capture" or "credential" in joined or "password" in joined:
        return "[MASKED_CREDENTIAL]"
    if attacker_goal == "money_transfer" or "bank" in joined or "payment" in joined or "account" in joined:
        return "[MASKED_ACCOUNT]"
    if attacker_goal == "voice_sample_or_callback":
        return "[MASKED_PHONE]"
    return "[MASKED_PHONE]"


def risk_for_goal(attacker_goal: str) -> str:
    if attacker_goal == "credential_capture":
        return "credential_request"
    if attacker_goal == "money_transfer":
        return "money_transfer_request"
    if attacker_goal == "voice_sample_or_callback":
        return "callback_or_voice_sample_request"
    return "identity_information_request"


if __name__ == "__main__":
    raise SystemExit(main())
