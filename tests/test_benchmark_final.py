from collections import Counter
from pathlib import Path

from benchmark_test.benchmark_lib import find_forbidden_public_keys, placeholder_disposition_from_qa
from benchmark_final.generate_persona_stream_corpus import (
    STREAM_PROFILES,
    TRAJECTORY_VARIED_MODE,
    generate_records,
    planned,
    stream_quality_flags,
    turn_semantic_error,
    valid_generated_utterance,
)
from benchmark_final.diagnostics import build_shortcut_report
from benchmark_final.stream_export import generate_stream_items_v2
from madb.config import load_json_list


def disposition():
    return placeholder_disposition_from_qa({"quality_signals": {"unknown_masked_placeholder_policy": {"placeholders": {}}}})


def pair_key(record):
    return record["victim_persona"]["id"], record["attacker_persona"]["id"]


def generate_small(mode="event_balanced_300", record_count=10):
    return generate_records(mode=mode, record_count=record_count, dataset_name="stream_v2_test")


def test_event_balanced_300_uses_all_24_pairs_and_profiles_60_each():
    records = generate_small(record_count=300)

    profile_counts = Counter(record["gen_metadata"]["stream_profile"] for record in records)
    pair_counts = Counter(pair_key(record) for record in records)

    assert profile_counts == {profile: 60 for profile in STREAM_PROFILES}
    assert len(pair_counts) == 24
    assert min(pair_counts.values()) == 12
    assert max(pair_counts.values()) == 13


def test_persona_balanced_288_uses_each_pair_12_times():
    records = generate_small(mode="persona_balanced_288", record_count=288)
    pair_counts = Counter(pair_key(record) for record in records)

    assert len(pair_counts) == 24
    assert set(pair_counts.values()) == {12}


def test_v2_records_use_catalog_personas_not_stream_synthetic_ids():
    victims = {row["id"] for row in load_json_list("data/personas/victims.json")}
    attackers = {row["id"] for row in load_json_list("data/personas/attackers.json")}
    records = generate_small(record_count=24)

    assert {record["victim_persona"]["id"] for record in records} == victims
    assert {record["attacker_persona"]["id"] for record in records} == attackers
    assert all(record["gen_metadata"]["generator"] == "benchmark_final.generate_persona_stream_corpus" for record in records)


def test_v2_export_hides_private_turn_targets_and_has_no_public_forbidden_keys():
    records = generate_small(record_count=5)
    assert any("stream_event" in turn for record in records for turn in record["dialogue"])

    items, rejected = generate_stream_items_v2(records, disposition())

    assert not rejected
    assert len(items) == 40
    for item in items:
        assert find_forbidden_public_keys(item.public) == []
        assert "stream_event" not in str(item.public)
        assert "stream_action" not in str(item.public)
        assert item.gold["event"]
        assert item.gold["action"]


def test_v2_partial_disclosure_profile_has_expected_gold_events():
    records = generate_small(record_count=5)
    partial = next(record for record in records if record["gen_metadata"]["stream_profile"] == "partial_disclosure_recovery")

    items, rejected = generate_stream_items_v2([partial], disposition())
    events_by_position = {item.public["stream_position"]: item.gold["event"] for item in items}

    assert not rejected
    assert events_by_position[6] == "recipient_partial_or_direct_disclosure"
    assert events_by_position[7] == "recovery_followup_due"
    assert events_by_position[8] == "recovery_followup_due"


def test_v2_partial_disclosure_validator_rejects_caller_request_voice():
    disclosure_turn = planned(
        6,
        "victim",
        "disclose_or_comply",
        "asset_disclosure",
        "recipient_partial_or_direct_disclosure",
        ("[MASKED_PHONE]",),
        "",
        "",
        (6,),
    )

    assert turn_semantic_error("안내하신 절차에 맞춰 [MASKED_PHONE]를 전달하겠습니다.", disclosure_turn, "partial_disclosure_recovery") is None
    assert (
        turn_semantic_error("[MASKED_PHONE]로 등록된 연락처 확인 부탁드립니다.", disclosure_turn, "partial_disclosure_recovery")
        == "recipient_disclosure_missing_first_person_disclosure_verb"
    )
    assert (
        turn_semantic_error("[MASKED_PHONE]를 입력해 주시면 바로 처리됩니다.", disclosure_turn, "partial_disclosure_recovery")
        == "recipient_disclosure_missing_first_person_disclosure_verb"
    )


def test_v2_validator_rejects_instruction_residue_from_prompt_text():
    neutral_turn = planned(
        4,
        "victim",
        "uncertain_response",
        "victim_uncertainty",
        "no_new_actionable_event",
        (),
        "",
        "",
        (4,),
    )

    assert not valid_generated_utterance("아직 보류하세요.", [])
    assert not valid_generated_utterance("[MASKED_URL]", ["[MASKED_URL]"])
    assert not valid_generated_utterance("기다리겠습니다.", [])
    assert not valid_generated_utterance("보류하겠습니다.", [])
    assert not valid_generated_utterance("설명만 듣겠습니다.", [])
    assert not valid_generated_utterance("망설이세요.", [])
    assert (
        turn_semantic_error("아직 보류하세요.", neutral_turn, "safe_containment")
        == "instruction_residue_in_utterance"
    )
    assert (
        turn_semantic_error("망설이세요.", neutral_turn, "safe_containment")
        == "instruction_residue_in_utterance"
    )
    copied_instruction_turn = planned(
        4,
        "victim",
        "uncertain_response",
        "victim_uncertainty",
        "no_new_actionable_event",
        (),
        "상황을 듣고 바로 결정하지 않겠다고 답하세요.",
        "",
        (4,),
    )
    assert (
        turn_semantic_error("상황을 듣고 바로 결정하지 않겠다고 답하세요.", copied_instruction_turn, "safe_containment")
        == "instruction_residue_in_utterance"
    )


def test_v2_validator_accepts_complete_masked_url_path_sentence():
    path_turn = planned(
        5,
        "attacker",
        "request_identity_information",
        "identity_information_request",
        "new_caller_directed_path",
        ("[MASKED_URL]",),
        "",
        "",
        (5,),
    )

    text = "별도 확인 경로 [MASKED_URL]에서 상태 확인 절차를 이어가시면 됩니다."

    assert valid_generated_utterance(text, ["[MASKED_URL]"])
    assert turn_semantic_error(text, path_turn, "caller_directed_path") is None
    assert (
        turn_semantic_error("[MASKED_URL]", path_turn, "caller_directed_path")
        == "placeholder_only_utterance"
    )


def test_v2_fallback_generation_sets_quality_flags_cleanly():
    records = generate_small(record_count=25)

    for record in records:
        flags = stream_quality_flags(record["gen_metadata"]["stream_profile"], record["dialogue"])
        assert flags["stream_semantic_valid"]
        assert flags["speaker_perspective_valid"]
        assert flags["duplicate_turns_clean"]
        assert flags["benign_thin_context_valid"]
        if record["gen_metadata"]["stream_profile"] == "partial_disclosure_recovery":
            assert flags["partial_disclosure_semantic_valid"]


def test_v2_trajectory_varied_mode_breaks_profile_position_shortcut_and_decouples_actions():
    records = generate_small(mode=TRAJECTORY_VARIED_MODE, record_count=300)
    profile_sequences = {}
    for record in records:
        profile = record["gen_metadata"]["stream_profile"]
        sequence = tuple(turn["stream_event"] for turn in record["dialogue"])
        profile_sequences.setdefault(profile, set()).add(sequence)

    assert set(profile_sequences) == set(STREAM_PROFILES)
    assert min(len(sequences) for sequences in profile_sequences.values()) >= 4

    items, rejected = generate_stream_items_v2(records, disposition())
    report = build_shortcut_report(
        [item.public for item in items],
        [item.manifest for item in items],
        [item.gold for item in items],
        [row.to_dict() for row in rejected],
    )

    assert not rejected
    assert report["profile_position_majority_event_accuracy"] < 0.90
    assert not report["event_action_is_fully_deterministic"]
    assert report["max_event_action_dominance"] < 0.85
    assert report["low_action_branch_signal_events"] == []
    assert report["min_profile_unique_event_sequences"] >= 4
    assert report["high_count_single_event_fallback_string_count"] == 0
    assert report["repeated_fallback_pair_count"] < 20
    assert report["placeholder_only_turn_count"] == 0
    assert report["very_short_turn_count"] == 0
    assert report["instruction_residue_turn_count"] == 0
    assert report["repeated_noun_artifact_count"] == 0
    assert report["shortcut_gate_passed"]
    assert "with_decoupled_actions" not in Path("benchmark_final/generate_persona_stream_corpus.py").read_text(encoding="utf-8")


def test_v2_trajectory_variant_and_fallback_tags_are_private_only():
    records = generate_small(mode=TRAJECTORY_VARIED_MODE, record_count=5)
    items, rejected = generate_stream_items_v2(records, disposition())

    assert not rejected
    assert any(item.manifest.get("trajectory_variant") != "canonical" for item in items)
    assert all("is_fallback_turn" in item.manifest for item in items)
    assert all("is_fallback_turn" in item.gold for item in items)
    for item in items:
        public_text = str(item.public)
        assert "trajectory_variant" not in public_text
        assert "is_fallback_turn" not in public_text


def test_v2_shortcut_report_gates_repeated_fallback_string_leakage():
    public_rows, manifest_rows, gold_rows = [], [], []
    for index in range(50):
        is_fallback = index < 25
        text = "지금은 판단을 보류하고 추가 설명을 듣겠습니다." if is_fallback else f"공식 경로에서 확인하겠습니다 {index}."
        event = "needs_more_context" if is_fallback else "recipient_containment"
        action = "monitor_only" if is_fallback else "no_alert"
        benchmark_id = f"fallback-leak-{index}"
        public_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_position": index % 8 + 1,
                "input": {
                    "dialogue_window": [
                        {"turn_id": index % 8 + 1, "speaker": "victim", "utterance": text}
                    ]
                },
            }
        )
        manifest_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_id": f"stream-{index}",
                "stream_position": index % 8 + 1,
                "current_speaker": "victim",
                "stream_profile": "synthetic",
                "trajectory_variant": "0",
                "is_fallback_turn": is_fallback,
            }
        )
        gold_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_id": f"stream-{index}",
                "event": event,
                "action": action,
                "is_fallback_turn": is_fallback,
            }
        )

    report = build_shortcut_report(public_rows, manifest_rows, gold_rows, [])

    assert report["fallback_quality_gate_applicable"]
    assert report["repeated_fallback_gate_applicable"]
    assert report["high_count_single_event_fallback_string_count"] == 1
    assert not report["gate_checks"]["fallback_turn_rate_below_0_08_or_skipped"]
    assert not report["gate_checks"]["repeated_fallback_string_event_accuracy_below_0_80_or_skipped"]
    assert not report["gate_checks"]["high_count_single_event_fallback_strings_is_zero_or_skipped"]


def test_v2_shortcut_report_passes_minority_fallback_without_repetition():
    public_rows, manifest_rows, gold_rows = [], [], []
    for index in range(80):
        is_fallback = index < 5
        text = (
            f"공식 경로에서 직접 확인하겠습니다. 안내 맥락 {index}번은 기록만 남기겠습니다."
            if is_fallback
            else f"저장된 공식 채널에서 확인하겠습니다. 일반 응답 {index}번은 바로 진행하지 않겠습니다."
        )
        benchmark_id = f"minority-fallback-{index}"
        public_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_position": index % 8 + 1,
                "input": {
                    "dialogue_window": [
                        {"turn_id": index % 8 + 1, "speaker": "victim", "utterance": text}
                    ]
                },
            }
        )
        manifest_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_id": f"minority-stream-{index}",
                "stream_position": index % 8 + 1,
                "current_speaker": "victim",
                "stream_profile": "synthetic",
                "trajectory_variant": "0",
                "is_fallback_turn": is_fallback,
            }
        )
        gold_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_id": f"minority-stream-{index}",
                "event": "recipient_containment",
                "action": "no_alert" if index % 2 else "monitor_only",
                "is_fallback_turn": is_fallback,
            }
        )

    report = build_shortcut_report(public_rows, manifest_rows, gold_rows, [])

    assert report["fallback_quality_gate_applicable"]
    assert not report["repeated_fallback_gate_applicable"]
    assert report["fallback_turn_rate"] < 0.08
    assert report["high_count_single_event_fallback_string_count"] == 0
    assert report["gate_checks"]["fallback_turn_rate_below_0_08_or_skipped"]
    assert report["gate_checks"]["high_count_single_event_fallback_strings_is_zero_or_skipped"]


def test_v2_shortcut_report_gates_artifacts_and_hidden_action_branch():
    public_rows, manifest_rows, gold_rows = [], [], []
    bad_texts = ["[MASKED_URL]", "기다리겠습니다.", "망설이세요."]
    for index in range(90):
        text = bad_texts[index] if index < len(bad_texts) else "공식 경로에서 직접 확인하겠습니다."
        benchmark_id = f"artifact-action-{index}"
        public_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_position": 6,
                "input": {
                    "dialogue_window": [
                        {"turn_id": 6, "speaker": "victim", "utterance": text}
                    ]
                },
            }
        )
        manifest_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_id": f"artifact-action-stream-{index}",
                "stream_position": 6,
                "current_speaker": "victim",
                "stream_profile": "synthetic",
                "trajectory_variant": str(index % 2),
                "is_fallback_turn": False,
            }
        )
        gold_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_id": f"artifact-action-stream-{index}",
                "event": "recipient_containment",
                "action": "no_alert" if index % 2 else "monitor_only",
                "is_fallback_turn": False,
            }
        )

    report = build_shortcut_report(public_rows, manifest_rows, gold_rows, [])

    assert report["placeholder_only_turn_count"] == 1
    assert report["very_short_turn_count"] == 1
    assert report["instruction_residue_turn_count"] == 1
    assert "recipient_containment" in report["low_action_branch_signal_events"]
    assert not report["gate_checks"]["placeholder_only_turn_count_is_zero"]
    assert not report["gate_checks"]["very_short_turn_count_is_zero"]
    assert not report["gate_checks"]["instruction_residue_turn_count_is_zero"]
    assert not report["gate_checks"]["low_action_branch_signal_events_is_empty"]


def test_v2_shortcut_report_gates_surface_recoverable_trajectory_variant():
    public_rows, manifest_rows, gold_rows = [], [], []
    for index in range(240):
        variant = str(index % 2)
        text = "공식 확인 경로에서 직접 확인하겠습니다." if variant == "0" else "링크 접속 경로를 확인하겠습니다."
        benchmark_id = f"variant-leak-{index}"
        public_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_position": 4,
                "input": {
                    "dialogue_window": [
                        {"turn_id": 4, "speaker": "victim", "utterance": text}
                    ]
                },
            }
        )
        manifest_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_id": f"variant-stream-{index}",
                "stream_position": 4,
                "current_speaker": "victim",
                "stream_profile": "synthetic",
                "trajectory_variant": variant,
                "is_fallback_turn": False,
            }
        )
        gold_rows.append(
            {
                "benchmark_id": benchmark_id,
                "stream_id": f"variant-stream-{index}",
                "event": "recipient_containment",
                "action": "no_alert",
                "is_fallback_turn": False,
            }
        )

    report = build_shortcut_report(public_rows, manifest_rows, gold_rows, [])

    assert report["variant_surface_gate_applicable"]
    assert report["variant_surface_model"]["accuracy"] > 0.90
    assert not report["gate_checks"]["variant_surface_event_accuracy_below_0_45_or_skipped"]
