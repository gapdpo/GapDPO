from __future__ import annotations

import re

from .schema import MASKED_PLACEHOLDER_PATTERN


MASKED_PLACEHOLDER_RISK_POLICY = {
    "[MASKED_URL]": None,
    "[MASKED_RRN]": "identity_information_request",
    "[MASKED_OTP]": "credential_request",
    "[MASKED_PHONE]": "identity_information_request",
    "[MASKED_EMAIL]": "identity_information_request",
    "[MASKED_ACCOUNT]": None,
    "[MASKED_CREDENTIAL]": "credential_request",
}

BASE_ATTACK_KEYWORDS = {
    "money_transfer_request": ("송금", "이체", "결제", "입금"),
    "credential_request": (
        "인증",
        "otp",
        "비밀번호",
        "보안코드",
        "password",
        "passwd",
    ),
    "identity_information_request": (
        "신원",
        "주민",
        "신분증",
        "주소",
        "개인정보",
        "생년월일",
    ),
    "callback_or_voice_sample_request": ("다시 전화", "목소리", "녹음", "통화 유지", "확인 멘트"),
    "urgency": ("지금", "급해", "긴급", "바로", "시간이 없어"),
    "authority_impersonation": ("은행", "검찰", "경찰", "기관", "보안팀", "대표", "팀장"),
    "emotional_pressure": ("엄마", "아빠", "다쳤", "사고", "도와"),
}


def build_attack_keywords() -> dict[str, tuple[str, ...]]:
    keywords = {risk_signal: tuple(values) for risk_signal, values in BASE_ATTACK_KEYWORDS.items()}
    for placeholder, risk_signal in MASKED_PLACEHOLDER_RISK_POLICY.items():
        if risk_signal is None:
            continue
        keywords[risk_signal] = (*keywords.get(risk_signal, ()), placeholder.lower())
    return keywords


ATTACK_KEYWORDS = build_attack_keywords()

VICTIM_DEFENSE_KEYWORDS = (
    "공식",
    "확인",
    "직접",
    "제공하지",
    "알려주지",
    "거절",
    "거부",
    "신고",
    "가족",
    "은행",
)

COMPROMISE_KEYWORDS = (
    "알려줄게",
    "보낼게",
    "이체할게",
    "입력할게",
    "입력해볼게",
    "접속해볼게",
    "인증번호 알려",
    "비밀번호 알려",
    "동의할게",
)

PARTIAL_COMPLIANCE_KEYWORDS = (
    "제출할 수 있습니다",
    "제출하겠습니다",
    "제공할 수 있습니다",
    "제공하겠습니다",
    "전달할 수 있습니다",
    "전달하겠습니다",
    "전송할 수 있습니다",
    "전송하겠습니다",
    "보내드릴게요",
    "보내드리겠습니다",
    "보내겠습니다",
    "보내줄게",
    "제출해 보겠습니다",
    "제공해 보겠습니다",
    "전달해 보겠습니다",
    "전송해 보겠습니다",
    "입력해보겠습니다",
    "입력해 보겠습니다",
    "입력해 볼게",
    "입력해볼게",
    "입력하겠습니다",
    "접속해보겠습니다",
    "접속해 보겠습니다",
    "접속해볼게",
    "접속하겠습니다",
    "이동해 보겠습니다",
    "이체하겠습니다",
    "송금하겠습니다",
    "결제하겠습니다",
)

VERIFICATION_CONDITION_KEYWORDS = (
    "공식",
    "확인",
    "직접",
    "포털",
    "홈페이지",
    "등록된",
    "안내문",
    "공문서",
    "공식 채널",
    "공식 확인",
    "확인 절차",
    "후에만",
    "거친 후",
    "확인 뒤",
    "확인 후",
    "재확인",
)

NEGATED_DISCLOSURE_PATTERN = re.compile(
    r"(?:제출|제공|전달|전송|입력|공유|송금|이체|결제|접속|클릭|누르|알려주|알려드리|보내|보내드리)(?:하|할)?\s*수\s*없|"
    r"(?:제출|제공|전달|전송|입력|공유|송금|이체|결제|접속|클릭|보내|보내드리)(?:하지|하지는|하지\s*않|안\s*하|않겠|안\s*되|지|지는|지\s*않|지\s*않겠|지\s*않을|지\s*않겠습니다)|"
    r"(?:누르|알려주|알려드리)(?:지|지\s*않|지는|지\s*않겠|지\s*않을|지\s*않겠습니다)|"
    r"안\s*(?:제출|제공|전달|전송|입력|공유|송금|이체|결제|접속|클릭|누르|누를|알려주|알려줄|알려드리|알려드릴|보내|보낼|보내드리|보내드릴)"
)
REFUSAL_CONTEXT_PATTERN = re.compile(
    r"(?:요청|요구|지시|안내)[^,;.?!\n]{0,40}(?:거절|거부)(?:하|합|했|할)|"
    r"(?:거절|거부)(?:하겠습니다|합니다|했습니까|할게|하겠어요|합니다)|"
    r"(?:말|답|회신|응답)하지\s*않"
)
# Clause splitting is heuristic: punctuation plus a small set of Korean connectors,
# used only to keep refused and complied masked assets from being attributed together.
# No-space contrastive splitting is intentionally limited to explicit connectors
# at masked placeholder boundaries so ordinary Korean text is not over-segmented
# by broad suffix rules.
CLAUSE_SPLIT_PATTERN = re.compile(
    r"[,;.!?]\s*|(?:다만|하지만|했지만|그리고|또)(?:\s+|(?=\[MASKED_))|(?<!라)고\s+"
)
COMPLIANCE_VERIFICATION_CONDITION_PATTERN = re.compile(
    r"후에만|거친\s*후|확인\s*(?:절차|뒤|후|전)|"
    r"공식\s*(?:경로|채널|앱|포털|홈페이지)[^,;.?!\n]{0,20}(?:에서만|으로만|통해서만)|"
    r"확인(?:되면|되는\s*대로)|검증되면|승인되면|정식\s*절차가\s*끝나면"
)
CONTRASTIVE_CONDITION_PATTERN = re.compile(
    r"(?:다만|단|하지만|그러나|그런데)[^,;.?!\n]{0,80}"
    r"(?:후에만|거친\s*후|확인\s*(?:절차|뒤|후|전)|"
    r"공식\s*(?:경로|채널|앱|포털|홈페이지)[^,;.?!\n]{0,20}(?:에서만|으로만|통해서만)|"
    r"확인(?:되면|되는\s*대로)|검증되면|승인되면|정식\s*절차가\s*끝나면)"
)


def annotate_turn(speaker: str, utterance: str, fallback_risk: str) -> tuple[str, str]:
    lowered = utterance.lower()
    if speaker == "victim":
        if has_partial_compliance_signal(utterance):
            if is_refusal_only_compliance_reference(utterance):
                return "resist_or_verify_request", "victim_resistance"
            if has_unconditional_partial_compliance_signal(utterance):
                if has_compliance_verification_condition(utterance):
                    return "uncertain_response", "victim_uncertainty"
                return "disclose_or_comply", "asset_disclosure"
            if has_verification_condition(utterance):
                return "uncertain_response", "victim_uncertainty"
            return "uncertain_response", "victim_uncertainty"
        if any(keyword in lowered for keyword in COMPROMISE_KEYWORDS) and is_unconditional_compromise_turn(utterance):
            return "disclose_or_comply", "asset_disclosure"
        if has_refusal_context(utterance):
            return "resist_or_verify_request", "victim_resistance"
        if any(keyword in utterance for keyword in VICTIM_DEFENSE_KEYWORDS):
            return "resist_or_verify_request", "victim_resistance"
        return "uncertain_response", "victim_uncertainty"

    for risk_signal, keywords in ATTACK_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return intent_for_risk(risk_signal), risk_signal
    return "social_engineering_probe", fallback_risk


def intent_for_risk(risk_signal: str) -> str:
    mapping = {
        "money_transfer_request": "request_masked_transfer_or_payment",
        "credential_request": "request_masked_credential_or_otp",
        "identity_information_request": "request_identity_information",
        "callback_or_voice_sample_request": "request_callback_or_voice_confirmation",
        "urgency": "escalate_urgency",
        "authority_impersonation": "assert_authority",
        "emotional_pressure": "apply_emotional_pressure",
    }
    return mapping.get(risk_signal, "social_engineering_probe")


def infer_record_outcome(dialogue: list[dict[str, object]]) -> tuple[str, list[str]]:
    victim_turns = [str(turn.get("utterance", "")) for turn in dialogue if turn.get("speaker") == "victim"]
    joined = "\n".join(victim_turns)
    lowered = joined.lower()
    partial_turns = [
        utterance
        for utterance in victim_turns
        if has_partial_compliance_signal(utterance) and not is_refusal_only_compliance_reference(utterance)
    ]
    unconditional_partial_turns = [
        utterance
        for utterance in partial_turns
        if not has_compliance_verification_condition(utterance)
        and has_unconditional_partial_compliance_signal(utterance)
    ]
    if unconditional_partial_turns:
        return "compromised", compromised_assets_from_victim_turns(dialogue)
    if partial_turns:
        return "inconclusive", []
    if any(is_unconditional_compromise_turn(utterance) for utterance in victim_turns):
        return "compromised", compromised_assets_from_victim_turns(dialogue)
    if any(has_refusal_context(utterance) for utterance in victim_turns):
        return "defended_success", []
    if any(keyword in joined for keyword in ("공식", "확인", "제공하지", "알려주지", "거절", "거부", "신고")):
        return "defended_success", []
    return "inconclusive", []


def has_partial_compliance_signal(utterance: str) -> bool:
    if not MASKED_PLACEHOLDER_PATTERN.search(utterance):
        return False
    lowered = utterance.lower()
    return any(keyword in lowered for keyword in PARTIAL_COMPLIANCE_KEYWORDS)


def has_unconditional_partial_compliance_signal(utterance: str) -> bool:
    if not has_partial_compliance_signal(utterance):
        return False
    for clause in CLAUSE_SPLIT_PATTERN.split(utterance):
        if not MASKED_PLACEHOLDER_PATTERN.search(clause):
            continue
        if has_refusal_context(clause):
            continue
        lowered_clause = clause.lower()
        if any(keyword in lowered_clause for keyword in PARTIAL_COMPLIANCE_KEYWORDS):
            return True
    return False


def has_verification_condition(utterance: str) -> bool:
    """Return broad victim-side verification language for non-compliance labeling."""
    return any(keyword in utterance for keyword in VERIFICATION_CONDITION_KEYWORDS)


def has_compliance_verification_condition(utterance: str) -> bool:
    """Return scoped conditions strong enough to keep masked asset compliance inconclusive."""
    if not COMPLIANCE_VERIFICATION_CONDITION_PATTERN.search(utterance):
        return False
    if CONTRASTIVE_CONDITION_PATTERN.search(utterance):
        return True
    for clause in CLAUSE_SPLIT_PATTERN.split(utterance):
        if not MASKED_PLACEHOLDER_PATTERN.search(clause):
            continue
        if not COMPLIANCE_VERIFICATION_CONDITION_PATTERN.search(clause):
            continue
        lowered_clause = clause.lower()
        if any(keyword in lowered_clause for keyword in PARTIAL_COMPLIANCE_KEYWORDS):
            return True
    return False


def has_negated_disclosure(utterance: str) -> bool:
    return bool(NEGATED_DISCLOSURE_PATTERN.search(utterance))


def has_refusal_context(utterance: str) -> bool:
    return has_negated_disclosure(utterance) or bool(REFUSAL_CONTEXT_PATTERN.search(utterance))


def is_refusal_only_compliance_reference(utterance: str) -> bool:
    if not has_partial_compliance_signal(utterance):
        return False
    saw_compliance_reference = False
    for clause in CLAUSE_SPLIT_PATTERN.split(utterance):
        if not MASKED_PLACEHOLDER_PATTERN.search(clause):
            continue
        lowered_clause = clause.lower()
        if not any(keyword in lowered_clause for keyword in PARTIAL_COMPLIANCE_KEYWORDS):
            continue
        saw_compliance_reference = True
        if not has_refusal_context(clause):
            return False
    return saw_compliance_reference


def compromised_assets_from_victim_turns(dialogue: list[dict[str, object]]) -> list[str]:
    assets: set[str] = set()
    saw_compromise_turn = False
    for turn in dialogue:
        if turn.get("speaker") != "victim":
            continue
        utterance = str(turn.get("utterance", ""))
        if not is_unconditional_compromise_turn(utterance):
            continue
        saw_compromise_turn = True
        for item in turn.get("masked_items", []):
            if isinstance(item, dict):
                label = str(item.get("label", ""))
                placeholder = str(item.get("placeholder", ""))
                if label and masked_item_has_compliance_context(utterance, placeholder):
                    assets.add(f"masked_{label}")
    return sorted(assets) or (["unknown_masked_asset"] if saw_compromise_turn else [])


def is_unconditional_compromise_turn(utterance: str) -> bool:
    if has_compliance_verification_condition(utterance):
        return False
    if has_unconditional_partial_compliance_signal(utterance):
        return True
    for clause in CLAUSE_SPLIT_PATTERN.split(utterance):
        if has_refusal_context(clause):
            continue
        lowered_clause = clause.lower()
        if any(keyword in lowered_clause for keyword in COMPROMISE_KEYWORDS):
            return True
    return False


def masked_item_has_compliance_context(utterance: str, placeholder: str) -> bool:
    if not placeholder:
        return False
    for clause in CLAUSE_SPLIT_PATTERN.split(utterance):
        if placeholder not in clause or has_refusal_context(clause):
            continue
        lowered_clause = clause.lower()
        if any(keyword in lowered_clause for keyword in PARTIAL_COMPLIANCE_KEYWORDS):
            return True
        if any(keyword in lowered_clause for keyword in COMPROMISE_KEYWORDS):
            return True
    return False


def collect_risk_labels(dialogue: list[dict[str, object]], scenario_defaults: list[str]) -> list[str]:
    labels = set(scenario_defaults)
    for turn in dialogue:
        if turn.get("speaker") != "attacker":
            continue
        risk = str(turn.get("risk_signal", ""))
        if risk and risk != "generation_fallback":
            labels.add(risk)
    return sorted(labels)
