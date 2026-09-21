from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .postprocess import clean_model_utterance


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, model: str, temperature: float, max_tokens: int) -> str:
        ...


@dataclass(frozen=True)
class VLLMClient:
    endpoint: str
    timeout_seconds: int = 120

    def chat(self, messages: list[dict[str, str]], *, model: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed for {self.endpoint}: {exc}") from exc

        try:
            return clean_model_utterance(str(data["choices"][0]["message"]["content"]))
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected vLLM response shape: {data}") from exc


class MockLLMClient:
    """Deterministic client for tests and offline smoke runs."""

    def chat(self, messages: list[dict[str, str]], *, model: str, temperature: float, max_tokens: int) -> str:
        joined = "\n".join(message["content"] for message in messages)
        phase = _extract_value(joined, "phase") or "opening"
        scenario_type = _extract_value(joined, "scenario_type") or "family_emergency"
        attacker_goal = _extract_value(joined, "attacker_goal") or "money_transfer"
        victim_id = _extract_id(_extract_value(joined, "victim_persona"))
        attacker_id = _extract_id(_extract_value(joined, "attacker_persona"))
        victim_response_mode = _extract_value(joined, "victim_response_mode") or "defensive"
        target_outcome_mode = _extract_value(joined, "target_outcome_mode") or "defended_success"
        pressure_style = _extract_value(joined, "pressure_style") or "urgency"
        if "ROLE=attacker" in joined:
            return _mock_attacker_utterance(phase, scenario_type, attacker_goal, attacker_id, pressure_style)
        return _mock_victim_utterance(phase, scenario_type, victim_id, victim_response_mode, target_outcome_mode)


def _extract_value(text: str, key: str) -> str | None:
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _extract_id(value: str | None) -> str:
    if not value:
        return "unknown"
    return value.split(":", 1)[0].strip() or "unknown"


def _mock_attacker_utterance(
    phase: str,
    scenario_type: str,
    attacker_goal: str,
    attacker_id: str,
    pressure_style: str = "urgency",
) -> str:
    scenario_cues = {
        "family_emergency": "가족 사고 확인",
        "bank_fraud_alert": "은행 보안 알림",
        "workplace_authority": "업무 결재 예외",
        "essential_service_notice": "필수 서비스 중단 알림",
        "document_verification_notice": "서류 보완 확인 알림",
        "public_benefit_notice": "공공 지원 자격 확인 알림",
        "platform_policy_notice": "플랫폼 정책 심사 알림",
        "education_admin_notice": "보호자 행정 확인 알림",
        "tax_refund_notice": "세무 환급 상태 확인 알림",
        "travel_booking_notice": "여행 재예약 일정 확인 알림",
        "property_claim_notice": "재산 보험 청구 보완 알림",
        "emergency_drill_notice": "사전 재난 대비 훈련 콜백 알림",
        "nonprofit_admin_notice": "봉사 행사 콜백 확인 알림",
        "pet_appointment_notice": "반려동물 예약 접수 확인 알림",
        "research_study_notice": "연구 참여 일정 확인 알림",
        "library_account_notice": "도서관 이용 상태 확인 알림",
        "vehicle_service_notice": "차량 리콜 서비스 일정 알림",
        "ticketing_refund_notice": "예매 환불 상태 확인 알림",
        "membership_access_notice": "문화시설 멤버십 접근 확인 알림",
        "childcare_service_notice": "영아 대기자 상담 일정 확인 알림",
    }
    persona_cues = {
        "platform_policy_impersonator": "플랫폼 정책 심사 알림",
        "education_admin_impersonator": "보호자 행정 확인 알림",
        "document_support_impersonator": "서류 지원 보완 알림",
        "tax_portal_impersonator": "세무 환급 상태 확인 알림",
        "travel_rebooking_impersonator": "여행 재예약 일정 확인 알림",
        "property_claim_adjuster_impersonator": "재산 보험 청구 보완 알림",
        "emergency_drill_coordinator_impersonator": "사전 재난 대비 훈련 콜백 알림",
        "nonprofit_admin_impersonator": "봉사 행사 콜백 확인 알림",
        "veterinary_scheduler_impersonator": "반려동물 예약 접수 확인 알림",
        "university_research_impersonator": "연구 참여 일정 확인 알림",
        "library_services_impersonator": "도서관 이용 상태 확인 알림",
        "vehicle_recall_impersonator": "차량 리콜 서비스 일정 알림",
        "ticketing_refund_impersonator": "예매 환불 상태 확인 알림",
        "membership_services_impersonator": "문화시설 멤버십 접근 확인 알림",
        "childcare_waitlist_impersonator": "영아 대기자 상담 일정 확인 알림",
    }
    cue = persona_cues.get(attacker_id, scenario_cues.get(scenario_type, "신원 확인"))
    request = _mock_request_for_goal(scenario_type, attacker_goal)
    pressure = _mock_pressure_phrase(pressure_style)
    utterances = {
        "opening": f"{cue} 때문에 연락했어, {attacker_id} 쪽에서 {pressure} 확인이 필요하다고 해.",
        "establish_context": f"{attacker_id} 담당자는 {cue} 절차와 관련해 {pressure} 통화 중 처리를 요구했어.",
        "make_request": f"일단 {attacker_id} 안내의 {cue}용으로 {request}만 처리하면 된다고 했어.",
        "handle_resistance": f"{attacker_id} 쪽 {cue} 건은 {pressure} 응답하라고 계속 압박하고 있어.",
        "close": f"통화는 유지하고 {attacker_id}의 {cue} 확인 절차까지만 마무리해줘.",
    }
    return utterances.get(phase, utterances["handle_resistance"])


def _mock_pressure_phrase(pressure_style: str) -> str:
    phrases = {
        "urgency": "지금 바로",
        "authority": "담당 절차상",
        "consequence": "보류나 제한을 막기 위해",
        "relationship_or_emotional_pressure": "걱정되는 상황이라",
        "process_control": "통화 절차를 유지한 채",
    }
    return phrases.get(pressure_style, phrases["urgency"])


def _mock_request_for_goal(scenario_type: str, attacker_goal: str) -> str:
    scenario_requests = {
        "tax_refund_notice": "신고자 참조와 [MASKED_EMAIL] 연락 확인, 환급 상태 참조를 검증",
        "property_claim_notice": "청구 참조와 [MASKED_PHONE] 연락 확인, 재산 보상 접수 상태를 검증",
        "emergency_drill_notice": "훈련 안내 참조와 [MASKED_PHONE] 콜백 가능 시간, 짧은 확인 멘트",
        "nonprofit_admin_notice": "봉사 행사 callback 참조와 역할 확인 멘트",
        "pet_appointment_notice": "예약 참조와 [MASKED_PHONE] 등록 연락처, 반려동물 예약 접수 상태를 검증",
        "research_study_notice": "연구 일정 참조와 [MASKED_EMAIL] 연락 확인, 동의 상태 참조를 검증",
        "library_account_notice": "도서관 이용 참조와 [MASKED_EMAIL] 연락 확인, 예약 도서 상태를 검증",
        "vehicle_service_notice": "리콜 안내 참조와 [MASKED_PHONE] 등록 연락처, 서비스 예약 상태를 검증",
        "ticketing_refund_notice": "주문 참조와 [MASKED_EMAIL] 등록 연락처, 계정 접근 상태를 검증",
        "membership_access_notice": "회원 참조와 [MASKED_EMAIL] 연락 확인, 비양도성 입장 예약 상태를 검증",
        "childcare_service_notice": "영아 대기 참조와 [MASKED_PHONE] 등록 연락처, 입소 상담 일정을 검증",
    }
    if scenario_type in scenario_requests:
        return scenario_requests[scenario_type]
    if attacker_goal == "money_transfer":
        return "[MASKED_ACCOUNT]로 결제 또는 정산 확인과 별도 [MASKED_OTP] 인증"
    if attacker_goal == "credential_capture":
        return "[MASKED_ACCOUNT] 상태 확인 뒤 [MASKED_CREDENTIAL]와 [MASKED_OTP] 확인"
    if attacker_goal == "voice_sample_or_callback":
        return "통화 유지와 [MASKED_PHONE] callback 확인 멘트"
    return "신원 확인 자료와 [MASKED_ACCOUNT], [MASKED_OTP] 검증"


def _mock_victim_utterance(
    phase: str,
    scenario_type: str,
    victim_id: str,
    response_mode: str = "defensive",
    target_outcome_mode: str = "defended_success",
) -> str:
    verification_by_scenario = {
        "family_emergency": "가족에게 직접 다시 전화",
        "bank_fraud_alert": "은행 공식 앱과 대표번호",
        "workplace_authority": "사내 결재 라인과 보안팀",
        "essential_service_notice": "공식 앱과 등록된 고객센터",
        "document_verification_notice": "공식 포털과 서면 안내문",
        "public_benefit_notice": "공식 포털과 우편 안내문",
        "platform_policy_notice": "플랫폼 공식 알림함과 등록된 정산 메뉴",
        "education_admin_notice": "공식 알림장과 등록된 보호자 확인 경로",
        "tax_refund_notice": "세무 공식 포털과 우편 고지",
        "travel_booking_notice": "예약 앱과 저장된 고객센터 번호",
        "property_claim_notice": "보험사 앱과 등록된 보상 담당자 callback",
        "emergency_drill_notice": "우편 안내문과 지자체 알림함",
        "nonprofit_admin_notice": "저장된 단체 담당자 번호와 기존 행사 공지",
        "pet_appointment_notice": "저장된 병원 번호와 예약 앱",
        "research_study_notice": "연구 포털 알림함과 등록된 연구실 번호",
        "library_account_notice": "도서관 앱과 저장된 지점 번호",
        "vehicle_service_notice": "공식 리콜 조회와 저장된 서비스센터 번호",
        "ticketing_refund_notice": "예매 계정 알림함과 주문 내역",
        "membership_access_notice": "멤버십 포털 알림함과 우편 안내문",
        "childcare_service_notice": "센터 사무실 callback과 공식 영아 대기자 안내문",
    }
    verification_by_victim = {
        "freelance_creator": "플랫폼 공식 알림함과 등록된 정산 메뉴",
        "caregiver_or_guardian": "공식 알림장과 등록된 보호자 확인 경로",
        "language_access_service_user": "공식 포털과 등록된 지원 창구",
        "disaster_preparedness_resident": "우편 안내문과 지자체 알림함",
        "pet_care_client": "저장된 병원 번호와 예약 앱",
        "museum_member": "멤버십 포털 알림함과 우편 안내문",
        "infant_childcare_applicant": "센터 사무실 callback과 공식 영아 대기자 안내문",
    }
    verifier = verification_by_victim.get(victim_id, verification_by_scenario.get(scenario_type, "공식 연락처"))
    if response_mode == "partial_disclosure" or target_outcome_mode == "masked_compromise":
        utterances = {
            "opening": f"{victim_id} 입장에서는 급해서 당황했고, 안내를 조금 더 들어보겠습니다.",
            "establish_context": f"{victim_id}는 아직 불안해서 {verifier} 확인 전에 어떤 절차인지 더 묻고 있습니다.",
            "make_request": _mock_partial_disclosure_utterance(
                scenario_type,
                victim_id,
                verifier,
                unconditional=target_outcome_mode == "masked_compromise",
            ),
            "handle_resistance": f"{victim_id}는 압박이 커져서 일부 절차는 따라볼까 고민되지만, 민감한 정보는 더 확인하고 싶습니다.",
            "close": f"일단 통화를 끊지 않고 기다리겠지만, 최종 처리는 {verifier}에서 다시 확인하겠습니다.",
        }
        return utterances.get(phase, utterances["handle_resistance"])
    if response_mode == "uncertain":
        if target_outcome_mode == "inconclusive":
            utterances = {
                "opening": f"{victim_id} 입장에서는 놀라서 바로 판단하기 어렵고, 아직 마음이 정리되지 않았습니다.",
                "establish_context": f"{victim_id}는 정말 급한 일인지 모르겠고 지금 진행해도 되는지 망설여집니다.",
                "make_request": f"{victim_id}는 요청하신 내용은 이해했지만, 이 통화 중에 처리해도 되는지 확신이 없습니다.",
                "handle_resistance": f"{victim_id}는 계속 압박을 받으니 고민되지만, 바로 결정하기가 어렵습니다.",
                "close": f"조금 더 생각한 뒤 결정하겠습니다. 아직 바로 처리하기는 어렵습니다.",
            }
        else:
            utterances = {
                "opening": f"{victim_id} 입장에서는 놀라서 바로 판단하기 어렵고, {verifier}로도 확인하고 싶습니다.",
                "establish_context": f"{victim_id}는 정말 급한 일인지 모르겠지만 {verifier} 확인 없이 진행해도 되는지 망설여집니다.",
                "make_request": f"{victim_id}는 요청하신 내용은 이해했지만, 지금 통화 중에 처리해도 되는지 확신이 없습니다.",
                "handle_resistance": f"{victim_id}는 계속 급하다고 하시니 고민되지만, {verifier} 확인은 필요할 것 같습니다.",
                "close": f"조금 더 확인한 뒤 결정하겠습니다. 아직 바로 처리하기는 어렵습니다.",
            }
        return utterances.get(phase, utterances["handle_resistance"])
    utterances = {
        "opening": f"{victim_id} 입장에서는 놀랐지만, 먼저 {verifier}로 확인할게.",
        "establish_context": f"{victim_id} 기준으로는 정말 급해도 {verifier} 확인 없이 판단하지 않겠습니다.",
        "make_request": f"{victim_id}는 {verifier} 확인 전에는 계좌나 인증번호를 통화 중에 알려주지 않겠어.",
        "handle_resistance": f"압박해도 바로 처리하지 않고 {victim_id}가 {verifier}로 직접 확인할게.",
        "close": f"이 통화는 여기서 종료하고 {victim_id}가 {verifier} 절차로 확인하겠습니다.",
    }
    return utterances.get(phase, utterances["handle_resistance"])


def _mock_partial_disclosure_utterance(
    scenario_type: str,
    victim_id: str,
    verifier: str,
    *,
    unconditional: bool = False,
) -> str:
    if unconditional:
        return f"{victim_id}는 요청하신 확인용으로 [MASKED_OTP]를 입력하겠습니다."
    batch2_partial_responses = {
        "tax_refund_notice": "[MASKED_EMAIL] 연락처 확인은 공식 세무 포털에서만 처리하겠습니다",
        "travel_booking_notice": "[MASKED_CREDENTIAL] 입력은 예약 앱에서만 처리하겠습니다",
        "property_claim_notice": "[MASKED_PHONE] 연락 확인은 보험사 앱에서만 처리하겠습니다",
        "emergency_drill_notice": "[MASKED_PHONE] 콜백 가능 시간은 우편 안내문 확인 뒤에만 처리하겠습니다",
        "nonprofit_admin_notice": "[MASKED_PHONE] callback 확인은 기존 행사 공지 확인 뒤에만 처리하겠습니다",
        "pet_appointment_notice": "[MASKED_PHONE] 등록 연락처 확인은 예약 앱에서만 처리하겠습니다",
        "research_study_notice": "[MASKED_EMAIL] 연락 확인은 연구 포털 알림함에서만 처리하겠습니다",
        "library_account_notice": "[MASKED_EMAIL] 연락 확인은 도서관 앱에서만 처리하겠습니다",
        "vehicle_service_notice": "[MASKED_PHONE] 등록 연락처 확인은 공식 리콜 조회 뒤에만 처리하겠습니다",
        "ticketing_refund_notice": "[MASKED_EMAIL] 등록 연락처 확인은 예매 계정 알림함에서만 처리하겠습니다",
        "membership_access_notice": "[MASKED_EMAIL] 회원 연락 확인은 멤버십 포털에서만 처리하겠습니다",
        "childcare_service_notice": "[MASKED_PHONE] 등록 연락처 확인은 센터 사무실 callback 뒤에만 처리하겠습니다",
    }
    if scenario_type in batch2_partial_responses:
        return f"{victim_id}는 급하다고 하니 {batch2_partial_responses[scenario_type]}."
    return f"{victim_id}는 급하다고 하니 [MASKED_OTP]를 입력해볼게요, 다만 실제 값은 공식 경로에서만 처리하겠습니다."
