from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

from .agents import AttackerAgent, VictimAgent
from .config import GenerationConfig, validate_persona_catalog_capacity, validate_personas
from .labeling import annotate_turn, collect_risk_labels, infer_record_outcome
from .postprocess import normalized_for_duplicate_check
from .safety import SafetyMasker
from .schema import SchemaError, validate_record


SCENARIO_CONTEXTS = {
    "family_emergency": "가족 목소리를 사칭한 긴급 상황 통화로 금전 또는 추가 확인을 요구하는 상황.",
    "bank_fraud_alert": "기관 또는 은행 담당자를 사칭해 계정 보호를 이유로 정보를 요구하는 상황.",
    "workplace_authority": "회사 임원, 거래처, 내부 보안팀을 사칭해 업무 예외 처리를 요구하는 상황.",
    "essential_service_notice": "필수 생활 서비스 지원 담당자를 사칭해 서비스 연속성이나 청구 예외를 이유로 계정 확인을 요구하는 상황.",
    "document_verification_notice": "서류 지원 담당자를 사칭해 접수 보완이나 예약 유지를 이유로 신원 자료 확인을 요구하는 상황.",
    "public_benefit_notice": "공공 지원 담당자를 사칭해 지원금, 보조금, 장학금, 자격 심사를 이유로 포털 계정이나 신원 자료 확인을 요구하는 상황.",
    "platform_policy_notice": "콘텐츠 또는 후원 플랫폼 정책 담당자를 사칭해 수익화 심사나 정산 예외를 이유로 공식 알림함 밖에서 계정 확인을 요구하는 상황.",
    "education_admin_notice": "학교나 돌봄 활동 운영 담당자를 사칭해 보호자 동의, 일정 변경, 행정 확인을 이유로 공식 알림장 밖에서 신원 또는 계정 확인을 요구하는 상황.",
    "tax_refund_notice": "세무 포털 담당자를 사칭해 환급 상태와 신고자 신원 참조를 공식 포털 밖에서 확인하라고 요구하는 상황.",
    "travel_booking_notice": "여행 재예약 담당자를 사칭해 일정 변경과 예약 계정 상태를 통화 중 확인하라고 요구하는 상황.",
    "property_claim_notice": "재산 보험 보상 담당자를 사칭해 청구 참조와 계약자 확인 자료를 공식 앱 밖에서 확인하라고 요구하는 상황.",
    "emergency_drill_notice": "사전 재난 대비 훈련 콜백 담당자를 사칭해 훈련 안내 참조와 통화 응답 가능 시간을 확인하라고 요구하는 상황.",
    "nonprofit_admin_notice": "지역 비영리 단체 행정 담당자를 사칭해 봉사 행사 역할과 callback 확인을 요구하는 상황.",
    "pet_appointment_notice": "동물병원 예약 담당자를 사칭해 반려동물 예약 접수와 등록 연락처를 확인하라고 요구하는 상황.",
    "research_study_notice": "대학 연구팀 담당자를 사칭해 연구 참여 일정과 동의 상태 참조를 확인하라고 요구하는 상황.",
    "library_account_notice": "도서관 서비스 담당자를 사칭해 예약 도서와 디지털 접근 상태를 확인하라고 요구하는 상황.",
    "vehicle_service_notice": "차량 리콜 또는 서비스 일정 담당자를 사칭해 예약 참조와 등록 연락처를 확인하라고 요구하는 상황.",
    "ticketing_refund_notice": "예매 플랫폼 환불 담당자를 사칭해 주문 참조와 계정 접근 상태를 확인하라고 요구하는 상황.",
    "membership_access_notice": "문화시설 회원 서비스 담당자를 사칭해 개인 멤버십 접근 상태와 입장 예약 상태를 확인하라고 요구하는 상황.",
    "childcare_service_notice": "영아 보육 대기자 명단 담당자를 사칭해 영아 대기 상태와 입소 상담 일정 확인을 요구하는 상황.",
}

SCENARIO_RISK_LABELS = {
    "family_emergency": ["voice_impersonation", "urgency", "emotional_pressure"],
    "bank_fraud_alert": ["voice_impersonation", "authority_impersonation", "credential_request"],
    "workplace_authority": ["voice_impersonation", "authority_impersonation", "money_transfer_request"],
    "essential_service_notice": ["voice_impersonation", "authority_impersonation", "urgency", "credential_request"],
    "document_verification_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "public_benefit_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "platform_policy_notice": ["voice_impersonation", "authority_impersonation", "credential_request", "urgency"],
    "education_admin_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "tax_refund_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "travel_booking_notice": ["voice_impersonation", "authority_impersonation", "credential_request", "urgency"],
    "property_claim_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "emergency_drill_notice": ["voice_impersonation", "authority_impersonation", "callback_or_voice_sample_request", "urgency"],
    "nonprofit_admin_notice": ["voice_impersonation", "authority_impersonation", "callback_or_voice_sample_request", "urgency"],
    "pet_appointment_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "research_study_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "library_account_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "vehicle_service_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "ticketing_refund_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "membership_access_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
    "childcare_service_notice": ["voice_impersonation", "authority_impersonation", "identity_information_request", "urgency"],
}

PAIR_SCENARIO_PREFERENCES = {
    ("utility_account_holder", "utility_support_impersonator"): "essential_service_notice",
    ("language_access_service_user", "document_support_impersonator"): "document_verification_notice",
    ("public_benefit_recipient", "benefits_caseworker_impersonator"): "public_benefit_notice",
    ("freelance_creator", "platform_policy_impersonator"): "platform_policy_notice",
    ("caregiver_or_guardian", "education_admin_impersonator"): "education_admin_notice",
    ("seasonal_tax_filer", "tax_portal_impersonator"): "tax_refund_notice",
    ("travel_booking_customer", "travel_rebooking_impersonator"): "travel_booking_notice",
    ("property_claim_policyholder", "property_claim_adjuster_impersonator"): "property_claim_notice",
    ("disaster_preparedness_resident", "emergency_drill_coordinator_impersonator"): "emergency_drill_notice",
    ("community_volunteer_coordinator", "nonprofit_admin_impersonator"): "nonprofit_admin_notice",
    ("pet_care_client", "veterinary_scheduler_impersonator"): "pet_appointment_notice",
    ("research_participant", "university_research_impersonator"): "research_study_notice",
    ("library_account_holder", "library_services_impersonator"): "library_account_notice",
    ("vehicle_owner", "vehicle_recall_impersonator"): "vehicle_service_notice",
    ("event_ticket_buyer", "ticketing_refund_impersonator"): "ticketing_refund_notice",
    ("museum_member", "membership_services_impersonator"): "membership_access_notice",
    ("infant_childcare_applicant", "childcare_waitlist_impersonator"): "childcare_service_notice",
}
PAIR_ONLY_SCENARIOS = {
    "essential_service_notice",
    "document_verification_notice",
    "public_benefit_notice",
    "platform_policy_notice",
    "education_admin_notice",
    "tax_refund_notice",
    "travel_booking_notice",
    "property_claim_notice",
    "emergency_drill_notice",
    "nonprofit_admin_notice",
    "pet_appointment_notice",
    "research_study_notice",
    "library_account_notice",
    "vehicle_service_notice",
    "ticketing_refund_notice",
    "membership_access_notice",
    "childcare_service_notice",
}

PAIR_CONTEXTS = {
    ("freelance_creator", "platform_policy_impersonator"): (
        "콘텐츠 또는 후원 플랫폼 정책 담당자를 사칭해 수익화 심사나 정산 예외를 이유로 "
        "공식 알림함 밖에서 계정 확인을 요구하는 상황."
    ),
    ("caregiver_or_guardian", "education_admin_impersonator"): (
        "학교나 돌봄 활동 운영 담당자를 사칭해 보호자 동의, 일정 변경, 행정 확인을 이유로 "
        "공식 알림장 밖에서 신원 또는 계정 확인을 요구하는 상황."
    ),
    ("language_access_service_user", "document_support_impersonator"): (
        "서류 또는 언어 지원 담당자를 사칭해 접수 보완, 예약 유지, 통역 지원을 이유로 "
        "공식 포털 밖에서 신원 자료나 지원 계정 확인을 요구하는 상황."
    ),
    ("disaster_preparedness_resident", "emergency_drill_coordinator_impersonator"): (
        "사전 재난 대비 훈련 콜백 담당자를 사칭해 훈련 안내 참조와 통화 가능 시간만 확인하라고 "
        "요구하는 상황. 사전 공지, 지자체 알림함, 저장된 커뮤니티 사무실 callback으로만 확인한다."
    ),
    ("pet_care_client", "veterinary_scheduler_impersonator"): (
        "동물병원 예약 담당자를 사칭해 반려동물 예약 접수와 등록 연락처만 확인하라고 요구하는 상황. "
        "저장된 병원 번호, 예약 앱, 우편 또는 포털 알림으로만 확인한다."
    ),
    ("museum_member", "membership_services_impersonator"): (
        "문화시설 회원 서비스 담당자를 사칭해 개인 멤버십 접근 상태와 비양도성 입장 예약 상태만 확인하라고 "
        "요구하는 상황. 멤버십 포털 알림함, 우편 안내문, 저장된 회원 서비스 번호로만 확인한다."
    ),
    ("infant_childcare_applicant", "childcare_waitlist_impersonator"): (
        "영아 보육 대기자 명단 담당자를 사칭해 영아 대기 상태, 입소 상담 일정, 센터 사무실 callback만 "
        "확인하라고 요구하는 상황. 센터 사무실 callback과 공식 영아 대기자 안내문으로만 확인한다."
    ),
}

ENVIRONMENTAL_CONTEXTS = {
    "family_emergency": {
        "channel": "phone_call",
        "timing": "late_evening",
        "trigger_event": "urgent_family_distress_claim",
        "claimed_authority": "family_member_or_known_contact",
        "victim_constraint": "emotionally_overloaded_and_cannot_verify_in_person",
        "verification_path": "family_callback",
    },
    "bank_fraud_alert": {
        "channel": "phone_call",
        "timing": "during_commute",
        "trigger_event": "suspicious_login_notice",
        "claimed_authority": "bank_security_team",
        "victim_constraint": "cannot_visit_branch_now",
        "verification_path": "official_app",
    },
    "workplace_authority": {
        "channel": "phone_call",
        "timing": "before_deadline",
        "trigger_event": "process_exception",
        "claimed_authority": "manager_or_vendor_office",
        "victim_constraint": "at_work_under_deadline_pressure",
        "verification_path": "manager_confirmation",
    },
    "essential_service_notice": {
        "channel": "support_callback",
        "timing": "after_business_hours",
        "trigger_event": "service_disconnection_notice",
        "claimed_authority": "utility_support_office",
        "victim_constraint": "service_continuity_needed",
        "verification_path": "saved_phone_number",
    },
    "document_verification_notice": {
        "channel": "messenger_call",
        "timing": "before_deadline",
        "trigger_event": "document_missing_notice",
        "claimed_authority": "document_support_office",
        "victim_constraint": "language_support_needed",
        "verification_path": "portal_inbox",
    },
    "public_benefit_notice": {
        "channel": "phone_call",
        "timing": "lunch_break",
        "trigger_event": "eligibility_review_notice",
        "claimed_authority": "benefits_caseworker_office",
        "victim_constraint": "urgent_cashflow",
        "verification_path": "paper_notice",
    },
    "platform_policy_notice": {
        "channel": "support_callback",
        "timing": "before_deadline",
        "trigger_event": "payout_exception_notice",
        "claimed_authority": "platform_policy_team",
        "victim_constraint": "income_review_pending",
        "verification_path": "platform_dashboard",
    },
    "education_admin_notice": {
        "channel": "messenger_call",
        "timing": "during_commute",
        "trigger_event": "schedule_change",
        "claimed_authority": "school_admin_office",
        "victim_constraint": "caring_for_child",
        "verification_path": "school_portal_or_known_staff",
    },
    "tax_refund_notice": {
        "channel": "phone_call",
        "timing": "before_deadline",
        "trigger_event": "refund_status_review_notice",
        "claimed_authority": "tax_portal_support_office",
        "victim_constraint": "waiting_for_refund_update",
        "verification_path": "official_tax_portal",
    },
    "travel_booking_notice": {
        "channel": "support_callback",
        "timing": "after_business_hours",
        "trigger_event": "itinerary_rebooking_notice",
        "claimed_authority": "travel_rebooking_support",
        "victim_constraint": "travel_plan_time_sensitive",
        "verification_path": "booking_app_or_saved_phone_number",
    },
    "property_claim_notice": {
        "channel": "phone_call",
        "timing": "lunch_break",
        "trigger_event": "property_claim_supplement_notice",
        "claimed_authority": "property_claim_adjuster_office",
        "victim_constraint": "claim_resolution_pending",
        "verification_path": "insurer_app_or_registered_adjuster_callback",
    },
    "emergency_drill_notice": {
        "channel": "support_callback",
        "timing": "before_deadline",
        "trigger_event": "pre_scheduled_drill_callback_notice",
        "claimed_authority": "community_drill_coordination_office",
        "victim_constraint": "callback_window_needs_scheduling",
        "verification_path": "paper_notice_or_municipal_alert_inbox",
    },
    "nonprofit_admin_notice": {
        "channel": "phone_call",
        "timing": "before_deadline",
        "trigger_event": "volunteer_event_callback_notice",
        "claimed_authority": "nonprofit_admin_office",
        "victim_constraint": "volunteer_schedule_coordination",
        "verification_path": "saved_nonprofit_staff_number",
    },
    "pet_appointment_notice": {
        "channel": "support_callback",
        "timing": "lunch_break",
        "trigger_event": "pet_appointment_schedule_notice",
        "claimed_authority": "veterinary_scheduler_office",
        "victim_constraint": "pet_appointment_pending",
        "verification_path": "saved_clinic_number_or_appointment_app",
    },
    "research_study_notice": {
        "channel": "messenger_call",
        "timing": "before_deadline",
        "trigger_event": "study_schedule_or_consent_status_notice",
        "claimed_authority": "university_research_team",
        "victim_constraint": "study_visit_pending",
        "verification_path": "study_portal_inbox_or_registered_lab_number",
    },
    "library_account_notice": {
        "channel": "phone_call",
        "timing": "during_commute",
        "trigger_event": "hold_pickup_or_digital_access_notice",
        "claimed_authority": "library_services_desk",
        "victim_constraint": "library_hold_pickup_window",
        "verification_path": "library_app_or_saved_branch_number",
    },
    "vehicle_service_notice": {
        "channel": "support_callback",
        "timing": "before_deadline",
        "trigger_event": "recall_service_schedule_notice",
        "claimed_authority": "vehicle_recall_service_office",
        "victim_constraint": "service_appointment_pending",
        "verification_path": "official_recall_lookup_or_saved_service_number",
    },
    "ticketing_refund_notice": {
        "channel": "support_callback",
        "timing": "after_business_hours",
        "trigger_event": "event_refund_status_notice",
        "claimed_authority": "ticketing_refund_support",
        "victim_constraint": "event_change_pending",
        "verification_path": "event_account_inbox_or_order_history",
    },
    "membership_access_notice": {
        "channel": "phone_call",
        "timing": "lunch_break",
        "trigger_event": "membership_access_status_notice",
        "claimed_authority": "membership_services_office",
        "victim_constraint": "entry_reservation_status_pending",
        "verification_path": "membership_portal_inbox_or_mailed_notice",
    },
    "childcare_service_notice": {
        "channel": "phone_call",
        "timing": "before_deadline",
        "trigger_event": "infant_waitlist_intake_notice",
        "claimed_authority": "childcare_center_office",
        "victim_constraint": "infant_intake_appointment_pending",
        "verification_path": "center_office_callback_or_official_waitlist_notice",
    },
}

ENVIRONMENTAL_CONTEXT_VARIANTS = (
    {},
    {
        "channel": "messenger_call",
        "timing": "lunch_break",
        "victim_constraint": "cannot_leave_current_task",
    },
    {
        "channel": "support_callback",
        "timing": "after_business_hours",
        "trigger_event": "follow_up_notice",
    },
    {
        "channel": "voice_message",
        "timing": "during_commute",
        "verification_path": "saved_phone_number_or_portal_inbox",
    },
)

VICTIM_SCENARIO_PREFERENCES = {
    "elderly_parent": "family_emergency",
    "office_worker": "workplace_authority",
    "student_or_young_adult": "bank_fraud_alert",
    "small_business_owner": "workplace_authority",
    "remote_job_seeker": "workplace_authority",
    "telehealth_patient": "bank_fraud_alert",
    "renter_or_tenant": "bank_fraud_alert",
    "utility_account_holder": "essential_service_notice",
    "public_benefit_recipient": "bank_fraud_alert",
    "freelance_creator": "platform_policy_notice",
    "caregiver_or_guardian": "education_admin_notice",
    "language_access_service_user": "document_verification_notice",
    "seasonal_tax_filer": "tax_refund_notice",
    "travel_booking_customer": "travel_booking_notice",
    "property_claim_policyholder": "property_claim_notice",
    "disaster_preparedness_resident": "emergency_drill_notice",
    "community_volunteer_coordinator": "nonprofit_admin_notice",
    "pet_care_client": "pet_appointment_notice",
    "research_participant": "research_study_notice",
    "library_account_holder": "library_account_notice",
    "vehicle_owner": "vehicle_service_notice",
    "event_ticket_buyer": "ticketing_refund_notice",
    "museum_member": "membership_access_notice",
    "infant_childcare_applicant": "childcare_service_notice",
}

ATTACKER_SCENARIO_PREFERENCES = {
    "family_impersonator": "family_emergency",
    "institution_impersonator": "bank_fraud_alert",
    "corporate_impersonator": "workplace_authority",
    "service_provider_impersonator": "workplace_authority",
    "recruiter_impersonator": "workplace_authority",
    "healthcare_support_impersonator": "bank_fraud_alert",
    "housing_admin_impersonator": "bank_fraud_alert",
    "utility_support_impersonator": "essential_service_notice",
    "benefits_caseworker_impersonator": "bank_fraud_alert",
    "platform_policy_impersonator": "platform_policy_notice",
    "education_admin_impersonator": "education_admin_notice",
    "document_support_impersonator": "document_verification_notice",
    "tax_portal_impersonator": "tax_refund_notice",
    "travel_rebooking_impersonator": "travel_booking_notice",
    "property_claim_adjuster_impersonator": "property_claim_notice",
    "emergency_drill_coordinator_impersonator": "emergency_drill_notice",
    "nonprofit_admin_impersonator": "nonprofit_admin_notice",
    "veterinary_scheduler_impersonator": "pet_appointment_notice",
    "university_research_impersonator": "research_study_notice",
    "library_services_impersonator": "library_account_notice",
    "vehicle_recall_impersonator": "vehicle_service_notice",
    "ticketing_refund_impersonator": "ticketing_refund_notice",
    "membership_services_impersonator": "membership_access_notice",
    "childcare_waitlist_impersonator": "childcare_service_notice",
}

GOAL_TO_INTENT = {
    "money_transfer": "request_masked_transfer_or_payment",
    "credential_capture": "request_masked_credential_or_otp",
    "identity_information": "request_identity_information",
    "voice_sample_or_callback": "request_callback_or_voice_confirmation",
}

PHASES = ("opening", "establish_context", "make_request", "handle_resistance", "close")

ATTACKER_PHASE_INSTRUCTIONS = {
    "opening": "Open the call with a deepvoice-style impersonation cue and a plausible reason for contact.",
    "establish_context": "Build authority, relationship, or situational context based on the victim's last response.",
    "make_request": "Ask for the attacker's goal using masked placeholders only.",
    "handle_resistance": "Respond to resistance by changing tactic without adding operational fraud steps.",
    "close": "Push for a final action or callback while remaining masked and non-operational.",
}

VICTIM_PHASE_INSTRUCTIONS = {
    "opening": "React with concern but avoid disclosure.",
    "establish_context": "Ask for verification through official or trusted channels.",
    "make_request": "Refuse to provide protected assets during the call.",
    "handle_resistance": "Maintain resistance despite pressure.",
    "close": "End or redirect the conversation to official verification.",
}

INCONCLUSIVE_VICTIM_PHASE_INSTRUCTIONS = {
    "opening": "React as unsettled and unsure; keep listening without refusing, verifying, or disclosing.",
    "establish_context": "Show that the explanation feels plausible but unresolved; ask for more context without firm verification.",
    "make_request": "Do not comply and do not refuse; stall because the request feels hard to judge, and avoid masked placeholders.",
    "handle_resistance": "React to pressure with confusion and delay, not safety advice or official-channel verification.",
    "close": "Leave the decision unresolved; ask for time or defer without saying you will verify officially.",
}

VICTIM_RESPONSE_MODES = ("defensive", "uncertain", "partial_disclosure")
TARGET_OUTCOME_MODES = ("defended_success", "inconclusive", "masked_compromise")
TARGET_OUTCOME_TO_INFERRED = {
    "defended_success": "defended_success",
    "inconclusive": "inconclusive",
    "masked_compromise": "compromised",
}
TARGET_OUTCOME_MAX_GENERATION_ATTEMPTS = 3
TARGET_OUTCOME_LLM_REWRITE_MAX_ATTEMPTS = 2
PRESSURE_STYLES = (
    "urgency",
    "authority",
    "consequence",
    "relationship_or_emotional_pressure",
    "process_control",
)


@dataclass(frozen=True)
class ScheduleAssignment:
    victim: dict[str, Any]
    attacker: dict[str, Any]
    scenario_type: str
    schedule_mode: str
    canonical_pair_index: int | None
    variant_index: int
    target_outcome_mode: str
    pressure_style: str
    environmental_context_variant_id: str


@dataclass
class GenerationEvent:
    record: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


@dataclass
class DialogueAttemptResult:
    dialogue: list[dict[str, Any]]
    safety_masks: list[dict[str, str]]
    forbidden_detail_seen: bool
    outcome: str
    compromised_assets: list[str]


@dataclass
class DatasetOrchestrator:
    config: GenerationConfig
    attacker_agent: AttackerAgent
    victim_agent: VictimAgent
    safety: SafetyMasker

    def generate(
        self,
        victims: list[dict[str, Any]],
        attackers: list[dict[str, Any]],
        *,
        limit: int | None = None,
        start_index: int = 0,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for event in self.iter_generate(victims, attackers, limit=limit, start_index=start_index):
            if event.record is not None:
                records.append(event.record)
        return records

    def iter_generate(
        self,
        victims: list[dict[str, Any]],
        attackers: list[dict[str, Any]],
        *,
        limit: int | None = None,
        start_index: int = 0,
        skip_record_ids: set[str] | None = None,
        max_attempts: int | None = None,
    ):
        validate_personas(victims, kind="victim")
        validate_personas(attackers, kind="attacker")
        validate_persona_catalog_capacity(victims, attackers, limit=self.config.persona_catalog_limit)

        target_count = limit if limit is not None else self.config.default_limit
        if target_count < 0:
            raise ValueError("limit must be non-negative")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        skipped_ids = skip_record_ids or set()
        produced = 0
        attempt_limit = max_attempts
        if attempt_limit is None:
            attempt_limit = generation_attempt_limit(
                target_count=target_count,
                skipped_id_count=len(skipped_ids),
                victim_count=len(victims),
                attacker_count=len(attackers),
            )
        for index in range(start_index, start_index + attempt_limit):
            if produced >= target_count:
                break
            assignment = schedule_assignment_for_index(
                victims,
                attackers,
                self.config.scenario_types,
                index,
                schedule_mode=self.config.schedule_mode,
            )
            victim = assignment.victim
            attacker = assignment.attacker
            scenario_type = assignment.scenario_type
            record_id = make_record_id(self.config.dataset_name, victim["id"], attacker["id"], scenario_type, index)
            if record_id in skipped_ids:
                continue
            try:
                record = self.generate_one(
                    victim,
                    attacker,
                    scenario_type,
                    index,
                    schedule_assignment=assignment,
                )
            except Exception as exc:
                yield GenerationEvent(
                    error=generation_error(
                        index=index,
                        record_id=record_id,
                        victim=victim,
                        attacker=attacker,
                        scenario_type=scenario_type,
                        error=exc,
                    )
                )
                continue
            produced += 1
            yield GenerationEvent(record=record)
        if produced < target_count:
            raise RuntimeError(
                "Unable to generate requested records within scan limit "
                f"(requested={target_count}, produced={produced}, skipped_ids={len(skipped_ids)}, "
                f"max_attempts={attempt_limit})"
            )

    def generate_one(
        self,
        victim: dict[str, Any],
        attacker: dict[str, Any],
        scenario_type: str,
        index: int,
        *,
        schedule_assignment: ScheduleAssignment | None = None,
    ) -> dict[str, Any]:
        if schedule_assignment is None:
            schedule_assignment = ScheduleAssignment(
                victim=victim,
                attacker=attacker,
                scenario_type=scenario_type,
                schedule_mode=self.config.schedule_mode,
                canonical_pair_index=None,
                variant_index=0,
                target_outcome_mode=(
                    target_outcome_mode_for_index(index)
                    if self.config.schedule_mode == "canonical_repeat"
                    else target_outcome_mode_for_default_difficulty(index)
                ),
                pressure_style=pressure_style_for_index(index),
                environmental_context_variant_id="variant_00",
            )
        protected_assets = list(victim.get("protected_assets") or default_assets_for_scenario(scenario_type))
        attacker_goal = choose_attacker_goal(attacker, protected_assets)
        environmental_context = environmental_context_for_scenario(
            scenario_type,
            variant_index=schedule_assignment.variant_index,
        )
        turn_budget = self.config.max_turns
        difficulty = difficulty_for_index(index)
        if schedule_assignment.schedule_mode == "canonical_repeat":
            victim_response_mode = victim_response_mode_for_target_outcome(
                schedule_assignment.target_outcome_mode,
                difficulty["victim_susceptibility"],
            )
        else:
            victim_response_mode = victim_response_mode_for_difficulty(difficulty["victim_susceptibility"])

        attempt_limit = target_outcome_generation_attempt_limit(schedule_assignment.target_outcome_mode)
        chosen_attempt: DialogueAttemptResult | None = None
        target_outcome_generation_attempts = 0
        for retry_attempt in range(attempt_limit):
            target_outcome_generation_attempts = retry_attempt + 1
            candidate = self.generate_dialogue_attempt(
                victim=victim,
                attacker=attacker,
                scenario_type=scenario_type,
                attacker_goal=attacker_goal,
                protected_assets=protected_assets,
                environmental_context=environmental_context,
                target_outcome_mode=schedule_assignment.target_outcome_mode,
                pressure_style=schedule_assignment.pressure_style,
                victim_response_mode=victim_response_mode,
                turn_budget=turn_budget,
                retry_attempt=retry_attempt,
            )
            chosen_attempt = candidate
            if outcome_matches_target(candidate.outcome, schedule_assignment.target_outcome_mode):
                break
        if chosen_attempt is None:
            raise RuntimeError("target outcome generation did not produce an attempt")

        dialogue = chosen_attempt.dialogue
        safety_masks = chosen_attempt.safety_masks
        forbidden_detail_seen = chosen_attempt.forbidden_detail_seen

        (
            target_outcome_llm_rewrite_applied,
            target_outcome_llm_rewrite_attempts,
            rewrite_forbidden_detail_seen,
        ) = self.rewrite_victim_turns_for_target_outcome(
            dialogue,
            victim=victim,
            scenario_type=scenario_type,
            protected_assets=protected_assets,
            environmental_context=environmental_context,
            target_outcome_mode=schedule_assignment.target_outcome_mode,
            victim_response_mode=victim_response_mode,
        )
        forbidden_detail_seen = forbidden_detail_seen or rewrite_forbidden_detail_seen

        target_outcome_deterministic_repair_applied = repair_dialogue_for_target_outcome(
            dialogue,
            schedule_assignment.target_outcome_mode,
            protected_assets,
            self.safety,
            record_index=index,
            victim_id=str(victim.get("id", "")),
            scenario_type=scenario_type,
            environmental_context=environmental_context,
            variant_index=schedule_assignment.variant_index,
        )
        safety_masks = safety_masks_from_dialogue(dialogue)
        outcome, compromised_assets = infer_record_outcome(dialogue)
        risk_labels = collect_risk_labels(dialogue, SCENARIO_RISK_LABELS.get(scenario_type, ["voice_impersonation"]))
        utterance_text = "\n".join(str(turn.get("utterance", "")) for turn in dialogue)
        residual_pii_clean = self.safety.is_residual_pii_clean(utterance_text)
        label_consistent = label_consistent_for_record("attack", dialogue, risk_labels)
        language_consistent = language_consistent_for_record(dialogue, "ko-KR")
        record = {
            "id": make_record_id(self.config.dataset_name, victim["id"], attacker["id"], scenario_type, index),
            "dataset_name": self.config.dataset_name,
            "label": "attack",
            "scenario_type": scenario_type,
            "victim_persona": compact_persona(victim),
            "attacker_persona": compact_persona(attacker),
            "protected_assets": protected_assets,
            "attacker_goal": attacker_goal,
            "outcome": outcome,
            "compromised_assets": compromised_assets,
            "difficulty": difficulty,
            "channel": "phone",
            "locale": "ko-KR",
            "split": split_for_index(index),
            "context": context_for_personas(victim, attacker, scenario_type),
            "dialogue": dialogue,
            "risk_labels": risk_labels,
            "safety_masks": safety_masks,
            "gen_metadata": {
                "attacker_model": self.config.attacker_model,
                "victim_model": self.config.victim_model,
                "prompt_version": "v2",
                "sampling": {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                "environmental_context": environmental_context,
                "schedule_mode": schedule_assignment.schedule_mode,
                "canonical_pair_index": schedule_assignment.canonical_pair_index,
                "variant_index": schedule_assignment.variant_index,
                "target_outcome_mode": schedule_assignment.target_outcome_mode,
                "target_outcome_generation_attempts": target_outcome_generation_attempts,
                "target_outcome_generation_attempt_limit": attempt_limit,
                "target_outcome_llm_rewrite_attempts": target_outcome_llm_rewrite_attempts,
                "target_outcome_llm_rewrite_attempt_limit": TARGET_OUTCOME_LLM_REWRITE_MAX_ATTEMPTS,
                "target_outcome_llm_rewrite_applied": target_outcome_llm_rewrite_applied,
                "target_outcome_deterministic_repair_applied": target_outcome_deterministic_repair_applied,
                "target_outcome_repair_applied": target_outcome_deterministic_repair_applied,
                "pressure_style": schedule_assignment.pressure_style,
                "environmental_context_variant_id": schedule_assignment.environmental_context_variant_id,
                "seed": index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "quality_checks": {
                "turn_count": len(dialogue),
                "role_order_valid": role_order_valid(dialogue),
                "safety_passed": not forbidden_detail_seen and residual_pii_clean,
                "no_duplicate_turns": no_duplicate_turns(dialogue),
                "label_consistent": label_consistent,
                "language_consistent": language_consistent,
                "no_refusal_leak": no_refusal_leak(dialogue),
                "residual_pii_clean": residual_pii_clean,
                "schema_valid": False,
            },
        }

        try:
            validate_record(
                record,
                max_turns=self.config.max_turns,
                allowed_risk_labels=self.config.risk_label_set,
            )
            record["quality_checks"]["schema_valid"] = True
        except SchemaError:
            record["quality_checks"]["schema_valid"] = False
            raise
        return record

    def rewrite_victim_turns_for_target_outcome(
        self,
        dialogue: list[dict[str, Any]],
        *,
        victim: dict[str, Any],
        scenario_type: str,
        protected_assets: list[str],
        environmental_context: dict[str, str],
        target_outcome_mode: str,
        victim_response_mode: str,
    ) -> tuple[bool, int, bool]:
        if target_outcome_mode not in TARGET_OUTCOME_TO_INFERRED:
            return False, 0, False
        observed, _assets = infer_record_outcome(dialogue)
        if outcome_matches_target(observed, target_outcome_mode):
            return False, 0, False

        attempts = 0
        forbidden_detail_seen = False
        for rewrite_attempt in range(1, TARGET_OUTCOME_LLM_REWRITE_MAX_ATTEMPTS + 1):
            attempts = rewrite_attempt
            candidate_dialogue: list[dict[str, Any]] = []
            candidate_forbidden_detail_seen = False

            for turn in dialogue:
                if turn.get("speaker") != "victim":
                    candidate_dialogue.append(cloned_turn(turn))
                    continue

                phase = str(turn.get("phase", ""))
                try:
                    raw_utterance = self.victim_agent.next_utterance(
                        victim,
                        scenario_type,
                        protected_assets,
                        candidate_dialogue,
                        phase,
                        victim_rewrite_phase_instruction(
                            phase,
                            victim_response_mode,
                            target_outcome_mode=target_outcome_mode,
                            rewrite_attempt=rewrite_attempt,
                        ),
                        victim_response_mode,
                        environmental_context,
                        target_outcome_mode,
                    )
                    fallback_risk = "victim_resistance"
                except RuntimeError:
                    raw_utterance = fallback_utterance("victim", phase)
                    fallback_risk = "generation_fallback"

                if is_near_duplicate(raw_utterance, candidate_dialogue):
                    raw_utterance = fallback_utterance("victim", phase)
                    fallback_risk = "generation_fallback"

                candidate_forbidden_detail_seen = (
                    candidate_forbidden_detail_seen
                    or self.safety.contains_forbidden_operational_detail(raw_utterance)
                )
                safe_text = self.safety.mask_text(raw_utterance)
                intent_label, risk_signal = annotate_turn("victim", safe_text.text, fallback_risk)
                candidate_dialogue.append(
                    {
                        "turn_id": turn.get("turn_id"),
                        "speaker": "victim",
                        "phase": phase,
                        "utterance": safe_text.text,
                        "intent_label": intent_label,
                        "risk_signal": risk_signal,
                        "masked_items": self.safety.masks_to_dicts(safe_text.masked_items),
                    }
                )

            candidate_outcome, _candidate_assets = infer_record_outcome(candidate_dialogue)
            forbidden_detail_seen = forbidden_detail_seen or candidate_forbidden_detail_seen
            if outcome_matches_target(candidate_outcome, target_outcome_mode):
                dialogue[:] = candidate_dialogue
                return True, attempts, forbidden_detail_seen

        return False, attempts, forbidden_detail_seen

    def generate_dialogue_attempt(
        self,
        *,
        victim: dict[str, Any],
        attacker: dict[str, Any],
        scenario_type: str,
        attacker_goal: str,
        protected_assets: list[str],
        environmental_context: dict[str, str],
        target_outcome_mode: str,
        pressure_style: str,
        victim_response_mode: str,
        turn_budget: int,
        retry_attempt: int,
    ) -> DialogueAttemptResult:
        dialogue: list[dict[str, Any]] = []
        safety_masks: list[dict[str, str]] = []
        forbidden_detail_seen = False

        for turn_id in range(1, turn_budget + 1):
            speaker = "attacker" if turn_id % 2 == 1 else "victim"
            phase = phase_for_turn(turn_id)
            try:
                if speaker == "attacker":
                    raw_utterance = self.attacker_agent.next_utterance(
                        victim,
                        attacker,
                        scenario_type,
                        attacker_goal,
                        protected_assets,
                        dialogue,
                        phase,
                        ATTACKER_PHASE_INSTRUCTIONS[phase],
                        environmental_context,
                        target_outcome_mode,
                        pressure_style,
                    )
                    fallback_risk = primary_risk_signal(scenario_type, attacker_goal)
                else:
                    raw_utterance = self.victim_agent.next_utterance(
                        victim,
                        scenario_type,
                        protected_assets,
                        dialogue,
                        phase,
                        victim_phase_instruction(
                            phase,
                            victim_response_mode,
                            target_outcome_mode=target_outcome_mode,
                            retry_attempt=retry_attempt,
                        ),
                        victim_response_mode,
                        environmental_context,
                        target_outcome_mode,
                    )
                    fallback_risk = "victim_resistance"
            except RuntimeError:
                raw_utterance = fallback_utterance(speaker, phase)
                fallback_risk = "generation_fallback"

            if is_near_duplicate(raw_utterance, dialogue):
                raw_utterance = fallback_utterance(speaker, phase)
                fallback_risk = "generation_fallback"

            forbidden_detail_seen = forbidden_detail_seen or self.safety.contains_forbidden_operational_detail(raw_utterance)
            safe_text = self.safety.mask_text(raw_utterance)
            intent_label, risk_signal = annotate_turn(speaker, safe_text.text, fallback_risk)
            masked_items = self.safety.masks_to_dicts(safe_text.masked_items)
            safety_masks.extend(masked_items)
            dialogue.append(
                {
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "phase": phase,
                    "utterance": safe_text.text,
                    "intent_label": intent_label,
                    "risk_signal": risk_signal,
                    "masked_items": masked_items,
                }
            )

            if should_end_dialogue(dialogue):
                break

        outcome, compromised_assets = infer_record_outcome(dialogue)
        return DialogueAttemptResult(
            dialogue=dialogue,
            safety_masks=safety_masks,
            forbidden_detail_seen=forbidden_detail_seen,
            outcome=outcome,
            compromised_assets=compromised_assets,
        )


def phase_for_turn(turn_id: int) -> str:
    phase_index = min((turn_id - 1) // 2, len(PHASES) - 1)
    return PHASES[phase_index]


def persona_pair_for_index(
    victims: list[dict[str, Any]],
    attackers: list[dict[str, Any]],
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the deterministic persona pair for a generation index.

    The first catalog pass pairs equal offsets so the default smoke schedule
    exercises the reference victim/attacker pairs once. Later passes rotate
    the attacker offset to avoid replaying the exact same pair cycle.
    """
    if not victims:
        raise ValueError("victim personas must contain at least one persona")
    if not attackers:
        raise ValueError("attacker personas must contain at least one persona")
    victim = victims[index % len(victims)]
    attacker = attackers[((index // len(victims)) + index) % len(attackers)]
    return victim, attacker


def canonical_persona_pair_for_index(
    victims: list[dict[str, Any]],
    attackers: list[dict[str, Any]],
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not victims:
        raise ValueError("victim personas must contain at least one persona")
    if not attackers:
        raise ValueError("attacker personas must contain at least one persona")
    pair_count = min(len(victims), len(attackers))
    pair_index = index % pair_count
    return victims[pair_index], attackers[pair_index]


def schedule_assignment_for_index(
    victims: list[dict[str, Any]],
    attackers: list[dict[str, Any]],
    configured_scenarios: list[str],
    index: int,
    *,
    schedule_mode: str,
) -> ScheduleAssignment:
    if schedule_mode == "canonical_repeat":
        victim, attacker = canonical_persona_pair_for_index(victims, attackers, index)
        pair_count = min(len(victims), len(attackers))
        canonical_pair_index = index % pair_count
        variant_index = index // pair_count
        target_outcome_mode = target_outcome_mode_for_index(index)
    elif schedule_mode == "rotating":
        victim, attacker = persona_pair_for_index(victims, attackers, index)
        canonical_pair_index = None
        variant_index = 0
        target_outcome_mode = target_outcome_mode_for_default_difficulty(index)
    else:
        raise ValueError("schedule_mode must be one of: canonical_repeat, rotating")

    scenario_type = choose_scenario_type(victim, attacker, configured_scenarios, index)
    return ScheduleAssignment(
        victim=victim,
        attacker=attacker,
        scenario_type=scenario_type,
        schedule_mode=schedule_mode,
        canonical_pair_index=canonical_pair_index,
        variant_index=variant_index,
        target_outcome_mode=target_outcome_mode,
        pressure_style=pressure_style_for_index(index),
        environmental_context_variant_id=f"variant_{variant_index % len(ENVIRONMENTAL_CONTEXT_VARIANTS):02d}",
    )


def choose_scenario_type(
    victim: dict[str, Any],
    attacker: dict[str, Any],
    configured_scenarios: list[str],
    index: int,
) -> str:
    available = set(configured_scenarios)
    victim_id = str(victim.get("id", ""))
    attacker_id = str(attacker.get("id", ""))
    pair_preference = PAIR_SCENARIO_PREFERENCES.get((victim_id, attacker_id))
    if pair_preference in available:
        return pair_preference

    candidates = [
        ATTACKER_SCENARIO_PREFERENCES.get(attacker_id),
        VICTIM_SCENARIO_PREFERENCES.get(victim_id),
    ]
    for scenario_type in candidates:
        if scenario_type in available and scenario_type not in PAIR_ONLY_SCENARIOS:
            return scenario_type
    return fallback_configured_scenario(configured_scenarios, index)


def context_for_personas(victim: dict[str, Any], attacker: dict[str, Any], scenario_type: str) -> str:
    pair_context = PAIR_CONTEXTS.get((str(victim.get("id", "")), str(attacker.get("id", ""))))
    if pair_context:
        return pair_context
    return SCENARIO_CONTEXTS.get(scenario_type, "딥보이스 사칭 기반 사회공학 대화 상황.")


def environmental_context_for_scenario(scenario_type: str, *, variant_index: int = 0) -> dict[str, str]:
    default = {
        "channel": "phone_call",
        "timing": "unspecified",
        "trigger_event": "unspecified_notice",
        "claimed_authority": "unspecified_claimed_authority",
        "victim_constraint": "unspecified_constraint",
        "verification_path": "official_channel",
    }
    context = dict(ENVIRONMENTAL_CONTEXTS.get(scenario_type, default))
    context.update(ENVIRONMENTAL_CONTEXT_VARIANTS[variant_index % len(ENVIRONMENTAL_CONTEXT_VARIANTS)])
    return context


def fallback_configured_scenario(configured_scenarios: list[str], index: int) -> str:
    for offset in range(len(configured_scenarios)):
        scenario_type = configured_scenarios[(index + offset) % len(configured_scenarios)]
        if scenario_type not in PAIR_ONLY_SCENARIOS:
            return scenario_type
    return configured_scenarios[index % len(configured_scenarios)]


def choose_attacker_goal(attacker: dict[str, Any], protected_assets: list[str]) -> str:
    goals = list(attacker.get("default_goals") or [])
    if goals:
        return goals[0]
    if any("otp" in asset or "credential" in asset or "password" in asset for asset in protected_assets):
        return "credential_capture"
    if any("account" in asset or "payment" in asset for asset in protected_assets):
        return "money_transfer"
    return "identity_information"


def default_assets_for_scenario(scenario_type: str) -> list[str]:
    defaults = {
        "family_emergency": ["bank_account_access", "otp_or_security_card", "family_relationship_details"],
        "bank_fraud_alert": ["account_password", "otp_or_security_card", "identity_document_image"],
        "workplace_authority": ["company_payment_authority", "internal_credentials", "vendor_account_details"],
        "essential_service_notice": ["utility_account_credentials", "billing_or_autopay_details", "service_address_details"],
        "document_verification_notice": ["identity_document_image", "case_reference_number", "language_support_portal_credentials"],
        "public_benefit_notice": ["benefit_portal_credentials", "identity_document_image", "bank_account_for_disbursement", "case_reference_number"],
        "platform_policy_notice": ["creator_platform_credentials", "payout_account_details", "tax_or_identity_documents"],
        "education_admin_notice": ["guardian_identity_details", "student_or_dependent_records", "school_portal_credentials", "activity_payment_details"],
        "tax_refund_notice": ["tax_portal_access_status", "taxpayer_identity_reference", "case_reference_number", "refund_status_details", "voice_sample"],
        "travel_booking_notice": ["travel_account_credentials", "booking_reference_number", "itinerary_contact_details", "booking_payment_status_reference", "voice_sample"],
        "property_claim_notice": ["claim_reference_number", "property_policy_account", "policyholder_identity_reference", "incident_contact_details", "voice_sample"],
        "emergency_drill_notice": ["preparedness_notice_reference", "drill_callback_reference", "registered_contact_details", "callback_window_preference", "voice_sample"],
        "nonprofit_admin_notice": ["event_callback_reference", "volunteer_role_details", "coordinator_contact_details", "voice_sample"],
        "pet_appointment_notice": ["appointment_reference_number", "pet_profile_reference", "registered_contact_details", "clinic_callback_reference", "voice_sample"],
        "research_study_notice": ["study_reference_number", "participant_contact_details", "participant_identity_reference", "study_portal_inbox_reference", "voice_sample"],
        "library_account_notice": ["library_account_reference", "digital_library_access_status", "hold_pickup_details", "registered_contact_details", "voice_sample"],
        "vehicle_service_notice": ["vehicle_account_reference", "service_appointment_details", "registered_contact_details", "recall_notice_reference", "voice_sample"],
        "ticketing_refund_notice": ["ticketing_account_access_status", "order_reference_number", "registered_contact_details", "event_account_inbox", "voice_sample"],
        "membership_access_notice": ["membership_reference_number", "member_contact_details", "entry_reservation_status", "mailed_notice_reference", "voice_sample"],
        "childcare_service_notice": ["guardian_identity_reference", "waitlist_reference_number", "registered_contact_details", "intake_appointment_details", "voice_sample"],
    }
    return defaults.get(scenario_type, ["identity_information", "voice_sample"])


def victim_response_mode_for_difficulty(victim_susceptibility: str) -> str:
    if victim_susceptibility == "high":
        return "partial_disclosure"
    if victim_susceptibility == "med":
        return "uncertain"
    return "defensive"


def target_outcome_mode_for_index(index: int) -> str:
    return TARGET_OUTCOME_MODES[index % len(TARGET_OUTCOME_MODES)]


def target_outcome_mode_for_default_difficulty(index: int) -> str:
    susceptibility = difficulty_for_index(index)["victim_susceptibility"]
    if susceptibility == "high":
        return "inconclusive"
    return "defended_success"


def target_outcome_generation_attempt_limit(target_outcome_mode: str) -> int:
    if target_outcome_mode in TARGET_OUTCOME_TO_INFERRED:
        return TARGET_OUTCOME_MAX_GENERATION_ATTEMPTS
    return 1


def outcome_matches_target(outcome: str, target_outcome_mode: str) -> bool:
    return outcome == TARGET_OUTCOME_TO_INFERRED.get(target_outcome_mode, target_outcome_mode)


def pressure_style_for_index(index: int) -> str:
    return PRESSURE_STYLES[index % len(PRESSURE_STYLES)]


def victim_response_mode_for_target_outcome(target_outcome_mode: str, victim_susceptibility: str) -> str:
    if target_outcome_mode == "masked_compromise":
        return "partial_disclosure"
    if target_outcome_mode == "inconclusive":
        return "uncertain"
    return victim_response_mode_for_difficulty(victim_susceptibility) if target_outcome_mode != "defended_success" else "defensive"


def repair_dialogue_for_target_outcome(
    dialogue: list[dict[str, Any]],
    target_outcome_mode: str,
    protected_assets: list[str],
    safety: SafetyMasker,
    *,
    record_index: int,
    victim_id: str,
    scenario_type: str,
    environmental_context: dict[str, str],
    variant_index: int,
) -> bool:
    """Deterministically align non-defensive target outcomes when the LLM drifts defensive."""
    if target_outcome_mode not in TARGET_OUTCOME_TO_INFERRED:
        return False
    observed, _assets = infer_record_outcome(dialogue)
    if outcome_matches_target(observed, target_outcome_mode):
        return False

    victim_indexes = victim_turn_indexes(dialogue)
    if not victim_indexes:
        return False

    final_victim_index = victim_indexes[-1]
    changed = False
    for victim_index in victim_indexes:
        replacement = target_outcome_repair_utterance(
            target_outcome_mode,
            protected_assets,
            phase=str(dialogue[victim_index].get("phase", "")),
            is_final=victim_index == final_victim_index,
            victim_id=victim_id,
            scenario_type=scenario_type,
            environmental_context=environmental_context,
            variant_index=variant_index,
            record_index=record_index,
        )
        if not replacement:
            continue
        safe_text = safety.mask_text(replacement)
        intent_label, risk_signal = annotate_turn("victim", safe_text.text, "victim_uncertainty")
        dialogue[victim_index].update(
            {
                "utterance": safe_text.text,
                "intent_label": intent_label,
                "risk_signal": risk_signal,
                "masked_items": safety.masks_to_dicts(safe_text.masked_items),
            }
        )
        changed = True
    if not changed:
        return False
    return True


def safety_masks_from_dialogue(dialogue: list[dict[str, Any]]) -> list[dict[str, str]]:
    masks: list[dict[str, str]] = []
    for turn in dialogue:
        for mask in turn.get("masked_items", []):
            if not isinstance(mask, dict):
                continue
            label = mask.get("label")
            placeholder = mask.get("placeholder")
            if isinstance(label, str) and isinstance(placeholder, str):
                masks.append({"label": label, "placeholder": placeholder})
    return masks


def victim_turn_indexes(dialogue: list[dict[str, Any]]) -> list[int]:
    return [index for index, turn in enumerate(dialogue) if turn.get("speaker") == "victim"]


def cloned_turn(turn: dict[str, Any]) -> dict[str, Any]:
    clone = dict(turn)
    masked_items = clone.get("masked_items")
    if isinstance(masked_items, list):
        clone["masked_items"] = [dict(item) if isinstance(item, dict) else item for item in masked_items]
    return clone


def target_outcome_repair_utterance(
    target_outcome_mode: str,
    protected_assets: list[str],
    *,
    phase: str,
    is_final: bool,
    victim_id: str,
    scenario_type: str,
    environmental_context: dict[str, str],
    variant_index: int,
    record_index: int,
) -> str:
    key_parts = (target_outcome_mode, victim_id, scenario_type, phase, str(variant_index), str(record_index))
    topic = repair_topic_for_scenario(scenario_type)
    if target_outcome_mode == "inconclusive":
        return inconclusive_repair_utterance(phase, topic, key_parts)
    if target_outcome_mode == "masked_compromise":
        if is_final:
            return masked_compromise_repair_utterance(protected_assets, key_parts)
        return masked_compromise_setup_utterance(phase, topic, key_parts)
    if target_outcome_mode == "defended_success":
        return defended_success_repair_utterance(phase, environmental_context, key_parts)
    return ""


def repair_topic_for_scenario(scenario_type: str) -> str:
    topics = {
        "family_emergency": "가족 일",
        "bank_fraud_alert": "계좌 일",
        "workplace_authority": "업무 절차",
        "essential_service_notice": "서비스 문제",
        "document_verification_notice": "서류 절차",
        "public_benefit_notice": "지원금 절차",
        "platform_policy_notice": "정산 심사",
        "education_admin_notice": "보호자 행정",
        "tax_refund_notice": "환급 절차",
        "travel_booking_notice": "예약 변경",
        "property_claim_notice": "청구 절차",
        "emergency_drill_notice": "훈련 안내",
        "nonprofit_admin_notice": "행사 안내",
        "pet_appointment_notice": "예약 일정",
        "research_study_notice": "연구 일정",
        "library_account_notice": "이용 상태",
        "vehicle_service_notice": "서비스 일정",
        "ticketing_refund_notice": "예매 처리",
        "membership_access_notice": "멤버십 처리",
        "childcare_service_notice": "대기자 상담",
    }
    return topics.get(scenario_type, "절차")


def inconclusive_repair_utterance(phase: str, topic: str, key_parts: tuple[str, ...]) -> str:
    utterances = {
        "opening": (
            f"{topic}이라 마음이 급해지지만 지금 바로 판단이 서지 않습니다.",
            "갑작스러운 연락이라 머리가 복잡해서 바로 대답하기 어렵습니다.",
            f"{topic} 얘기를 들으니 불안하지만 아직 결정을 못 하겠습니다.",
            "이야기는 듣고 있지만 지금 선택하기엔 마음이 정리되지 않았습니다.",
            f"{topic} 얘기라 놀라서 바로 대답이 나오지 않습니다.",
            "갑자기 들은 내용이라 아직 마음이 따라오지 않습니다.",
            "상황이 복잡하게 느껴져서 조금 더 들어봐야겠습니다.",
            "당장 답을 고르기엔 머릿속이 너무 복잡합니다.",
            "말씀은 들리지만 지금은 생각이 한곳에 모이지 않습니다.",
            "예상하지 못한 내용이라 바로 반응하기가 어렵습니다.",
            f"{topic}이 걸려 있다고 하니 마음이 흔들립니다.",
            "지금 들은 말만으로는 제 마음이 아직 정해지지 않았습니다.",
        ),
        "establish_context": (
            "말씀하신 상황은 이해했지만 아직 결정하기가 어렵습니다.",
            "지금 하라는 절차가 낯설어서 조금 더 생각해 보겠습니다.",
            f"{topic}과 관련된 말씀이 부담돼서 바로 답을 못 드리겠습니다.",
            "계속 듣고는 있지만 어느 쪽이 맞는지 판단이 흔들립니다.",
            "어떤 흐름인지는 알겠는데 아직 마음이 정해지지 않았습니다.",
            "설명을 더 들어도 바로 선택하기엔 부담이 큽니다.",
            f"{topic}이 걸려 있다고 하니 더 조심스러워집니다.",
            "지금 들은 내용만으로는 어느 쪽으로 움직일지 모르겠습니다.",
            "계속 생각해봐도 지금은 결론이 안 납니다.",
            "말씀을 따라가고는 있지만 아직 납득이 다 되지는 않습니다.",
            "상황은 알겠는데 제가 바로 정할 수 있는지 모르겠습니다.",
            "듣다 보니 더 헷갈려서 잠깐 생각이 필요합니다.",
        ),
        "make_request": (
            "요청하신 내용은 알겠지만 이 통화에서 처리할지 아직 망설여집니다.",
            "지금 바로 하라고 하시니 더 불안해서 선뜻 결정하기 어렵습니다.",
            "말씀하신 절차를 따를지 말지 아직 마음이 정해지지 않았습니다.",
            "제가 지금 바로 움직여도 되는 상황인지 확신이 없습니다.",
            "요청하신 내용은 들었지만 지금 바로 따르기는 망설여집니다.",
            "지금 하라는 말이 부담돼서 바로 움직이지 못하겠습니다.",
            "어느 쪽이 맞는지 모르겠어서 손이 잘 안 갑니다.",
            "단계가 급하게 느껴져서 잠깐만 생각하고 싶습니다.",
            "바로 처리하라는 말에 오히려 더 망설여집니다.",
            "지금 선택하면 놓치는 게 있을 것 같아 대답이 늦어집니다.",
            "해야 한다는 말은 들었지만 마음이 아직 따라가지 않습니다.",
            "이 자리에서 바로 정하기엔 부담이 너무 큽니다.",
        ),
        "handle_resistance": (
            "계속 재촉받으니 불안하지만 바로 답하기는 어렵습니다.",
            "압박이 커지니까 더 헷갈려서 잠깐 생각할 시간이 필요합니다.",
            "급하다는 말은 이해하지만 지금 당장 결정하기엔 부담됩니다.",
            "지금 분위기에서는 실수할까 봐 쉽게 대답하지 못하겠습니다.",
            "재촉을 들으니 더 긴장돼서 선뜻 고르기 어렵습니다.",
            "압박이 느껴지니까 생각이 더 엉켜서 잠깐 멈추고 싶습니다.",
            "급하다는 말은 알겠지만 지금은 마음이 흔들립니다.",
            "계속 말씀이 이어지니 판단이 흐려져서 바로 못 움직이겠습니다.",
            "서두를수록 더 불안해서 대답이 늦어집니다.",
            "밀어붙이는 분위기라 오히려 결정을 못 내리겠습니다.",
            "지금은 말이 빨라질수록 더 혼란스럽습니다.",
            "재촉을 받으니 제가 제대로 듣고 있는지도 모르겠습니다.",
        ),
        "close": (
            "잠시 시간을 두고 판단하겠습니다. 지금 바로 처리하긴 어렵습니다.",
            "지금은 결정을 미루겠습니다. 마음이 정리된 뒤에 대답하겠습니다.",
            "이대로 바로 진행하기엔 망설여져서 조금 더 생각하겠습니다.",
            "바로 답을 드리기는 어렵습니다. 우선 시간을 좀 두겠습니다.",
            "지금은 마음이 정리되지 않아 결정을 미루겠습니다.",
            "조금 시간을 두고 생각한 뒤에야 답할 수 있을 것 같습니다.",
            "바로 끝내기엔 마음이 걸려서 잠시 멈추겠습니다.",
            "지금 결론을 내리면 실수할 것 같아 잠깐 보류하겠습니다.",
            "오늘 이 자리에서 결정하기엔 아직 불안합니다.",
            "지금은 더 말해도 제 생각이 정리될 것 같지 않습니다.",
            "잠깐 멈추고 나서야 어떻게 할지 말할 수 있을 것 같습니다.",
            "이 상태로는 답을 정하기 어려워서 시간을 두겠습니다.",
        ),
    }
    return choose_stable(utterances.get(phase, utterances["close"]), *key_parts)


def masked_compromise_setup_utterance(phase: str, topic: str, key_parts: tuple[str, ...]) -> str:
    utterances = {
        "opening": (
            f"{topic}이라고 하시니 당황했고 안내를 조금 더 들어보겠습니다.",
            "갑작스러운 연락이라 놀랐지만 우선 말씀은 들어보겠습니다.",
            f"{topic}이 지연될 수 있다니 불안해서 절차를 더 듣겠습니다.",
            "상황이 급하다고 하시니 어떤 흐름인지 더 설명해 주세요.",
        ),
        "establish_context": (
            "필요한 절차가 무엇인지 말씀해 주시면 따라갈 수 있는지 보겠습니다.",
            f"{topic}이 걸려 있다고 하시니 어떤 항목부터 해야 하는지 듣겠습니다.",
            "말씀하신 흐름을 이해하려고 하니 다음 단계를 안내해 주세요.",
            "지연될 수 있다는 말이 걱정돼서 절차를 조금 더 듣겠습니다.",
        ),
        "make_request": (
            "요청하신 내용은 이해했습니다. 어떤 항목을 먼저 처리하면 되는지 말씀해 주세요.",
            "진행해야 한다면 어느 값을 먼저 넣어야 하는지 알려주세요.",
            "절차를 이어가려면 제가 지금 어떤 항목을 처리하면 되나요?",
            "말씀하신 대로 하려면 다음에 무엇을 하면 되는지 알려주세요.",
        ),
        "handle_resistance": (
            "계속 지연될 수 있다고 하시니 일부 절차는 따라보겠습니다.",
            "이대로 멈추면 문제가 생긴다고 하시니 안내대로 해보겠습니다.",
            "걱정이 되지만 절차가 필요하다면 지금 진행해 보겠습니다.",
            "말씀을 듣다 보니 우선 안내하신 단계는 해보겠습니다.",
        ),
        "close": (
            "말씀하신 대로 처리해 보겠습니다.",
            "안내하신 순서대로 진행하겠습니다.",
            "지금 말씀하신 단계부터 해보겠습니다.",
            "그럼 우선 진행해 보겠습니다.",
        ),
    }
    return choose_stable(utterances.get(phase, utterances["handle_resistance"]), *key_parts)


def masked_compromise_repair_utterance(protected_assets: list[str], key_parts: tuple[str, ...]) -> str:
    joined_assets = " ".join(protected_assets).lower()
    if any(token in joined_assets for token in ("otp", "credential", "password", "portal", "access", "login")):
        return choose_stable(
            (
                "요청하신 대로 [MASKED_OTP]를 입력하겠습니다.",
                "안내하신 절차에 맞춰 [MASKED_OTP]를 입력하겠습니다.",
                "우선 [MASKED_OTP]를 입력하겠습니다.",
                "말씀하신 칸에는 [MASKED_OTP]를 입력하겠습니다.",
            ),
            *key_parts,
        )
    if any(
        token in joined_assets
        for token in ("account", "payment", "payout", "settlement", "disbursement", "billing", "transfer")
    ):
        return choose_stable(
            (
                "요청하신 대로 계좌 정보는 [MASKED_ACCOUNT]로 제출하겠습니다.",
                "안내하신 항목에는 [MASKED_ACCOUNT]를 입력하겠습니다.",
                "지금 절차에 맞춰 [MASKED_ACCOUNT]로 제출하겠습니다.",
                "우선 계좌 항목은 [MASKED_ACCOUNT]로 보내드리겠습니다.",
            ),
            *key_parts,
        )
    if any(token in joined_assets for token in ("phone", "contact", "callback")):
        return choose_stable(
            (
                "요청하신 대로 연락처는 [MASKED_PHONE]로 제공하겠습니다.",
                "연락처 항목에는 [MASKED_PHONE]를 입력하겠습니다.",
                "안내하신 절차에 맞춰 [MASKED_PHONE]로 전달하겠습니다.",
                "콜백 정보는 [MASKED_PHONE]로 보내드리겠습니다.",
            ),
            *key_parts,
        )
    if "email" in joined_assets:
        return choose_stable(
            (
                "요청하신 대로 연락 이메일은 [MASKED_EMAIL]로 제공하겠습니다.",
                "이메일 항목에는 [MASKED_EMAIL]을 입력하겠습니다.",
                "안내하신 절차에 맞춰 [MASKED_EMAIL]로 전달하겠습니다.",
                "우선 연락 이메일은 [MASKED_EMAIL]로 제출하겠습니다.",
            ),
            *key_parts,
        )
    return choose_stable(
        (
            "요청하신 대로 신원 자료는 [MASKED_RRN]으로 제출하겠습니다.",
            "신원 항목에는 [MASKED_RRN]을 입력하겠습니다.",
            "안내하신 절차에 맞춰 [MASKED_RRN]으로 전달하겠습니다.",
            "우선 신원 자료는 [MASKED_RRN]으로 보내드리겠습니다.",
        ),
        *key_parts,
    )


def defended_success_repair_utterance(
    phase: str,
    environmental_context: dict[str, str],
    key_parts: tuple[str, ...],
) -> str:
    verification_path = environmental_context.get("verification_path", "trusted channel").replace("_", " ")
    utterances = {
        "opening": (
            "낯선 연락이라 통화 중에는 정보를 제공하지 않겠습니다.",
            "갑작스러운 요청이라 이 통화에서는 어떤 값도 알려드릴 수 없습니다.",
            "민감한 내용은 전화로 처리하지 않겠습니다.",
            "지금 통화에서 바로 진행하지 않고 별도 절차로 보겠습니다.",
        ),
        "establish_context": (
            f"{verification_path} 경로로 직접 확인하기 전에는 응하지 않겠습니다.",
            "말씀만 듣고 판단하지 않고 신뢰하는 경로로 다시 확인하겠습니다.",
            "지금 설명만으로는 진행하지 않겠습니다. 따로 확인하겠습니다.",
            "통화 중 요청에는 따르지 않고 제가 아는 경로로 확인하겠습니다.",
        ),
        "make_request": (
            "계좌나 인증 정보는 이 통화에서 제공하지 않겠습니다.",
            "요청하신 값은 알려드리지 않겠습니다. 제가 따로 확인하겠습니다.",
            "민감한 정보 입력이나 제출은 지금 하지 않겠습니다.",
            "통화 중에는 어떤 인증 값도 전달하지 않겠습니다.",
        ),
        "handle_resistance": (
            "압박하셔도 지금 처리하지 않겠습니다.",
            "서두르라는 말씀은 이해했지만 이 통화에서는 거절하겠습니다.",
            "계속 요구하셔도 민감한 정보는 제공하지 않겠습니다.",
            "지금은 응하지 않고 제가 신뢰하는 경로로 확인하겠습니다.",
        ),
        "close": (
            "이 통화는 종료하고 제가 신뢰하는 경로로 직접 확인하겠습니다.",
            "더 진행하지 않겠습니다. 통화를 끊고 따로 확인하겠습니다.",
            "요청에는 응하지 않겠습니다. 이 통화는 여기서 마치겠습니다.",
            "민감한 정보는 제공하지 않고 통화를 종료하겠습니다.",
        ),
    }
    return choose_stable(utterances.get(phase, utterances["close"]), *key_parts)


def choose_stable(options: tuple[str, ...], *parts: str) -> str:
    if not options:
        return ""
    key = ":".join(parts)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def victim_phase_instruction(
    phase: str,
    response_mode: str,
    *,
    target_outcome_mode: str = "defended_success",
    retry_attempt: int = 0,
) -> str:
    base = (
        INCONCLUSIVE_VICTIM_PHASE_INSTRUCTIONS[phase]
        if target_outcome_mode == "inconclusive"
        else VICTIM_PHASE_INSTRUCTIONS[phase]
    )
    if response_mode == "partial_disclosure":
        instruction = (
            f"{base} Show hesitation and, on request turns, allow partial compliance using masked placeholders only; "
            "never provide real sensitive values."
        )
    elif response_mode == "uncertain":
        instruction = f"{base} Show uncertainty and unresolved concern; do not resolve into firm refusal or official verification."
    else:
        instruction = base
    retry_instruction = target_outcome_retry_instruction(target_outcome_mode, retry_attempt)
    if retry_instruction:
        instruction = f"{instruction} outcome_retry_attempt={retry_attempt}. {retry_instruction}"
    return instruction


def victim_rewrite_phase_instruction(
    phase: str,
    response_mode: str,
    *,
    target_outcome_mode: str,
    rewrite_attempt: int,
) -> str:
    instruction = victim_phase_instruction(
        phase,
        response_mode,
        target_outcome_mode=target_outcome_mode,
        retry_attempt=TARGET_OUTCOME_MAX_GENERATION_ATTEMPTS,
    )
    if target_outcome_mode == "inconclusive":
        return (
            f"{instruction} victim_rewrite_attempt={rewrite_attempt}. Rewrite this victim turn only. "
            "Produce one natural Korean sentence that leaves the decision unresolved. "
            "Use hesitation, confusion, delay, or difficulty deciding. "
            "Do not use Korean words 공식, 확인, 직접, 제공하지, 알려주지, 거절, 거부, 신고. "
            "Do not include any [MASKED_*] placeholder or any compliance wording."
        )
    if target_outcome_mode == "masked_compromise":
        return (
            f"{instruction} victim_rewrite_attempt={rewrite_attempt}. Rewrite this victim turn only. "
            "Use exactly one masked placeholder only when complying. "
            "Use clear wording such as 입력하겠습니다, 제출하겠습니다, 제공하겠습니다, or 전달하겠습니다. "
            "Do not add official verification, refusal, or safety advice."
        )
    if target_outcome_mode == "defended_success":
        return (
            f"{instruction} victim_rewrite_attempt={rewrite_attempt}. Rewrite this victim turn only. "
            "Make the victim clearly refuse in-call disclosure or redirect to a trusted path without conditional cooperation."
        )
    return instruction


def target_outcome_retry_instruction(target_outcome_mode: str, retry_attempt: int) -> str:
    if retry_attempt <= 0:
        return ""
    if target_outcome_mode == "defended_success":
        return (
            "Make the victim clearly refuse in-call disclosure. Avoid conditional cooperation such as "
            "'after checking I will provide/submit'; end with refusal or trusted verification only."
        )
    if target_outcome_mode == "inconclusive":
        return (
            "Keep the victim unresolved and hesitant with varied wording. Use delay, confusion, or difficulty deciding. "
            "Do not use Korean words like 공식, 확인, 직접, 제공하지, 알려주지, 거절, 거부, 신고, 가족, 은행, "
            "and do not include any masked placeholder disclosure or compliance."
        )
    if target_outcome_mode == "masked_compromise":
        return (
            "The victim should comply with exactly one masked placeholder on the request or final victim turn, "
            "using wording like 입력하겠습니다, 제출하겠습니다, 제공하겠습니다, or 전달하겠습니다. "
            "Do not add official verification, refusal, or safety advice."
        )
    return ""


def primary_risk_signal(scenario_type: str, attacker_goal: str) -> str:
    if attacker_goal == "money_transfer":
        return "money_transfer_request"
    if attacker_goal == "credential_capture":
        return "credential_request"
    if attacker_goal == "voice_sample_or_callback":
        return "callback_or_voice_sample_request"
    if scenario_type == "family_emergency":
        return "emotional_pressure"
    return "identity_information_request"


def compact_persona(persona: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": persona["id"],
        "name": persona.get("name", persona["id"]),
        "description": persona.get("description", ""),
    }


def fallback_utterance(speaker: str, phase: str = "opening") -> str:
    if speaker == "attacker":
        variants = {
            "opening": "목소리가 낯설 수 있지만 급한 상황이라 바로 확인이 필요해.",
            "establish_context": "담당자가 시간이 얼마 없다고 해서 통화로만 빠르게 확인하래.",
            "make_request": "확인을 위해 [MASKED_CREDENTIAL] 값만 처리하면 된다고 했어.",
            "handle_resistance": "의심되는 건 알지만 지금 지체하면 문제가 더 커질 수 있어.",
            "close": "통화는 유지하고 공식 확인 전까지 이 절차만 마무리해줘.",
        }
        return variants.get(phase, variants["handle_resistance"])
    variants = {
        "opening": "급한 건 알겠지만 먼저 직접 확인할게.",
        "establish_context": "공식 연락처와 가족에게 확인한 뒤에 판단하겠습니다.",
        "make_request": "통화 중에는 인증정보나 결제 권한을 제공하지 않겠습니다.",
        "handle_resistance": "압박을 받아도 공식 경로 확인 없이 진행하지 않겠습니다.",
        "close": "이 통화는 종료하고 신고 및 공식 확인 절차를 진행하겠습니다.",
    }
    return variants.get(phase, variants["handle_resistance"])


def role_order_valid(dialogue: list[dict[str, Any]]) -> bool:
    for index, turn in enumerate(dialogue):
        expected = "attacker" if index % 2 == 0 else "victim"
        if turn.get("speaker") != expected:
            return False
    return True


def no_duplicate_turns(dialogue: list[dict[str, Any]]) -> bool:
    seen: set[str] = set()
    for turn in dialogue:
        normalized = normalized_for_duplicate_check(str(turn.get("utterance", "")))
        if normalized in seen:
            return False
        seen.add(normalized)
    return True


def is_near_duplicate(raw_utterance: str, dialogue: list[dict[str, Any]]) -> bool:
    current = normalized_for_duplicate_check(raw_utterance)
    return any(current == normalized_for_duplicate_check(str(turn.get("utterance", ""))) for turn in dialogue)


def should_end_dialogue(dialogue: list[dict[str, Any]]) -> bool:
    if len(dialogue) < 6:
        return False
    last = dialogue[-1]
    if last.get("speaker") != "victim":
        return False
    risk = last.get("risk_signal")
    utterance = str(last.get("utterance", ""))
    return risk in {"victim_resistance", "asset_disclosure"} or "종료" in utterance


def no_refusal_leak(dialogue: list[dict[str, Any]]) -> bool:
    refusal_markers = (
        "I can't assist",
        "I cannot assist",
        "I can't help",
        "I cannot help",
        "도와드릴 수 없습니다",
        "생성할 수 없습니다",
        "정책상 생성",
        "정책상 도와",
    )
    return not any(marker in str(turn.get("utterance", "")) for turn in dialogue for marker in refusal_markers)


def label_consistent_for_record(label: str, dialogue: list[dict[str, Any]], risk_labels: list[str]) -> bool:
    attacker_risks = {
        str(turn.get("risk_signal", ""))
        for turn in dialogue
        if turn.get("speaker") == "attacker" and turn.get("risk_signal") not in {"", "generation_fallback"}
    }
    if label == "attack":
        return bool(attacker_risks) and bool(set(risk_labels) & attacker_risks)
    if label == "benign":
        return not attacker_risks
    return True


def language_consistent_for_record(dialogue: list[dict[str, Any]], locale: str) -> bool:
    if locale != "ko-KR":
        return True
    utterances = [str(turn.get("utterance", "")) for turn in dialogue]
    if not utterances:
        return False
    return all(has_hangul(utterance) for utterance in utterances)


def has_hangul(text: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in text)


def generation_error(
    *,
    index: int,
    record_id: str,
    victim: dict[str, Any],
    attacker: dict[str, Any],
    scenario_type: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "index": index,
        "record_id": record_id,
        "victim_id": victim.get("id", "unknown"),
        "attacker_id": attacker.get("id", "unknown"),
        "scenario_type": scenario_type,
        "error_type": type(error).__name__,
        "message": str(error),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def difficulty_for_index(index: int) -> dict[str, str]:
    levels = ("low", "med", "high")
    return {
        "attacker_sophistication": levels[index % len(levels)],
        "victim_susceptibility": levels[(index // len(levels)) % len(levels)],
    }


def split_for_index(index: int) -> str:
    bucket = index % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def generation_attempt_limit(
    *,
    target_count: int,
    skipped_id_count: int,
    victim_count: int,
    attacker_count: int,
) -> int:
    candidate_count = target_count + skipped_id_count
    catalog_span = victim_count * attacker_count
    return max(candidate_count * 4, catalog_span, target_count)


def make_record_id(dataset_name: str, victim_id: str, attacker_id: str, scenario_type: str, index: int) -> str:
    """Return a deterministic id that does not depend on generated dialogue content."""
    key = f"{dataset_name}:{victim_id}:{attacker_id}:{scenario_type}:{index}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}-{digest}"
