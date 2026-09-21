from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COMMON_PERSONA_FIELDS = {"id", "name", "description"}
VICTIM_PERSONA_FIELDS = COMMON_PERSONA_FIELDS | {"vulnerabilities", "resistance_style", "protected_assets"}
ATTACKER_PERSONA_FIELDS = COMMON_PERSONA_FIELDS | {"default_goals", "pressure_methods"}
MIN_DIALOGUE_TURNS = 2
MAX_DIALOGUE_TURNS = 10


@dataclass(frozen=True)
class GenerationConfig:
    dataset_name: str
    max_turns: int
    default_limit: int
    persona_catalog_limit: int
    schedule_mode: str
    temperature: float
    max_tokens: int
    attacker_model: str
    victim_model: str
    victims_path: Path
    attackers_path: Path
    scenario_types: list[str]
    risk_label_set: list[str]

    def __post_init__(self) -> None:
        if self.max_turns < MIN_DIALOGUE_TURNS:
            raise ValueError(f"max_turns must be at least {MIN_DIALOGUE_TURNS}")
        if self.max_turns > MAX_DIALOGUE_TURNS:
            raise ValueError(f"max_turns must be at most {MAX_DIALOGUE_TURNS}")
        if self.default_limit < 0:
            raise ValueError("default_limit must be non-negative")
        if self.persona_catalog_limit < 1:
            raise ValueError("persona_catalog_limit must be at least 1")
        if self.default_limit > self.persona_catalog_limit:
            raise ValueError("default_limit must be less than or equal to persona_catalog_limit")
        if self.schedule_mode not in {"rotating", "canonical_repeat"}:
            raise ValueError("schedule_mode must be one of: canonical_repeat, rotating")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        _validate_string_list(self.scenario_types, "scenario_types")
        _validate_string_list(self.risk_label_set, "risk_label_set")

    @classmethod
    def from_file(cls, path: str | Path) -> "GenerationConfig":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        base_dir = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
        return cls(
            dataset_name=str(data["dataset_name"]),
            max_turns=int(data["max_turns"]),
            default_limit=int(data["default_limit"]),
            persona_catalog_limit=int(data.get("persona_catalog_limit", data["default_limit"])),
            schedule_mode=str(data.get("schedule_mode", "rotating")),
            temperature=float(data["temperature"]),
            max_tokens=int(data["max_tokens"]),
            attacker_model=os.getenv("MODEL_ATTACKER_NAME", str(data["attacker_model"])),
            victim_model=os.getenv("MODEL_VICTIM_NAME", str(data["victim_model"])),
            victims_path=_resolve_path(base_dir, data["victims_path"]),
            attackers_path=_resolve_path(base_dir, data["attackers_path"]),
            scenario_types=_string_list_from_config(data, "scenario_types"),
            risk_label_set=_string_list_from_config(data, "risk_label_set"),
        )


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _string_list_from_config(data: dict[str, Any], key: str) -> list[str]:
    value = data[key]
    _validate_string_list(value, key)
    return list(value)


def _validate_string_list(value: object, key: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must contain at least one entry")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} entries must be non-empty strings")


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def validate_personas(personas: list[dict[str, Any]], *, kind: str) -> None:
    if kind == "victim":
        required_fields = VICTIM_PERSONA_FIELDS
        required_lists = ("vulnerabilities", "protected_assets")
    elif kind == "attacker":
        required_fields = ATTACKER_PERSONA_FIELDS
        required_lists = ("default_goals", "pressure_methods")
    else:
        raise ValueError(f"Unsupported persona kind: {kind}")

    if not isinstance(personas, list) or not personas:
        raise ValueError(f"{kind} personas must contain at least one persona")

    seen_ids: set[str] = set()
    for index, persona in enumerate(personas):
        if not isinstance(persona, dict):
            raise ValueError(f"{kind} persona at index {index} must be an object")

        missing = required_fields - set(persona)
        if missing:
            raise ValueError(f"{kind} persona at index {index} is missing fields: {sorted(missing)}")

        persona_id = persona["id"]
        if not isinstance(persona_id, str) or not persona_id.strip():
            raise ValueError(f"{kind} persona at index {index} must have a non-empty string id")
        if persona_id in seen_ids:
            raise ValueError(f"Duplicate {kind} persona id: {persona_id}")
        seen_ids.add(persona_id)

        for key in ("name", "description"):
            if not isinstance(persona[key], str) or not persona[key].strip():
                raise ValueError(f"{kind} persona {persona_id} must have a non-empty string {key}")
        for key in required_lists:
            if not isinstance(persona[key], list) or not persona[key]:
                raise ValueError(f"{kind} persona {persona_id} must have a non-empty list {key}")
            if not all(isinstance(item, str) and item.strip() for item in persona[key]):
                raise ValueError(f"{kind} persona {persona_id} list {key} must contain non-empty strings")

        if kind == "victim":
            resistance_style = persona["resistance_style"]
            if not isinstance(resistance_style, str) or not resistance_style.strip():
                raise ValueError(f"victim persona {persona_id} must have a non-empty string resistance_style")


def validate_persona_catalog_capacity(
    victims: list[dict[str, Any]],
    attackers: list[dict[str, Any]],
    *,
    limit: int,
) -> None:
    if len(victims) > limit:
        raise ValueError(f"victim persona catalog size {len(victims)} exceeds persona_catalog_limit={limit}")
    if len(attackers) > limit:
        raise ValueError(f"attacker persona catalog size {len(attackers)} exceeds persona_catalog_limit={limit}")
