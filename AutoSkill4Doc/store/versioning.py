"""
LLM-driven registry/version registration stage for the standalone document pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from autoskill import AutoSkill
from autoskill.llm.base import LLM
from autoskill.llm.factory import build_llm
from autoskill.models import Skill, SkillStatus
from autoskill.utils.time import now_iso

from ..core.common import (
    StageLogger,
    StageProgressCallback,
    emit_stage_log,
    emit_stage_progress,
    normalize_text,
    summarize_names,
)
from ..core.config import DEFAULT_DOC_SKILL_USER_ID, DEFAULT_RETRIEVAL_SCORE_THRESHOLD
from ..core.llm_utils import (
    clip_confidence,
    coerce_str_list,
    compact_text_list,
    llm_complete_json,
    maybe_json_dict,
)
from ..models import (
    DocumentRecord,
    SkillLifecycle,
    SkillSpec,
    SupportRecord,
    SupportRelation,
    VersionState,
)
from ..stages.compiler import _build_structured_prompt, _coerce_examples
from ..taxonomy import load_skill_taxonomy
from .registry import DocumentRegistry
from .retrieval import DEFAULT_RETRIEVAL_LIMIT, DocumentSkillRetriever, build_document_skill_retriever
from .staging import plain_skill_specs, write_registration_staging
from .visible_tree import sync_visible_skill_tree

_ACTIVE_STORE_STATES = {
    VersionState.CANDIDATE,
    VersionState.DRAFT,
    VersionState.EVALUATING,
    VersionState.ACTIVE,
    VersionState.WATCHLIST,
}


def _bump_patch(version: str) -> str:
    """Bumps a semantic version patch number."""

    parts = [p for p in str(version or "").split(".") if p.strip().isdigit()]
    if len(parts) != 3:
        return "0.1.1"
    major, minor, patch = (int(parts[0]), int(parts[1]), int(parts[2]))
    return f"{major}.{minor}.{patch + 1}"


def _plain_skill(skill: Any) -> Dict[str, Any]:
    """Serializes one persisted AutoSkill store record into a compact dict."""

    return {
        "id": str(getattr(skill, "id", "") or ""),
        "name": str(getattr(skill, "name", "") or ""),
        "description": str(getattr(skill, "description", "") or ""),
        "version": str(getattr(skill, "version", "") or ""),
        "status": str(getattr(getattr(skill, "status", None), "value", getattr(skill, "status", "")) or ""),
    }


def _copy_skill(
    skill: SkillSpec,
    *,
    skill_id: Optional[str] = None,
    version: Optional[str] = None,
    status: Optional[VersionState] = None,
    support_ids: Optional[List[str]] = None,
    metadata_update: Optional[Dict[str, Any]] = None,
) -> SkillSpec:
    """Creates a skill copy with updated identity/version/status fields."""

    payload = skill.to_dict()
    if skill_id is not None:
        payload["skill_id"] = str(skill_id or "").strip()
    if version is not None:
        payload["version"] = str(version or "0.1.0")
    if status is not None:
        payload["status"] = status.value
    if support_ids is not None:
        payload["support_ids"] = list(support_ids or [])
    md = dict(payload.get("metadata") or {})
    if metadata_update:
        md.update(dict(metadata_update or {}))
    payload["metadata"] = md
    return SkillSpec.from_dict(payload)


def _copy_skill_hierarchy(
    skill: SkillSpec,
    *,
    parent_skill_id: Optional[str] = None,
    parent_candidate_ids: Optional[List[str]] = None,
    child_skill_ids: Optional[List[str]] = None,
    hierarchy_confidence: Optional[float] = None,
    hierarchy_status: Optional[str] = None,
    visible_role: Optional[str] = None,
) -> SkillSpec:
    """Returns one skill copy with updated hierarchy fields."""

    payload = skill.to_dict()
    if parent_skill_id is not None:
        payload["parent_skill_id"] = str(parent_skill_id or "").strip()
    if parent_candidate_ids is not None:
        payload["parent_candidate_ids"] = list(parent_candidate_ids or [])
    if child_skill_ids is not None:
        payload["child_skill_ids"] = list(child_skill_ids or [])
    if hierarchy_confidence is not None:
        payload["hierarchy_confidence"] = float(hierarchy_confidence or 0.0)
    if hierarchy_status is not None:
        payload["hierarchy_status"] = str(hierarchy_status or "").strip()
    if visible_role is not None:
        payload["visible_role"] = str(visible_role or "").strip()
    md = dict(payload.get("metadata") or {})
    if parent_skill_id is not None:
        md["parent_skill_id"] = str(parent_skill_id or "").strip()
    if parent_candidate_ids is not None:
        md["parent_candidate_ids"] = list(parent_candidate_ids or [])
    if child_skill_ids is not None:
        md["child_skill_ids"] = list(child_skill_ids or [])
    if hierarchy_confidence is not None:
        md["hierarchy_confidence"] = float(hierarchy_confidence or 0.0)
    if hierarchy_status is not None:
        md["hierarchy_status"] = str(hierarchy_status or "").strip()
    if visible_role is not None:
        md["visible_role"] = str(visible_role or "").strip()
    payload["metadata"] = md
    return SkillSpec.from_dict(payload)


def _copy_support(
    support: SupportRecord,
    *,
    skill_id: str,
    relation_type: Optional[SupportRelation] = None,
    metadata_update: Optional[Dict[str, Any]] = None,
) -> SupportRecord:
    """Creates a support copy rebound to one canonical skill id."""

    payload = support.to_dict()
    payload["skill_id"] = str(skill_id or "").strip()
    if relation_type is not None:
        payload["relation_type"] = relation_type.value
    md = dict(payload.get("metadata") or {})
    if metadata_update:
        md.update(dict(metadata_update or {}))
    payload["metadata"] = md
    return SupportRecord.from_dict(payload)


def _same_asset_layer(left: SkillSpec, right: SkillSpec) -> bool:
    """Checks whether two skills live at the same asset type and granularity."""

    return (
        str(left.asset_type or "").strip() == str(right.asset_type or "").strip()
        and str(left.granularity or "").strip() == str(right.granularity or "").strip()
    )


def _same_asset_node(left: SkillSpec, right: SkillSpec) -> bool:
    """Checks whether two skills live under the same configured hierarchy node."""

    left_node = str(getattr(left, "asset_node_id", "") or (left.metadata or {}).get("asset_node_id") or "").strip()
    right_node = str(getattr(right, "asset_node_id", "") or (right.metadata or {}).get("asset_node_id") or "").strip()
    if left_node and right_node:
        return left_node == right_node
    return True


def _granularity_rank(value: str) -> int:
    """Maps granularity labels into a stable coarse-to-fine rank."""

    raw = str(value or "").strip().lower()
    if raw == "macro":
        return 0
    if raw == "micro":
        return 2
    return 1


def _doc_ids_from_support_ids(support_ids: Sequence[str], support_by_id: Dict[str, SupportRecord]) -> List[str]:
    """Collects source document ids for one support reference set."""

    seen = set()
    out: List[str] = []
    for support_id in support_ids or []:
        support = support_by_id.get(str(support_id or "").strip())
        if support is None:
            continue
        doc_id = str(support.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _plausible_parent_candidate(child: SkillSpec, parent: SkillSpec, *, score: float) -> bool:
    """Returns whether one parent hit is strong enough for automatic fallback attachment."""

    if float(score or 0.0) >= 0.05:
        return True
    for field in ("task_family", "method_family", "stage"):
        left = normalize_text(getattr(child, field, ""), lower=True)
        right = normalize_text(getattr(parent, field, ""), lower=True)
        if left and right and left == right:
            return True
    child_tokens = {token for token in normalize_text(f"{child.name} {child.objective}", lower=True).split() if token}
    parent_tokens = {token for token in normalize_text(f"{parent.name} {parent.objective}", lower=True).split() if token}
    return len(child_tokens & parent_tokens) >= 2


def _needs_peer_context(
    skill: SkillSpec,
    *,
    peer_skills: Sequence[SkillSpec],
    existing_skills: Sequence[SkillSpec],
) -> bool:
    """Returns whether classify_change() should include same-batch peers."""

    if not peer_skills or not existing_skills:
        return False
    child_rank = _granularity_rank(skill.granularity)
    child_level = int(skill.asset_level or 0)
    for existing in existing_skills or []:
        if _granularity_rank(existing.granularity) < child_rank:
            return True
        existing_level = int(existing.asset_level or 0)
        if child_level > 0 and existing_level > 0 and existing_level < child_level:
            return True
    return False


def _conflicting_support_ids(support_ids: Sequence[str], support_by_id: Dict[str, SupportRecord]) -> List[str]:
    """Collects support ids explicitly marked as conflicts."""

    out: List[str] = []
    seen = set()
    for support_id in support_ids or []:
        support = support_by_id.get(str(support_id or "").strip())
        if support is None or support.relation_type != SupportRelation.CONFLICT:
            continue
        if support.support_id in seen:
            continue
        seen.add(support.support_id)
        out.append(support.support_id)
    return out


def _support_lookup(
    registry: DocumentRegistry,
    support_records: Sequence[SupportRecord],
) -> Dict[str, SupportRecord]:
    """Builds a combined support index from registry state and the current batch."""

    out = {support.support_id: support for support in registry.list_supports()}
    for support in support_records or []:
        out[support.support_id] = support
    return out


@dataclass
class ChangeDecision:
    """Decision returned by LLM-backed skill change classification."""

    action: str
    skill: SkillSpec
    matched_skill_ids: List[str] = field(default_factory=list)
    reason: str = ""
    split_parent_id: str = ""


@dataclass
class VersionRegistrationResult:
    """Output of the registry/version registration stage."""

    documents: List[DocumentRecord] = field(default_factory=list)
    support_records: List[SupportRecord] = field(default_factory=list)
    skill_specs: List[SkillSpec] = field(default_factory=list)
    hierarchy_updates: List[SkillSpec] = field(default_factory=list)
    lifecycles: List[SkillLifecycle] = field(default_factory=list)
    change_logs: List[Dict[str, Any]] = field(default_factory=list)
    version_history: List[Dict[str, Any]] = field(default_factory=list)
    provenance_links: List[Dict[str, Any]] = field(default_factory=list)
    upserted_store_skills: List[Dict[str, Any]] = field(default_factory=list)
    staging_runs: List[Dict[str, Any]] = field(default_factory=list)
    visible_tree: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)
    dry_run: bool = False


def _layout_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Collects visible-layout metadata carried by one run."""

    md = dict(metadata or {})
    family_name = str(md.get("family_name") or md.get("school_name") or "").strip()
    return {
        "family_name": family_name,
        "family_id": str(md.get("family_id") or "").strip(),
        "profile_id": str(md.get("profile_id") or "").strip(),
        "taxonomy_axis": str(md.get("taxonomy_axis") or "").strip(),
        "taxonomy_class": str(md.get("taxonomy_class") or "").strip(),
        "domain_root_name": str(md.get("domain_root_name") or "").strip(),
        "domain_root_id": str(md.get("domain_root_id") or "").strip(),
        "family_bucket_label": str(md.get("family_bucket_label") or "").strip(),
        "visible_levels": dict(md.get("visible_levels") or {}) if isinstance(md.get("visible_levels"), dict) else {},
        "child_type": str(md.get("child_type") or "").strip(),
    }


def _merge_layout_metadata(
    skill: SkillSpec,
    *,
    metadata: Optional[Dict[str, Any]],
) -> SkillSpec:
    """Attaches run-level layout metadata to a skill spec for later visible-tree sync."""

    layout_md = {key: value for key, value in _layout_metadata(metadata).items() if value}
    if not layout_md:
        return skill
    existing = dict(skill.metadata or {})
    merged_update: Dict[str, Any] = {}
    for key, value in layout_md.items():
        current = existing.get(key)
        if isinstance(current, dict) and current:
            continue
        if isinstance(current, list) and current:
            continue
        if str(current or "").strip():
            continue
        merged_update[key] = value
    if not merged_update:
        return skill
    return _copy_skill(skill, metadata_update=merged_update)


def _store_root_from_context(*, registry: DocumentRegistry, sdk: Optional[AutoSkill]) -> str:
    """Infers the visible skill library root from the SDK or registry location."""

    if sdk is not None:
        raw = str(getattr(getattr(sdk, "config", None), "store", {}).get("path") or "").strip()
        if raw:
            return os.path.abspath(os.path.expanduser(raw))
    registry_root = os.path.abspath(os.path.expanduser(str(registry.root_dir or "").strip()))
    runtime_dir = os.path.dirname(registry_root)
    if os.path.basename(runtime_dir) == ".runtime":
        return os.path.dirname(runtime_dir)
    if os.path.basename(registry_root) == ".runtime":
        return os.path.dirname(registry_root)
    return os.path.dirname(registry_root)


def _store_status_for_skill(skill: SkillSpec) -> SkillStatus:
    """Maps one document lifecycle state into the final AutoSkill store status."""

    return SkillStatus.ACTIVE if skill.status in _ACTIVE_STORE_STATES else SkillStatus.ARCHIVED


def _store_metadata_for_skill(skill: SkillSpec, *, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Builds final store metadata from one reconciled document skill."""

    merged = dict(metadata or {})
    merged.update(dict(skill.metadata or {}))
    merged["_autoskill_asset_type"] = str(skill.asset_type or "").strip()
    merged["_autoskill_granularity"] = str(skill.granularity or "").strip()
    merged["_autoskill_asset_node_id"] = str(skill.asset_node_id or "").strip()
    merged["asset_type"] = str(skill.asset_type or "").strip()
    merged["granularity"] = str(skill.granularity or "").strip()
    merged["asset_node_id"] = str(skill.asset_node_id or "").strip()
    merged["asset_path"] = str(skill.asset_path or "").strip()
    merged["asset_level"] = int(skill.asset_level or 0)
    merged["parent_skill_id"] = str(skill.parent_skill_id or "").strip()
    merged["child_skill_ids"] = list(skill.child_skill_ids or [])
    merged["hierarchy_confidence"] = float(skill.hierarchy_confidence or 0.0)
    merged["hierarchy_status"] = str(skill.hierarchy_status or "").strip()
    merged["visible_role"] = str(skill.visible_role or "").strip()
    merged["source_type"] = "document_skill"
    return merged


def _store_source_for_skill(skill: SkillSpec) -> Dict[str, Any]:
    """Builds stable source metadata carried into the final store skill."""

    return {
        "source_type": "document_skill",
        "skill_spec_id": skill.skill_id,
        "asset_type": skill.asset_type,
        "granularity": skill.granularity,
        "asset_node_id": skill.asset_node_id,
        "asset_path": skill.asset_path,
        "asset_level": skill.asset_level,
        "parent_skill_id": skill.parent_skill_id,
        "child_skill_ids": list(skill.child_skill_ids or []),
        "hierarchy_confidence": skill.hierarchy_confidence,
        "hierarchy_status": skill.hierarchy_status,
        "visible_role": skill.visible_role,
        "objective": skill.objective,
        "support_ids": list(skill.support_ids or []),
        "domain": skill.domain,
        "task_family": skill.task_family,
        "method_family": skill.method_family,
        "stage": skill.stage,
        "version": skill.version,
        "status": skill.status.value,
    }


def _store_files_for_skill(skill: SkillSpec) -> Dict[str, str]:
    """Returns any extra files that should be persisted with the final store skill."""

    files = maybe_json_dict((skill.metadata or {}).get("files"))
    if not isinstance(files, dict):
        return {}
    return {str(path): str(content) for path, content in files.items()}


def _store_skill_from_spec(
    skill: SkillSpec,
    *,
    user_id: str,
    metadata: Optional[Dict[str, Any]],
) -> Skill:
    """Builds one final AutoSkill store skill directly from a reconciled document skill."""

    return Skill(
        id=str(skill.skill_id or "").strip(),
        user_id=str(user_id or "").strip() or DEFAULT_DOC_SKILL_USER_ID,
        name=str(skill.name or "").strip(),
        description=str(skill.description or "").strip(),
        instructions=str(skill.skill_body or "").strip(),
        triggers=list(skill.triggers or []),
        examples=list(skill.examples or []),
        tags=list(skill.tags or []),
        version=str(skill.version or "0.1.0"),
        status=_store_status_for_skill(skill),
        files=_store_files_for_skill(skill),
        source=_store_source_for_skill(skill),
        metadata=_store_metadata_for_skill(skill, metadata=metadata),
        updated_at=now_iso(),
    )


def _sync_store_skills(
    *,
    sdk: AutoSkill,
    skill_specs: Sequence[SkillSpec],
    user_id: str,
    metadata: Optional[Dict[str, Any]],
    logger: StageLogger,
) -> Tuple[List[Skill], List[Skill]]:
    """
    Persists reconciled document skills directly into the final store.

    This bypasses the generic SDK maintainer so AutoSkill4Doc keeps control over
    asset-layer boundaries such as `asset_type` and `granularity`.
    """

    effective_user = str(user_id or "").strip() or DEFAULT_DOC_SKILL_USER_ID
    deduped: Dict[str, SkillSpec] = {}
    for item in list(skill_specs or []):
        key = str(item.skill_id or "").strip()
        if key:
            deduped[key] = item

    touched: List[Skill] = []
    for item in deduped.values():
        persisted = _store_skill_from_spec(item, user_id=effective_user, metadata=metadata)
        existing = sdk.store.get(persisted.id)
        if persisted.status == SkillStatus.ARCHIVED and existing is None:
            continue
        sdk.store.upsert(persisted)
        touched.append(persisted)

    try:
        current_active = list(sdk.store.list(user_id=effective_user) or [])
    except Exception:
        current_active = []

    emit_stage_log(
        logger,
        f"[register_versions] store_sync touched={len(touched)} active={len(current_active)} names={summarize_names([skill.name for skill in touched])}",
    )
    return touched, current_active


def _staging_bucket_for_skill(skill: SkillSpec, *, metadata: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    """Builds one `(profile_id, family_id, child_type)` staging bucket tuple."""

    md = dict(metadata or {})
    skill_md = dict(skill.metadata or {})
    profile_id = (
        str(skill_md.get("profile_id") or "").strip()
        or str(md.get("profile_id") or "").strip()
        or "document_profile"
    )
    family_id = (
        str(skill_md.get("family_name") or "").strip()
        or str(skill_md.get("school_name") or "").strip()
        or str(md.get("family_name") or "").strip()
        or str(skill_md.get("taxonomy_class") or "").strip()
        or str(md.get("school_name") or "").strip()
        or str(skill.domain or "").strip()
        or str(skill.method_family or "").strip()
        or "unknown_family"
    )
    child_type = (
        str(skill_md.get("child_type") or "").strip()
        or str(skill.task_family or "").strip()
        or str(skill.asset_type or "").strip()
        or str(md.get("child_type") or "").strip()
        or "general_child"
    )
    return profile_id, family_id, child_type


class VersionManager:
    """LLM-backed version and lifecycle manager for document builds."""

    def __init__(
        self,
        *,
        registry: DocumentRegistry,
        llm: LLM,
        retriever: Optional[DocumentSkillRetriever] = None,
        retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
        logger: StageLogger = None,
        progress_callback: StageProgressCallback = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.retriever = retriever
        self.retrieval_limit = max(1, int(retrieval_limit or DEFAULT_RETRIEVAL_LIMIT))
        self.logger = logger
        self.progress_callback = progress_callback

    @staticmethod
    def _compact_text(value: Any, *, limit: int = 240) -> str:
        """Returns one short normalized text field suitable for hot-path LLM payloads."""

        text = normalize_text(str(value or ""))
        if len(text) <= max(0, int(limit or 0)):
            return text
        return text[: max(0, int(limit or 0))].rstrip() + "..."

    def _support_summary_for_llm(self, skill: SkillSpec, *, support_by_id: Dict[str, SupportRecord]) -> Dict[str, Any]:
        """Builds one compact support summary for version decisions."""

        supports = [
            support_by_id.get(str(support_id or "").strip())
            for support_id in list(skill.support_ids or [])
        ]
        resolved = [support for support in supports if support is not None]
        relation_counts: Dict[str, int] = {}
        conflict_count = 0
        for support in resolved:
            relation = str(support.relation_type.value or "").strip()
            relation_counts[relation] = int(relation_counts.get(relation) or 0) + 1
            if support.relation_type == SupportRelation.CONFLICT:
                conflict_count += 1
        return {
            "count": len(resolved),
            "conflict_count": conflict_count,
            "relation_counts": relation_counts,
            "sections": compact_text_list([str(support.section or "").strip() for support in resolved], limit=4),
        }

    def _skill_for_llm(self, skill: SkillSpec, *, support_by_id: Dict[str, SupportRecord]) -> Dict[str, Any]:
        return {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "prompt": skill.skill_body,
            "asset_type": skill.asset_type,
            "granularity": skill.granularity,
            "asset_node_id": skill.asset_node_id,
            "asset_path": skill.asset_path,
            "asset_level": skill.asset_level,
            "visible_role": skill.visible_role,
            "objective": skill.objective,
            "domain": skill.domain,
            "task_family": skill.task_family,
            "method_family": skill.method_family,
            "stage": skill.stage,
            "applicable_signals": list(skill.applicable_signals or []),
            "contraindications": list(skill.contraindications or []),
            "intervention_moves": list(skill.intervention_moves or []),
            "triggers": list(skill.triggers or []),
            "workflow_steps": list(skill.workflow_steps or []),
            "constraints": list(skill.constraints or []),
            "cautions": list(skill.cautions or []),
            "output_contract": list(skill.output_contract or []),
            "examples": [
                {
                    "input": str(example.input or ""),
                    "output": str(example.output or ""),
                    "notes": str(example.notes or ""),
                }
                for example in list(skill.examples or [])
            ],
            "tags": list(skill.tags or []),
            "support_ids": list(skill.support_ids or []),
            "support_excerpt_summaries": [
                {
                    "support_id": support.support_id,
                    "relation_type": support.relation_type.value,
                    "section": support.section,
                    "excerpt": support.excerpt,
                }
                for support_id in list(skill.support_ids or [])
                for support in [support_by_id.get(str(support_id or "").strip())]
                if support is not None
            ],
            "version": skill.version,
            "status": skill.status.value,
            "metadata": dict(skill.metadata or {}),
        }

    def _skill_for_change_llm(self, skill: SkillSpec, *, support_by_id: Dict[str, SupportRecord]) -> Dict[str, Any]:
        """Builds one compact skill payload for high-frequency classify_change() calls."""

        return {
            "skill_id": skill.skill_id,
            "name": self._compact_text(skill.name, limit=120),
            "description": self._compact_text(skill.description, limit=220),
            "asset_type": str(skill.asset_type or "").strip(),
            "granularity": str(skill.granularity or "").strip(),
            "asset_node_id": str(skill.asset_node_id or "").strip(),
            "asset_level": int(skill.asset_level or 0),
            "visible_role": str(skill.visible_role or "").strip(),
            "objective": self._compact_text(skill.objective, limit=220),
            "domain": str(skill.domain or "").strip(),
            "task_family": str(skill.task_family or "").strip(),
            "method_family": str(skill.method_family or "").strip(),
            "stage": str(skill.stage or "").strip(),
            "triggers": list(skill.triggers or [])[:3],
            "workflow_steps": list(skill.workflow_steps or [])[:6],
            "constraints": list(skill.constraints or [])[:4],
            "cautions": list(skill.cautions or [])[:3],
            "output_contract": list(skill.output_contract or [])[:3],
            "tags": list(skill.tags or [])[:4],
            "support_summary": self._support_summary_for_llm(skill, support_by_id=support_by_id),
            "version": str(skill.version or "").strip(),
            "status": skill.status.value,
        }

    def _resolved_skill(self, raw: Any, *, fallback: SkillSpec) -> SkillSpec:
        item = maybe_json_dict(raw)
        if not item:
            return fallback
        prompt = str(item.get("prompt") or item.get("skill_body") or fallback.skill_body).strip()
        intervention_moves = compact_text_list(coerce_str_list(item.get("intervention_moves")), limit=12) or list(
            fallback.intervention_moves or []
        )
        workflow_steps = compact_text_list(coerce_str_list(item.get("workflow_steps")), limit=12) or list(fallback.workflow_steps or [])
        constraints = compact_text_list(coerce_str_list(item.get("constraints")), limit=12) or list(fallback.constraints or [])
        cautions = compact_text_list(coerce_str_list(item.get("cautions")), limit=12) or list(fallback.cautions or [])
        applicable_signals = compact_text_list(coerce_str_list(item.get("applicable_signals")), limit=12) or list(
            fallback.applicable_signals or []
        )
        contraindications = compact_text_list(coerce_str_list(item.get("contraindications")), limit=12) or list(
            fallback.contraindications or []
        )
        output_contract = compact_text_list(coerce_str_list(item.get("output_contract")), limit=12) or list(
            fallback.output_contract or []
        )
        examples = _coerce_examples(item.get("examples")) or list(fallback.examples or [])
        if not prompt or (not workflow_steps and not intervention_moves and not constraints and not cautions):
            return fallback
        objective = str(item.get("objective") or fallback.objective or fallback.description).strip()
        structured_prompt = _build_structured_prompt(
            prompt=prompt,
            objective=objective,
            applicable_signals=applicable_signals,
            contraindications=contraindications,
            intervention_moves=intervention_moves,
            workflow_steps=workflow_steps,
            constraints=constraints,
            cautions=cautions,
            output_contract=output_contract,
            examples=examples,
        )
        return SkillSpec(
            skill_id=fallback.skill_id,
            name=str(item.get("name") or fallback.name).strip(),
            description=str(item.get("description") or fallback.description).strip(),
            skill_body=structured_prompt,
            asset_type=str(item.get("asset_type") or fallback.asset_type).strip(),
            granularity=str(item.get("granularity") or fallback.granularity).strip(),
            asset_node_id=str(item.get("asset_node_id") or fallback.asset_node_id).strip(),
            asset_path=str(item.get("asset_path") or fallback.asset_path).strip(),
            asset_level=int(item.get("asset_level", fallback.asset_level) or fallback.asset_level or 0),
            visible_role=str(item.get("visible_role") or fallback.visible_role).strip(),
            objective=objective,
            domain=str(item.get("domain") or fallback.domain).strip(),
            task_family=str(item.get("task_family") or fallback.task_family).strip(),
            method_family=str(item.get("method_family") or fallback.method_family).strip(),
            stage=str(item.get("stage") or fallback.stage).strip(),
            applicable_signals=applicable_signals,
            contraindications=contraindications,
            intervention_moves=intervention_moves,
            triggers=compact_text_list(coerce_str_list(item.get("triggers")), limit=5) or list(fallback.triggers or []),
            workflow_steps=workflow_steps,
            constraints=constraints,
            cautions=cautions,
            output_contract=output_contract,
            examples=examples,
            tags=compact_text_list(coerce_str_list(item.get("tags")), limit=6) or list(fallback.tags or []),
            support_ids=compact_text_list(coerce_str_list(item.get("support_ids")), limit=64)
            or list(fallback.support_ids or []),
            metadata={
                **dict(fallback.metadata or {}),
                "files": maybe_json_dict(item.get("files")) or maybe_json_dict((fallback.metadata or {}).get("files")),
                "resources": maybe_json_dict(item.get("resources")) or maybe_json_dict((fallback.metadata or {}).get("resources")),
                "llm_reason": str(item.get("reason") or "").strip(),
            },
            version=fallback.version,
            status=fallback.status,
        )

    def classify_change(
        self,
        skill: SkillSpec,
        *,
        peer_skills: Sequence[SkillSpec],
        existing_skills: Sequence[SkillSpec],
        support_by_id: Dict[str, SupportRecord],
    ) -> ChangeDecision:
        """Uses an LLM to classify how one candidate should affect registry state."""

        compact_candidate = self._skill_for_change_llm(skill, support_by_id=support_by_id)
        compact_existing = [
            self._skill_for_change_llm(existing, support_by_id=support_by_id)
            for existing in list(existing_skills or [])
        ][:DEFAULT_RETRIEVAL_LIMIT]
        if not compact_existing:
            payload = {"candidate_skill": compact_candidate}
            system = (
                "You are AutoSkill's Document Skill Version Manager.\n"
                "Task: decide whether one brand-new candidate skill should be created or discarded.\n"
                "Output ONLY strict JSON parseable by json.loads.\n"
                "Rules:\n"
                "- Use discard only when the candidate is too weak, redundant, or not a reusable executable skill.\n"
                "- Use create when it is a reusable skill worth registering.\n"
                "Return schema:\n"
                "{\n"
                '  "action": "create"|"discard",\n'
                '  "reason": "short reason",\n'
                '  "resolved_skill": {optional canonical skill payload}\n'
                "}\n"
            )
            repair_system = (
                "You are a JSON output fixer for document skill version decisions.\n"
                "Given DATA and DRAFT, output ONLY strict JSON with fields action, reason, resolved_skill.\n"
            )
            repaired_payload = f"DATA:\n{json.dumps(payload, ensure_ascii=False)}\n\nDRAFT:\n__DRAFT__"
            parsed = llm_complete_json(
                llm=self.llm,
                system=system,
                payload=payload,
                repair_system=repair_system,
                repair_payload=repaired_payload,
            )
            obj = maybe_json_dict(parsed)
            action = str(obj.get("action") or "").strip().lower()
            if action not in {"create", "discard"}:
                action = "create"
            return ChangeDecision(
                action=action,
                skill=self._resolved_skill(obj.get("resolved_skill"), fallback=skill),
                matched_skill_ids=[],
                reason=str(obj.get("reason") or "").strip() or action,
                split_parent_id="",
            )

        compact_peers: List[Dict[str, Any]] = []
        if _needs_peer_context(skill, peer_skills=peer_skills, existing_skills=existing_skills):
            compact_peers = [
                self._skill_for_change_llm(peer, support_by_id=support_by_id)
                for peer in list(peer_skills or [])
                if peer.skill_id != skill.skill_id
            ][:1]

        payload = {"candidate_skill": compact_candidate, "existing_skills": compact_existing}
        if compact_peers:
            payload["peer_candidates"] = compact_peers
        system = (
            "You are AutoSkill's Document Skill Version Manager.\n"
            "Task: decide how one candidate document skill should update the registry.\n"
            "Output ONLY strict JSON parseable by json.loads.\n"
            "Actions:\n"
            "- create: a new distinct capability\n"
            "- strengthen: same capability, mostly stronger evidence or minor additive guidance\n"
            "- revise: same capability, materially updated instructions or constraints\n"
            "- merge: candidate should merge into one or more existing skills and may deprecate overlapping older skills\n"
            "- split: candidate is one child workflow split out from an older broader parent skill\n"
            "- unchanged: candidate does not materially change the existing skill\n"
            "- discard: do not persist candidate\n"
            "Rules:\n"
            "- Decide from semantic capability identity, not lexical similarity.\n"
            "- candidate_skill and existing_skills are compact summaries; do not assume omitted fields are meaningful differences.\n"
            "- Do not merge, strengthen, revise, or mark unchanged across different asset_type, granularity, or asset_node_id.\n"
            "- Treat macro_protocol, session_skill, micro_skill, safety_rule, and knowledge_reference as different asset layers unless there is an explicit split relationship.\n"
            "- peer_candidates, when present, are weak context only for split detection.\n"
            "- Use support conflict evidence to avoid preserving outdated or unsafe guidance.\n"
            "- If action is strengthen/revise/unchanged/split, target_skill_ids should contain exactly one existing skill id.\n"
            "- If action is merge, target_skill_ids may contain one or more existing skill ids.\n"
            "- Provide resolved_skill when action is strengthen/revise/merge/unchanged and when rewriting the candidate makes the result clearer.\n"
            "Return schema:\n"
            "{\n"
            '  "action": "create"|"strengthen"|"revise"|"merge"|"split"|"unchanged"|"discard",\n'
            '  "target_skill_ids": ["..."],\n'
            '  "reason": "short reason",\n'
            '  "resolved_skill": {optional canonical skill payload}\n'
            "}\n"
        )
        repair_system = (
            "You are a JSON output fixer for document skill version decisions.\n"
            "Given DATA and DRAFT, output ONLY strict JSON with fields action, target_skill_ids, reason, resolved_skill.\n"
        )
        repaired_payload = (
            f"DATA:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            "DRAFT:\n__DRAFT__"
        )
        parsed = llm_complete_json(
            llm=self.llm,
            system=system,
            payload=payload,
            repair_system=repair_system,
            repair_payload=repaired_payload,
        )
        obj = maybe_json_dict(parsed)
        action = str(obj.get("action") or "").strip().lower()
        if action not in {"create", "strengthen", "revise", "merge", "split", "unchanged", "discard"}:
            action = "create"
        matched = compact_text_list(coerce_str_list(obj.get("target_skill_ids")), limit=8)
        resolved = self._resolved_skill(obj.get("resolved_skill"), fallback=skill)
        reason = str(obj.get("reason") or "").strip() or action
        existing_by_id = {existing.skill_id: existing for existing in list(existing_skills or [])}
        matched_existing = [existing_by_id[skill_id] for skill_id in matched if skill_id in existing_by_id]
        if action in {"strengthen", "revise", "merge", "unchanged"} and not matched_existing:
            action = "create"
            matched = []
            reason = "create (matched existing targets missing from retrieved candidates)"
        if action in {"strengthen", "revise", "merge", "unchanged"} and matched_existing:
            if any(not _same_asset_layer(resolved, existing) for existing in matched_existing):
                action = "create"
                matched = []
                reason = "create (cross-layer merge blocked)"
            elif any(not _same_asset_node(resolved, existing) for existing in matched_existing):
                action = "create"
                matched = []
                reason = "create (cross-node merge blocked)"
        if action == "split":
            if not matched_existing:
                action = "create"
                matched = []
                reason = "create (split parent missing from retrieved candidates)"
            else:
                parent = matched_existing[0]
                if _granularity_rank(resolved.granularity) <= _granularity_rank(parent.granularity):
                    action = "create"
                    matched = []
                    reason = "create (split requires narrower child)"
        return ChangeDecision(
            action=action,
            skill=resolved,
            matched_skill_ids=matched,
            reason=reason,
            split_parent_id=(matched[0] if action == "split" and matched else ""),
        )

    def _taxonomy_for_skill(self, skill: SkillSpec) -> Any:
        """Loads the configured taxonomy used by one persisted skill."""

        md = dict(skill.metadata or {})
        requested_path = str(md.get("skill_taxonomy_path") or md.get("skill_taxonomy") or "").strip()
        requested_domain = str(md.get("domain_type") or skill.domain or "").strip()
        try:
            return load_skill_taxonomy(domain_type=requested_domain, taxonomy_path=requested_path)
        except Exception:
            return load_skill_taxonomy()

    def classify_parent_link(
        self,
        child: SkillSpec,
        *,
        parent_hits: Sequence[Any],
        allowed_parent_nodes: Sequence[str],
    ) -> Dict[str, Any]:
        """Chooses one parent candidate for a skill using constrained LLM + safe fallback."""

        hits = [item for item in list(parent_hits or []) if getattr(item, "skill", None) is not None]
        if not hits:
            return {"decision": "defer", "parent_skill_id": "", "confidence": 0.0, "reason": "no parent candidates"}

        payload = {
            "child_skill": {
                "skill_id": child.skill_id,
                "name": child.name,
                "asset_type": child.asset_type,
                "asset_node_id": child.asset_node_id,
                "asset_level": child.asset_level,
                "objective": child.objective,
                "workflow_steps": list(child.workflow_steps or []),
                "constraints": list(child.constraints or []),
                "triggers": list(child.triggers or []),
            },
            "allowed_parent_nodes": list(allowed_parent_nodes or []),
            "parent_candidates": [
                {
                    "skill_id": hit.skill.skill_id,
                    "name": hit.skill.name,
                    "asset_node_id": hit.skill.asset_node_id,
                    "asset_level": hit.skill.asset_level,
                    "objective": hit.skill.objective,
                    "workflow_steps": list(hit.skill.workflow_steps or [])[:6],
                    "score": float(getattr(hit, "score", 0.0) or 0.0),
                }
                for hit in hits[:5]
            ],
        }
        system = (
            "You are AutoSkill4Doc's hierarchy linker.\n"
            "Task: attach one child skill to the best broader parent candidate.\n"
            "Rules:\n"
            "- Choose ONLY one parent_skill_id from parent_candidates when a clear broader parent exists.\n"
            "- Parent must be a broader reusable skill that should call or contain the child skill.\n"
            "- If no parent is clearly suitable, return decision=defer.\n"
            "- Do not invent ids.\n"
            "Return ONLY strict JSON:\n"
            "{\n"
            '  "decision": "attach"|"defer",\n'
            '  "parent_skill_id": "candidate-id-or-empty",\n'
            '  "confidence": 0.0,\n'
            '  "reason": "short reason"\n'
            "}\n"
        )
        repair_system = (
            "You are a JSON fixer for AutoSkill4Doc hierarchy linking.\n"
            "Return ONLY strict JSON with decision, parent_skill_id, confidence, reason.\n"
        )
        repaired_payload = f"DATA:\n{json.dumps(payload, ensure_ascii=False)}\n\nDRAFT:\n__DRAFT__"
        parsed = llm_complete_json(
            llm=self.llm,
            system=system,
            payload=payload,
            repair_system=repair_system,
            repair_payload=repaired_payload,
        )
        obj = maybe_json_dict(parsed)
        decision = str(obj.get("decision") or "").strip().lower()
        parent_skill_id = str(obj.get("parent_skill_id") or "").strip()
        confidence = clip_confidence(obj.get("confidence"), default=0.0)
        reason = str(obj.get("reason") or "").strip()
        valid_ids = {str(hit.skill.skill_id or "").strip() for hit in hits}
        if decision == "attach" and parent_skill_id in valid_ids:
            return {
                "decision": "attach",
                "parent_skill_id": parent_skill_id,
                "confidence": confidence,
                "reason": reason or "llm hierarchy attachment",
            }

        # Safe fallback: if there is exactly one candidate, or one clearly outranks the rest, use it.
        if len(hits) == 1 and _plausible_parent_candidate(child, hits[0].skill, score=float(hits[0].score or 0.0)):
            return {
                "decision": "attach",
                "parent_skill_id": str(hits[0].skill.skill_id or "").strip(),
                "confidence": max(confidence, 0.6),
                "reason": reason or "single eligible parent candidate",
            }
        if (
            len(hits) >= 2
            and float(hits[0].score or 0.0) >= float(hits[1].score or 0.0) + 0.15
            and _plausible_parent_candidate(child, hits[0].skill, score=float(hits[0].score or 0.0))
        ):
            return {
                "decision": "attach",
                "parent_skill_id": str(hits[0].skill.skill_id or "").strip(),
                "confidence": max(confidence, 0.55),
                "reason": reason or "top parent candidate clearly outranks remaining hits",
            }
        return {"decision": "defer", "parent_skill_id": "", "confidence": 0.0, "reason": reason or "no confident parent"}

    def link_hierarchy(
        self,
        *,
        skills: Sequence[SkillSpec],
        existing_skills: Sequence[SkillSpec],
    ) -> Tuple[List[SkillSpec], List[SkillSpec]]:
        """Assigns parent-child links and returns current skills plus touched existing parents."""

        current = [skill for skill in list(skills or []) if isinstance(skill, SkillSpec)]
        if not current:
            return [], []

        corpus_by_id: Dict[str, SkillSpec] = {}
        for skill in list(existing_skills or []):
            if skill.status in {VersionState.DEPRECATED, VersionState.RETIRED}:
                continue
            corpus_by_id[skill.skill_id] = skill
        for skill in current:
            if skill.status in {VersionState.DEPRECATED, VersionState.RETIRED}:
                continue
            corpus_by_id[skill.skill_id] = skill

        hierarchy_retriever = self.retriever or build_document_skill_retriever()
        hierarchy_retriever.refresh(list(corpus_by_id.values()))

        updated: Dict[str, SkillSpec] = {}
        touched_existing: Dict[str, SkillSpec] = {}
        for skill in current:
            if skill.status in {VersionState.DEPRECATED, VersionState.RETIRED}:
                updated[skill.skill_id] = skill
                continue
            taxonomy = self._taxonomy_for_skill(skill)
            asset_level = max(0, int(skill.asset_level or 0))
            visible_role = str(skill.visible_role or "").strip()
            if asset_level <= 1:
                updated[skill.skill_id] = _copy_skill_hierarchy(
                    skill,
                    parent_skill_id="",
                    parent_candidate_ids=[],
                    hierarchy_confidence=1.0 if asset_level == 1 else 0.0,
                    hierarchy_status="root",
                    visible_role=visible_role or "parent",
                )
                continue

            node = taxonomy.get_asset_node(skill.asset_node_id)
            allowed_parent_nodes = [str(node.parent or "").strip()] if node is not None and str(node.parent or "").strip() else []
            hits = hierarchy_retriever.search_parents(
                skill,
                allowed_parent_nodes=allowed_parent_nodes,
                limit=min(5, self.retrieval_limit),
                exclude_ids={skill.skill_id},
            )
            linked = self.classify_parent_link(
                skill,
                parent_hits=hits,
                allowed_parent_nodes=allowed_parent_nodes,
            )
            parent_skill_id = str(linked.get("parent_skill_id") or "").strip()
            confidence = clip_confidence(linked.get("confidence"), default=0.0)
            status = "linked" if parent_skill_id else "unresolved"
            if parent_skill_id:
                parent_skill = corpus_by_id.get(parent_skill_id)
                parent_level = max(1, int(getattr(parent_skill, "asset_level", 0) or 0)) if parent_skill is not None else asset_level - 1
                if parent_level != max(1, asset_level - 1):
                    parent_skill_id = ""
                    status = "unresolved"
                    confidence = 0.0
            updated[skill.skill_id] = _copy_skill_hierarchy(
                skill,
                parent_skill_id=parent_skill_id,
                parent_candidate_ids=[str(hit.skill.skill_id or "").strip() for hit in hits],
                hierarchy_confidence=confidence,
                hierarchy_status=status,
                visible_role=visible_role or ("leaf" if asset_level >= 3 else "parent"),
            )

        children_by_parent: Dict[str, List[str]] = {}
        for skill in updated.values():
            parent_skill_id = str(skill.parent_skill_id or "").strip()
            if not parent_skill_id:
                continue
            children_by_parent.setdefault(parent_skill_id, []).append(skill.skill_id)
        for parent_skill_id, child_ids in children_by_parent.items():
            parent = updated.get(parent_skill_id) or corpus_by_id.get(parent_skill_id)
            if parent is None:
                continue
            merged_child_ids = sorted(set(list(parent.child_skill_ids or []) + list(child_ids or [])))
            linked_parent = _copy_skill_hierarchy(
                parent,
                child_skill_ids=merged_child_ids,
                hierarchy_status="parent",
                visible_role=str(parent.visible_role or "").strip() or "parent",
            )
            if parent_skill_id in updated:
                updated[parent_skill_id] = linked_parent
            else:
                touched_existing[parent_skill_id] = linked_parent

        return [updated.get(skill.skill_id, skill) for skill in current], list(touched_existing.values())

    def create_new_version(self, *, current_version: str, action: str) -> str:
        """Creates the next version string for one lifecycle action."""

        action_s = str(action or "").strip().lower()
        if action_s in {"", "create"}:
            return str(current_version or "").strip() or "0.1.0"
        if action_s == "unchanged":
            return str(current_version or "").strip() or "0.1.0"
        return _bump_patch(str(current_version or "").strip() or "0.1.0")

    def update_lifecycle(
        self,
        *,
        skill_id: str,
        current_state: Optional[VersionState],
        action: str,
        target_state: VersionState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillLifecycle:
        """Creates one lifecycle transition consistent with the requested action."""

        action_s = str(action or "").strip().lower()
        if action_s == "deprecate":
            next_state = (
                target_state
                if target_state in {VersionState.WATCHLIST, VersionState.DEPRECATED, VersionState.RETIRED}
                else VersionState.DEPRECATED
            )
        else:
            next_state = target_state
        from_state = current_state if current_state is not None and current_state != next_state else None
        return SkillLifecycle(
            lifecycle_id=str(uuid.uuid4()),
            skill_id=str(skill_id or "").strip(),
            from_state=from_state,
            to_state=next_state,
            reason=action_s or "update",
            metadata=dict(metadata or {}),
        )

    def mark_deprecated(
        self,
        *,
        skill: SkillSpec,
        reason: str,
        state: VersionState = VersionState.DEPRECATED,
        related_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Marks one skill as deprecated/watchlist/retired."""

        related = list(related_ids or [])
        next_version = self.create_new_version(current_version=skill.version, action="deprecate")
        updated_skill = _copy_skill(
            skill,
            version=next_version,
            status=state,
            metadata_update={
                "change_action": "deprecate",
                "deprecation_reason": str(reason or "").strip(),
                "related_skill_ids": related,
            },
        )
        provenance = {
            "entity_type": "skill",
            "entity_id": updated_skill.skill_id,
            "doc_ids": [],
            "support_added": [],
            "support_conflicts": [],
            "related_entity_ids": related,
        }
        lifecycle = self.update_lifecycle(
            skill_id=updated_skill.skill_id,
            current_state=skill.status,
            action="deprecate",
            target_state=state,
            metadata={"reason": str(reason or "").strip(), "related_entity_ids": related},
        )
        change_log = self._change_payload(
            entity_type="skill",
            entity_id=updated_skill.skill_id,
            action="deprecate",
            from_version=skill.version,
            to_version=updated_skill.version,
            from_state=skill.status.value,
            to_state=updated_skill.status.value,
            summary=str(reason or "").strip(),
            provenance=provenance,
            related_entity_ids=related,
        )
        history = self._history_payload(
            entity_type="skill",
            entity_id=updated_skill.skill_id,
            version=updated_skill.version,
            action="deprecate",
            status=updated_skill.status,
            related_entity_ids=related,
        )
        return {
            "skill": updated_skill,
            "lifecycle": lifecycle,
            "change_log": change_log,
            "version_history": history,
            "provenance_links": provenance,
        }

    def merge_skills(
        self,
        *,
        decision: ChangeDecision,
        skill: SkillSpec,
        existing_skills_by_id: Dict[str, SkillSpec],
        target_state: VersionState,
    ) -> Dict[str, Any]:
        """Merges multiple existing skills into one updated primary skill."""

        matched_ids = [skill_id for skill_id in decision.matched_skill_ids if skill_id in existing_skills_by_id]
        if not matched_ids:
            raise ValueError("merge requires at least one existing matched skill")
        primary_existing = existing_skills_by_id[matched_ids[0]]
        secondary_existing = [existing_skills_by_id[skill_id] for skill_id in matched_ids[1:]]
        next_version = self.create_new_version(current_version=primary_existing.version, action="merge")
        merged_skill = _copy_skill(
            skill,
            skill_id=primary_existing.skill_id,
            version=next_version,
            status=target_state,
            support_ids=list(primary_existing.support_ids or []) + list(skill.support_ids or []),
            metadata_update={
                "change_action": "merge",
                "merged_from_skill_ids": [existing.skill_id for existing in secondary_existing],
                "llm_reason": decision.reason,
            },
        )
        deprecated_secondary = [
            self.mark_deprecated(
                skill=secondary,
                reason="merged_into_other_skill",
                state=VersionState.DEPRECATED,
                related_ids=[primary_existing.skill_id],
            )
            for secondary in secondary_existing
        ]
        return {
            "primary_skill": merged_skill,
            "secondary": deprecated_secondary,
        }

    def split_skill(
        self,
        *,
        parent_skill: SkillSpec,
        child_skills: Sequence[SkillSpec],
        target_state: VersionState,
    ) -> Dict[str, Any]:
        """Splits one existing skill into multiple more focused skills."""

        updated_children: List[SkillSpec] = []
        lifecycles: List[SkillLifecycle] = []
        change_logs: List[Dict[str, Any]] = []
        version_history: List[Dict[str, Any]] = []
        provenance_links: List[Dict[str, Any]] = []

        for child in list(child_skills or []):
            updated_child = _copy_skill(
                child,
                version="0.1.0",
                status=target_state,
                metadata_update={
                    "change_action": "split",
                    "split_from_skill_id": parent_skill.skill_id,
                },
            )
            updated_children.append(updated_child)
            lifecycles.append(
                self.update_lifecycle(
                    skill_id=updated_child.skill_id,
                    current_state=None,
                    action="split",
                    target_state=target_state,
                    metadata={"split_from_skill_id": parent_skill.skill_id},
                )
            )
            provenance = {
                "entity_type": "skill",
                "entity_id": updated_child.skill_id,
                "doc_ids": [],
                "support_added": list(updated_child.support_ids or []),
                "support_conflicts": [],
                "related_entity_ids": [parent_skill.skill_id],
            }
            provenance_links.append(provenance)
            version_history.append(
                self._history_payload(
                    entity_type="skill",
                    entity_id=updated_child.skill_id,
                    version=updated_child.version,
                    action="split",
                    status=updated_child.status,
                    related_entity_ids=[parent_skill.skill_id],
                )
            )
            change_logs.append(
                self._change_payload(
                    entity_type="skill",
                    entity_id=updated_child.skill_id,
                    action="split",
                    from_version="",
                    to_version=updated_child.version,
                    from_state="",
                    to_state=updated_child.status.value,
                    summary="split_from_existing_skill",
                    provenance=provenance,
                    related_entity_ids=[parent_skill.skill_id],
                )
            )

        deprecated_parent = self.mark_deprecated(
            skill=parent_skill,
            reason="split_into_more_specific_skills",
            state=VersionState.DEPRECATED,
            related_ids=[child.skill_id for child in updated_children],
        )
        return {
            "children": updated_children,
            "deprecated_parent": deprecated_parent,
            "lifecycles": lifecycles,
            "change_logs": change_logs,
            "version_history": version_history,
            "provenance_links": provenance_links,
        }

    def _change_payload(
        self,
        *,
        entity_type: str,
        entity_id: str,
        action: str,
        from_version: str,
        to_version: str,
        from_state: str,
        to_state: str,
        summary: str,
        provenance: Dict[str, Any],
        related_entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Builds one normalized change log payload."""

        return {
            "change_id": str(uuid.uuid4()),
            "entity_type": str(entity_type or "").strip(),
            "entity_id": str(entity_id or "").strip(),
            "action": str(action or "").strip(),
            "changed_at": now_iso(),
            "from_version": str(from_version or "").strip(),
            "to_version": str(to_version or "").strip(),
            "from_state": str(from_state or "").strip(),
            "to_state": str(to_state or "").strip(),
            "summary": str(summary or "").strip(),
            "related_entity_ids": list(related_entity_ids or []),
            "provenance": dict(provenance or {}),
        }

    def _history_payload(
        self,
        *,
        entity_type: str,
        entity_id: str,
        version: str,
        action: str,
        status: VersionState,
        related_entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Builds one normalized version history entry."""

        return {
            "entity_type": str(entity_type or "").strip(),
            "entity_id": str(entity_id or "").strip(),
            "version": str(version or "").strip(),
            "action": str(action or "").strip(),
            "changed_at": now_iso(),
            "status": status.value,
            "related_entity_ids": list(related_entity_ids or []),
        }

    def _conflict_review(
        self,
        *,
        existing_skill: SkillSpec,
        incoming_skills: Sequence[SkillSpec],
        support_by_id: Dict[str, SupportRecord],
    ) -> Dict[str, str]:
        """Asks the LLM whether incoming conflicting evidence should downgrade an existing skill."""

        payload = {
            "existing_skill": self._skill_for_llm(existing_skill, support_by_id=support_by_id),
            "incoming_skills": [
                self._skill_for_llm(skill, support_by_id=support_by_id)
                for skill in list(incoming_skills or [])
            ][:8],
        }
        system = (
            "You are AutoSkill's Document Conflict Judge.\n"
            "Task: decide whether new incoming document skills/support should keep, watchlist, or deprecate an existing skill.\n"
            "Output ONLY strict JSON parseable by json.loads.\n"
            "Use watchlist when conflict signals are meaningful but not strong enough for deprecation.\n"
            "Use deprecate when newer evidence materially contradicts or replaces the older skill's guidance.\n"
            "Return schema: {\"action\": \"keep\"|\"watchlist\"|\"deprecate\", \"reason\": \"short reason\"}\n"
        )
        repair_system = (
            "You are a JSON output fixer for document conflict review.\n"
            "Given DATA and DRAFT, output ONLY strict JSON with fields action and reason.\n"
        )
        repaired_payload = (
            f"DATA:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            "DRAFT:\n__DRAFT__"
        )
        parsed = llm_complete_json(
            llm=self.llm,
            system=system,
            payload=payload,
            repair_system=repair_system,
            repair_payload=repaired_payload,
        )
        obj = maybe_json_dict(parsed)
        action = str(obj.get("action") or "").strip().lower()
        if action not in {"keep", "watchlist", "deprecate"}:
            action = "keep"
        return {"action": action, "reason": str(obj.get("reason") or "").strip() or action}

    def reconcile(
        self,
        *,
        skills: Sequence[SkillSpec],
        support_records: Sequence[SupportRecord],
        target_state: VersionState,
    ) -> VersionRegistrationResult:
        """Reconciles one compiled batch against registry state."""

        result = VersionRegistrationResult(support_records=[], dry_run=False)
        support_by_id = _support_lookup(self.registry, support_records)
        existing_skills = list(self.registry.list_skills())
        existing_skills_by_id = {skill.skill_id: skill for skill in existing_skills}
        retriever = self.retriever or build_document_skill_retriever()
        retriever.refresh(existing_skills)

        decisions: List[ChangeDecision] = []
        total_skills = len(list(skills or []))
        emit_stage_progress(
            self.progress_callback,
            {
                "stage": "register",
                "kind": "start",
                "phase": "reconcile",
                "completed_skills": 0,
                "total_skills": total_skills,
                "errors": 0,
            },
        )
        for idx, skill in enumerate(list(skills or []), start=1):
            hits = retriever.search(
                skill,
                limit=self.retrieval_limit,
            )
            retrieved_existing = [hit.skill for hit in list(hits or [])]
            emit_stage_log(
                self.logger,
                (
                    f"[register_versions] retrieve skill={skill.skill_id} "
                    f"hits={len(retrieved_existing)} "
                    f"names={summarize_names([item.name for item in retrieved_existing])}"
                ),
            )
            decision = self.classify_change(
                skill,
                peer_skills=skills,
                existing_skills=retrieved_existing,
                support_by_id=support_by_id,
            )
            decisions.append(decision)
            emit_stage_progress(
                self.progress_callback,
                {
                    "stage": "register",
                    "kind": "reconcile_progress",
                    "phase": "reconcile",
                    "completed_skills": idx,
                    "total_skills": total_skills,
                    "current_skill_id": str(skill.skill_id or "").strip(),
                    "current_name": str(skill.name or "").strip(),
                    "hits": len(retrieved_existing),
                    "action": str(decision.action or "").strip(),
                    "errors": len(list(result.errors or [])),
                },
            )

        processed_split_parents: Set[str] = set()
        consumed_existing_ids: Set[str] = set()
        consumed_new_ids: Set[str] = set()
        incoming_support_by_id = {support.support_id: support for support in list(support_records or [])}

        for decision in [item for item in decisions if item.action == "merge" and item.matched_skill_ids]:
            new_skill = decision.skill
            application = self.merge_skills(
                decision=decision,
                skill=new_skill,
                existing_skills_by_id=existing_skills_by_id,
                target_state=target_state,
            )
            merged_skill = application["primary_skill"]
            bound_supports = [
                _copy_support(support, skill_id=merged_skill.skill_id)
                for support in list(incoming_support_by_id.values())
                if support.support_id in set(new_skill.support_ids or [])
            ]
            merged_skill = _copy_skill(
                merged_skill,
                support_ids=list(dict.fromkeys(list(merged_skill.support_ids or []) + [support.support_id for support in bound_supports])),
                metadata_update={"llm_reason": decision.reason},
            )
            provenance = {
                "entity_type": "skill",
                "entity_id": merged_skill.skill_id,
                "doc_ids": _doc_ids_from_support_ids(merged_skill.support_ids, support_by_id),
                "support_added": [support.support_id for support in bound_supports],
                "support_conflicts": _conflicting_support_ids(merged_skill.support_ids, support_by_id),
                "related_entity_ids": decision.matched_skill_ids,
            }
            result.skill_specs.append(merged_skill)
            result.support_records.extend(bound_supports)
            result.lifecycles.append(
                self.update_lifecycle(
                    skill_id=merged_skill.skill_id,
                    current_state=existing_skills_by_id[merged_skill.skill_id].status,
                    action="merge",
                    target_state=target_state,
                    metadata={"merged_from_skill_ids": decision.matched_skill_ids, "llm_reason": decision.reason},
                )
            )
            result.provenance_links.append(provenance)
            result.change_logs.append(
                self._change_payload(
                    entity_type="skill",
                    entity_id=merged_skill.skill_id,
                    action="merge",
                    from_version=existing_skills_by_id[merged_skill.skill_id].version,
                    to_version=merged_skill.version,
                    from_state=existing_skills_by_id[merged_skill.skill_id].status.value,
                    to_state=merged_skill.status.value,
                    summary=decision.reason,
                    provenance=provenance,
                    related_entity_ids=decision.matched_skill_ids,
                )
            )
            result.version_history.append(
                self._history_payload(
                    entity_type="skill",
                    entity_id=merged_skill.skill_id,
                    version=merged_skill.version,
                    action="merge",
                    status=merged_skill.status,
                    related_entity_ids=decision.matched_skill_ids,
                )
            )
            for secondary in list(application["secondary"] or []):
                result.skill_specs.append(secondary["skill"])
                result.lifecycles.append(secondary["lifecycle"])
                result.change_logs.append(secondary["change_log"])
                result.version_history.append(secondary["version_history"])
                result.provenance_links.append(secondary["provenance_links"])
            consumed_new_ids.add(new_skill.skill_id)
            consumed_existing_ids.update(decision.matched_skill_ids)
            emit_stage_log(
                self.logger,
                f"[register_versions] merge name={merged_skill.name} skill={merged_skill.skill_id} from={decision.matched_skill_ids}",
            )

        split_groups: Dict[str, List[ChangeDecision]] = {}
        for decision in decisions:
            if decision.action != "split" or not decision.matched_skill_ids:
                continue
            split_groups.setdefault(decision.matched_skill_ids[0], []).append(decision)

        for parent_id, bucket in split_groups.items():
            if parent_id in processed_split_parents:
                continue
            parent = existing_skills_by_id.get(parent_id)
            if parent is None:
                continue
            application = self.split_skill(
                parent_skill=parent,
                child_skills=[item.skill for item in bucket],
                target_state=target_state,
            )
            processed_split_parents.add(parent_id)
            consumed_existing_ids.add(parent_id)
            for item in bucket:
                consumed_new_ids.add(item.skill.skill_id)
                bound_supports = [
                    _copy_support(support, skill_id=item.skill.skill_id)
                    for support in list(incoming_support_by_id.values())
                    if support.support_id in set(item.skill.support_ids or [])
                ]
                result.support_records.extend(bound_supports)
            result.skill_specs.extend(list(application["children"] or []))
            result.skill_specs.append(application["deprecated_parent"]["skill"])
            result.lifecycles.extend(list(application["lifecycles"] or []))
            result.lifecycles.append(application["deprecated_parent"]["lifecycle"])
            result.change_logs.extend(list(application["change_logs"] or []))
            result.change_logs.append(application["deprecated_parent"]["change_log"])
            result.version_history.extend(list(application["version_history"] or []))
            result.version_history.append(application["deprecated_parent"]["version_history"])
            result.provenance_links.extend(list(application["provenance_links"] or []))
            result.provenance_links.append(application["deprecated_parent"]["provenance_links"])
            emit_stage_log(
                self.logger,
                f"[register_versions] split parent={parent_id} children={len(bucket)} names={summarize_names([item.skill.name for item in bucket])}",
            )

        for decision in decisions:
            if decision.skill.skill_id in consumed_new_ids or decision.action == "discard":
                continue
            if decision.action not in {"create", "strengthen", "revise", "unchanged"}:
                continue

            new_skill = decision.skill
            incoming_supports = [
                support
                for support in list(incoming_support_by_id.values())
                if support.support_id in set(new_skill.support_ids or [])
            ]

            if decision.action == "create" or not decision.matched_skill_ids:
                updated_skill = _copy_skill(
                    new_skill,
                    version=self.create_new_version(current_version=new_skill.version, action="create"),
                    status=target_state,
                    metadata_update={"change_action": "create", "llm_reason": decision.reason},
                )
                bound_supports = [_copy_support(support, skill_id=updated_skill.skill_id) for support in incoming_supports]
                updated_skill = _copy_skill(updated_skill, support_ids=[support.support_id for support in bound_supports])
                provenance = {
                    "entity_type": "skill",
                    "entity_id": updated_skill.skill_id,
                    "doc_ids": _doc_ids_from_support_ids(updated_skill.support_ids, support_by_id),
                    "support_added": list(updated_skill.support_ids or []),
                    "support_conflicts": _conflicting_support_ids(updated_skill.support_ids, support_by_id),
                    "related_entity_ids": [],
                }
                result.skill_specs.append(updated_skill)
                result.support_records.extend(bound_supports)
                result.lifecycles.append(
                    self.update_lifecycle(
                        skill_id=updated_skill.skill_id,
                        current_state=None,
                        action="create",
                        target_state=target_state,
                        metadata={"doc_ids": provenance["doc_ids"], "llm_reason": decision.reason},
                    )
                )
                result.change_logs.append(
                    self._change_payload(
                        entity_type="skill",
                        entity_id=updated_skill.skill_id,
                        action="create",
                        from_version="",
                        to_version=updated_skill.version,
                        from_state="",
                        to_state=updated_skill.status.value,
                        summary=decision.reason,
                        provenance=provenance,
                    )
                )
                result.version_history.append(
                    self._history_payload(
                        entity_type="skill",
                        entity_id=updated_skill.skill_id,
                        version=updated_skill.version,
                        action="create",
                        status=updated_skill.status,
                    )
                )
                result.provenance_links.append(provenance)
                emit_stage_log(
                    self.logger,
                    f"[register_versions] create name={updated_skill.name} skill={updated_skill.skill_id}",
                )
                continue

            existing_skill = existing_skills_by_id.get(decision.matched_skill_ids[0])
            if existing_skill is None:
                result.errors.append({"skill_id": new_skill.skill_id, "error": "matched skill missing from registry"})
                continue
            current_version = existing_skill.version
            next_version = self.create_new_version(current_version=current_version, action=decision.action)
            if decision.action == "unchanged":
                next_version = current_version
            bound_supports = [_copy_support(support, skill_id=existing_skill.skill_id) for support in incoming_supports]
            merged_support_ids = list(dict.fromkeys(list(existing_skill.support_ids or []) + [support.support_id for support in bound_supports]))
            updated_skill = _copy_skill(
                new_skill,
                skill_id=existing_skill.skill_id,
                version=next_version,
                status=(existing_skill.status if decision.action == "unchanged" else target_state),
                support_ids=merged_support_ids,
                metadata_update={
                    "change_action": decision.action,
                    "previous_skill_id": existing_skill.skill_id,
                    "llm_reason": decision.reason,
                },
            )
            provenance = {
                "entity_type": "skill",
                "entity_id": updated_skill.skill_id,
                "doc_ids": _doc_ids_from_support_ids(updated_skill.support_ids, support_by_id),
                "support_added": [support.support_id for support in bound_supports],
                "support_conflicts": _conflicting_support_ids(updated_skill.support_ids, support_by_id),
                "related_entity_ids": [existing_skill.skill_id],
            }
            result.skill_specs.append(updated_skill)
            result.support_records.extend(bound_supports)
            consumed_existing_ids.add(existing_skill.skill_id)
            if decision.action != "unchanged":
                result.lifecycles.append(
                    self.update_lifecycle(
                        skill_id=updated_skill.skill_id,
                        current_state=existing_skill.status,
                        action=decision.action,
                        target_state=target_state,
                        metadata={**provenance, "llm_reason": decision.reason},
                    )
                )
                result.change_logs.append(
                    self._change_payload(
                        entity_type="skill",
                        entity_id=updated_skill.skill_id,
                        action=decision.action,
                        from_version=existing_skill.version,
                        to_version=updated_skill.version,
                        from_state=existing_skill.status.value,
                        to_state=updated_skill.status.value,
                        summary=decision.reason,
                        provenance=provenance,
                        related_entity_ids=[existing_skill.skill_id],
                    )
                )
                result.version_history.append(
                    self._history_payload(
                        entity_type="skill",
                        entity_id=updated_skill.skill_id,
                        version=updated_skill.version,
                        action=decision.action,
                        status=updated_skill.status,
                        related_entity_ids=[existing_skill.skill_id],
                    )
                )
                result.provenance_links.append(provenance)
                emit_stage_log(
                    self.logger,
                    f"[register_versions] {decision.action} name={updated_skill.name} skill={updated_skill.skill_id}",
                )

        incoming_related = [decision.skill for decision in decisions if decision.action != "discard"]
        if any(support.relation_type == SupportRelation.CONFLICT for support in support_records or []):
            for existing_skill in existing_skills:
                if existing_skill.skill_id in consumed_existing_ids:
                    continue
                if existing_skill.status in {VersionState.DEPRECATED, VersionState.RETIRED}:
                    continue
                review = self._conflict_review(
                    existing_skill=existing_skill,
                    incoming_skills=incoming_related,
                    support_by_id=support_by_id,
                )
                if review["action"] == "keep":
                    continue
                deprecated_state = (
                    VersionState.WATCHLIST if review["action"] == "watchlist" else VersionState.DEPRECATED
                )
                deprecated = self.mark_deprecated(
                    skill=existing_skill,
                    reason=review["reason"],
                    state=deprecated_state,
                    related_ids=[skill.skill_id for skill in incoming_related],
                )
                result.skill_specs.append(deprecated["skill"])
                result.lifecycles.append(deprecated["lifecycle"])
                result.change_logs.append(deprecated["change_log"])
                result.version_history.append(deprecated["version_history"])
                result.provenance_links.append(deprecated["provenance_links"])
                consumed_existing_ids.add(existing_skill.skill_id)
                emit_stage_log(
                    self.logger,
                    f"[register_versions] deprecate name={existing_skill.name} skill={existing_skill.skill_id}",
                )

        deduped_skills: Dict[str, SkillSpec] = {}
        for skill in result.skill_specs:
            deduped_skills[f"{skill.skill_id}:{skill.status.value}"] = skill
        result.skill_specs = list(deduped_skills.values())

        deduped_supports: Dict[str, SupportRecord] = {}
        for support in result.support_records:
            deduped_supports[support.support_id] = support
        result.support_records = list(deduped_supports.values())
        emit_stage_progress(
            self.progress_callback,
            {
                "stage": "register",
                "kind": "reconcile_done",
                "phase": "reconcile",
                "completed_skills": total_skills,
                "total_skills": total_skills,
                "actions": {
                    "create": len([item for item in decisions if item.action == "create"]),
                    "strengthen": len([item for item in decisions if item.action == "strengthen"]),
                    "revise": len([item for item in decisions if item.action == "revise"]),
                    "merge": len([item for item in decisions if item.action == "merge"]),
                    "split": len([item for item in decisions if item.action == "split"]),
                    "unchanged": len([item for item in decisions if item.action == "unchanged"]),
                    "discard": len([item for item in decisions if item.action == "discard"]),
                },
                "errors": len(list(result.errors or [])),
            },
        )
        return result


def register_versions(
    *,
    registry: DocumentRegistry,
    documents: Sequence[DocumentRecord],
    support_records: Sequence[SupportRecord],
    skill_specs: Sequence[SkillSpec],
    sdk: Optional[AutoSkill] = None,
    llm: Optional[LLM] = None,
    user_id: str = DEFAULT_DOC_SKILL_USER_ID,
    metadata: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    target_state: VersionState = VersionState.ACTIVE,
    logger: StageLogger = None,
    progress_callback: StageProgressCallback = None,
    retrieval_score_threshold: float = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
) -> VersionRegistrationResult:
    """
    Registers a compiled batch into the document registry and optionally the skill store.
    """

    effective_state = VersionState.DRAFT if dry_run else target_state
    preexisting_skills = list(registry.list_skills())
    llm_impl = llm or build_llm(dict(getattr(getattr(sdk, "config", None), "llm", {}) or {"provider": "mock"}))
    sdk_config = getattr(sdk, "config", None)
    embeddings_config = dict(getattr(sdk_config, "embeddings", {}) or {"provider": "hashing", "dims": 256})
    bm25_weight = float(getattr(sdk_config, "bm25_weight", 0.1) or 0.1)
    store_root = _store_root_from_context(registry=registry, sdk=sdk)
    retriever = build_document_skill_retriever(
        embeddings_config=embeddings_config,
        bm25_weight=bm25_weight,
        score_threshold=max(0.0, float(retrieval_score_threshold or DEFAULT_RETRIEVAL_SCORE_THRESHOLD)),
        base_store_root=store_root,
    )
    manager = VersionManager(
        registry=registry,
        llm=llm_impl,
        retriever=retriever,
        retrieval_limit=DEFAULT_RETRIEVAL_LIMIT,
        logger=logger,
        progress_callback=progress_callback,
    )
    reconciled = manager.reconcile(
        skills=skill_specs,
        support_records=support_records,
        target_state=effective_state,
    )
    reconciled.skill_specs = [_merge_layout_metadata(skill, metadata=metadata) for skill in list(reconciled.skill_specs or [])]
    reconciled.skill_specs, reconciled.hierarchy_updates = manager.link_hierarchy(
        skills=reconciled.skill_specs,
        existing_skills=preexisting_skills,
    )
    emit_stage_progress(
        progress_callback,
        {
            "stage": "register",
            "kind": "hierarchy_done",
            "phase": "hierarchy",
            "completed_skills": len(list(reconciled.skill_specs or [])),
            "total_skills": len(list(reconciled.skill_specs or [])),
            "hierarchy_updates": len(list(reconciled.hierarchy_updates or [])),
            "errors": len(list(reconciled.errors or [])),
        },
    )
    reconciled.documents = list(documents or [])
    reconciled.dry_run = bool(dry_run)
    current_store_skills: List[Any] = []

    if not dry_run:
        for document in documents or []:
            registry.upsert_document(document)
        for support in reconciled.support_records:
            registry.upsert_support(support)
        seen_skills: Set[str] = set()
        for skill in list(reconciled.skill_specs or []) + list(reconciled.hierarchy_updates or []):
            key = f"{skill.skill_id}:{skill.status.value}"
            if key in seen_skills:
                continue
            registry.upsert_skill(skill)
            seen_skills.add(key)
        for lifecycle in reconciled.lifecycles:
            registry.append_lifecycle(lifecycle)
        for payload in reconciled.change_logs:
            registry.append_change_log(str(payload.get("change_id") or str(uuid.uuid4())), payload)
        for entry in reconciled.version_history:
            registry.append_version_history(
                entity_type=str(entry.get("entity_type") or ""),
                entity_id=str(entry.get("entity_id") or ""),
                entry=entry,
            )
        for payload in reconciled.provenance_links:
            registry.upsert_provenance_links(
                entity_type=str(payload.get("entity_type") or ""),
                entity_id=str(payload.get("entity_id") or ""),
                payload=payload,
            )
        try:
            retriever.refresh(list(registry.list_skills()))
        except Exception as e:
            reconciled.errors.append({"stage": "retrieval_cache_refresh", "error": str(e)})
            emit_stage_log(logger, f"[register_versions] retrieval cache refresh error: {e}")

    if sdk is not None and reconciled.skill_specs:
        md = dict(metadata or {})
        md.setdefault("channel", "offline_extract_from_doc")
        md.setdefault("source_type", "document")
        md["document_registry_root"] = registry.root_dir
        if reconciled.skill_specs:
            if not dry_run:
                try:
                    touched_store_skills, current_store_skills = _sync_store_skills(
                        sdk=sdk,
                        skill_specs=list(reconciled.skill_specs or []) + list(reconciled.hierarchy_updates or []),
                        user_id=str(user_id or "").strip() or DEFAULT_DOC_SKILL_USER_ID,
                        metadata=md,
                        logger=logger,
                    )
                    reconciled.upserted_store_skills = [_plain_skill(skill) for skill in list(touched_store_skills or [])]
                    emit_stage_log(
                        logger,
                        f"[register_versions] store_upserts={len(reconciled.upserted_store_skills)} names={summarize_names([str(skill.get('name') or '') for skill in reconciled.upserted_store_skills])}",
                    )
                    emit_stage_progress(
                        progress_callback,
                        {
                            "stage": "register",
                            "kind": "store_done",
                            "phase": "store_sync",
                            "completed_skills": len(list(reconciled.skill_specs or [])),
                            "total_skills": len(list(reconciled.skill_specs or [])),
                            "store_upserts": len(list(reconciled.upserted_store_skills or [])),
                            "errors": len(list(reconciled.errors or [])),
                        },
                    )
                except Exception as e:
                    reconciled.errors.append({"stage": "store_upsert", "error": str(e)})
                    emit_stage_log(logger, f"[register_versions] store upsert error: {e}")
            else:
                emit_stage_log(
                    logger,
                    f"[register_versions] dry-run store_upserts={len(reconciled.skill_specs)} names={summarize_names([spec.name for spec in reconciled.skill_specs])}",
                )
        if not dry_run and not current_store_skills:
            try:
                current_store_skills = list(sdk.store.list(user_id=str(user_id or "").strip() or DEFAULT_DOC_SKILL_USER_ID) or [])
            except Exception:
                current_store_skills = []

    if not dry_run:
        store_root = _store_root_from_context(registry=registry, sdk=sdk)
        if reconciled.skill_specs:
            bucketed: Dict[Tuple[str, str, str], List[SkillSpec]] = {}
            for skill in list(reconciled.skill_specs or []):
                bucket = _staging_bucket_for_skill(skill, metadata=metadata)
                bucketed.setdefault(bucket, []).append(skill)
            raw_bucketed: Dict[Tuple[str, str, str], List[SkillSpec]] = {}
            for skill in list(skill_specs or []):
                merged = _merge_layout_metadata(skill, metadata=metadata)
                bucket = _staging_bucket_for_skill(merged, metadata=metadata)
                raw_bucketed.setdefault(bucket, []).append(merged)

            support_by_id = {support.support_id: support for support in list(reconciled.support_records or [])}
            document_by_id = {document.doc_id: document for document in list(documents or [])}
            for (profile_id, family_id, child_type), bucket_skills in bucketed.items():
                bucket_skill_ids = {skill.skill_id for skill in list(bucket_skills or [])}
                bucket_support_ids = {
                    str(support_id or "").strip()
                    for skill in list(bucket_skills or [])
                    for support_id in list(skill.support_ids or [])
                    if str(support_id or "").strip()
                }
                bucket_supports = [
                    support.to_dict()
                    for support_id, support in support_by_id.items()
                    if support_id in bucket_support_ids
                ]
                bucket_documents = [
                    document.to_dict()
                    for document_id, document in document_by_id.items()
                    if document_id in {
                        str(support.doc_id or "").strip()
                        for support in support_by_id.values()
                        if support.support_id in bucket_support_ids
                    }
                ]
                bucket_existing_active = [
                    skill.to_dict()
                    for skill in preexisting_skills
                    if skill.status in _ACTIVE_STORE_STATES and _staging_bucket_for_skill(skill, metadata=metadata) == (profile_id, family_id, child_type)
                ]
                bucket_change_logs = [
                    dict(payload)
                    for payload in list(reconciled.change_logs or [])
                    if str(payload.get("entity_id") or "").strip() in bucket_skill_ids
                ]
                staging_summary = write_registration_staging(
                    base_store_root=store_root,
                    profile_id=profile_id,
                    family_id=family_id,
                    child_type=child_type,
                    run_id="",
                    documents=bucket_documents,
                    support_records=bucket_supports,
                    raw_candidates=plain_skill_specs(raw_bucketed.get((profile_id, family_id, child_type)) or []),
                    existing_active=bucket_existing_active,
                    canonical_results=plain_skill_specs(bucket_skills),
                    change_logs=bucket_change_logs,
                )
                reconciled.staging_runs.append(staging_summary.to_dict())
                emit_stage_log(
                    logger,
                    f"[register_versions] staging profile={profile_id} family={family_id} child_type={child_type} skills={len(bucket_skills)}",
                )
                emit_stage_progress(
                    progress_callback,
                    {
                        "stage": "register",
                        "kind": "staging_progress",
                        "phase": "staging",
                        "completed_buckets": len(list(reconciled.staging_runs or [])),
                        "total_buckets": len(bucketed),
                        "family_name": family_id,
                        "child_type": child_type,
                        "errors": len(list(reconciled.errors or [])),
                    },
                )

        try:
            visible_tree = sync_visible_skill_tree(
                registry=registry,
                store_root=store_root,
                documents=documents,
                support_records=reconciled.support_records,
                skill_specs=reconciled.skill_specs,
                user_id=str(user_id or "").strip() or DEFAULT_DOC_SKILL_USER_ID,
                metadata=metadata,
                store_skills=(current_store_skills if sdk is not None else None),
                logger=logger,
            )
            reconciled.visible_tree = visible_tree.to_dict()
            emit_stage_progress(
                progress_callback,
                {
                    "stage": "register",
                    "kind": "visible_tree_done",
                    "phase": "visible_tree",
                    "completed_skills": len(list(reconciled.skill_specs or [])),
                    "total_skills": len(list(reconciled.skill_specs or [])),
                    "affected_families": len(list((reconciled.visible_tree or {}).get("affected_families") or [])),
                    "visible_children": len(list((reconciled.visible_tree or {}).get("child_paths") or [])),
                    "errors": len(list(reconciled.errors or [])),
                },
            )
        except Exception as e:
            reconciled.errors.append({"stage": "visible_tree_sync", "error": str(e)})
            emit_stage_log(logger, f"[register_versions] visible tree sync error: {e}")

    return reconciled
