"""
Staged offline document pipeline orchestration.

The pipeline is organized as explicit stages so callers can rerun or override
any stage independently:
- ingest_document
- extract_skills
- compile_skills
- register_versions
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from autoskill import AutoSkill

from .core.common import StageLogger, compact_metadata
from .core.config import (
    DEFAULT_DOC_SKILL_USER_ID,
    DEFAULT_EXTRACT_STRATEGY,
    DEFAULT_MAX_SECTION_CHARS,
    DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
    default_store_path,
)
from .core.provider_config import allow_mock_provider, ensure_runtime_llm_provider
from .stages.compiler import (
    SkillCompilationResult,
    SkillCompiler,
    build_skill_compiler,
    compile_skills,
)
from .stages.extractor import (
    LLMDocumentSkillExtractor,
    DocumentSkillExtractor,
    SkillExtractionResult,
    build_document_skill_extractor,
    extract_skills,
)
from .ingest import (
    DocumentIngestResult,
    DocumentIngestor,
    HeuristicDocumentIngestor,
    ingest_document,
)
from .models import DocumentRecord, SkillDraft, SkillSpec, SupportRecord, VersionState
from .models import StrictWindow
from .family_resolver import DocumentFamilyResolver, build_document_family_resolver
from .store.intermediate import (
    IntermediateRunWriter,
    build_resume_key,
    find_intermediate_run_by_resume_key,
    new_intermediate_run_id,
)
from .store.registry import DocumentRegistry, build_registry_from_store_config
from .store.versioning import VersionRegistrationResult, register_versions
from .taxonomy import SkillTaxonomy, load_skill_taxonomy


@dataclass
class DocumentBuildResult:
    """Top-level result of a full offline document build run."""

    ingest: DocumentIngestResult
    extracted: SkillExtractionResult
    compiled: SkillCompilationResult
    registration: VersionRegistrationResult
    dry_run: bool = False
    intermediate: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a compact build summary suitable for CLI/API output."""

        return {
            "dry_run": bool(self.dry_run),
            "documents": len(self.ingest.documents),
            "skipped_documents": len(self.ingest.skipped_documents),
            "windows": len(self.ingest.windows),
            "support_records": len(self.compiled.support_records),
            "skill_drafts": len(self.extracted.skill_drafts),
            "skill_specs": len(self.compiled.skill_specs),
            "lifecycles": len(self.registration.lifecycles),
            "change_logs": len(self.registration.change_logs),
            "version_history_entries": len(self.registration.version_history),
            "provenance_links": len(self.registration.provenance_links),
            "store_upserts": len(self.registration.upserted_store_skills),
            "staging_runs": len(self.registration.staging_runs),
            "visible_families": len(list((self.registration.visible_tree or {}).get("affected_families") or [])),
            "visible_children": len(list((self.registration.visible_tree or {}).get("child_paths") or [])),
            "intermediate": dict(self.intermediate or {}),
            "errors": (
                list(self.ingest.errors)
                + list(self.extracted.errors)
                + list(self.compiled.errors)
                + list(self.registration.errors)
            ),
        }


class DocumentBuildPipeline:
    """Composable staged pipeline for offline document compilation."""

    def __init__(
        self,
        *,
        registry: DocumentRegistry,
        sdk: Optional[AutoSkill] = None,
        document_ingestor: Optional[DocumentIngestor] = None,
        document_skill_extractor: Optional[DocumentSkillExtractor] = None,
        skill_compiler: Optional[SkillCompiler] = None,
        taxonomy: Optional[SkillTaxonomy] = None,
        family_resolver: Optional[DocumentFamilyResolver] = None,
        logger: StageLogger = None,
        retrieval_score_threshold: float = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
    ) -> None:
        """Builds a pipeline with replaceable stage implementations."""

        self.registry = registry
        self.sdk = sdk
        self.document_ingestor = document_ingestor or HeuristicDocumentIngestor()
        self.document_skill_extractor = document_skill_extractor or build_document_skill_extractor("llm")
        self.skill_compiler = skill_compiler or build_skill_compiler("llm")
        self.taxonomy = taxonomy or load_skill_taxonomy()
        self.family_resolver = family_resolver or build_document_family_resolver(taxonomy=self.taxonomy)
        self.logger = logger
        self.retrieval_score_threshold = max(0.0, float(retrieval_score_threshold or DEFAULT_RETRIEVAL_SCORE_THRESHOLD))

    def _ensure_runtime_llm(self, *, context: str) -> None:
        """Prevents user-facing extraction paths from silently using mock."""

        llm_cfg = dict(getattr(getattr(self.sdk, "config", None), "llm", {}) or {})
        if not llm_cfg:
            return
        ensure_runtime_llm_provider(
            str(llm_cfg.get("provider") or ""),
            context=context,
            allow_mock=allow_mock_provider(llm_cfg),
        )

    def _run_taxonomy(
        self,
        *,
        documents: Sequence[DocumentRecord],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillTaxonomy:
        """Loads the most specific taxonomy for one build run."""

        md = dict(metadata or {})
        requested_domain_type = (
            str(md.get("domain_type") or "").strip()
            or str(md.get("domain") or "").strip()
            or next((str(doc.domain or "").strip() for doc in list(documents or []) if str(doc.domain or "").strip()), "")
        )
        requested_taxonomy_path = str(md.get("skill_taxonomy_path") or md.get("skill_taxonomy") or "").strip()
        if not requested_domain_type and not requested_taxonomy_path:
            return self.taxonomy
        current_path = str(getattr(self.taxonomy, "taxonomy_id", "") or "").strip()
        if not requested_taxonomy_path and requested_domain_type == str(self.taxonomy.domain_type or "").strip():
            return self.taxonomy
        try:
            return load_skill_taxonomy(
                domain_type=requested_domain_type,
                taxonomy_path=requested_taxonomy_path,
            )
        except Exception:
            return self.taxonomy

    def _family_resolver_for_taxonomy(self, taxonomy: SkillTaxonomy) -> DocumentFamilyResolver:
        """Builds one compatible family resolver when the run taxonomy changes."""

        if str(taxonomy.taxonomy_id or "").strip() == str(self.taxonomy.taxonomy_id or "").strip():
            return self.family_resolver
        return build_document_family_resolver(
            taxonomy=taxonomy,
            llm=getattr(self.family_resolver, "llm", None),
        )

    def _extractor_for_taxonomy(self, taxonomy: SkillTaxonomy) -> DocumentSkillExtractor:
        """Builds one compatible extractor when the run taxonomy differs."""

        current = self.document_skill_extractor
        current_taxonomy_id = str(getattr(getattr(current, "taxonomy", None), "taxonomy_id", "") or "").strip()
        if current_taxonomy_id == str(taxonomy.taxonomy_id or "").strip():
            return current
        if isinstance(current, LLMDocumentSkillExtractor):
            return build_document_skill_extractor(
                "llm",
                llm=getattr(current, "_llm", None),
                max_section_chars=int(getattr(current, "max_section_chars", DEFAULT_MAX_SECTION_CHARS) or DEFAULT_MAX_SECTION_CHARS),
                overlap_chars=int(getattr(current, "overlap_chars", 0) or 0),
                max_candidates_per_unit=int(getattr(current, "max_candidates_per_unit", 3) or 3),
                max_units_per_document=int(getattr(current, "max_units_per_document", 0) or 0),
                extract_workers=int(getattr(current, "extract_workers", 1) or 1),
                extract_retries=int(getattr(current, "extract_retries", 3) or 0),
                extract_retry_backoff_s=float(getattr(current, "extract_retry_backoff_s", 1.0) or 0.0),
                taxonomy=taxonomy,
            )
        return current

    def resolve_run_metadata(
        self,
        *,
        documents: Sequence[DocumentRecord],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolves family/domain display metadata after ingest."""

        md = dict(metadata or {})
        effective_taxonomy = self._run_taxonomy(documents=documents, metadata=md)
        resolved_family = str(md.get("family_name") or "").strip()
        if not resolved_family and self.family_resolver is not None:
            resolver = self._family_resolver_for_taxonomy(effective_taxonomy)
            resolution = resolver.resolve(
                documents=list(documents or []),
                metadata=md,
            )
            if len(list(documents or [])) > 1:
                per_doc_matches = []
                for document in list(documents or []):
                    doc_resolution = resolver.resolve(
                        documents=[document],
                        metadata=md,
                        allow_llm=False,
                    )
                    family_name = str(doc_resolution.family_name or "").strip()
                    if family_name and family_name != str(effective_taxonomy.default_family_name or "").strip():
                        per_doc_matches.append(family_name)
                detected_families = sorted({name for name in per_doc_matches if name})
                if len(detected_families) > 1:
                    default_candidate = effective_taxonomy.resolve_family_candidate(
                        requested=str(effective_taxonomy.default_family_name or "").strip()
                    )
                    resolution.family_id = ""
                    resolution.family_name = str(effective_taxonomy.default_family_name or "").strip()
                    if default_candidate is not None:
                        resolution.family_id = str(default_candidate.get("id") or "").strip()
                        resolution.family_name = str(
                            default_candidate.get("visible_name") or default_candidate.get("name") or resolution.family_name
                        ).strip()
                    resolution.confidence = min(float(resolution.confidence or 0.0), 0.3)
                    resolution.source = "mixed_rule"
                    resolution.reason = (
                        "documents matched multiple configured families; "
                        "fallback to taxonomy default family for this batch"
                    )
                    md["family_candidates_detected"] = detected_families
            if resolution.family_name:
                md["family_name"] = resolution.family_name
            if resolution.family_id:
                md["family_id"] = resolution.family_id
            md["family_confidence"] = float(resolution.confidence or 0.0)
            md["family_source"] = str(resolution.source or "").strip()
            if str(resolution.reason or "").strip():
                md["family_reason"] = str(resolution.reason or "").strip()
        elif resolved_family:
            candidate = effective_taxonomy.resolve_family_candidate(requested=resolved_family, metadata=md)
            if candidate is not None:
                md["family_name"] = str(candidate.get("visible_name") or candidate.get("name") or "").strip()
                if str(candidate.get("id") or "").strip():
                    md["family_id"] = str(candidate.get("id") or "").strip()
        if not str(md.get("taxonomy_axis") or "").strip():
            axis = effective_taxonomy.resolve_axis_label()
            if axis:
                md["taxonomy_axis"] = axis
        if not str(md.get("domain_root_name") or "").strip():
            md["domain_root_name"] = effective_taxonomy.domain_root_name()
        if not str(md.get("domain_root_id") or "").strip():
            md["domain_root_id"] = effective_taxonomy.domain_root_id()
        if not str(md.get("family_bucket_label") or "").strip():
            md["family_bucket_label"] = effective_taxonomy.family_bucket_label()
        if not isinstance(md.get("visible_levels"), dict):
            md["visible_levels"] = dict(effective_taxonomy.to_dict().get("visible_levels") or {})
        if not str(md.get("profile_id") or "").strip():
            md["profile_id"] = effective_taxonomy.derive_profile_id(
                requested="",
                family_name=str(md.get("family_name") or "").strip(),
            )
        md["taxonomy_id"] = str(md.get("taxonomy_id") or effective_taxonomy.taxonomy_id).strip()
        md["domain_type"] = str(md.get("domain_type") or effective_taxonomy.domain_type).strip()
        return md

    def ingest_document(
        self,
        *,
        data: Optional[Any] = None,
        file_path: str = "",
        title: str = "",
        source_type: str = "document",
        domain: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        continue_on_error: bool = True,
        dry_run: bool = False,
        max_documents: int = 0,
        extract_strategy: str = DEFAULT_EXTRACT_STRATEGY,
    ) -> DocumentIngestResult:
        """Runs the ingestion stage only."""

        return ingest_document(
            data=data,
            file_path=file_path,
            title=title,
            source_type=source_type,
            domain=domain,
            metadata=metadata,
            registry=self.registry,
            ingestor=self.document_ingestor,
            continue_on_error=continue_on_error,
            dry_run=dry_run,
            max_documents=max_documents,
            extract_strategy=extract_strategy,
            logger=self.logger,
        )

    def extract_skills(
        self,
        *,
        documents: List[DocumentRecord],
        windows: Optional[List[StrictWindow]] = None,
        taxonomy: Optional[SkillTaxonomy] = None,
        extract_workers: Optional[int] = None,
        progress_callback=None,
        stage_progress_callback=None,
        accumulate_result: bool = True,
    ) -> SkillExtractionResult:
        """Runs the direct skill extraction stage only."""

        self._ensure_runtime_llm(context="AutoSkill4Doc extract_skills")
        extractor = self._extractor_for_taxonomy(taxonomy or self.taxonomy)
        if (
            extract_workers is not None
            and isinstance(extractor, LLMDocumentSkillExtractor)
            and int(getattr(extractor, "extract_workers", 1) or 1) != max(1, int(extract_workers or 1))
        ):
            extractor = build_document_skill_extractor(
                "llm",
                llm=getattr(extractor, "_llm", None),
                llm_config=dict(getattr(extractor, "_llm_config", {}) or {}),
                max_section_chars=int(getattr(extractor, "max_section_chars", DEFAULT_MAX_SECTION_CHARS) or DEFAULT_MAX_SECTION_CHARS),
                overlap_chars=int(getattr(extractor, "overlap_chars", 0) or 0),
                max_candidates_per_unit=int(getattr(extractor, "max_candidates_per_unit", 3) or 3),
                max_units_per_document=int(getattr(extractor, "max_units_per_document", 0) or 0),
                extract_workers=max(1, int(extract_workers or 1)),
                extract_retries=int(getattr(extractor, "extract_retries", 3) or 0),
                extract_retry_backoff_s=float(getattr(extractor, "extract_retry_backoff_s", 1.0) or 0.0),
                taxonomy=getattr(extractor, "taxonomy", None),
            )
        return extract_skills(
            documents=list(documents or []),
            windows=list(windows or []),
            extractor=extractor,
            logger=self.logger,
            progress_callback=progress_callback,
            stage_progress_callback=stage_progress_callback,
            accumulate_result=accumulate_result,
        )

    def compile_skills(
        self,
        *,
        skill_drafts: List[SkillDraft],
        support_records: List[SupportRecord],
        target_state: VersionState = VersionState.DRAFT,
        progress_callback=None,
    ) -> SkillCompilationResult:
        """Runs the skill compilation stage only."""

        self._ensure_runtime_llm(context="AutoSkill4Doc compile_skills")
        return compile_skills(
            skill_drafts=list(skill_drafts or []),
            support_records=list(support_records or []),
            compiler=self.skill_compiler,
            target_state=target_state,
            logger=self.logger,
            progress_callback=progress_callback,
        )

    def register_versions(
        self,
        *,
        documents: List[DocumentRecord],
        support_records: List[SupportRecord],
        skill_specs: List[SkillSpec],
        user_id: str = DEFAULT_DOC_SKILL_USER_ID,
        metadata: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        target_state: VersionState = VersionState.ACTIVE,
        progress_callback=None,
    ) -> VersionRegistrationResult:
        """Runs the registry/version registration stage only."""

        self._ensure_runtime_llm(context="AutoSkill4Doc register_versions")
        return register_versions(
            registry=self.registry,
            documents=list(documents or []),
            support_records=list(support_records or []),
            skill_specs=list(skill_specs or []),
            sdk=self.sdk,
            user_id=str(user_id or "").strip() or DEFAULT_DOC_SKILL_USER_ID,
            metadata=metadata,
            dry_run=dry_run,
            target_state=target_state,
            logger=self.logger,
            progress_callback=progress_callback,
            retrieval_score_threshold=self.retrieval_score_threshold,
        )

    def build(
        self,
        *,
        user_id: str = DEFAULT_DOC_SKILL_USER_ID,
        data: Optional[Any] = None,
        file_path: str = "",
        title: str = "",
        source_type: str = "document",
        domain: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        continue_on_error: bool = True,
        dry_run: bool = False,
        target_state: Optional[VersionState] = None,
        max_documents: int = 0,
        extract_strategy: str = DEFAULT_EXTRACT_STRATEGY,
        extract_progress_callback=None,
        stage_progress_callback=None,
    ) -> DocumentBuildResult:
        """Runs the full offline document build pipeline."""

        self._ensure_runtime_llm(context="AutoSkill4Doc build")
        effective_state = target_state or (VersionState.DRAFT if dry_run else VersionState.ACTIVE)
        intermediate_writer = None
        intermediate_summary: Dict[str, Any] = {}
        resumed_run = False
        if not dry_run:
            store_root = ""
            if self.sdk is not None:
                store_root = str(getattr(getattr(self.sdk, "config", None), "store", {}).get("path") or "").strip()
            if not store_root:
                registry_root = os.path.abspath(os.path.expanduser(str(self.registry.root_dir or "").strip()))
                runtime_dir = os.path.dirname(registry_root)
                if os.path.basename(runtime_dir) == ".runtime":
                    store_root = os.path.dirname(runtime_dir)
            if store_root:
                llm_cfg = dict(getattr(getattr(self.sdk, "config", None), "llm", {}) or {})
                emb_cfg = dict(getattr(getattr(self.sdk, "config", None), "embeddings", {}) or {})
                if file_path:
                    input_signature = self._file_input_signature(file_path=file_path)
                else:
                    input_signature = self._data_input_signature(data)
                resume_payload = {
                    "input_signature": input_signature,
                    "title": str(title or "").strip(),
                    "source_type": str(source_type or "").strip(),
                    "domain": str(domain or "").strip(),
                    "metadata": compact_metadata(dict(metadata or {})),
                    "max_documents": int(max_documents or 0),
                    "extract_strategy": str(extract_strategy or "").strip(),
                    "llm_provider": str(llm_cfg.get("provider") or "").strip(),
                    "llm_model": str(llm_cfg.get("model") or "").strip(),
                    "embeddings_provider": str(emb_cfg.get("provider") or "").strip(),
                    "embeddings_model": str(emb_cfg.get("model") or "").strip(),
                }
                resume_key = build_resume_key(resume_payload)
                resume_metadata = dict(metadata or {})
                resume_metadata["resume_key"] = resume_key
                resumable = find_intermediate_run_by_resume_key(base_store_root=store_root, resume_key=resume_key)
                intermediate_writer = IntermediateRunWriter(
                    base_store_root=store_root,
                    run_id=str((resumable or {}).get("run_id") or new_intermediate_run_id()),
                    metadata=resume_metadata,
                    resume_existing=bool(resumable),
                )
                if resumable:
                    resumed_run = True
                    intermediate_writer.update_metadata({"resumed_from_run_id": str((resumable or {}).get("run_id") or "").strip()})
                intermediate_summary = intermediate_writer.summary().to_dict()
        base_metadata = dict(metadata or {})
        if intermediate_writer is not None:
            base_metadata["resume_key"] = str((intermediate_writer._state.get("metadata") or {}).get("resume_key") or "").strip()
        if intermediate_writer is not None and intermediate_writer.has_completed_stage("ingest"):
            ingest_result = intermediate_writer.load_ingest()
        else:
            ingest_result = self.ingest_document(
                data=data,
                file_path=file_path,
                title=title,
                source_type=source_type,
                domain=domain,
                metadata=base_metadata,
                continue_on_error=continue_on_error,
                dry_run=dry_run,
                max_documents=max_documents,
                extract_strategy=extract_strategy,
            )
            if intermediate_writer is not None:
                intermediate_writer.write_ingest(ingest_result)
                intermediate_summary = intermediate_writer.summary().to_dict()
        resolved_metadata = self.resolve_run_metadata(documents=ingest_result.documents, metadata=base_metadata)
        resolved_metadata = compact_metadata(resolved_metadata)
        if intermediate_writer is not None:
            intermediate_writer.update_metadata(resolved_metadata)
            intermediate_summary = intermediate_writer.summary().to_dict()
        effective_taxonomy = self._run_taxonomy(documents=ingest_result.documents, metadata=resolved_metadata)
        for document in list(ingest_result.documents or []):
            document.metadata.update(resolved_metadata)
        for document in list(ingest_result.skipped_documents or []):
            document.metadata.update(resolved_metadata)
        progress_owner = extract_progress_callback
        if (
            progress_owner is not None
            and not hasattr(progress_owner, "set_total_documents")
            and hasattr(progress_owner, "__self__")
        ):
            progress_owner = getattr(progress_owner, "__self__", progress_owner)
        if progress_owner is not None and hasattr(progress_owner, "set_total_documents"):
            completed_docs = 0
            if intermediate_writer is not None:
                completed_docs = len(intermediate_writer.processed_extract_doc_ids())
            try:
                progress_owner.set_total_documents(
                    len(list(ingest_result.documents or [])),
                    completed=completed_docs,
                    total_windows=len(list(ingest_result.windows or [])),
                    completed_windows=len(
                        [
                            window
                            for window in list(ingest_result.windows or [])
                            if str(window.doc_id or "").strip() in set(intermediate_writer.processed_extract_doc_ids())
                        ]
                    )
                    if intermediate_writer is not None
                    else 0,
                )
            except Exception:
                pass
        try:
            if intermediate_writer is not None:
                extracted_result = self._resume_or_extract(
                    intermediate_writer=intermediate_writer,
                    ingest_result=ingest_result,
                    taxonomy=effective_taxonomy,
                    extract_progress_callback=extract_progress_callback,
                    stage_progress_callback=stage_progress_callback,
                )
                intermediate_summary = intermediate_writer.summary().to_dict()
            else:
                extracted_result = self.extract_skills(
                    documents=ingest_result.documents,
                    windows=ingest_result.windows,
                    taxonomy=effective_taxonomy,
                    extract_workers=int(getattr(self.document_skill_extractor, "extract_workers", 1) or 1),
                    accumulate_result=True,
                    progress_callback=extract_progress_callback,
                    stage_progress_callback=stage_progress_callback,
                )
            if intermediate_writer is not None and intermediate_writer.has_completed_stage("compile"):
                compiled_result = intermediate_writer.load_compile()
            else:
                compiled_result = self.compile_skills(
                    skill_drafts=extracted_result.skill_drafts,
                    support_records=extracted_result.support_records,
                    target_state=effective_state,
                    progress_callback=stage_progress_callback,
                )
                if intermediate_writer is not None:
                    intermediate_writer.write_compile(compiled_result)
                    intermediate_summary = intermediate_writer.summary().to_dict()
            if intermediate_writer is not None and intermediate_writer.has_completed_stage("register"):
                registration_result = intermediate_writer.load_registration()
            else:
                registration_result = self.register_versions(
                    documents=ingest_result.documents,
                    support_records=compiled_result.support_records,
                    skill_specs=compiled_result.skill_specs,
                    user_id=user_id,
                    metadata=resolved_metadata,
                    dry_run=dry_run,
                    target_state=effective_state,
                    progress_callback=stage_progress_callback,
                )
                if intermediate_writer is not None:
                    intermediate_writer.write_registration(registration_result)
                    intermediate_summary = intermediate_writer.summary().to_dict()
            build_result = DocumentBuildResult(
                ingest=ingest_result,
                extracted=extracted_result,
                compiled=compiled_result,
                registration=registration_result,
                dry_run=bool(dry_run),
                intermediate=dict(intermediate_summary or {}),
            )
            if intermediate_writer is not None:
                if resumed_run:
                    build_result.intermediate["resumed"] = True
                intermediate_writer.complete(summary=build_result.to_dict())
                intermediate_summary = intermediate_writer.summary().to_dict()
                build_result.intermediate = dict(intermediate_summary or {})
                if resumed_run:
                    build_result.intermediate["resumed"] = True
            return build_result
        except Exception as exc:
            if intermediate_writer is not None:
                intermediate_writer.fail(error=str(exc))
            raise

    @staticmethod
    def _data_input_signature(data: Optional[Any]) -> str:
        """Builds a stable content signature for in-memory input payloads."""

        if data is None:
            return ""
        if isinstance(data, bytes):
            raw = data
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raw = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _file_input_signature(*, file_path: str) -> Dict[str, Any]:
        """Builds a lightweight change signature for a file or directory input."""

        abs_path = os.path.abspath(os.path.expanduser(str(file_path or "").strip()))
        if not abs_path or not os.path.exists(abs_path):
            return {"path": abs_path, "exists": False}
        if os.path.isfile(abs_path):
            stat = os.stat(abs_path)
            return {"path": abs_path, "type": "file", "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        items: List[Dict[str, Any]] = []
        for root, _, files in os.walk(abs_path):
            for name in sorted(files):
                path = os.path.join(root, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                items.append(
                    {
                        "relative_path": os.path.relpath(path, abs_path),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
        return {"path": abs_path, "type": "directory", "files": items}

    def _resume_or_extract(
        self,
        *,
        intermediate_writer: IntermediateRunWriter,
        ingest_result: DocumentIngestResult,
        taxonomy: SkillTaxonomy,
        extract_progress_callback=None,
        stage_progress_callback=None,
    ) -> SkillExtractionResult:
        """Loads persisted extract progress when possible and continues remaining docs only."""

        if intermediate_writer.has_completed_stage("extract"):
            return intermediate_writer.load_extract()
        persisted = intermediate_writer.load_extract()
        processed_doc_ids = set(intermediate_writer.processed_extract_doc_ids())
        remaining_docs = [doc for doc in list(ingest_result.documents or []) if str(doc.doc_id or "").strip() not in processed_doc_ids]
        if not remaining_docs:
            intermediate_writer.write_extract(persisted)
            return persisted
        remaining_docs_by_id = {str(doc.doc_id or "").strip(): doc for doc in remaining_docs}
        remaining_doc_ids = {str(doc.doc_id or "").strip() for doc in remaining_docs}
        remaining_windows = [window for window in list(ingest_result.windows or []) if str(window.doc_id or "").strip() in remaining_doc_ids]
        fresh = self.extract_skills(
            documents=remaining_docs,
            windows=remaining_windows,
            taxonomy=taxonomy,
            extract_workers=int(getattr(self.document_skill_extractor, "extract_workers", 1) or 1),
            accumulate_result=True,
            progress_callback=lambda record, supports, drafts, cumulative: self._on_resume_extract_progress(
                intermediate_writer=intermediate_writer,
                ingest_result=ingest_result,
                extract_progress_callback=extract_progress_callback,
                record=record,
                supports=supports,
                drafts=drafts,
                cumulative=cumulative,
            ),
            stage_progress_callback=stage_progress_callback,
        )
        written_error_doc_ids = set()
        for error in list(fresh.errors or []):
            if not isinstance(error, dict):
                continue
            doc_id = str(error.get("doc_id") or "").strip()
            record = remaining_docs_by_id.get(doc_id)
            if record is None or doc_id in written_error_doc_ids:
                continue
            intermediate_writer.write_extract_error(
                record=record,
                error=dict(error or {}),
                total_documents=len(ingest_result.documents),
            )
            written_error_doc_ids.add(doc_id)
        supports_by_id = {support.support_id: support for support in list(persisted.support_records or [])}
        for support in list(fresh.support_records or []):
            supports_by_id[support.support_id] = support
        drafts_by_id = {draft.draft_id: draft for draft in list(persisted.skill_drafts or [])}
        for draft in list(fresh.skill_drafts or []):
            drafts_by_id[draft.draft_id] = draft
        errors = list(persisted.errors or []) + list(fresh.errors or [])
        merged = SkillExtractionResult(
            documents=list(ingest_result.documents or []),
            windows=list(ingest_result.windows or []),
            support_records=list(supports_by_id.values()),
            skill_drafts=list(drafts_by_id.values()),
            errors=errors,
            extractor_name=fresh.extractor_name or persisted.extractor_name,
        )
        intermediate_writer.write_extract(merged)
        return merged

    @staticmethod
    def _on_resume_extract_progress(
        *,
        intermediate_writer: IntermediateRunWriter,
        ingest_result: DocumentIngestResult,
        extract_progress_callback,
        record: DocumentRecord,
        supports: List[SupportRecord],
        drafts: List[SkillDraft],
        cumulative: SkillExtractionResult,
    ) -> None:
        """Combines intermediate persistence with optional external progress reporting."""

        intermediate_writer.write_extract_progress(
            record=record,
            supports=supports,
            drafts=drafts,
            total_documents=len(ingest_result.documents),
        )
        if extract_progress_callback is not None:
            extract_progress_callback(record, supports, drafts, cumulative)


def build_default_document_pipeline(
    *,
    sdk: Optional[AutoSkill] = None,
    registry_root: str = "",
    logger: StageLogger = None,
    document_ingestor: Optional[DocumentIngestor] = None,
    document_skill_extractor: Optional[DocumentSkillExtractor] = None,
    skill_compiler: Optional[SkillCompiler] = None,
    taxonomy: Optional[SkillTaxonomy] = None,
    family_resolver: Optional[DocumentFamilyResolver] = None,
    extract_workers: int = 1,
    extract_retries: int = 3,
    extract_retry_backoff_s: float = 1.0,
    retrieval_score_threshold: float = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
) -> DocumentBuildPipeline:
    """Builds the default staged document pipeline."""

    if registry_root:
        registry = DocumentRegistry(root_dir=registry_root)
    elif sdk is not None:
        registry = build_registry_from_store_config(dict(getattr(getattr(sdk, "config", None), "store", {}) or {}))
    else:
        from .store.registry import default_registry_root

        registry = DocumentRegistry(root_dir=default_registry_root(default_store_path()))
    effective_taxonomy = taxonomy or load_skill_taxonomy()
    return DocumentBuildPipeline(
        registry=registry,
        sdk=sdk,
        logger=logger,
        document_ingestor=document_ingestor
        or HeuristicDocumentIngestor(
            llm_config=dict(getattr(getattr(sdk, "config", None), "llm", {}) or {}),
            max_section_chars=DEFAULT_MAX_SECTION_CHARS,
            outline_fallback_mode="auto",
        ),
        document_skill_extractor=document_skill_extractor
        or build_document_skill_extractor(
            "llm",
            llm_config=dict(getattr(getattr(sdk, "config", None), "llm", {}) or {}),
            extract_workers=max(1, int(extract_workers or 1)),
            extract_retries=max(0, int(extract_retries or 0)),
            extract_retry_backoff_s=max(0.0, float(extract_retry_backoff_s or 0.0)),
        ),
        skill_compiler=skill_compiler
        or build_skill_compiler(
            "llm",
            llm_config=dict(getattr(getattr(sdk, "config", None), "llm", {}) or {}),
        ),
        taxonomy=effective_taxonomy,
        family_resolver=family_resolver
        or build_document_family_resolver(
            taxonomy=effective_taxonomy,
            llm_config=dict(getattr(getattr(sdk, "config", None), "llm", {}) or {}),
        ),
        retrieval_score_threshold=max(0.0, float(retrieval_score_threshold or DEFAULT_RETRIEVAL_SCORE_THRESHOLD)),
    )
