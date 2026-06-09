#!/usr/bin/env python3
"""Unified RAG: Hierarchical Segment Router + Calibrated 6-Stage Pipeline.

Pipeline:
1. LLM routes query to the best segment (for long PDFs)
2. HybridPageExperiment retrieves and calibrates candidates
3. Candidates are filtered to the segment page range
4. Run all 6 stages (A~F) including anchor / neighbor / BGE reranker
5. LLM generates the final answer from the winning page + neighbors
"""

from __future__ import annotations

import os
from typing import Iterable

from rag.llm.llm import LLMService
from rag.retriever.page_calibration import HybridPageExperiment
from rag.retriever.rag import RAGService, RetrievedPage, StructuredAnswer
from rag.retriever.segment_router import SegmentRouter
from rag.vector.vector_db import VectorDB

_HIERARCHICAL_ENABLED = os.getenv("HIERARCHICAL_RAG_ENABLED", "1").strip() not in (
    "0", "false", "False", "",
)


class UnifiedRAGService:
    """Combines Hierarchical segment routing with the full 6-stage calibration."""

    def __init__(
        self,
        vector_db=None,
        llm=None,
        page_index=None,
        enable_hierarchical: bool | None = None,
        candidate_pages: int = 50,
    ):
        self.base = RAGService(
            vector_db=vector_db,
            llm=llm,
            page_index=page_index,
        )
        self._enable = (
            enable_hierarchical
            if enable_hierarchical is not None
            else _HIERARCHICAL_ENABLED
        )
        self.router = SegmentRouter(llm=self.base.llm)
        self.experiment = HybridPageExperiment(
            rag=self.base,
            candidate_pages=candidate_pages,
        )

    # ------------------------------------------------------------------
    # Core answer flow
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        filename: str | None = None,
        run_llm: bool = True,
        include_prompt: bool = False,
        neighbor_pages: int = 1,
        max_chars_per_page: int = 4500,
    ) -> dict:
        """Run the full unified pipeline and return a structured answer."""
        # 1) Segment routing
        segment = None
        if filename and self._enable:
            segment = self.router.route(query, filename)

        route_info: dict = {
            "hierarchical_enabled": self._enable,
            "segment": None,
        }
        if segment:
            route_info["segment"] = {
                "id": segment.segment_id,
                "title": segment.title,
                "start_page": segment.start_page,
                "end_page": segment.end_page,
            }

        # 2) Retrieve hybrid candidates (global)
        candidates = self.experiment.retrieve_hybrid_candidates(
            query,
            initial_k=200,
            max_chars_per_page=max_chars_per_page,
        )

        # 3) Filter to segment range
        if segment and segment.segment_id != "all":
            start = max(1, segment.start_page - 2)
            end = segment.end_page + 2
            filtered = [
                p for p in candidates
                if start <= int(p.page_number) <= end
            ]
            # Safety fallback: keep at least 5 pages
            if len(filtered) < 5:
                filtered = candidates
            candidates = filtered

        # 4) Run all 6 calibration stages (A~F)
        stage_results = self.experiment.run_all_stages(query, candidates)
        selected_stage = stage_results["F_advanced_guarded"]
        top_page = selected_stage.page
        top_filename = selected_stage.filename

        # 5) Build context pages (center page + neighbors)
        context_pages = self.base.build_forced_pages(
            query=query,
            filename=top_filename,
            page_number=top_page,
            neighbor_pages=neighbor_pages,
            max_chars_per_page=max_chars_per_page,
        )
        if not context_pages:
            return {
                "filename": top_filename or "",
                "page": top_page if top_page is not None else -1,
                "answer": "未检索到相关片段",
                "sources": [],
                "llm_used": False,
                "raw_answer": "",
                "error": "no_context_pages",
                "route_info": route_info,
                **({"prompt": ""} if include_prompt else {}),
            }

        # 6) Build prompt and run LLM
        prompt = self.base.build_prompt(query, context_pages)
        sources = self.base._format_sources(context_pages)
        fallback = StructuredAnswer(
            filename=top_filename,
            page=top_page,
            answer="",
            sources=sources,
            prompt=prompt,
            llm_used=False,
        )

        if not run_llm:
            fallback.error = "llm_not_run"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["route_info"] = route_info
            result["stage_debug"] = selected_stage.debug
            return result

        if not self.base.llm.available:
            fallback.error = "llm_unavailable: OPENAI_API_KEY is empty"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["route_info"] = route_info
            result["stage_debug"] = selected_stage.debug
            return result

        try:
            raw_answer = self.base.llm.generate(prompt)
        except Exception as exc:
            fallback.error = f"llm_exception: {exc}"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["route_info"] = route_info
            result["stage_debug"] = selected_stage.debug
            return result

        parsed = self.base.parse_structured_answer(raw_answer)
        if not parsed:
            fallback.answer = raw_answer.strip()
            fallback.raw_answer = raw_answer
            fallback.llm_used = True
            fallback.error = "llm_output_not_json"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["route_info"] = route_info
            result["stage_debug"] = selected_stage.debug
            return result

        answer = parsed.get("answer") or ""
        parsed_filename = parsed.get("filename") or top_filename
        parsed_page = self.base._normalize_page_value(
            parsed.get("page") or top_page
        )
        valid_sources = {
            (source["filename"], str(source["page"])) for source in sources
        }
        if (parsed_filename, str(parsed_page)) not in valid_sources:
            parsed_filename = top_filename
            parsed_page = top_page
        else:
            parsed_filename, parsed_page = self.base._maybe_override_source(
                parsed_filename, parsed_page, sources
            )

        result = StructuredAnswer(
            filename=parsed_filename,
            page=parsed_page,
            answer=answer,
            sources=sources,
            prompt=prompt,
            raw_answer=raw_answer,
            llm_used=True,
        ).to_dict(include_prompt=include_prompt)
        result["route_info"] = route_info
        result["stage_debug"] = selected_stage.debug
        return result

    # ------------------------------------------------------------------
    # Batch helper (used by run script)
    # ------------------------------------------------------------------

    def answer_for_question_item(
        self,
        item: dict,
        **kwargs,
    ) -> dict:
        """Convenience wrapper for a single test.json item."""
        return self.answer(
            query=item["question"],
            filename=item.get("filename"),
            **kwargs,
        )
