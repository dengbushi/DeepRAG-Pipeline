#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ContextChunk:
    text: str
    source_url: str
    source_title: str
    source_type: str
    score: float
    query: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "source_type": self.source_type,
            "score": self.score,
            "query": self.query,
        }


class ContextManager:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120, top_k: int = 8, similarity_threshold: float = 0.35):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def select_relevant_context(
        self,
        query: str,
        search_results: List[Any],
        content_results: List[Any],
        max_chunks: Optional[int] = None,
    ) -> List[ContextChunk]:
        chunks: List[ContextChunk] = []

        for item in search_results:
            title = getattr(item, "title", "") if not isinstance(item, dict) else item.get("title", "")
            url = getattr(item, "url", "") if not isinstance(item, dict) else item.get("url", "")
            snippet = getattr(item, "snippet", "") if not isinstance(item, dict) else item.get("snippet", "")
            source = getattr(item, "source", "search") if not isinstance(item, dict) else item.get("source", "search")
            if snippet:
                chunks.append(ContextChunk(snippet, url, title, source, self._score(query, title + "\n" + snippet), query))

        for item in content_results:
            success = getattr(item, "success", False) if not isinstance(item, dict) else item.get("success", False)
            if not success:
                continue
            title = getattr(item, "title", "") if not isinstance(item, dict) else item.get("title", "")
            url = getattr(item, "url", "") if not isinstance(item, dict) else item.get("url", "")
            content = getattr(item, "content", "") if not isinstance(item, dict) else item.get("content", "")
            for piece in self._chunk_text(content):
                chunks.append(ContextChunk(piece, url, title, "content", self._score(query, title + "\n" + piece), query))

        ranked = sorted(chunks, key=lambda item: item.score, reverse=True)
        deduped = self._deduplicate_chunks([item for item in ranked if item.score >= self.similarity_threshold])
        return deduped[: max_chunks or self.top_k]

    def select_context_with_budget(
        self,
        query: str,
        contexts: List[Any],
        max_chars: int,
        max_chunks: Optional[int] = None,
        source_mapping: Optional[Dict[str, int]] = None,
        per_source_limit: int = 2,
    ) -> List[ContextChunk]:
        ranked = self.rerank_context_chunks(query=query, contexts=contexts, max_chunks=None)
        if not ranked:
            return []

        selected: List[ContextChunk] = []
        used_signatures = set()
        source_counts: Dict[str, int] = {}
        remaining_budget = max_chars
        chunk_limit = max_chunks or self.top_k

        for enforce_diversity in (True, False):
            for chunk in ranked:
                if len(selected) >= chunk_limit or remaining_budget <= 0:
                    return selected

                signature = (chunk.source_url, chunk.text[:120].lower())
                if signature in used_signatures:
                    continue

                source_key = chunk.source_url or f"__unknown__::{chunk.source_title}"
                current_source_count = source_counts.get(source_key, 0)
                if enforce_diversity and current_source_count >= 1:
                    continue
                if current_source_count >= per_source_limit:
                    continue

                estimated_size = self._estimate_evidence_block_size(
                    chunk,
                    ref_num=source_mapping.get(chunk.source_url) if source_mapping else None,
                )
                if selected and estimated_size > remaining_budget:
                    continue
                if not selected and estimated_size > max_chars:
                    compressed_text = self._truncate_text(chunk.text, max(180, max_chars // 2))
                    if not compressed_text:
                        continue
                    chunk = ContextChunk(
                        compressed_text,
                        chunk.source_url,
                        chunk.source_title,
                        chunk.source_type,
                        chunk.score,
                        chunk.query,
                    )
                    estimated_size = self._estimate_evidence_block_size(
                        chunk,
                        ref_num=source_mapping.get(chunk.source_url) if source_mapping else None,
                    )
                    if estimated_size > max_chars:
                        continue

                selected.append(chunk)
                used_signatures.add(signature)
                source_counts[source_key] = current_source_count + 1
                remaining_budget -= estimated_size

        return selected

    def rerank_context_chunks(
        self,
        query: str,
        contexts: List[Any],
        max_chunks: Optional[int] = None,
    ) -> List[ContextChunk]:
        chunks: List[ContextChunk] = []
        for item in contexts:
            if isinstance(item, ContextChunk):
                text = item.text
                title = item.source_title
                url = item.source_url
                source_type = item.source_type
            else:
                text = item.get("text", "") if isinstance(item, dict) else ""
                title = item.get("source_title", "") if isinstance(item, dict) else ""
                url = item.get("source_url", "") if isinstance(item, dict) else ""
                source_type = item.get("source_type", "content") if isinstance(item, dict) else "content"

            normalized_text = re.sub(r"\s+", " ", text or "").strip()
            if not normalized_text:
                continue

            chunks.append(
                ContextChunk(
                    normalized_text,
                    url,
                    title,
                    source_type,
                    self._score(query, f"{title}\n{normalized_text}"),
                    query,
                )
            )

        ranked = sorted(chunks, key=lambda item: item.score, reverse=True)
        deduped = self._deduplicate_chunks([item for item in ranked if item.score >= self.similarity_threshold])
        return deduped[: max_chunks or self.top_k]

    def count_high_quality_contexts(self, contexts: List[Any]) -> int:
        count = 0
        for item in contexts:
            score = item.score if isinstance(item, ContextChunk) else float(item.get("score", 0.0))
            if score >= self.similarity_threshold:
                count += 1
        return count

    def build_context_text(self, chunks: Iterable[ContextChunk], max_chars: int = 12000) -> str:
        parts: List[str] = []
        total = 0
        for index, chunk in enumerate(chunks, 1):
            block = (
                f"[{index}] {chunk.source_title or '未知标题'}\n"
                f"URL: {chunk.source_url or '未知URL'}\n"
                f"相关度: {chunk.score:.3f}\n"
                f"内容: {chunk.text.strip()}"
            )
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

    def build_evidence_packet(
        self,
        chunks: Iterable[Any],
        source_mapping: Optional[Dict[str, int]] = None,
        max_chars: int = 12000,
    ) -> str:
        parts: List[str] = []
        total = 0
        for index, item in enumerate(chunks, 1):
            chunk = item if isinstance(item, ContextChunk) else ContextChunk(
                text=item.get("text", ""),
                source_url=item.get("source_url", ""),
                source_title=item.get("source_title", ""),
                source_type=item.get("source_type", "content"),
                score=float(item.get("score", 0.0)),
                query=item.get("query", ""),
            )
            ref_num = source_mapping.get(chunk.source_url) if source_mapping else None
            ref_text = f" [^{ref_num}]" if ref_num else ""
            content_text = self._truncate_text(chunk.text.strip(), 700)
            block = (
                f"[{index}] {chunk.source_title or '未知标题'}{ref_text}\n"
                f"来源: {chunk.source_type}\n"
                f"相关度: {chunk.score:.3f}\n"
                f"内容: {content_text}"
            )
            separator = 0 if not parts else 2
            if total + len(block) + separator > max_chars:
                break
            parts.append(block)
            total += len(block) + separator
        return "\n\n".join(parts)

    def estimate_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def _chunk_text(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: List[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + self.chunk_size)
            chunks.append(normalized[start:end])
            if end >= len(normalized):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def _score(self, query: str, text: str) -> float:
        query_terms = self._tokenize(query)
        text_terms = self._tokenize(text)
        if not query_terms or not text_terms:
            return 0.0

        overlap = sum(1 for term in query_terms if term in text_terms)
        phrase_bonus = 0.2 if query.strip() and query.strip().lower() in text.lower() else 0.0
        density_bonus = min(len(text_terms), 200) / 1000.0
        return overlap / len(query_terms) + phrase_bonus + density_bonus

    def _deduplicate_chunks(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        deduped: List[ContextChunk] = []
        seen = set()
        for chunk in chunks:
            signature = (chunk.source_url, chunk.text[:200].lower())
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(chunk)
        return deduped

    def _estimate_evidence_block_size(self, chunk: ContextChunk, ref_num: Optional[int] = None) -> int:
        ref_text = f" [^{ref_num}]" if ref_num else ""
        content_text = self._truncate_text(chunk.text.strip(), 700)
        block = (
            f"[1] {chunk.source_title or '未知标题'}{ref_text}\n"
            f"来源: {chunk.source_type}\n"
            f"相关度: {chunk.score:.3f}\n"
            f"内容: {content_text}"
        )
        return len(block) + 2

    def _truncate_text(self, text: str, limit: int) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)].rstrip() + "..."

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()) if len(token) > 1]
