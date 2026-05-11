#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .serper_search import SerperSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class UnifiedSearchResult:
    title: str
    url: str
    snippet: str
    source: str
    position: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "position": self.position,
            "metadata": self.metadata or {},
        }


class BaseRetriever(ABC):
    name = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int) -> List[UnifiedSearchResult]:
        raise NotImplementedError


class SerperWebRetriever(BaseRetriever):
    name = "serper"

    def __init__(self, engine: SerperSearchEngine):
        self.engine = engine

    async def search(self, query: str, max_results: int) -> List[UnifiedSearchResult]:
        results = await self.engine.search(query, max_results=max_results)
        return [
            UnifiedSearchResult(
                title=item.title,
                url=item.url,
                snippet=item.snippet,
                source=self.name,
                position=item.position,
                metadata={"date": item.date} if item.date else {},
            )
            for item in results
        ]


class SerperScholarRetriever(BaseRetriever):
    name = "serper_scholar"

    def __init__(self, engine: SerperSearchEngine):
        self.engine = engine

    async def search(self, query: str, max_results: int) -> List[UnifiedSearchResult]:
        results = await self.engine.search_scholar(query, max_results=max_results)
        return [
            UnifiedSearchResult(
                title=item.title,
                url=item.url,
                snippet=item.snippet,
                source=self.name,
                position=item.position,
                metadata={"date": item.date} if item.date else {},
            )
            for item in results
        ]


class RetrieverManager:
    def __init__(self, retrievers: List[BaseRetriever]):
        self.retrievers = retrievers

    async def search(self, query: str, max_results: int) -> List[UnifiedSearchResult]:
        all_results: List[UnifiedSearchResult] = []
        errors: List[Exception] = []
        if not self.retrievers:
            raise RuntimeError("没有可用的检索器")
        for retriever in self.retrievers:
            try:
                results = await retriever.search(query, max_results=max_results)
                all_results.extend(results)
            except Exception as e:
                errors.append(e)
                logger.warning(f"Retriever {retriever.name} 搜索失败: {e}")
        if errors and not all_results:
            raise RuntimeError("所有检索器搜索失败") from errors[0]
        return self._deduplicate(all_results)

    def list_retrievers(self) -> List[str]:
        return [retriever.name for retriever in self.retrievers]

    @staticmethod
    def _deduplicate(results: Iterable[UnifiedSearchResult]) -> List[UnifiedSearchResult]:
        deduped: List[UnifiedSearchResult] = []
        seen_urls = set()
        seen_title_snippets = set()

        for result in results:
            url = (result.url or "").strip()
            title_snippet = ((result.title or "") + "|" + (result.snippet or "")[:120]).strip().lower()
            if url and url in seen_urls:
                continue
            if title_snippet and title_snippet in seen_title_snippets:
                continue
            if url:
                seen_urls.add(url)
            if title_snippet:
                seen_title_snippets.add(title_snippet)
            deduped.append(result)
        return deduped


def build_retriever_manager(search_config: Dict[str, Any], serper_engine: Optional[SerperSearchEngine]) -> RetrieverManager:
    retriever_names = search_config.get("retrievers") or ["serper"]
    if isinstance(retriever_names, str):
        retriever_names = [item.strip() for item in retriever_names.split(",") if item.strip()]

    retrievers: List[BaseRetriever] = []
    for name in retriever_names:
        normalized = name.lower()
        if normalized == "serper":
            if serper_engine:
                retrievers.append(SerperWebRetriever(serper_engine))
        elif normalized in {"serper_scholar", "scholar"}:
            if serper_engine:
                retrievers.append(SerperScholarRetriever(serper_engine))
        else:
            logger.warning(f"未知 retriever: {name}")

    return RetrieverManager(retrievers)
