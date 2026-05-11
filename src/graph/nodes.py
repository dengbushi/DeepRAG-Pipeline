#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph节点实现
"""

import asyncio
import logging
from typing import Dict, Any, List
from ..agents.agent_manager import AgentManager
from ..search.serper_search import SerperSearchEngine, SerperSearchResult
from ..search.web_reader import WebReader
from ..search.retrievers import UnifiedSearchResult, RetrieverManager, build_retriever_manager
from ..rag.context_manager import ContextChunk
from ..rag.context_manager import ContextManager

logger = logging.getLogger(__name__)


class ResearchNodes:
    """研究工作流节点"""

    def __init__(
        self, 
        agent_manager: AgentManager,
        serper_engine: SerperSearchEngine,
        web_reader: WebReader,
        config: Dict[str, Any]
    ):
        self.agent_manager = agent_manager
        self.serper_engine = serper_engine
        self.web_reader = web_reader
        self.config = config
        
        # 搜索配置
        search_config = config.get('search', {})
        self.max_rounds = search_config.get('max_rounds', 3)
        self.max_total_queries = search_config.get('max_total_queries', 6)
        self.max_urls_per_step = search_config.get('max_urls_to_scrape_per_query', search_config.get('max_urls_per_step', 3))
        self.token_budget = search_config.get('token_budget', 50000)
        self.allow_early_termination = search_config.get('allow_early_termination', True)
        self.max_results = search_config.get('max_results_per_query', search_config.get('max_results', 10))
        self.context_top_k = search_config.get('context_top_k', 8)
        self.context_max_chars = search_config.get('context_max_chars', 12000)
        self.report_context_max_chars = search_config.get('report_context_max_chars', min(self.context_max_chars, 9000))
        self.evidence_similarity_threshold = search_config.get('evidence_similarity_threshold', 0.35)
        self.max_planned_queries = search_config.get('max_planned_queries', 3)
        self.retriever_manager: RetrieverManager = build_retriever_manager(search_config, serper_engine)
        self.context_manager = ContextManager(
            chunk_size=search_config.get('chunk_size', 900),
            chunk_overlap=search_config.get('chunk_overlap', 120),
            top_k=self.context_top_k,
            similarity_threshold=self.evidence_similarity_threshold,
        )

    async def extract_question_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点1: 提取核心问题"""
        logger.info("=== 节点: 提取核心问题 ===")
        
        question = await self.agent_manager.extract_question(
            state["original_query"]
        )
        
        logger.info(f"提取的问题: {question}")
        return {
            "extracted_question": question,
            "events": [self._event("extract_question", state, {"question": question})],
        }

    async def extract_keywords_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点2: 提取搜索关键词"""
        logger.info("=== 节点: 提取关键词 ===")
        
        keywords = await self.agent_manager.extract_keywords(
            state["extracted_question"]
        )
        
        logger.info(f"搜索关键词: {keywords}")
        return {
            "keywords": keywords,
            "current_question": keywords,
            "events": [self._event("extract_keywords", state, {"keywords": keywords})],
        }

    async def search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点3: 执行搜索"""
        logger.info(f"=== 节点: 搜索 (第{state['current_round']}轮) ===")
        
        query = state["current_question"]
        logger.info(f"搜索查询: {query}")
        
        results = await self.retriever_manager.search(query, max_results=self.max_results)
        context = self._format_search_results(results)
        selected_contexts = self.context_manager.select_relevant_context(
            query=query,
            search_results=results,
            content_results=[],
            max_chunks=min(self.context_top_k, 4),
        )
        selected_context_text = self.context_manager.build_context_text(selected_contexts, max_chars=min(self.context_max_chars, 4000))
        
        step_log = {
            "step": state["current_step"],
            "action": "search",
            "query": query,
            "results_count": len(results),
            "round": state["current_round"],
            "retrievers": self.retriever_manager.list_retrievers(),
        }
        
        logger.info(f"搜索完成，获得 {len(results)} 个结果")
        
        high_quality_count = self.context_manager.count_high_quality_contexts(selected_contexts)
        return {
            "search_results": results,
            "source_registry": self._build_source_entries(results, source_type="search", query=query),
            "all_context": [f"搜索结果 (第{state['current_round']}轮): {context}", selected_context_text] if selected_context_text else [f"搜索结果 (第{state['current_round']}轮): {context}"],
            "selected_contexts": [chunk.to_dict() for chunk in selected_contexts],
            "search_history": state.get("search_history", []) + [query],
            "step_questions": [item for item in state.get("step_questions", []) if item != query],
            "steps_log": [step_log],
            "latest_evidence_count": high_quality_count,
            "report_ready": False,
            "current_tokens": state.get("current_tokens", 0) + self.context_manager.estimate_tokens(context + selected_context_text),
            "events": [self._event("search_completed", state, {"query": query, "results_count": len(results), "retrievers": self.retriever_manager.list_retrievers()})],
        }

    async def visit_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点4: 访问URL获取内容"""
        logger.info(f"=== 节点: 访问URL (第{state['current_round']}轮) ===")

        gather_queries = self._build_gather_queries(state)
        urls_to_visit = await self._gather_urls_for_queries(
            queries=gather_queries,
            visited_urls=state["visited_urls"],
        )
        
        if not urls_to_visit:
            logger.warning("没有URL可访问")
            return {
                "steps_log": [{
                    "step": state["current_step"],
                    "action": "visit",
                    "queries": gather_queries,
                    "urls": [],
                    "results_count": 0
                }],
                "events": [self._event("visit_skipped", state, {"reason": "no_urls", "queries": gather_queries})],
            }
        
        content_results = await self.web_reader.read_urls(
            urls_to_visit,
            max_concurrent=3
        )
        
        successful_results = [r for r in content_results if r.success]
        content_context = ""
        selected_contexts = []
        if successful_results:
            selected_contexts = self.context_manager.select_relevant_context(
                query=state.get("current_question", state.get("original_query", "")),
                search_results=[],
                content_results=successful_results,
                max_chunks=self.context_top_k,
            )
            content_context = self.context_manager.build_context_text(selected_contexts, max_chars=self.context_max_chars)
        
        step_log = {
            "step": state["current_step"],
            "action": "visit",
            "queries": gather_queries,
            "urls": urls_to_visit,
            "results_count": len(successful_results),
            "round": state["current_round"],
        }
        
        completed_rounds = state.get("completed_rounds", 0) + 1
        logger.info(
            f"访问完成，成功访问 {len(successful_results)} 个URL，累计轮次: {completed_rounds}"
        )
        
        high_quality_count = self.context_manager.count_high_quality_contexts(selected_contexts)
        report_ready = len(state.get("completed_queries", []) + gather_queries) >= state.get("max_total_queries", self.max_total_queries)
        return {
            "content_results": content_results,
            "source_registry": self._build_source_entries(successful_results, source_type="content", query=state.get("current_question", state.get("original_query", ""))),
            "all_context": [f"网页内容 (第{state['current_round']}轮): {content_context}"] if content_context else [],
            "selected_contexts": [chunk.to_dict() for chunk in selected_contexts],
            "visited_urls": state.get("visited_urls", []) + urls_to_visit,
            "completed_rounds": completed_rounds,
            "completed_queries": state.get("completed_queries", []) + gather_queries,
            "steps_log": [step_log],
            "latest_evidence_count": high_quality_count,
            "report_ready": report_ready,
            "current_tokens": state.get("current_tokens", 0) + self.context_manager.estimate_tokens(content_context),
            "events": [self._event("visit_completed", state, {"queries": gather_queries, "urls": urls_to_visit, "success_count": len(successful_results)})],
        }

    async def reflect_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点5: 反思并生成步骤问题"""
        logger.info(f"=== 节点: 反思 (第{state['current_round']}轮) ===")
        
        context_summary = self._build_planning_context(state)
        remaining_budget = state.get("token_budget", self.token_budget) - state.get("current_tokens", 0)
        completed_rounds = state.get("completed_rounds", 0)
        max_rounds = state.get("max_rounds", self.max_rounds)
        completed_queries = len(state.get("search_history", []))
        max_total_queries = state.get("max_total_queries", self.max_total_queries)
        latest_evidence_count = state.get("latest_evidence_count", 0)
        report_budget = state.get("report_context_max_chars", self.report_context_max_chars)
        selected_contexts = state.get("selected_contexts", [])
        evidence_ready = self._has_sufficient_evidence(selected_contexts, report_budget)

        if completed_queries >= max_total_queries:
            logger.info(f"累计查询数达到上限 {completed_queries}/{max_total_queries}，停止继续规划")
            return {
                "step_questions": [],
                "early_termination": True,
                "report_ready": True,
                "steps_log": [{
                    "step": state["current_step"],
                    "action": "reflect",
                    "new_questions_count": 0,
                    "early_termination": True,
                    "round": state["current_round"],
                    "remaining_budget": remaining_budget,
                }],
                "events": [self._event("planning_completed", state, {"planned_queries": [], "early_termination": True, "remaining_budget": remaining_budget, "reason": "max_total_queries_reached"})],
            }

        if completed_rounds >= max_rounds:
            logger.info(f"已完成最大轮次 {completed_rounds}/{max_rounds}，停止继续规划")
            return {
                "step_questions": [],
                "early_termination": True,
                "report_ready": True,
                "steps_log": [{
                    "step": state["current_step"],
                    "action": "reflect",
                    "new_questions_count": 0,
                    "early_termination": True,
                    "round": state["current_round"],
                    "remaining_budget": remaining_budget,
                }],
                "events": [self._event("planning_completed", state, {"planned_queries": [], "early_termination": True, "remaining_budget": remaining_budget, "reason": "max_rounds_reached"})],
            }

        new_steps = await self.agent_manager.plan_queries(
            state["original_query"],
            context_summary,
            max_queries=self.max_planned_queries,
        )

        seen_queries = {
            self._normalize_query(query)
            for query in (
                state.get("search_history", [])
                + state.get("completed_queries", [])
                + state.get("step_questions", [])
                + [state.get("current_question", "")]
            )
            if query
        }
        deduped_steps: List[str] = []
        for step in new_steps:
            normalized = self._normalize_query(step)
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            deduped_steps.append(step)

        new_steps = deduped_steps[: self.max_planned_queries]
        early_termination = (
            (len(new_steps) == 0 and self.allow_early_termination)
            or remaining_budget <= 0
            or (latest_evidence_count == 0 and evidence_ready)
        )
        
        step_log = {
            "step": state["current_step"],
            "action": "reflect",
            "new_questions_count": len(new_steps),
            "early_termination": early_termination,
            "round": state["current_round"],
            "remaining_budget": remaining_budget,
        }
        
        if early_termination:
            logger.info("LLM判断信息充足，提前结束搜索")
        else:
            logger.info(f"反思完成，生成 {len(new_steps)} 个新问题")
        
        return {
            "step_questions": state.get("step_questions", []) + new_steps,
            "early_termination": early_termination,
            "report_ready": state.get("report_ready", False) or early_termination or (evidence_ready and len(new_steps) == 0) or (latest_evidence_count == 0 and len(new_steps) == 0),
            "steps_log": [step_log],
            "events": [self._event("planning_completed", state, {"planned_queries": new_steps, "early_termination": early_termination, "remaining_budget": remaining_budget})],
        }

    async def decide_action_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点6: 决定下一步动作"""
        current_step = state.get("current_step", 0)
        logger.info(f"=== 节点: 决定动作 (步骤{current_step}) ===")
        
        steps_log = state.get("steps_log", [])
        
        max_steps = state.get("max_steps", 8)
        if current_step >= max_steps:
            logger.info(f"达到最大步骤 {max_steps}，生成报告")
            return {"next_action": "answer"}
        
        completed_rounds = state.get("completed_rounds", 0)
        completed_queries = len(state.get("search_history", []))
        max_total_queries = state.get("max_total_queries", self.max_total_queries)
        if completed_queries >= max_total_queries:
            logger.info(f"累计查询数达到上限 {completed_queries}/{max_total_queries}，生成报告")
            return {"next_action": "answer"}

        if state.get("report_ready", False):
            logger.info("已满足报告生成条件，生成报告")
            return {"next_action": "answer"}

        if state.get("early_termination", False):
            logger.info("提前终止，生成报告")
            return {"next_action": "answer"}

        if state.get("current_tokens", 0) >= state.get("token_budget", self.token_budget):
            logger.info("达到 token budget，生成报告")
            return {"next_action": "answer"}

        if not steps_log:
            logger.info("初始状态，开始搜索")
            return {"next_action": "search"}
        
        last_step = steps_log[-1]
        last_action = last_step.get("action")
        
        # 决策逻辑
        if last_action == "search":
            next_action = "visit"
        elif last_action == "visit":
            if completed_rounds >= self.max_rounds:
                logger.info(
                    f"完成 {completed_rounds} 轮访问，达到最大轮次 {self.max_rounds}，生成报告"
                )
                next_action = "answer"
            else:
                next_action = "reflect"
        elif last_action == "reflect":
            if completed_rounds >= state.get("max_rounds", self.max_rounds):
                logger.info("反思后已达到最大轮次，生成报告")
                next_action = "answer"
            else:
                # 检查是否有新的步骤问题
                step_questions = state.get("step_questions", [])
                if step_questions:
                    next_action = "search"
                else:
                    logger.info("没有新的高价值步骤问题，生成报告")
                    next_action = "answer"
        else:
            next_action = "search"
        
        logger.info(f"下一步动作: {next_action}")
        return {"next_action": next_action}

    async def increment_step_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点7: 更新步骤和轮次"""
        current_step = state.get("current_step", 0) + 1
        last_action = ""
        if state.get("steps_log"):
            last_action = state["steps_log"][-1].get("action", "")
        
        # 计算当前轮次
        if current_step <= 2:
            current_round = 1
        else:
            current_round = 1 + (current_step - 2 + 2) // 3
        
        # 更新当前问题
        step_questions = state.get("step_questions", [])
        if step_questions:
            current_question = step_questions[0]
            remaining_questions = step_questions if last_action == "reflect" else step_questions[1:]
        else:
            current_question = state.get("keywords", state.get("original_query", ""))
            remaining_questions = []
        
        logger.info(f"步骤更新: {current_step}, 轮次: {current_round}")
        
        return {
            "current_step": current_step,
            "current_round": current_round,
            "current_question": current_question,
            "step_questions": remaining_questions,
            "events": [self._event("step_incremented", state, {"current_step": current_step, "current_round": current_round, "current_question": current_question})],
        }

    async def generate_report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点8: 生成最终研究报告"""
        logger.info("=== 节点: 生成研究报告 ===")
        
        source_registry = self._dedupe_source_registry(state.get("source_registry", []))
        source_mapping = {
            item.get("url"): item.get("number")
            for item in source_registry
            if item.get("url") and item.get("number")
        }
        evidence_budget = state.get("report_context_max_chars", self.report_context_max_chars)
        evidence_query = " ".join([
            state.get("extracted_question", state.get("original_query", "")),
            state.get("keywords", ""),
            " ".join(state.get("search_history", [])),
        ]).strip()
        selected_evidence = self.context_manager.select_context_with_budget(
            query=evidence_query,
            contexts=state.get("selected_contexts", []),
            max_chars=evidence_budget,
            max_chunks=self.context_top_k,
            source_mapping=source_mapping,
        )
        structured_context = self.context_manager.build_evidence_packet(
            selected_evidence,
            source_mapping=source_mapping,
            max_chars=evidence_budget,
        )
        final_answer = await self.agent_manager.generate_research_report(
            state["extracted_question"],
            state.get("search_results", []),
            state.get("content_results", []),
            selected_contexts=[chunk.to_dict() if isinstance(chunk, ContextChunk) else chunk for chunk in selected_evidence],
            structured_context=structured_context,
            source_registry=source_registry,
        )
        
        logger.info("研究报告生成完成")
        
        return {
            "final_answer": final_answer,
            "confidence": 0.9,
            "success": True,
            "events": [self._event("report_generated", state, {"selected_contexts": len(selected_evidence), "sources": len(source_registry), "evidence_budget": evidence_budget})],
        }

    # === 辅助方法 ===

    def _build_gather_queries(self, state: Dict[str, Any]) -> List[str]:
        queries: List[str] = []
        current_question = state.get("current_question", "")
        if current_question:
            queries.append(current_question)

        for query in state.get("step_questions", []):
            if query and query not in queries and query not in state.get("completed_queries", []):
                queries.append(query)
            if len(queries) >= self.max_planned_queries:
                break

        return queries or [state.get("keywords", state.get("original_query", ""))]

    def _normalize_query(self, query: str) -> str:
        return " ".join((query or "").strip().lower().split())

    def _has_sufficient_evidence(self, contexts: List[Any], report_context_max_chars: int) -> bool:
        high_quality_count = self.context_manager.count_high_quality_contexts(contexts)
        source_urls = set()
        for item in contexts:
            source_url = item.get("source_url", "") if isinstance(item, dict) else getattr(item, "source_url", "")
            if source_url:
                source_urls.add(source_url)
        if high_quality_count >= 10 and len(source_urls) >= 8:
            return True
        return False

    def _build_source_entries(self, items: List[Any], source_type: str, query: str) -> List[dict]:
        entries: List[dict] = []
        for item in items:
            url = getattr(item, "url", "") if not isinstance(item, dict) else item.get("url", "")
            if not url:
                continue
            title = getattr(item, "title", "") if not isinstance(item, dict) else item.get("title", "")
            snippet = getattr(item, "snippet", "") if not isinstance(item, dict) else item.get("snippet", "")
            entries.append({
                "url": url,
                "title": title or url,
                "source_type": source_type,
                "query": query,
                "snippet": snippet,
            })
        return entries

    def _dedupe_source_registry(self, entries: List[dict]) -> List[dict]:
        deduped: List[dict] = []
        seen_urls = set()
        for entry in entries:
            url = entry.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            normalized = dict(entry)
            normalized["number"] = len(deduped) + 1
            deduped.append(normalized)
        return deduped

    async def _gather_urls_for_queries(self, queries: List[str], visited_urls: List[str]) -> List[str]:
        if not queries:
            return []

        search_tasks = [
            self.retriever_manager.search(query, max_results=self.max_results)
            for query in queries
        ]
        search_batches = await asyncio.gather(*search_tasks, return_exceptions=True)

        gathered_results: List[Any] = []
        for query, batch in zip(queries, search_batches):
            if isinstance(batch, Exception):
                logger.warning(f"子查询搜索失败: {query}, 错误: {batch}")
                continue
            gathered_results.extend(batch)

        return await self._select_urls_to_visit(gathered_results, visited_urls)

    async def _select_urls_to_visit(
        self, 
        search_results: List[dict], 
        visited_urls: List[str]
    ) -> List[str]:
        """智能URL选择"""
        if not search_results:
            return []
        
        results = []
        for i, r in enumerate(search_results):
            if isinstance(r, SerperSearchResult):
                results.append(r)
            elif isinstance(r, UnifiedSearchResult):
                results.append(SerperSearchResult(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                    position=r.position or i + 1
                ))
            elif isinstance(r, dict):
                results.append(SerperSearchResult(
                    title=r.get('title', ''),
                    url=r.get('url', ''),
                    snippet=r.get('snippet', ''),
                    position=r.get('position', i + 1)
                ))
        
        # 选择未访问的URL
        unvisited_results = [r for r in results if r.url not in visited_urls]
        
        selected_urls = []
        seen_domains = set()
        for result in unvisited_results:
            domain = result.url.split('/')[2] if '://' in result.url else result.url
            if domain in seen_domains and len(selected_urls) < self.max_urls_per_step - 1:
                continue
            seen_domains.add(domain)
            selected_urls.append(result.url)
            if len(selected_urls) >= self.max_urls_per_step:
                break
        
        return selected_urls

    def _format_search_results(self, results: List[Any]) -> str:
        if not results:
            return "没有找到相关搜索结果。"
        context_parts = ["搜索结果:"]
        for i, result in enumerate(results, 1):
            title = getattr(result, 'title', '') if not isinstance(result, dict) else result.get('title', '')
            url = getattr(result, 'url', '') if not isinstance(result, dict) else result.get('url', '')
            snippet = getattr(result, 'snippet', '') if not isinstance(result, dict) else result.get('snippet', '')
            source = getattr(result, 'source', '') if not isinstance(result, dict) else result.get('source', '')
            context_parts.append(f"\n{i}. {title}")
            context_parts.append(f"   URL: {url}")
            context_parts.append(f"   来源: {source}")
            context_parts.append(f"   摘要: {snippet}")
        return "\n".join(context_parts)

    def _build_planning_context(self, state: Dict[str, Any]) -> str:
        selected_contexts = state.get('selected_contexts', [])
        if selected_contexts:
            top_contexts = selected_contexts[-self.context_top_k:]
            lines = []
            for index, item in enumerate(top_contexts, 1):
                if isinstance(item, dict):
                    lines.append(f"[{index}] {item.get('source_title', '未知标题')} | {item.get('source_url', '未知URL')} | score={item.get('score', 0):.3f}\n{item.get('text', '')}")
                else:
                    lines.append(str(item))
            return "\n\n".join(lines)
        return "\n\n".join(state.get('all_context', [])[-3:])

    def _build_structured_report_context(self, selected_contexts: List[Any]) -> str:
        lines = []
        for index, item in enumerate(selected_contexts[: self.context_top_k], 1):
            if isinstance(item, dict):
                lines.append(
                    f"[{index}] {item.get('source_title', '未知标题')}\n"
                    f"URL: {item.get('source_url', '未知URL')}\n"
                    f"相关度: {item.get('score', 0):.3f}\n"
                    f"内容: {item.get('text', '')}"
                )
        return "\n\n".join(lines)

    def _event(self, event_type: str, state: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": event_type,
            "step": state.get('current_step', 0),
            "round": state.get('current_round', 0),
            "payload": payload,
        }
