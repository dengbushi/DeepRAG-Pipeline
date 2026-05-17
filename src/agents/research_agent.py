#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研究报告生成代理
"""

import re
import logging
from typing import List, Dict, Any, Optional
from .base import BaseAgent
from ..llm.base import BaseLLM
from ..search.serper_search import SerperSearchResult
from ..search.web_reader import WebReaderResult

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """研究报告生成代理"""
    
    def __init__(self, llm: BaseLLM):
        super().__init__(
            llm=llm,
            name="research_agent",
            role_description="专业的研究分析师，负责整合多源信息并生成结构化研究报告",
            task_description="""基于搜索结果和网页内容，生成客观、准确、结构化的研究报告。

要求：
1. 使用清晰的标题结构（# ## ###）
2. 内容要有逻辑层次，包含概述、主要发现、详细分析、结论等部分
3. **重要**: 使用内联引用格式，在每个相关内容后面紧跟引用，格式如：某个观点或信息 [^1]
4. 在报告末尾添加"参考来源"部分，按编号列出所有引用的链接，格式如：[^1]: 网站标题 - URL
5. 使用Markdown格式，包含适当的加粗、列表等格式
6. 内容要客观、准确，基于提供的信息进行分析
7. 确保每个重要信息都有对应的引用编号，让读者能清楚知道信息来源"""
        )
        self.description = "基于搜索结果和网页内容生成结构化研究报告"
    
    async def process(self, question: str, **kwargs) -> str:
        """
        生成研究报告
        
        Args:
            question: 研究问题
            **kwargs: 包含search_results和content_results
            
        Returns:
            结构化研究报告
        """
        search_results = kwargs.get('search_results', [])
        content_results = kwargs.get('content_results', [])
        selected_contexts = kwargs.get('selected_contexts', [])
        structured_context = kwargs.get('structured_context', '')
        source_registry = kwargs.get('source_registry', [])
        
        return await self.generate_research_report(
            question,
            search_results,
            content_results,
            selected_contexts=selected_contexts,
            structured_context=structured_context,
            source_registry=source_registry,
        )
    
    async def generate_research_report(
        self, 
        question: str, 
        search_results: List[SerperSearchResult], 
        content_results: List[WebReaderResult],
        selected_contexts: Optional[List[Dict[str, Any]]] = None,
        structured_context: str = "",
        source_registry: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        生成结构化研究报告
        
        Args:
            question: 研究问题
            search_results: 搜索结果列表
            content_results: 网页内容结果列表
            
        Returns:
            结构化研究报告
        """
        max_context_length = 12000
        references = self._collect_references(search_results, content_results, source_registry or [])
        context = self._build_research_context(
            search_results,
            content_results,
            selected_contexts or [],
            structured_context,
            references,
            max_chars=max_context_length,
        )
        
        if not context.strip():
            raise ValueError(f"没有找到关于 '{question}' 的相关信息来生成研究报告")

        logger.info(f"开始生成研究报告，问题: {question}")
        logger.info(f"上下文长度: {len(context)} 字符")

        outline = await self._generate_outline(question, context)
        body = await self._generate_body(question, outline, context)
        body = self._strip_reference_section(body)
        used_references = self._select_used_references(body, references)
        body, used_references = self._renumber_references(body, used_references)
        references_text = self._build_reference_section(used_references)
        report = f"{body}\n\n{references_text}".strip()
        
        logger.info(f"研究报告生成完成，长度: {len(report)} 字符")
        return report
    
    def _build_research_context(
        self, 
        search_results: List[SerperSearchResult], 
        content_results: List[WebReaderResult],
        selected_contexts: List[Dict[str, Any]],
        structured_context: str,
        references: List[Dict[str, Any]],
        max_chars: int = 12000,
    ) -> str:
        """构建研究上下文"""
        context_parts: List[str] = []
        source_mapping = {ref['url']: ref['number'] for ref in references if ref.get('url')}
        current_length = 0

        def append_block(block: str) -> bool:
            nonlocal current_length
            text = (block or '').strip()
            if not text:
                return True

            candidate = text if not context_parts else f"\n\n{text}"
            remaining = max_chars - current_length
            if remaining <= 0:
                return False

            if len(candidate) <= remaining:
                context_parts.append(text)
                current_length += len(candidate)
                return True

            minimum_useful = min(240, max(remaining - 20, 0))
            if minimum_useful < 120:
                return False

            trimmed = candidate[:minimum_useful].rstrip()
            if not trimmed:
                return False
            if context_parts and trimmed.startswith("\n\n"):
                trimmed = trimmed[2:]
            if not trimmed:
                return False
            context_parts.append(trimmed + "\n...(内容已压缩)")
            current_length = max_chars
            return False

        compact_references = references[: min(len(references), 30)]
        if references and len(references) > len(compact_references):
            logger.info(f"来源较多，压缩来源索引展示: {len(compact_references)}/{len(references)}")
        
        if structured_context:
            if not append_block("## 精选证据\n" + structured_context):
                return "\n\n".join(context_parts)

        if compact_references:
            reference_lines = ["## 可用引用来源"]
            for ref in compact_references:
                title = self._truncate_text(ref['title'], 80)
                reference_lines.append(f"[^{ref['number']}] {title}")
            if len(references) > len(compact_references):
                reference_lines.append(f"... 其余 {len(references) - len(compact_references)} 个来源可继续使用相同编号规则引用")
            if not append_block("\n".join(reference_lines)):
                return "\n\n".join(context_parts)

        include_context_fragments = not structured_context or len(structured_context) < max_chars // 3
        if include_context_fragments and selected_contexts:
            fragment_lines = ["## 关键上下文片段"]
            for i, item in enumerate(selected_contexts[:5], 1):
                source_url = item.get('source_url', '未知URL')
                ref_num = source_mapping.get(source_url)
                ref_suffix = f" [^{ref_num}]" if ref_num else ""
                fragment_lines.append(f"[{i}] {self._truncate_text(item.get('source_title', '未知标题'), 80)}{ref_suffix} | score={item.get('score', 0):.3f}")
                fragment_lines.append(self._truncate_text(item.get('text', ''), 500))
            if not append_block("\n".join(fragment_lines)):
                return "\n\n".join(context_parts)

        return "\n\n".join(context_parts)
    
    async def _generate_outline(self, question: str, context: str) -> str:
        prompt = (
            f"研究问题：{question}\n\n"
            f"证据上下文：\n{context[:8000]}\n\n"
            "请生成一个结构化研究报告大纲，使用 Markdown 标题，包含：概述、关键发现、详细分析、结论。不要输出‘参考来源’章节标题。"
        )
        messages = self.create_messages(prompt)
        return await self.invoke_llm(messages, temperature=0.2)

    async def _generate_body(self, question: str, outline: str, context: str) -> str:
        prompt = (
            f"研究问题：{question}\n\n"
            f"报告大纲：\n{outline}\n\n"
            f"证据上下文：\n{context[:10000]}\n\n"
            "请根据大纲生成完整研究报告正文，要求使用 Markdown。"
            "正文必须包含‘概述’‘关键发现’‘详细分析’‘结论’几个部分，并让‘详细分析’成为篇幅最长的部分。"
            "请尽量综合多个来源展开分析、比较概念差异、说明发展脉络与应用影响，而不是只做简短摘要。"
            "关键信息附带引用标记 [^n]，引用编号只能使用上下文中给出的来源编号。"
            "不要输出‘参考来源’或参考文献列表，这部分将由系统在正文后统一追加。"
            "避免编造，若证据不足则明确说明局限。"
        )
        messages = self.create_messages(prompt)
        return await self.invoke_llm(messages, temperature=0.2)

    def _collect_references(self, search_results: List[SerperSearchResult], content_results: List[WebReaderResult], source_registry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        references = []
        seen = set()

        for item in source_registry:
            url = item.get('url', '')
            title = item.get('title', '')
            if url and url not in seen:
                seen.add(url)
                references.append({'title': title or url, 'url': url})

        for result in search_results:
            title = getattr(result, 'title', '')
            url = getattr(result, 'url', '')
            if url and url not in seen:
                seen.add(url)
                references.append({'title': title or url, 'url': url})

        for result in content_results:
            title = getattr(result, 'title', '')
            url = getattr(result, 'url', '')
            success = getattr(result, 'success', False)
            if success and url and url not in seen:
                seen.add(url)
                references.append({'title': title or url, 'url': url})

        for index, ref in enumerate(references, 1):
            ref['number'] = index

        return references

    def _build_reference_section(self, references: List[Dict[str, Any]]) -> str:
        if not references:
            return "## 参考来源\n\n暂无可用来源。"

        lines = ["## 参考来源", ""]
        for ref in references:
            lines.append(f"[^{ref['number']}]: {ref['title']} - {ref['url']}")
        return "\n".join(lines)

    def _strip_reference_section(self, body: str) -> str:
        markers = ["\n## 参考来源", "\n# 参考来源", "\n参考来源"]
        cleaned = body.strip()
        for marker in markers:
            position = cleaned.find(marker)
            if position != -1:
                cleaned = cleaned[:position].rstrip()
        return cleaned

    def _truncate_text(self, text: str, limit: int) -> str:
        normalized = " ".join((text or '').split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)].rstrip() + "..."

    def _select_used_references(self, body: str, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        used_numbers = self._extract_reference_numbers(body)
        if not used_numbers:
            return references[: min(8, len(references))]
        return [ref for ref in references if ref.get('number') in used_numbers]

    def _renumber_references(self, body: str, references: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        number_mapping = {
            ref.get('number'): index
            for index, ref in enumerate(references, 1)
            if ref.get('number')
        }

        def replace_reference(match):
            old_number = int(match.group(1))
            new_number = number_mapping.get(old_number)
            return f"[^{new_number}]" if new_number else match.group(0)

        renumbered_body = re.sub(r"\[\^(\d+)\]", replace_reference, body or "")
        renumbered_references = []
        for index, ref in enumerate(references, 1):
            renumbered = dict(ref)
            renumbered['number'] = index
            renumbered_references.append(renumbered)
        return renumbered_body, renumbered_references

    def _extract_reference_numbers(self, body: str) -> List[int]:
        found = []
        seen = set()
        for match in re.finditer(r"\[\^(\d+)\]", body or ""):
            number = int(match.group(1))
            if number in seen:
                continue
            seen.add(number)
            found.append(number)
        return found
    
    def get_info(self) -> Dict[str, Any]:
        """获取代理信息"""
        return {
            "name": self.name,
            "description": self.description,
            "type": "research_agent",
            "capabilities": [
                "研究报告生成",
                "信息整合分析", 
                "结构化输出",
                "多源信息综合"
            ]
        }
