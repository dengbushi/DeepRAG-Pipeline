#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研究报告生成代理
"""

import logging
from typing import List, Dict, Any, Optional
from .base import BaseAgent
from ..llm.base import BaseLLM
from ..search.serper_search import SerperSearchResult
from ..search.jina_reader import JinaReaderResult

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
        
        return await self.generate_research_report(question, search_results, content_results)
    
    async def generate_research_report(
        self, 
        question: str, 
        search_results: List[SerperSearchResult], 
        content_results: List[JinaReaderResult]
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
        try:
            # 构建研究上下文
            context = self._build_research_context(search_results, content_results)
            
            if not context.strip():
                return f"抱歉，没有找到关于 '{question}' 的相关信息来生成研究报告。"
            
            # 限制上下文长度（避免触发内容审核）
            max_context_length = 10000
            if len(context) > max_context_length:
                logger.warning(f"上下文过长（{len(context)}字符），截取到{max_context_length}字符")
                context = context[:max_context_length] + "\n\n...(内容已截断)"
            
            # 使用BaseAgent的标准方法调用LLM
            logger.info(f"开始生成研究报告，问题: {question}")
            logger.info(f"上下文长度: {len(context)} 字符")
            
            # 创建消息并调用LLM
            messages = self.create_messages(question, context)
            report = await self.invoke_llm(messages)
            
            logger.info(f"研究报告生成完成，长度: {len(report)} 字符")
            return report
            
        except Exception as e:
            logger.error(f"研究报告生成失败: {e}")
            return f"研究报告生成过程中出现错误: {str(e)}"
    
    def _build_research_context(
        self, 
        search_results: List[SerperSearchResult], 
        content_results: List[JinaReaderResult]
    ) -> str:
        """构建研究上下文"""
        context_parts = []
        source_mapping = {}  # URL到编号的映射
        reference_list = []  # 引用列表
        ref_counter = 1
        
        # 添加搜索结果摘要
        if search_results:
            context_parts.append("## 搜索结果摘要")
            for i, result in enumerate(search_results, 1):
                try:
                    # 处理搜索结果对象
                    if hasattr(result, 'title'):
                        # 为每个来源分配引用编号
                        if result.url not in source_mapping:
                            source_mapping[result.url] = ref_counter
                            reference_list.append({
                                'number': ref_counter,
                                'title': result.title,
                                'url': result.url
                            })
                            ref_counter += 1
                        
                        ref_num = source_mapping[result.url]
                        context_parts.append(f"**来源[^{ref_num}]: {result.title}**")
                        context_parts.append(f"   URL: {result.url}")
                        if hasattr(result, 'snippet') and result.snippet:
                            context_parts.append(f"   摘要: {result.snippet}")
                    else:
                        # 如果是字符串或其他格式，直接使用
                        context_parts.append(f"{i}. {str(result)}")
                    context_parts.append("")
                except Exception as e:
                    logger.warning(f"处理搜索结果 {i} 时出错: {e}")
                    context_parts.append(f"{i}. [处理错误的搜索结果]")
                    context_parts.append("")
        
        # 添加详细网页内容
        if content_results:
            successful_content = [r for r in content_results if hasattr(r, 'success') and r.success and hasattr(r, 'content') and r.content]
            if successful_content:
                context_parts.append("## 详细网页内容")
                for i, result in enumerate(successful_content, 1):
                    try:
                        title = getattr(result, 'title', '未知标题')
                        url = getattr(result, 'url', '未知URL')
                        content = getattr(result, 'content', '')
                        
                        # 为每个来源分配引用编号
                        if url not in source_mapping:
                            source_mapping[url] = ref_counter
                            reference_list.append({
                                'number': ref_counter,
                                'title': title,
                                'url': url
                            })
                            ref_counter += 1
                        
                        ref_num = source_mapping[url]
                        context_parts.append(f"### 来源[^{ref_num}]: {title}")
                        context_parts.append(f"URL: {url}")
                        
                        # 截取内容前2000字符避免过长
                        if content:
                            content_preview = content[:2000]
                            if len(content) > 2000:
                                content_preview += "..."
                            context_parts.append(f"内容: {content_preview}")
                        context_parts.append("")
                    except Exception as e:
                        logger.warning(f"处理内容结果 {i} 时出错: {e}")
                        context_parts.append(f"### 来源 {i}: [处理错误的内容结果]")
                        context_parts.append("")
        
        # 添加引用指导和参考列表
        if reference_list:
            context_parts.append("## 引用编号对应表")
            for ref in reference_list:
                context_parts.append(f"[^{ref['number']}]: {ref['title']} - {ref['url']}")
            context_parts.append("")
            
            context_parts.append("## 重要引用指导")
            context_parts.append("1. 在报告中引用信息时，请在相关内容后添加引用编号，如：某个观点[^1]")
            context_parts.append("2. 在报告末尾添加'参考来源'部分，按编号列出所有引用")
            context_parts.append("3. 确保每个重要信息都有对应的引用，让读者知道信息来源")
            context_parts.append("4. 引用格式：[^编号]: 网站标题 - URL")
            context_parts.append("")
            
            # 提供引用示例
            context_parts.append("## 引用格式示例")
            context_parts.append("正文中：ChatGPT是基于Transformer架构的大型语言模型[^1]。")
            context_parts.append("参考来源部分：[^1]: OpenAI官网 - https://openai.com/chatgpt")
        
        return "\n".join(context_parts)
    
    
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
