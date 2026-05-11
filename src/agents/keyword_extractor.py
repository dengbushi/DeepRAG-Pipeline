#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键词提取代理
"""

import logging
import re
from .base import BaseAgent
from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)

class KeywordExtractorAgent(BaseAgent):
    """关键词提取代理"""
    
    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(
            llm=llm,
            name="KeywordExtractor",
            role_description=(
                "你是一个专业的关键词提取专家。你擅长从问题中提取出最适合搜索的关键词。"
                "你了解搜索引擎的工作原理，能够选择最有效的搜索词汇。"
                "使用中文时只会使用简体中文来回答问题。"
            ),
            task_description=(
                "请从以下问题中提取出最适合进行网络搜索的关键词。"
                "要求：\n"
                "1. 只返回关键词，用空格分隔\n"
                "2. 选择最重要和最具搜索价值的词汇\n"
                "3. 避免停用词（如：的、是、在、等）\n"
                "4. 关键词数量控制在2-10个\n"
                "5. 优先选择名词、专有名词和核心概念"
            ),
            **kwargs
        )
    
    async def process(self, input_data: str, **kwargs) -> str:
        """提取搜索关键词"""
        logger.info(f"开始提取关键词: {input_data}")
        
        messages = self.create_messages(input_data)
        
        keywords = await self.invoke_llm(messages, **kwargs)
        
        # 清理和验证关键词
        cleaned_keywords = self.clean_keywords(keywords)
        
        if not cleaned_keywords:
            raise ValueError("未提取到有效关键词")
        
        logger.info(f"关键词提取完成: {cleaned_keywords}")
        return cleaned_keywords
    
    def clean_keywords(self, keywords: str) -> str:
        """清理关键词"""
        if not keywords:
            return ""
        
        # 移除标点符号和多余空格
        keywords = re.sub(r'[，。！？、；：""''（）【】]', ' ', keywords)
        keywords = re.sub(r'\s+', ' ', keywords).strip()
        
        # 分割关键词并过滤
        keyword_list = [kw.strip() for kw in keywords.split() if kw.strip()]
        
        # 过滤停用词
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '然而', '因为', '所以', '如果', '那么'}
        filtered_keywords = [kw for kw in keyword_list if kw not in stop_words and len(kw) > 1]
        
        # 限制关键词数量
        if len(filtered_keywords) > 10:
            filtered_keywords = filtered_keywords[:10]
        
        return ' '.join(filtered_keywords)
    
