#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题提取代理
"""

import logging
from .base import BaseAgent
from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)

class QuestionExtractorAgent(BaseAgent):
    """问题提取代理"""
    
    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(
            llm=llm,
            name="QuestionExtractor",
            role_description=(
                "你是一个专业的问题分析专家。你擅长从复杂的描述中提取出核心问题。"
                "你的任务是理解用户的真实意图，去除无关信息，提取出最核心的问题。"
                "使用中文时只会使用简体中文来回答问题。"
            ),
            task_description=(
                "请从以下文本中提取出最核心的问题，去除无关的描述和背景信息。"
                "要求：\n"
                "1. 只返回提取出的核心问题，不需要额外说明\n"
                "2. 保持问题的完整性和准确性\n"
                "3. 如果有多个问题，请提取最主要的一个\n"
                "4. 确保问题表述清晰、具体"
            ),
            **kwargs
        )
    
    async def process(self, input_data: str, **kwargs) -> str:
        """提取核心问题"""
        logger.info(f"开始提取问题: {input_data[:50]}...")
        
        messages = self.create_messages(input_data)
        
        try:
            extracted_question = await self.invoke_llm(messages, **kwargs)
            
            # 简单验证提取的问题
            if not extracted_question or len(extracted_question.strip()) < 3:
                logger.warning("提取的问题过短，使用原始输入")
                return input_data
            
            logger.info(f"问题提取完成: {extracted_question}")
            return extracted_question
            
        except Exception as e:
            logger.error(f"问题提取失败: {e}")
            # 降级处理：返回原始输入
            return input_data
    
    def validate_question(self, question: str) -> bool:
        """验证问题质量"""
        if not question or len(question.strip()) < 3:
            return False
        
        # 检查是否包含问号或疑问词
        question_indicators = ['？', '?', '什么', '如何', '为什么', '哪里', '谁', '何时']
        return any(indicator in question for indicator in question_indicators)
