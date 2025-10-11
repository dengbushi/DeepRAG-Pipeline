#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问答代理
"""

import logging
from typing import Optional
from .base import BaseAgent
from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)

class QAAgent(BaseAgent):
    """问答代理"""
    
    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(
            llm=llm,
            name="QAAgent",
            role_description=(
                "你是一个专业的问答助手。你会根据提供的信息来回答问题，"
                "如果没有足够的信息，你会诚实地说不知道。"
                "你的回答准确、有用、简洁明了。"
                "使用中文时只会使用简体中文来回答问题。"
            ),
            task_description=(
                "请根据以下信息回答问题。"
                "要求：\n"
                "1. 基于提供的信息进行回答\n"
                "2. 如果信息不足，请明确说明需要更多信息\n"
                "3. 回答要准确、客观、有条理\n"
                "4. 避免编造不存在的信息\n"
                "5. 如果有多个可能的答案，请都列出来"
            ),
            **kwargs
        )
    
    async def process(self, input_data: str, context: Optional[str] = None, **kwargs) -> str:
        """回答问题"""
        logger.info(f"开始回答问题: {input_data[:50]}...")
        
        messages = self.create_messages(input_data, context)
        
        try:
            answer = await self.invoke_llm(messages, **kwargs)
            
            # 验证答案质量
            if self.validate_answer(answer):
                logger.info("问题回答完成")
                return answer
            else:
                logger.warning("答案质量不佳，尝试重新生成")
                # 可以在这里实现重试逻辑
                return answer
                
        except Exception as e:
            logger.error(f"问题回答失败: {e}")
            return "抱歉，我无法回答这个问题。请稍后重试或重新表述您的问题。"
    
    def validate_answer(self, answer: str) -> bool:
        """验证答案质量"""
        if not answer or len(answer.strip()) < 5:
            return False
        
        # 检查是否包含常见的无效回答
        invalid_patterns = [
            "我不知道",
            "无法回答",
            "抱歉",
            "sorry",
            "I don't know"
        ]
        
        answer_lower = answer.lower()
        if any(pattern in answer_lower for pattern in invalid_patterns):
            # 如果包含这些词但答案较长，可能是有效的解释性回答
            return len(answer) > 50
        
        return True
    
    async def answer_with_confidence(self, question: str, context: Optional[str] = None) -> dict:
        """带置信度的回答"""
        answer = await self.process(question, context)
        
        # 简单的置信度评估
        confidence = self.estimate_confidence(answer, context)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "has_context": context is not None and len(context.strip()) > 0
        }
    
    def estimate_confidence(self, answer: str, context: Optional[str] = None) -> float:
        """估算回答置信度"""
        confidence = 0.5  # 基础置信度
        
        # 如果有上下文信息，增加置信度
        if context and len(context.strip()) > 100:
            confidence += 0.3
        
        # 根据答案长度调整
        if len(answer) > 100:
            confidence += 0.1
        elif len(answer) < 20:
            confidence -= 0.2
        
        # 检查答案中的不确定性表达
        uncertainty_words = ["可能", "也许", "大概", "似乎", "可能是"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in answer)
        confidence -= uncertainty_count * 0.1
        
        return max(0.0, min(1.0, confidence))
