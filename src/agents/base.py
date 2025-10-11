#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代理基础抽象类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from ..llm.base import BaseLLM, Message

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """代理基础抽象类"""
    
    def __init__(
        self, 
        llm: BaseLLM, 
        name: str,
        role_description: str,
        task_description: str,
        **kwargs
    ):
        self.llm = llm
        self.name = name
        self.role_description = role_description
        self.task_description = task_description
        self.config = kwargs
        
        logger.info(f"初始化代理: {self.name}")
    
    @abstractmethod
    async def process(self, input_data: str, **kwargs) -> str:
        """处理输入数据"""
        pass
    
    def create_messages(self, user_input: str, context: Optional[str] = None) -> list:
        """创建消息列表"""
        messages = [
            Message(role="system", content=self.role_description)
        ]
        
        # 构建用户消息
        user_content = f"{self.task_description}\n\n"
        
        if context:
            user_content += f"上下文信息：\n{context}\n\n"
        
        user_content += f"输入：{user_input}"
        
        messages.append(Message(role="user", content=user_content))
        
        return messages
    
    async def invoke_llm(self, messages: list, **kwargs) -> str:
        """调用LLM"""
        try:
            response = await self.llm.generate(messages, **kwargs)
            return response.content.strip()
        except Exception as e:
            logger.error(f"代理 {self.name} LLM调用失败: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """获取代理信息"""
        return {
            "name": self.name,
            "role_description": self.role_description,
            "task_description": self.task_description,
            "config": self.config
        }
