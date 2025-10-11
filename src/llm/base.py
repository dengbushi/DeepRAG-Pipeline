#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM基础抽象类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Message:
    """消息类"""
    role: str  # system, user, assistant
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

@dataclass
class LLMResponse:
    """LLM响应类"""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None

class BaseLLM(ABC):
    """LLM基础抽象类"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    async def generate(
        self, 
        messages: List[Message], 
        **kwargs
    ) -> LLMResponse:
        """生成响应"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置"""
        pass
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """格式化消息"""
        return [msg.to_dict() for msg in messages]
