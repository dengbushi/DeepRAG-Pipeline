#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM工厂类
"""

import logging
from typing import Dict, Type
from .base import BaseLLM
from .deepseek import DeepSeekLLM
from ..config import LLMConfig

logger = logging.getLogger(__name__)

class LLMFactory:
    """LLM工厂类"""
    
    _providers: Dict[str, Type[BaseLLM]] = {
        "deepseek": DeepSeekLLM,
    }
    
    @classmethod
    def create_llm(cls, config: LLMConfig) -> BaseLLM:
        """创建LLM实例"""
        provider = config.provider.lower()
        
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"不支持的LLM提供商: {provider}. 可用的: {available}")
        
        llm_class = cls._providers[provider]
        
        try:
            # 根据不同提供商传递不同参数
            if provider == "deepseek":
                return llm_class(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    model_name=config.model_name,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    timeout=config.timeout
                )
            else:
                return llm_class(**config.__dict__)
                
        except Exception as e:
            logger.error(f"创建LLM实例失败: {e}")
            raise
    
    @classmethod
    def register_provider(cls, name: str, llm_class: Type[BaseLLM]):
        """注册新的LLM提供商"""
        cls._providers[name.lower()] = llm_class
        logger.info(f"已注册LLM提供商: {name}")
    
    @classmethod
    def get_available_providers(cls) -> list:
        """获取可用的提供商列表"""
        return list(cls._providers.keys())
