#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek LLM实现
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
from .base import BaseLLM, Message, LLMResponse

logger = logging.getLogger(__name__)

class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM实现"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model_name: str = "deepseek-chat",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        if not self.validate_config():
            raise ValueError("DeepSeek配置无效")
    
    def validate_config(self) -> bool:
        """验证配置"""
        if not self.api_key:
            logger.error("DeepSeek API密钥未设置")
            return False
        
        if not self.base_url:
            logger.error("DeepSeek API地址未设置")
            return False
        
        return True
    
    async def generate(
        self, 
        messages: List[Message], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """生成响应"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 格式化消息
        formatted_messages = self.format_messages(messages)
        
        # 检查消息内容长度
        total_content_length = sum(len(msg.get('content', '')) for msg in formatted_messages)
        logger.debug(f"请求消息总长度: {total_content_length} 字符")
        
        data = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stream": False
        }
        
        url = f"{self.base_url}/chat/completions"
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        choice = result["choices"][0]
                        content = choice["message"]["content"]
                        
                        return LLMResponse(
                            content=content,
                            usage=result.get("usage"),
                            model=result.get("model"),
                            finish_reason=choice.get("finish_reason")
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API错误 {response.status}: {error_text}")
                        logger.error(f"请求数据: model={self.model_name}, max_tokens={max_tokens or self.max_tokens}, messages_count={len(formatted_messages)}")
                        logger.error(f"消息内容长度: {total_content_length} 字符")
                        
                        # 如果是400错误，记录更多细节
                        if response.status == 400:
                            logger.error("400错误可能原因:")
                            logger.error("1. 请求体格式错误")
                            logger.error("2. max_tokens超出范围（应在1-8192之间）")
                            logger.error("3. 消息内容包含不支持的字符")
                            logger.error("4. 消息格式不符合要求")
                        
                        raise Exception(f"API请求失败: {response.status}")
                        
        except asyncio.TimeoutError:
            logger.error("DeepSeek API请求超时")
            raise Exception("API请求超时")
        except Exception as e:
            logger.error(f"DeepSeek API请求失败: {e}")
            raise Exception(f"API请求失败: {str(e)}")
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            test_messages = [
                Message(role="system", content="你是一个有用的助手。"),
                Message(role="user", content="请说'连接测试成功'")
            ]
            
            response = await self.generate(test_messages, max_tokens=50)
            logger.info("DeepSeek连接测试成功")
            return True
            
        except Exception as e:
            logger.error(f"DeepSeek连接测试失败: {e}")
            return False
