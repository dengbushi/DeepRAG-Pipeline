#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serper.dev搜索引擎 - Google搜索API封装
"""

import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SerperSearchResult:
    """Serper搜索结果数据类"""
    title: str
    url: str
    snippet: str
    position: int
    date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "date": self.date
        }

class SerperSearchEngine:
    """Serper.dev搜索引擎"""
    
    def __init__(self, api_key: str, max_results: int = 10):
        """
        初始化Serper搜索引擎
        
        Args:
            api_key: Serper.dev API密钥
            max_results: 最大搜索结果数量
        """
        self.api_key = api_key
        self.max_results = max_results
        self.base_url = "https://google.serper.dev/search"
        
        # 请求头
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        logger.info(f"Serper搜索引擎初始化完成，最大结果数: {max_results}")
    
    async def search(self, query: str, max_results: Optional[int] = None) -> List[SerperSearchResult]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量，如果为None则使用默认值
            
        Returns:
            搜索结果列表
        """
        if not query.strip():
            logger.warning("搜索查询为空")
            return []
        
        # 使用指定的max_results或默认值
        num_results = max_results if max_results is not None else self.max_results
        
        # 构建请求数据
        payload = {
            "q": query,
            "num": min(num_results, 100),  # Serper API限制最多100个结果
            "gl": "us",  # 地理位置
            "hl": "en"   # 语言
        }
        
        try:
            logger.info(f"开始Serper搜索: '{query}', 最大结果数: {num_results}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_search_results(data)
                        logger.info(f"Serper搜索成功，获得 {len(results)} 个结果")
                        return results
                    
                    elif response.status == 401:
                        logger.error("Serper API认证失败，请检查API密钥")
                        return []
                    
                    elif response.status == 429:
                        logger.error("Serper API请求频率限制，请稍后重试")
                        return []
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"Serper API请求失败，状态码: {response.status}, 错误: {error_text}")
                        return []
                        
        except asyncio.TimeoutError:
            logger.error("Serper搜索请求超时")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"Serper搜索网络错误: {e}")
            return []
        except Exception as e:
            logger.error(f"Serper搜索未知错误: {e}")
            return []
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[SerperSearchResult]:
        """
        解析搜索结果
        
        Args:
            data: Serper API返回的原始数据
            
        Returns:
            解析后的搜索结果列表
        """
        results = []
        
        # 解析有机搜索结果
        organic_results = data.get("organic", [])
        
        for i, item in enumerate(organic_results):
            try:
                result = SerperSearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    position=item.get("position", i + 1),
                    date=item.get("date")
                )
                
                # 验证必要字段
                if result.title and result.url:
                    results.append(result)
                else:
                    logger.warning(f"跳过无效搜索结果: {item}")
                    
            except Exception as e:
                logger.warning(f"解析搜索结果项失败: {e}, 数据: {item}")
                continue
        
        return results
    
    def format_results_for_context(self, results: List[SerperSearchResult]) -> str:
        """
        格式化搜索结果为上下文字符串
        
        Args:
            results: 搜索结果列表
            
        Returns:
            格式化的上下文字符串
        """
        if not results:
            return "没有找到相关搜索结果。"
        
        context_parts = ["搜索结果:"]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n{i}. {result.title}")
            context_parts.append(f"   URL: {result.url}")
            context_parts.append(f"   摘要: {result.snippet}")
            if result.date:
                context_parts.append(f"   日期: {result.date}")
        
        return "\n".join(context_parts)
    
    async def search_scholar(self, query: str, max_results: Optional[int] = None) -> List[SerperSearchResult]:
        """
        执行Google Scholar学术搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            学术搜索结果列表
        """
        # 使用Google Scholar搜索端点
        scholar_url = "https://google.serper.dev/scholar"
        
        num_results = max_results if max_results is not None else self.max_results
        
        payload = {
            "q": query,
            "num": min(num_results, 20)  # Scholar API限制
        }
        
        try:
            logger.info(f"开始Google Scholar搜索: '{query}'")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    scholar_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_scholar_results(data)
                        logger.info(f"Scholar搜索成功，获得 {len(results)} 个结果")
                        return results
                    else:
                        logger.error(f"Scholar搜索失败，状态码: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Scholar搜索错误: {e}")
            return []
    
    def _parse_scholar_results(self, data: Dict[str, Any]) -> List[SerperSearchResult]:
        """解析Google Scholar搜索结果"""
        results = []
        
        organic_results = data.get("organic", [])
        
        for i, item in enumerate(organic_results):
            try:
                result = SerperSearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    position=i + 1,
                    date=item.get("publicationInfo", {}).get("summary", "")
                )
                
                if result.title and result.url:
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"解析Scholar结果失败: {e}")
                continue
        
        return results
