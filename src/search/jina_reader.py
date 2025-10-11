#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jina Reader API - 网页内容抓取和处理
"""

import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

@dataclass
class JinaReaderResult:
    """Jina Reader抓取结果数据类"""
    url: str
    title: str
    content: str
    links: List[str]
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "links": self.links,
            "success": self.success,
            "error": self.error
        }

class JinaReader:
    """Jina Reader API客户端"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Jina Reader 
        
        Args:
            api_key: Jina API密钥（可选，免费版本不需要）
        """
        self.api_key = api_key
        self.base_url = "https://r.jina.ai/"
        
        # 配置参数
        self.max_retries = 3
        self.timeout = 30  # 
        self.max_content_length = 150000  # 
        
        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 访问过的URL集合，用于去重
        self.visited_urls: Set[str] = set()
        
        logger.info("Jina Reader初始化完成")
    
    async def read_url(self, url: str, include_links: bool = True) -> JinaReaderResult:
        """
        读取单个URL的内容
        
        Args:
            url: 要读取的URL
            include_links: 是否包含页面中的链接
            
        Returns:
            抓取结果
        """
        if not url or not url.startswith(('http://', 'https://')):
            logger.warning(f"无效的URL: {url}")
            return JinaReaderResult(
                url=url,
                title="",
                content="",
                links=[],
                success=False,
                error="无效的URL"
            )
        
        # 检查是否已访问过
        if url in self.visited_urls:
            logger.info(f"URL已访问过，跳过: {url}")
            return JinaReaderResult(
                url=url,
                title="",
                content="",
                links=[],
                success=False,
                error="URL已访问过"
            )
        
        # URL构建
        jina_url = f"https://r.jina.ai/{url}"
        
        # 添加参数
        headers = self.headers.copy()
        if include_links:
            headers["X-With-Links-Summary"] = "all"
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                logger.info(f"开始抓取URL (尝试 {attempt + 1}/{self.max_retries}): {url}")
                
                # 优化的连接器配置
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=10,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    ssl=False  # 暂时禁用SSL验证以避免SSL超时问题
                )
                
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(
                        jina_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            content = await response.text()
                            
                            # 内容长度限制
                            if len(content) > self.max_content_length:
                                content = content[:self.max_content_length]
                                logger.info(f"内容过长，截取到 {self.max_content_length} 字符")
                            
                            result = self._parse_content(url, content)
                            
                            # 标记为已访问
                            self.visited_urls.add(url)
                            
                            logger.info(f"URL抓取成功: {url}, 内容长度: {len(result.content)}")
                            return result
                        
                        elif response.status == 429:
                            logger.warning(f"Jina Reader请求频率限制 (尝试 {attempt + 1}/{self.max_retries})")
                            if attempt == self.max_retries - 1:
                                return JinaReaderResult(
                                    url=url,
                                    title="",
                                    content="",
                                    links=[],
                                    success=False,
                                    error="请求频率限制"
                                )
                        
                        else:
                            error_text = await response.text()
                            logger.warning(f"Jina Reader请求失败 (尝试 {attempt + 1}/{self.max_retries})，状态码: {response.status}")
                            if attempt == self.max_retries - 1:
                                return JinaReaderResult(
                                    url=url,
                                    title="",
                                    content="",
                                    links=[],
                                    success=False,
                                    error=f"HTTP {response.status}: {error_text}"
                                )
                            
            except asyncio.TimeoutError:
                logger.warning(f"URL抓取超时 (尝试 {attempt + 1}/{self.max_retries}): {url}")
                if attempt == self.max_retries - 1:
                    return JinaReaderResult(
                        url=url,
                        title="",
                        content="",
                        links=[],
                        success=False,
                        error="请求超时"
                    )
            except Exception as e:
                logger.warning(f"URL抓取错误 (尝试 {attempt + 1}/{self.max_retries}): {url}, 错误: {e}")
                if attempt == self.max_retries - 1:
                    return JinaReaderResult(
                        url=url,
                        title="",
                        content="",
                        links=[],
                        success=False,
                        error=str(e)
                    )
        
        # 如果所有重试都失败了
        return JinaReaderResult(
            url=url,
            title="",
            content="",
            links=[],
            success=False,
            error="所有重试失败"
        )
    
    async def read_urls(self, urls: List[str], max_concurrent: int = 5) -> List[JinaReaderResult]:
        """
        批量读取多个URL
        
        Args:
            urls: URL列表
            max_concurrent: 最大并发数
            
        Returns:
            抓取结果列表
        """
        if not urls:
            return []
        
        # 过滤已访问的URL
        new_urls = [url for url in urls if url not in self.visited_urls]
        
        if not new_urls:
            logger.info("所有URL都已访问过")
            return []
        
        logger.info(f"开始批量抓取 {len(new_urls)} 个URL")
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def read_with_semaphore(url: str) -> JinaReaderResult:
            async with semaphore:
                return await self.read_url(url)
        
        # 并发执行
        tasks = [read_with_semaphore(url) for url in new_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"URL抓取异常: {new_urls[i]}, 错误: {result}")
                valid_results.append(JinaReaderResult(
                    url=new_urls[i],
                    title="",
                    content="",
                    links=[],
                    success=False,
                    error=str(result)
                ))
            else:
                valid_results.append(result)
        
        successful_count = sum(1 for r in valid_results if r.success)
        logger.info(f"批量抓取完成，成功: {successful_count}/{len(new_urls)}")
        
        return valid_results
    
    def _parse_content(self, url: str, content: str) -> JinaReaderResult:
        """
        解析Jina Reader返回的内容
        
        Args:
            url: 原始URL
            content: Jina Reader返回的内容
            
        Returns:
            解析后的结果
        """
        try:
            # Jina Reader返回的是Markdown格式的内容
            lines = content.split('\n')
            
            # 提取标题（通常是第一行）
            title = ""
            content_lines = []
            links = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 提取标题
                if line.startswith('# ') and not title:
                    title = line[2:].strip()
                elif line.startswith('Title: ') and not title:
                    title = line[7:].strip()
                else:
                    content_lines.append(line)
                
                # 提取链接
                link_matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line)
                for _, link_url in link_matches:
                    if link_url.startswith(('http://', 'https://')):
                        links.append(link_url)
            
            # 清理内容
            cleaned_content = self._clean_content('\n'.join(content_lines))
            
            # 去重链接
            unique_links = list(set(links))
            
            return JinaReaderResult(
                url=url,
                title=title or self._extract_title_from_url(url),
                content=cleaned_content,
                links=unique_links,
                success=True
            )
            
        except Exception as e:
            logger.error(f"内容解析失败: {e}")
            return JinaReaderResult(
                url=url,
                title="",
                content=content,  # 返回原始内容
                links=[],
                success=True,
                error=f"解析失败: {e}"
            )
    
    def _clean_content(self, content: str) -> str:
        """
        清理内容，移除不必要的格式和噪音
        
        Args:
            content: 原始内容
            
        Returns:
            清理后的内容
        """
        if not content:
            return ""
        
        # 移除多余的空行
        lines = [line.strip() for line in content.split('\n')]
        lines = [line for line in lines if line]
        
        # 移除Markdown格式标记
        cleaned_lines = []
        for line in lines:
            # 移除图片标记
            line = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', line)
            # 移除链接格式，保留文本
            line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
            # 移除粗体和斜体标记
            line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            line = re.sub(r'\*([^*]+)\*', r'\1', line)
            # 移除标题标记
            line = re.sub(r'^#+\s*', '', line)
            
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_title_from_url(self, url: str) -> str:
        """从URL中提取标题"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path.strip('/')
            
            if path:
                # 使用路径的最后一部分作为标题
                title = path.split('/')[-1].replace('-', ' ').replace('_', ' ')
                return f"{title} - {domain}"
            else:
                return domain
        except:
            return url
    
    def format_results_for_context(self, results: List[JinaReaderResult]) -> str:
        """
        格式化抓取结果为上下文字符串
        
        Args:
            results: 抓取结果列表
            
        Returns:
            格式化的上下文字符串
        """
        if not results:
            return "没有成功抓取到网页内容。"
        
        context_parts = ["网页内容:"]
        
        for i, result in enumerate(results, 1):
            if result.success and result.content:
                context_parts.append(f"\n{i}. {result.title}")
                context_parts.append(f"   URL: {result.url}")
                
                # 截取内容前500字符作为摘要
                content_preview = result.content[:500]
                if len(result.content) > 500:
                    content_preview += "..."
                
                context_parts.append(f"   内容: {content_preview}")
                
                if result.links:
                    context_parts.append(f"   相关链接数: {len(result.links)}")
            else:
                context_parts.append(f"\n{i}. [抓取失败] {result.url}")
                if result.error:
                    context_parts.append(f"   错误: {result.error}")
        
        return "\n".join(context_parts)
    
    def get_all_links(self, results: List[JinaReaderResult]) -> List[str]:
        """
        从抓取结果中提取所有链接
        
        Args:
            results: 抓取结果列表
            
        Returns:
            所有链接的列表
        """
        all_links = []
        for result in results:
            if result.success:
                all_links.extend(result.links)
        
        # 去重并过滤
        unique_links = []
        seen = set()
        
        for link in all_links:
            if link not in seen and link not in self.visited_urls:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links
    
    def clear_visited_urls(self):
        """清空已访问URL记录"""
        self.visited_urls.clear()
        logger.info("已清空访问记录")
