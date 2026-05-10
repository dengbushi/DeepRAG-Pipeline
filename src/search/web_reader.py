#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class WebReaderResult:
    url: str
    title: str
    content: str
    links: List[str]
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "links": self.links,
            "success": self.success,
            "error": self.error,
        }


class WebReader:
    def __init__(self):
        self.timeout = 20
        self.max_content_length = 150000
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.visited_urls: Set[str] = set()
        logger.info("Web Reader初始化完成")

    async def read_url(self, url: str, include_links: bool = True) -> WebReaderResult:
        if not url or not url.startswith(("http://", "https://")):
            logger.warning(f"无效的URL: {url}")
            return WebReaderResult(url=url, title="", content="", links=[], success=False, error="无效的URL")

        if url in self.visited_urls:
            logger.info(f"URL已访问过，跳过: {url}")
            return WebReaderResult(url=url, title="", content="", links=[], success=False, error="URL已访问过")

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=5, ttl_dns_cache=300, use_dns_cache=True, ssl=False)
            async with aiohttp.ClientSession(headers=self.headers, connector=connector, timeout=timeout) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status != 200:
                        return WebReaderResult(url=url, title="", content="", links=[], success=False, error=f"HTTP {response.status}")
                    html = await response.text(errors="ignore")

            result = self._parse_html_content(url, html, include_links=include_links)
            if result.success:
                self.visited_urls.add(url)
                logger.info(f"网页抓取成功: {url}, 内容长度: {len(result.content)}")
            return result
        except asyncio.TimeoutError:
            return WebReaderResult(url=url, title="", content="", links=[], success=False, error="请求超时")
        except Exception as e:
            logger.warning(f"网页抓取失败: {url}, 错误: {e}")
            return WebReaderResult(url=url, title="", content="", links=[], success=False, error=str(e))

    async def read_urls(self, urls: List[str], max_concurrent: int = 5) -> List[WebReaderResult]:
        if not urls:
            return []

        new_urls = [url for url in urls if url not in self.visited_urls]
        if not new_urls:
            logger.info("所有URL都已访问过")
            return []

        logger.info(f"开始批量抓取 {len(new_urls)} 个URL")
        semaphore = asyncio.Semaphore(max_concurrent)

        async def read_with_semaphore(url: str) -> WebReaderResult:
            async with semaphore:
                return await self.read_url(url)

        tasks = [read_with_semaphore(url) for url in new_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results: List[WebReaderResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"URL抓取异常: {new_urls[i]}, 错误: {result}")
                valid_results.append(WebReaderResult(url=new_urls[i], title="", content="", links=[], success=False, error=str(result)))
            else:
                valid_results.append(result)

        successful_count = sum(1 for r in valid_results if r.success)
        logger.info(f"批量抓取完成，成功: {successful_count}/{len(new_urls)}")
        return valid_results

    def _parse_html_content(self, url: str, html: str, include_links: bool = True) -> WebReaderResult:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            if not title:
                title = self._extract_title_from_url(url)

            links: List[str] = []
            if include_links:
                for link in soup.find_all("a", href=True):
                    href = urljoin(url, link.get("href", "").strip())
                    if href.startswith(("http://", "https://")):
                        links.append(href)

            text = soup.get_text("\n", strip=True)
            cleaned_content = self._clean_content(text)
            if len(cleaned_content) > self.max_content_length:
                cleaned_content = cleaned_content[: self.max_content_length]

            return WebReaderResult(
                url=url,
                title=title,
                content=cleaned_content,
                links=list(dict.fromkeys(links)),
                success=len(cleaned_content) >= 100,
                error=None if len(cleaned_content) >= 100 else "内容过短",
            )
        except Exception as e:
            logger.error(f"HTML解析失败: {e}")
            return WebReaderResult(url=url, title="", content="", links=[], success=False, error=f"HTML解析失败: {e}")

    def _clean_content(self, content: str) -> str:
        if not content:
            return ""

        lines = [line.strip() for line in content.split("\n")]
        lines = [line for line in lines if line]
        cleaned_lines: List[str] = []
        for line in lines:
            line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", line)
            line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
            line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            line = re.sub(r"\*([^*]+)\*", r"\1", line)
            line = re.sub(r"^#+\s*", "", line)
            if line:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _extract_title_from_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path.strip("/")
            if path:
                title = path.split("/")[-1].replace("-", " ").replace("_", " ")
                return f"{title} - {domain}"
            return domain
        except Exception:
            return url

    def clear_visited_urls(self):
        self.visited_urls.clear()
        logger.info("已清空访问记录")
