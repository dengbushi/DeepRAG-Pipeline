#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索模块 - 基于Serper + 网页抓取 + LangGraph的深度搜索架构
"""

from .serper_search import SerperSearchEngine, SerperSearchResult
from .web_reader import WebReader, WebReaderResult
from .cache import SearchCache
from .retrievers import UnifiedSearchResult, BaseRetriever, RetrieverManager, build_retriever_manager

__all__ = [
    # 搜索引擎
    'SerperSearchEngine',
    'SerperSearchResult', 
    'WebReader',
    'WebReaderResult',
    'UnifiedSearchResult',
    'BaseRetriever',
    'RetrieverManager',
    'build_retriever_manager',
    
    # 缓存
    'SearchCache',
]
