#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统主类 - LangGraph版本
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from ..config import AppConfig, config_manager
from ..llm.factory import LLMFactory
from ..agents.agent_manager import AgentManager
from ..search.serper_search import SerperSearchEngine
from ..search.web_reader import WebReader
from ..graph.workflow import ResearchWorkflow
from ..search.cache import SearchCache
from .pipeline import RAGPipeline, RAGResult

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG系统主类"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or config_manager.get_config()
        
        # 初始化组件
        self.llm = None
        self.agent_manager = None
        self.serper_engine = None
        self.web_reader = None
        self.workflow = None
        self.cache = None
        self.pipeline = None
        
        self._initialized = False
        
        logger.info("RAG系统创建完成 (LangGraph)")
    
    async def initialize(self):
        """初始化系统"""
        if self._initialized:
            logger.warning("系统已经初始化")
            return
        
        try:
            logger.info("开始初始化RAG系统 (LangGraph)...")
            
            # 初始化LLM
            self.llm = LLMFactory.create_llm(self.config.llm)
            logger.info(f"LLM初始化完成: {self.config.llm.provider}")
            
            # 测试LLM连接
            if hasattr(self.llm, 'test_connection'):
                connection_ok = await self.llm.test_connection()
                if not connection_ok:
                    raise Exception("LLM连接测试失败")
            
            # 初始化代理管理器
            self.agent_manager = AgentManager(self.llm)
            logger.info("代理管理器初始化完成")
            
            # 初始化搜索引擎
            search_config = self.config.to_dict().get('search', {})
            serper_api_key = search_config.get('serper_api_key')
            if serper_api_key and serper_api_key != "YOUR_SERPER_API_KEY_HERE":
                self.serper_engine = SerperSearchEngine(
                    api_key=serper_api_key,
                    max_results=search_config.get('max_results_per_query', search_config.get('max_results', 10))
                )
                logger.info("Serper搜索引擎初始化完成")
            
            # 初始化网页读取器
            self.web_reader = WebReader()
            logger.info("Web Reader初始化完成")
            
            # 初始化LangGraph工作流
            self.workflow = ResearchWorkflow(
                agent_manager=self.agent_manager,
                serper_engine=self.serper_engine,
                web_reader=self.web_reader,
                config=self.config.to_dict()
            )
            logger.info("LangGraph工作流初始化完成")
            
            # 初始化缓存
            if self.config.cache.enabled:
                self.cache = SearchCache(self.config.cache)
                logger.info("缓存系统初始化完成")
            
            # 初始化管道
            self.pipeline = RAGPipeline(
                agent_manager=self.agent_manager,
                workflow=self.workflow,
                cache=self.cache
            )
            logger.info("RAG管道初始化完成")
            
            self._initialized = True
            logger.info("RAG系统初始化完成 (LangGraph)")
            
        except Exception as e:
            logger.error(f"RAG系统初始化失败: {e}")
            raise
    
    async def ask(
        self, 
        question: str, 
        use_cache: bool = True
    ) -> RAGResult:
        """提问"""
        if not self._initialized:
            await self.initialize()
        
        return await self.pipeline.process(
            question=question,
            use_cache=use_cache
        )
    
    async def batch_ask(
        self, 
        questions: List[str], 
        use_cache: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[RAGResult]:
        """批量提问"""
        if not self._initialized:
            await self.initialize()
        
        return await self.pipeline.batch_process(
            questions=questions,
            use_cache=use_cache,
            progress_callback=progress_callback
        )
    
    async def test_system(self) -> Dict[str, Any]:
        """测试系统"""
        if not self._initialized:
            await self.initialize()
        
        test_results = {}
        
        # 测试LLM
        try:
            if hasattr(self.llm, 'test_connection'):
                test_results['llm'] = await self.llm.test_connection()
            else:
                test_results['llm'] = True
        except Exception as e:
            test_results['llm'] = False
            logger.error(f"LLM测试失败: {e}")
        
        # 测试代理
        try:
            agent_results = await self.agent_manager.test_all_agents()
            test_results['agents'] = agent_results
        except Exception as e:
            test_results['agents'] = {}
            logger.error(f"代理测试失败: {e}")
        
        # 测试搜索引擎
        try:
            if self.serper_engine:
                search_results = await self.serper_engine.search("测试搜索")
                test_results['search'] = len(search_results) > 0
            else:
                test_results['search'] = False
        except Exception as e:
            test_results['search'] = False
            logger.error(f"搜索测试失败: {e}")
        
        # 测试缓存
        if self.cache:
            try:
                self.cache.set("test", "data", ttl=1)
                cached_data = self.cache.get("test")
                test_results['cache'] = cached_data == "data"
            except Exception as e:
                test_results['cache'] = False
                logger.error(f"缓存测试失败: {e}")
        else:
            test_results['cache'] = None
        
        # 测试完整管道
        try:
            test_question = "什么是人工智能？"
            result = await self.ask(test_question, use_cache=False)
            test_results['pipeline'] = result.confidence > 0
        except Exception as e:
            test_results['pipeline'] = False
            logger.error(f"管道测试失败: {e}")
        
        return test_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "initialized": self._initialized,
            "config": self.config.to_dict(),
            "components": {
                "llm": self.llm is not None,
                "agent_manager": self.agent_manager is not None,
                "serper_engine": self.serper_engine is not None,
                "web_reader": self.web_reader is not None,
                "workflow": self.workflow is not None,
                "cache": self.cache is not None,
                "pipeline": self.pipeline is not None
            }
        }
        
        if self._initialized and self.pipeline:
            info["pipeline_stats"] = self.pipeline.get_pipeline_stats()
        
        return info
    
    async def cleanup(self):
        """清理资源"""
        logger.info("开始清理RAG系统资源...")
        
        if self.cache:
            self.cache.cleanup_expired()
        
        logger.info("RAG系统资源清理完成")
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        if self._initialized:
            logger.warning("系统已初始化，配置更新需要重新初始化")
        
        config_manager.update_config(new_config)
        self.config = config_manager.get_config()
        
        # 重置初始化状态
        self._initialized = False
        
        logger.info("配置已更新")

# 全局RAG系统实例
rag_system = RAGSystem()
