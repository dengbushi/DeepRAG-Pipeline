#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG处理管道 - LangGraph版本
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from ..agents.agent_manager import AgentManager
from ..graph.workflow import ResearchWorkflow
from ..search.cache import SearchCache

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """RAG结果 - LangGraph版本"""
    question: str
    extracted_question: str
    keywords: str
    answer: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    cached: bool = False
    search_results: List[Any] = None
    content_results: List[Any] = None
    steps_log: List[dict] = None
    observability_events: List[dict] = None
    total_steps: int = 0
    success: bool = True
    
    def __post_init__(self):
        """初始化默认值"""
        if self.search_results is None:
            self.search_results = []
        if self.content_results is None:
            self.content_results = []
        if self.steps_log is None:
            self.steps_log = []
        if self.observability_events is None:
            self.observability_events = []
    
    def _serialize_list(self, items: List[Any]) -> List[Any]:
        """序列化列表中的元素"""
        serialized = []
        for item in items:
            if hasattr(item, "to_dict") and callable(getattr(item, "to_dict")):
                serialized.append(item.to_dict())
            else:
                serialized.append(item)
        return serialized

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        serialized_search = self._serialize_list(self.search_results)
        serialized_content = self._serialize_list(self.content_results)

        return {
            "question": self.question,
            "extracted_question": self.extracted_question,
            "keywords": self.keywords,
            "answer": self.answer,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "cached": self.cached,
            "search_results": serialized_search,
            "content_results": serialized_content,
            "steps_log": self.steps_log,
            "observability_events": self.observability_events,
            "total_steps": self.total_steps,
            "deep_search": {
                "total_steps": self.total_steps,
                "search_steps": len([s for s in self.steps_log if s.get("action") == "search"]),
                "visit_steps": len([s for s in self.steps_log if s.get("action") == "visit"]),
                "content_results_count": len(serialized_content),
                "success": self.success
            }
        }

class RAGPipeline:
    """RAG处理管道 - LangGraph版本"""
    
    def __init__(
        self, 
        agent_manager: AgentManager,
        workflow: ResearchWorkflow,
        cache: Optional[SearchCache] = None
    ):
        self.agent_manager = agent_manager
        self.workflow = workflow
        self.cache = cache
        
        logger.info("RAG管道已初始化 (LangGraph)")
    
    async def process(
        self, 
        question: str, 
        use_cache: bool = True
    ) -> RAGResult:
        """处理问题并返回答案 - 使用LangGraph工作流"""
        start_time = time.time()
        
        logger.info(f"开始处理问题 (LangGraph): {question}")
        
        try:
            # 检查缓存
            if use_cache and self.cache:
                cached_result = self.cache.get_by_namespace("answer", question)
                if cached_result:
                    logger.info("使用缓存结果")
                    cached_result.cached = True
                    cached_result.processing_time = time.time() - start_time
                    return cached_result
            
            # 使用LangGraph工作流执行完整的研究流程
            logger.info("启动LangGraph研究工作流...")
            workflow_result = await self.workflow.run(question, use_cache)
            
            logger.info(f"LangGraph工作流完成:")
            logger.info(f"  - 总步骤: {workflow_result.get('total_steps', 0)}")
            logger.info(f"  - 搜索结果: {len(workflow_result.get('search_results', []))}")
            logger.info(f"  - 内容结果: {len(workflow_result.get('content_results', []))}")
            logger.info(f"  - 成功状态: {workflow_result.get('success', True)}")
            
            # 创建结果
            result = RAGResult(
                question=question,
                extracted_question=workflow_result.get('extracted_question', ''),
                keywords=workflow_result.get('keywords', ''),
                answer=workflow_result.get('answer', ''),
                confidence=workflow_result.get('confidence', 0.9),
                processing_time=time.time() - start_time,
                cached=False,
                search_results=workflow_result.get('search_results', []),
                content_results=workflow_result.get('content_results', []),
                steps_log=workflow_result.get('steps_log', []),
                observability_events=workflow_result.get('observability_events', []),
                total_steps=workflow_result.get('total_steps', 0),
                success=workflow_result.get('success', True)
            )
            
            # 缓存结果
            if use_cache and self.cache:
                self.cache.set_by_namespace("answer", question, result)
            
            logger.info(f"问题处理完成，耗时: {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"LangGraph处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            result = RAGResult(
                question=question,
                extracted_question=question,
                keywords="",
                answer="抱歉，我无法处理这个问题。请稍后重试或重新表述您的问题。",
                confidence=0.0,
                processing_time=time.time() - start_time,
                cached=False,
                success=False
            )
            
            return result
    
    async def batch_process(
        self, 
        questions: List[str], 
        use_cache: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[RAGResult]:
        """批量处理问题"""
        results = []
        
        logger.info(f"开始批量处理 {len(questions)} 个问题")
        
        for i, question in enumerate(questions):
            try:
                result = await self.process(question, use_cache)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(questions), result)
                
                logger.info(f"完成问题 {i+1}/{len(questions)}")
                
            except Exception as e:
                logger.error(f"处理问题 {i+1} 失败: {e}")
                # 添加错误结果
                error_result = RAGResult(
                    question=question,
                    extracted_question=question,
                    keywords="",
                    answer=f"处理失败: {str(e)}",
                    confidence=0.0,
                    processing_time=0.0,
                    cached=False
                )
                results.append(error_result)
        
        logger.info(f"批量处理完成，成功: {len([r for r in results if r.confidence > 0])}/{len(questions)}")
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取管道统计信息"""
        stats = {
            "agents": self.agent_manager.list_agents(),
            "cache_enabled": self.cache is not None and self.cache.config.enabled
        }
        
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats
