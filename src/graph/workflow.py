#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph工作流高层API
"""

import logging
import time
from typing import Dict, Any, Optional, AsyncIterator
from .graph import create_research_graph
from .state import ResearchState

logger = logging.getLogger(__name__)


class ResearchWorkflow:
    """研究工作流 - LangGraph版本"""
    
    def __init__(
        self,
        agent_manager,
        serper_engine,
        web_reader,
        config: Dict[str, Any]
    ):
        """
        初始化研究工作流
        
        Args:
            agent_manager: Agent管理器
            serper_engine: Serper搜索引擎
            web_reader: 网页读取器
            config: 系统配置
        """
        self.agent_manager = agent_manager
        self.serper_engine = serper_engine
        self.web_reader = web_reader
        self.config = config
        
        # 创建图
        self.app = create_research_graph(
            agent_manager,
            serper_engine,
            web_reader,
            config
        )
        
        # 搜索配置
        search_config = config.get('search', {})
        self.max_rounds = search_config.get('max_rounds', 3)
        self.max_total_queries = search_config.get('max_total_queries', 6)
        self.report_context_max_chars = search_config.get('report_context_max_chars', min(search_config.get('context_max_chars', 12000), 9000))
        self.max_steps = 2 + 3 * (self.max_rounds - 1)
        self.token_budget = search_config.get('token_budget', 50000)
        
        logger.info(f"研究工作流初始化完成 (LangGraph)")
        logger.info(f"  最大轮次: {self.max_rounds}")
        logger.info(f"  最大查询数: {self.max_total_queries}")
        logger.info(f"  最大步骤: {self.max_steps}")
        logger.info(f"  Token预算: {self.token_budget}")
    
    async def run(
        self,
        question: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        执行研究工作流
        
        Args:
            question: 用户问题
            use_cache: 是否使用缓存
            
        Returns:
            研究结果
        """
        start_time = time.time()
        
        logger.info(f"开始执行研究工作流: '{question}'")
        
        # 初始化状态
        initial_state: ResearchState = {
            # 输入
            "original_query": question,
            "extracted_question": "",
            "keywords": "",
            
            # 控制
            "current_round": 0,
            "current_step": 0,
            "current_question": "",
            "max_rounds": self.max_rounds,
            "max_total_queries": self.max_total_queries,
            "max_steps": self.max_steps,
            "completed_rounds": 0,
            "report_context_max_chars": self.report_context_max_chars,
            
            # 累积数据
            "search_results": [],
            "content_results": [],
            "source_registry": [],
            "all_context": [],
            "selected_contexts": [],
            "steps_log": [],
            "events": [],
            
            # 状态
            "step_questions": [],
            "visited_urls": [],
            "search_history": [],
            "completed_queries": [],
            "latest_evidence_count": 0,
            "report_ready": False,
            
            # 控制流
            "next_action": "",
            "should_continue": True,
            "early_termination": False,
            
            # Token管理
            "current_tokens": 0,
            "token_budget": self.token_budget,
            
            # 输出
            "final_answer": "",
            "confidence": 0.0,
            "success": False,
            "processing_time": 0.0,
            "cached": False
        }
        
        try:
            # 执行图（设置递归限制）
            recursion_limit = max(50, self.max_steps * 6)
            config = {"recursion_limit": recursion_limit}
            final_state = await self.app.ainvoke(initial_state, config=config)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 构建结果
            result = {
                "question": question,
                "extracted_question": final_state.get("extracted_question", ""),
                "keywords": final_state.get("keywords", ""),
                "answer": final_state.get("final_answer", ""),
                "confidence": final_state.get("confidence", 0.0),
                "processing_time": processing_time,
                "search_results": final_state.get("search_results", []),
                "content_results": final_state.get("content_results", []),
                "steps_log": final_state.get("steps_log", []),
                "observability_events": final_state.get("events", []),
                "total_steps": final_state.get("current_step", 0),
                "success": final_state.get("success", True),
                "cached": False
            }
            
            logger.info(f"研究工作流完成，耗时: {processing_time:.2f}s")
            logger.info(f"  总步骤: {result['total_steps']}")
            logger.info(f"  搜索结果: {len(result['search_results'])}")
            logger.info(f"  内容结果: {len(result['content_results'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"研究工作流执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def stream(
        self,
        question: str,
        use_cache: bool = True
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式执行研究工作流
        
        Args:
            question: 用户问题
            use_cache: 是否使用缓存
            
        Yields:
            中间状态更新
        """
        logger.info(f"开始流式执行研究工作流: '{question}'")
        
        # 初始化状态（同run方法）
        initial_state: ResearchState = {
            "original_query": question,
            "extracted_question": "",
            "keywords": "",
            "current_round": 0,
            "current_step": 0,
            "current_question": "",
            "max_rounds": self.max_rounds,
            "max_total_queries": self.max_total_queries,
            "max_steps": self.max_steps,
            "completed_rounds": 0,
            "report_context_max_chars": self.report_context_max_chars,
            "search_results": [],
            "content_results": [],
            "source_registry": [],
            "all_context": [],
            "selected_contexts": [],
            "steps_log": [],
            "events": [],
            "step_questions": [],
            "visited_urls": [],
            "search_history": [],
            "completed_queries": [],
            "latest_evidence_count": 0,
            "report_ready": False,
            "next_action": "",
            "should_continue": True,
            "early_termination": False,
            "current_tokens": 0,
            "token_budget": self.token_budget,
            "final_answer": "",
            "confidence": 0.0,
            "success": False,
            "processing_time": 0.0,
            "cached": False
        }
        
        try:
            start_time = time.time()
            recursion_limit = max(50, self.max_steps * 6)
            config = {"recursion_limit": recursion_limit}
            async for state in self.app.astream(initial_state, config=config, stream_mode="values"):
                state["processing_time"] = time.time() - start_time
                yield state
                
        except Exception as e:
            logger.error(f"流式执行失败: {e}")
            yield {"error": str(e)}
    
    def visualize(self, output_path: str = "workflow.png"):
        """
        可视化工作流图
        
        Args:
            output_path: 输出文件路径
        """
        try:
            from langchain_core.runnables.graph import MermaidDrawMethod
            
            # 生成Mermaid图
            mermaid_png = self.app.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API
            )
            
            # 保存为PNG
            with open(output_path, 'wb') as f:
                f.write(mermaid_png)
            
            logger.info(f"工作流图已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"可视化失败: {e}")
            logger.info("提示: 可能需要安装graphviz或使用在线Mermaid工具")
