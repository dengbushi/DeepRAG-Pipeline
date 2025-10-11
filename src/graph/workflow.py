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
        jina_reader,
        config: Dict[str, Any]
    ):
        """
        初始化研究工作流
        
        Args:
            agent_manager: Agent管理器
            serper_engine: Serper搜索引擎
            jina_reader: Jina Reader
            config: 系统配置
        """
        self.agent_manager = agent_manager
        self.serper_engine = serper_engine
        self.jina_reader = jina_reader
        self.config = config
        
        # 创建图
        self.app = create_research_graph(
            agent_manager,
            serper_engine,
            jina_reader,
            config
        )
        
        # 搜索配置
        search_config = config.get('search', {})
        self.max_rounds = search_config.get('max_rounds', 3)
        self.min_rounds = search_config.get('min_rounds', 1)
        if self.min_rounds < 1:
            logger.warning("最小轮次配置小于1，自动调整为1")
            self.min_rounds = 1
        if self.min_rounds > self.max_rounds:
            logger.warning("最小轮次大于最大轮次，自动调整为最大轮次")
            self.min_rounds = self.max_rounds
        self.max_steps = 2 + 3 * (self.max_rounds - 1)
        self.token_budget = search_config.get('token_budget', 50000)
        
        logger.info(f"研究工作流初始化完成 (LangGraph)")
        logger.info(f"  最小轮次: {self.min_rounds}")
        logger.info(f"  最大轮次: {self.max_rounds}")
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
            "min_rounds": self.min_rounds,
            "max_rounds": self.max_rounds,
            "max_steps": self.max_steps,
            "completed_rounds": 0,
            
            # 累积数据
            "search_results": [],
            "content_results": [],
            "all_context": [],
            "steps_log": [],
            
            # 状态
            "step_questions": [],
            "visited_urls": [],
            "search_history": [],
            
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
            config = {"recursion_limit": 50}  # 增加递归限制
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
            
            # 返回错误结果
            return {
                "question": question,
                "extracted_question": "",
                "keywords": "",
                "answer": f"研究失败: {str(e)}",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "search_results": [],
                "content_results": [],
                "steps_log": [],
                "total_steps": 0,
                "success": False,
                "cached": False
            }
    
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
            "min_rounds": self.min_rounds,
            "max_rounds": self.max_rounds,
            "max_steps": self.max_steps,
            "completed_rounds": 0,
            "search_results": [],
            "content_results": [],
            "all_context": [],
            "steps_log": [],
            "step_questions": [],
            "visited_urls": [],
            "search_history": [],
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
            # 流式执行
            async for chunk in self.app.astream(initial_state):
                # 返回每个节点的输出
                yield chunk
                
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
