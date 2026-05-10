#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph图构建
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .state import ResearchState
from .nodes import ResearchNodes
from .edges import route_action

logger = logging.getLogger(__name__)


def create_research_graph(
    agent_manager,
    serper_engine,
    web_reader,
    config: Dict[str, Any]
):
    """
    创建研究工作流图
    
    Args:
        agent_manager: Agent管理器
        serper_engine: Serper搜索引擎
        web_reader: 网页读取器
        config: 系统配置
        
    Returns:
        编译后的图
    """
    logger.info("开始构建LangGraph研究工作流...")
    
    # 初始化节点
    nodes = ResearchNodes(agent_manager, serper_engine, web_reader, config)
    
    # 创建图
    workflow = StateGraph(ResearchState)
    
    # 添加节点
    workflow.add_node("extract_question", nodes.extract_question_node)
    workflow.add_node("extract_keywords", nodes.extract_keywords_node)
    workflow.add_node("decide_action", nodes.decide_action_node)
    workflow.add_node("search", nodes.search_node)
    workflow.add_node("visit", nodes.visit_node)
    workflow.add_node("reflect", nodes.reflect_node)
    workflow.add_node("increment_step", nodes.increment_step_node)
    workflow.add_node("generate_report", nodes.generate_report_node)
    
    # 设置入口点
    workflow.set_entry_point("extract_question")
    
    # 添加边
    workflow.add_edge("extract_question", "extract_keywords")
    workflow.add_edge("extract_keywords", "decide_action")
    
    # 条件路由：从decide_action到不同的执行节点
    workflow.add_conditional_edges(
        "decide_action",
        route_action,
        {
            "search": "search",
            "visit": "visit",
            "reflect": "reflect",
            "generate_report": "generate_report"
        }
    )
    
    # 执行节点返回increment_step
    workflow.add_edge("search", "increment_step")
    workflow.add_edge("visit", "increment_step")
    workflow.add_edge("reflect", "increment_step")
    
    # increment_step返回decide_action（形成循环）
    workflow.add_edge("increment_step", "decide_action")
    
    # 结束节点
    workflow.add_edge("generate_report", END)
    
    # 编译图
    app = workflow.compile()
    
    logger.info("LangGraph研究工作流构建完成")
    logger.info("工作流结构:")
    logger.info("  入口: extract_question")
    logger.info("  循环: decide_action → [search/visit/reflect] → increment_step → decide_action")
    logger.info("  出口: generate_report → END")
    
    return app
