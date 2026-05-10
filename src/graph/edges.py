#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph边和路由逻辑
"""

import logging
from typing import Dict, Any, Literal

logger = logging.getLogger(__name__)


def route_action(state: Dict[str, Any]) -> Literal["search", "visit", "reflect", "generate_report"]:
    """
    根据next_action路由到不同节点
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    # 检查提前终止
    if state.get("early_termination", False):
        logger.info("路由: 提前终止 → 生成报告")
        return "generate_report"

    # 检查报告准备状态
    if state.get("report_ready", False):
        logger.info("路由: 已满足报告条件 → 生成报告")
        return "generate_report"

    # 检查累计查询预算
    completed_queries = len(state.get("completed_queries", []))
    max_total_queries = state.get("max_total_queries", 6)
    if completed_queries >= max_total_queries:
        logger.info(f"路由: 达到最大查询数({max_total_queries}) → 生成报告")
        return "generate_report"
    
    # 检查最大步骤数
    max_steps = state.get("max_steps", 8)
    if state.get("current_step", 0) >= max_steps:
        logger.info(f"路由: 达到最大步骤({max_steps}) → 生成报告")
        return "generate_report"
    
    # 检查token预算
    token_budget = state.get("token_budget", 50000)
    if state.get("current_tokens", 0) >= token_budget:
        logger.info(f"路由: 达到token预算({token_budget}) → 生成报告")
        return "generate_report"
    
    # 根据next_action路由
    action = state.get("next_action", "search")
    
    if action == "search":
        logger.info("路由: 搜索")
        return "search"
    elif action == "visit":
        logger.info("路由: 访问URL")
        return "visit"
    elif action == "reflect":
        logger.info("路由: 反思")
        return "reflect"
    elif action == "answer":
        logger.info("路由: 生成报告")
        return "generate_report"
    else:
        logger.warning(f"未知动作: {action}，默认生成报告")
        return "generate_report"


def should_continue(state: Dict[str, Any]) -> bool:
    """
    判断是否应该继续循环
    
    Args:
        state: 当前状态
        
    Returns:
        是否继续
    """
    # 检查提前终止
    if state.get("early_termination", False):
        return False
    
    # 检查最大步骤数
    max_steps = state.get("max_steps", 8)
    if state.get("current_step", 0) >= max_steps:
        return False
    
    # 检查token预算
    token_budget = state.get("token_budget", 50000)
    if state.get("current_tokens", 0) >= token_budget:
        return False
    
    # 检查next_action
    next_action = state.get("next_action", "")
    if next_action == "answer":
        return False
    
    return True
