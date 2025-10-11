#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph状态定义
"""

from typing import TypedDict, List, Annotated, Optional, Any
from operator import add


class ResearchState(TypedDict):
    """研究工作流状态"""
    
    # === 输入 ===
    original_query: str              # 原始用户问题
    extracted_question: str          # 提取的核心问题
    keywords: str                    # 搜索关键词
    
    # === 搜索控制 ===
    current_round: int               # 当前搜索轮次 (1-3)
    current_step: int                # 当前步骤编号
    current_question: str            # 当前要搜索的问题
    min_rounds: int                  # 最少执行轮次
    max_rounds: int                  # 最大轮次
    max_steps: int                   # 最大步骤数
    completed_rounds: int            # 已完成的访问轮次计数
    
    # === 累积数据（使用add注解自动合并） ===
    search_results: Annotated[List[Any], add]      # 所有搜索结果
    content_results: Annotated[List[Any], add]     # 所有网页内容
    all_context: Annotated[List[str], add]         # 所有上下文信息
    steps_log: Annotated[List[dict], add]          # 步骤日志
    
    # === 状态追踪 ===
    step_questions: List[str]        # 待搜索的步骤问题队列
    visited_urls: List[str]          # 已访问的URL列表
    search_history: List[str]        # 搜索历史
    
    # === 控制流 ===
    next_action: str                 # 下一步动作: search/visit/reflect/answer
    should_continue: bool            # 是否继续循环
    early_termination: bool          # 是否提前结束
    
    # === Token管理 ===
    current_tokens: int              # 当前token消耗
    token_budget: int                # Token预算
    
    # === 输出 ===
    final_answer: str                # 最终答案
    confidence: float                # 置信度
    success: bool                    # 是否成功
    processing_time: float           # 处理时间
    cached: bool                     # 是否使用缓存
