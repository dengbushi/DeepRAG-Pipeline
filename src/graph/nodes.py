#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph节点实现
"""

import logging
from typing import Dict, Any, List
from ..agents.agent_manager import AgentManager
from ..search.serper_search import SerperSearchEngine
from ..search.jina_reader import JinaReader

logger = logging.getLogger(__name__)


class ResearchNodes:
    """研究工作流节点"""
    
    def __init__(
        self, 
        agent_manager: AgentManager,
        serper_engine: SerperSearchEngine,
        jina_reader: JinaReader,
        config: Dict[str, Any]
    ):
        self.agent_manager = agent_manager
        self.serper_engine = serper_engine
        self.jina_reader = jina_reader
        self.config = config
        
        # 搜索配置
        search_config = config.get('search', {})
        self.min_rounds = search_config.get('min_rounds', 1)
        self.max_rounds = search_config.get('max_rounds', 3)
        self.max_urls_per_step = search_config.get('max_urls_per_step', 3)
        self.token_budget = search_config.get('token_budget', 50000)
        self.allow_early_termination = search_config.get('allow_early_termination', True)
    
    async def extract_question_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点1: 提取核心问题"""
        logger.info("=== 节点: 提取核心问题 ===")
        
        question = await self.agent_manager.extract_question(
            state["original_query"]
        )
        
        logger.info(f"提取的问题: {question}")
        return {"extracted_question": question}
    
    async def extract_keywords_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点2: 提取搜索关键词"""
        logger.info("=== 节点: 提取关键词 ===")
        
        keywords = await self.agent_manager.extract_keywords(
            state["extracted_question"]
        )
        
        logger.info(f"搜索关键词: {keywords}")
        return {
            "keywords": keywords,
            "current_question": keywords  # 初始化当前问题
        }
    
    async def search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点3: 执行搜索"""
        logger.info(f"=== 节点: 搜索 (第{state['current_round']}轮) ===")
        
        query = state["current_question"]
        logger.info(f"搜索查询: {query}")
        
        # 执行搜索
        results = await self.serper_engine.search(query)
        
        # 格式化上下文
        context = self.serper_engine.format_results_for_context(results)
        
        # 记录步骤
        step_log = {
            "step": state["current_step"],
            "action": "search",
            "query": query,
            "results_count": len(results),
            "round": state["current_round"]
        }
        
        logger.info(f"搜索完成，获得 {len(results)} 个结果")
        
        return {
            "search_results": results,
            "all_context": [f"搜索结果 (第{state['current_round']}轮): {context}"],
            "steps_log": [step_log],
            "current_tokens": state.get("current_tokens", 0) + len(context) // 4
        }
    
    async def visit_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点4: 访问URL获取内容"""
        logger.info(f"=== 节点: 访问URL (第{state['current_round']}轮) ===")
        
        # 选择要访问的URL
        urls_to_visit = await self._select_urls_to_visit(
            state["search_results"],
            state["visited_urls"]
        )
        
        if not urls_to_visit:
            logger.warning("没有URL可访问")
            return {
                "steps_log": [{
                    "step": state["current_step"],
                    "action": "visit",
                    "urls": [],
                    "results_count": 0
                }]
            }
        
        # 访问URL
        content_results = await self.jina_reader.read_urls(
            urls_to_visit,
            max_concurrent=3
        )
        
        # 更新上下文
        successful_results = [r for r in content_results if r.success]
        content_context = ""
        if successful_results:
            content_context = self.jina_reader.format_results_for_context(successful_results)
        
        # 记录步骤
        step_log = {
            "step": state["current_step"],
            "action": "visit",
            "urls": urls_to_visit,
            "results_count": len(successful_results),
            "round": state["current_round"]
        }
        
        completed_rounds = state.get("completed_rounds", 0) + 1
        logger.info(
            f"访问完成，成功访问 {len(successful_results)} 个URL，累计轮次: {completed_rounds}"
        )
        
        return {
            "content_results": content_results,
            "all_context": [f"网页内容 (第{state['current_round']}轮): {content_context}"] if content_context else [],
            "visited_urls": state.get("visited_urls", []) + urls_to_visit,
            "completed_rounds": completed_rounds,
            "steps_log": [step_log],
            "current_tokens": state.get("current_tokens", 0) + len(content_context) // 4
        }
    
    async def reflect_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点5: 反思并生成步骤问题"""
        logger.info(f"=== 节点: 反思 (第{state['current_round']}轮) ===")
        
        # 生成步骤问题
        new_steps = await self._generate_step_questions(
            state["original_query"],
            state["all_context"],
            state["current_round"]
        )
        
        # 检查是否提前结束
        early_termination = len(new_steps) == 0 and self.allow_early_termination
        
        # 记录步骤
        step_log = {
            "step": state["current_step"],
            "action": "reflect",
            "new_questions_count": len(new_steps),
            "early_termination": early_termination,
            "round": state["current_round"]
        }
        
        if early_termination:
            logger.info("LLM判断信息充足，提前结束搜索")
        else:
            logger.info(f"反思完成，生成 {len(new_steps)} 个新问题")
        
        return {
            "step_questions": state.get("step_questions", []) + new_steps,
            "early_termination": early_termination,
            "steps_log": [step_log]
        }
    
    async def decide_action_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点6: 决定下一步动作"""
        current_step = state.get("current_step", 0)
        logger.info(f"=== 节点: 决定动作 (步骤{current_step}) ===")
        
        steps_log = state.get("steps_log", [])
        
        # 检查是否达到最大步骤
        max_steps = state.get("max_steps", 8)
        if current_step >= max_steps:
            logger.info(f"达到最大步骤 {max_steps}，生成报告")
            return {"next_action": "answer"}
        
        # 检查提前终止
        current_round = state.get("current_round", 0)
        min_rounds = state.get("min_rounds", self.min_rounds)
        completed_rounds = state.get("completed_rounds", 0)
        if state.get("early_termination", False):
            if completed_rounds >= min_rounds:
                logger.info("提前终止，生成报告")
                return {"next_action": "answer"}
            logger.info(
                f"提前终止请求，但仅完成 {completed_rounds}/{min_rounds} 轮，继续执行"
            )

        # 如果没有步骤，开始搜索
        if not steps_log:
            logger.info("初始状态，开始搜索")
            return {"next_action": "search"}
        
        last_step = steps_log[-1]
        last_action = last_step.get("action")
        
        # 决策逻辑
        if last_action == "search":
            next_action = "visit"
        elif last_action == "visit":
            visit_count = len([s for s in steps_log if s["action"] == "visit"])
            if completed_rounds >= self.max_rounds:
                logger.info(
                    f"完成 {completed_rounds} 轮访问，达到最大轮次 {self.max_rounds}，生成报告"
                )
                next_action = "answer"
            else:
                next_action = "reflect"
        elif last_action == "reflect":
            # 检查是否有新的步骤问题
            step_questions = state.get("step_questions", [])
            if step_questions:
                next_action = "search"
            else:
                if completed_rounds >= min_rounds:
                    logger.info("没有新的步骤问题，满足最小轮次，生成报告")
                    next_action = "answer"
                else:
                    logger.info(
                        f"没有新的步骤问题，但仅完成 {completed_rounds}/{min_rounds} 轮，继续搜索"
                    )
                    next_action = "search"
        else:
            next_action = "search"
        
        logger.info(f"下一步动作: {next_action}")
        return {"next_action": next_action}
    
    async def increment_step_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点7: 更新步骤和轮次"""
        current_step = state.get("current_step", 0) + 1
        
        # 计算当前轮次
        if current_step <= 2:
            current_round = 1
        else:
            current_round = 1 + (current_step - 2 + 2) // 3
        
        # 更新当前问题
        step_questions = state.get("step_questions", [])
        if step_questions:
            current_question = step_questions[0]
            remaining_questions = step_questions[1:]
        else:
            current_question = state.get("keywords", state.get("original_query", ""))
            remaining_questions = []
        
        logger.info(f"步骤更新: {current_step}, 轮次: {current_round}")
        
        return {
            "current_step": current_step,
            "current_round": current_round,
            "current_question": current_question,
            "step_questions": remaining_questions
        }
    
    async def generate_report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点8: 生成最终研究报告"""
        logger.info("=== 节点: 生成研究报告 ===")
        
        # 使用Research Agent生成报告
        final_answer = await self.agent_manager.generate_research_report(
            state["extracted_question"],
            state.get("search_results", []),
            state.get("content_results", [])
        )
        
        logger.info("研究报告生成完成")
        
        return {
            "final_answer": final_answer,
            "confidence": 0.9,
            "success": True
        }
    
    # === 辅助方法 ===
    
    async def _select_urls_to_visit(
        self, 
        search_results: List[dict], 
        visited_urls: List[str]
    ) -> List[str]:
        """智能URL选择"""
        if not search_results:
            return []
        
        # 转换为对象（简化处理）
        from ..search.serper_search import SerperSearchResult
        results = []
        for i, r in enumerate(search_results):
            if isinstance(r, SerperSearchResult):
                results.append(r)
            elif isinstance(r, dict):
                results.append(SerperSearchResult(
                    title=r.get('title', ''),
                    url=r.get('url', ''),
                    snippet=r.get('snippet', ''),
                    position=r.get('position', i + 1)
                ))
        
        # 选择未访问的URL
        unvisited_results = [r for r in results if r.url not in visited_urls]
        
        # 选择前N个
        selected_urls = [r.url for r in unvisited_results[:self.max_urls_per_step]]
        
        return selected_urls
    
    async def _generate_step_questions(
        self, 
        original_query: str, 
        context: List[str],
        current_round: int
    ) -> List[str]:
        """生成步骤问题"""
        # 导入LLM相关模块
        from ..llm.deepseek import DeepSeekLLM
        from ..config import ConfigManager
        from ..llm.base import Message
        
        try:
            # 摘要上下文
            context_summary = "\n".join(context) if context else "暂无已收集信息"
            
            # 构建prompt
            prompt = f"""作为深度研究助手，分析当前研究进展并生成下一步搜索查询。

原始问题: {original_query}

已收集信息摘要:
{context_summary[:8000]}

请分析信息缺口，判断是否需要继续搜索：

**如果需要继续搜索：**
- 生成1个最重要的搜索查询
- 查询要与原始问题相关且不重复已有信息
- 查询要适合搜索引擎，使用关键词形式
- 直接返回查询文本

**如果信息已经充足：**
- 使用 <sufficient></sufficient> 标签说明理由

示例格式：
- 需要继续：[关键词1] [关键词2] [关键词3]
- 信息充足：<sufficient>已收集足够信息</sufficient>
"""
            
            # 获取配置
            config_manager = ConfigManager()
            app_config = config_manager.get_config()
            llm_config = app_config.llm
            
            # 创建LLM客户端
            llm_client = DeepSeekLLM(
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                model_name=llm_config.model_name,
                max_tokens=2000,
                temperature=0.3,
                timeout=llm_config.timeout
            )
            
            # 调用LLM
            messages = [Message(role="user", content=prompt)]
            llm_response = await llm_client.generate(messages)
            response = llm_response.content.strip()
            
            # 检测是否包含<sufficient>标签
            if '<sufficient>' in response and '</sufficient>' in response:
                logger.info("LLM判断信息充足，提前结束搜索")
                return []
            
            # 解析步骤问题
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('**') or line.startswith('-'):
                    continue
                if line.startswith('<thinking>') or line.startswith('</thinking>'):
                    continue
                # 找到第一个有效内容
                step_question = line.replace('<thinking>', '').replace('</thinking>', '').strip()
                if step_question:
                    logger.info(f"生成步骤问题: {step_question}")
                    return [step_question]
            
            # 回退策略
            fallback = f"{original_query} 详细信息"
            logger.info(f"回退步骤问题: {fallback}")
            return [fallback]
            
        except Exception as e:
            logger.warning(f"步骤问题生成失败: {e}，使用回退策略")
            fallback = f"{original_query} 详细信息"
            return [fallback]
