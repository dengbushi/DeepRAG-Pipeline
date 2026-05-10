#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代理管理器
"""

import logging
from typing import Dict, Any, Optional, List
from .base import BaseAgent
from .question_extractor import QuestionExtractorAgent
from .keyword_extractor import KeywordExtractorAgent
from .research_agent import ResearchAgent
from .query_planner import QueryPlannerAgent
from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)

class AgentManager:
    """代理管理器"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """初始化代理"""
        try:
            self.agents = {
                "question_extractor": QuestionExtractorAgent(self.llm),
                "keyword_extractor": KeywordExtractorAgent(self.llm),
                "research_agent": ResearchAgent(self.llm),
                "query_planner": QueryPlannerAgent(self.llm)
            }
            logger.info(f"已初始化 {len(self.agents)} 个代理")
        except Exception as e:
            logger.error(f"代理初始化失败: {e}")
            raise
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """获取代理"""
        return self.agents.get(name)
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """列出所有代理信息"""
        return {name: agent.get_info() for name, agent in self.agents.items()}
    
    async def extract_question(self, text: str, **kwargs) -> str:
        """提取问题"""
        agent = self.get_agent("question_extractor")
        if not agent:
            raise ValueError("问题提取代理未找到")
        return await agent.process(text, **kwargs)
    
    async def extract_keywords(self, question: str, **kwargs) -> str:
        """提取关键词"""
        agent = self.get_agent("keyword_extractor")
        if not agent:
            raise ValueError("关键词提取代理未找到")
        return await agent.process(question, **kwargs)
    
    async def generate_research_report(self, question: str, search_results: List, content_results: List, **kwargs) -> str:
        """
        生成研究报告
        
        Args:
            question: 研究问题
            search_results: 搜索结果列表
            content_results: 网页内容结果列表
            
        Returns:
            结构化研究报告
        """
        research_agent = self.get_agent("research_agent")
        if not isinstance(research_agent, ResearchAgent):
            raise ValueError("研究代理未找到或类型错误")
        
        return await research_agent.generate_research_report(question, search_results, content_results, **kwargs)
    
    async def plan_queries(self, question: str, context: str, max_queries: int = 3) -> List[str]:
        """规划后续检索查询"""
        planner = self.get_agent("query_planner")
        if not isinstance(planner, QueryPlannerAgent):
            raise ValueError("查询规划代理未找到或类型错误")
        return await planner.plan_queries(question, context, max_queries=max_queries)
    
    def add_agent(self, name: str, agent: BaseAgent):
        """添加自定义代理"""
        self.agents[name] = agent
        logger.info(f"已添加代理: {name}")
    
    def remove_agent(self, name: str) -> bool:
        """移除代理"""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"已移除代理: {name}")
            return True
        return False
    
    async def test_all_agents(self) -> Dict[str, bool]:
        """测试所有代理"""
        results = {}
        test_input = "请问什么是人工智能？"
        
        for name, agent in self.agents.items():
            try:
                result = await agent.process(test_input)
                
                results[name] = bool(result and len(result.strip()) > 0)
                logger.info(f"代理 {name} 测试成功")
                
            except Exception as e:
                results[name] = False
                logger.error(f"代理 {name} 测试失败: {e}")
        
        return results
