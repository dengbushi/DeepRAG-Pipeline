#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代理管理器
"""

import logging
from typing import Dict, Any, Optional
from .base import BaseAgent
from .question_extractor import QuestionExtractorAgent
from .keyword_extractor import KeywordExtractorAgent
from .qa_agent import QAAgent
from .research_agent import ResearchAgent
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
                "qa_agent": QAAgent(self.llm),
                "research_agent": ResearchAgent(self.llm)
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
    
    async def answer_question(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """回答问题"""
        agent = self.get_agent("qa_agent")
        if not agent:
            raise ValueError("问答代理未找到")
        return await agent.process(question, context=context, **kwargs)
    
    async def get_answer_with_confidence(self, question: str, context: Optional[str] = None) -> dict:
        """获取带置信度的答案"""
        qa_agent = self.get_agent("qa_agent")
        if not isinstance(qa_agent, QAAgent):
            raise ValueError("问答代理类型错误")
        return await qa_agent.answer_with_confidence(question, context)
    
    async def generate_research_report(self, question: str, search_results: list, content_results: list) -> str:
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
        
        return await research_agent.generate_research_report(question, search_results, content_results)
    
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
                if name == "qa_agent":
                    result = await agent.process(test_input, context="人工智能是模拟人类智能的技术")
                else:
                    result = await agent.process(test_input)
                
                results[name] = bool(result and len(result.strip()) > 0)
                logger.info(f"代理 {name} 测试成功")
                
            except Exception as e:
                results[name] = False
                logger.error(f"代理 {name} 测试失败: {e}")
        
        return results
