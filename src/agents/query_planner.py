#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
from typing import List

from .base import BaseAgent
from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)


class QueryPlannerAgent(BaseAgent):
    def __init__(self, llm: BaseLLM):
        super().__init__(
            llm=llm,
            name="query_planner",
            role_description="你是研究规划助手，负责分析已有证据并提出下一轮最有价值的检索问题。",
            task_description="基于用户原始问题、已有上下文和检索历史，生成一个 JSON 数组，包含 0 到 3 个下一步检索查询。若信息已充分，返回空数组 []。",
        )

    async def process(self, input_data: str, **kwargs) -> str:
        context = kwargs.get("context", "")
        return await self.plan_queries(input_data, context=context)

    async def plan_queries(self, original_query: str, context: str, max_queries: int = 3) -> List[str]:
        prompt = (
            f"原始问题：{original_query}\n\n"
            f"已有上下文：\n{context[:6000]}\n\n"
            f"请输出一个 JSON 数组，包含 0 到 {max_queries} 个后续搜索查询。要求："
            "\n1. 查询必须互补，避免重复"
            "\n2. 应优先覆盖证据缺口"
            "\n3. 适合搜索引擎检索"
            "\n4. 必须检查原始问题中的专有名词、英文术语和枚举项是否已覆盖；如未覆盖，优先生成对应查询"
            "\n5. 如果信息已经充分，直接返回 []"
        )
        messages = self.create_messages(prompt)
        response = await self.invoke_llm(messages, temperature=0.2)
        return self._parse_json_array(response, max_queries)

    def _parse_json_array(self, response: str, max_queries: int) -> List[str]:
        try:
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                data = json.loads(response[start : end + 1])
                if not isinstance(data, list):
                    raise ValueError("query planner 响应不是 JSON 数组")
                queries = [str(item).strip() for item in data if str(item).strip()]
                return queries[:max_queries]
        except Exception as e:
            raise ValueError(f"解析 query planner 响应失败: {e}") from e

        raise ValueError("query planner 响应缺少 JSON 数组")
