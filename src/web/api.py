#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web API接口 - 简化版
"""

import asyncio
import contextvars
import json
import logging
import time
import uuid
from flask import Blueprint, request, jsonify, Response, stream_with_context
from ..config import reset_log_request_id, set_log_request_id
from ..rag.pipeline import RAGResult
from ..rag.system import rag_system

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

def parse_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}

def completed_stages_from_state(state: dict) -> list:
    completed = []
    steps = state.get("steps_log", [])
    if state.get("extracted_question") or state.get("question"):
        completed.append("extract")
    if state.get("keywords"):
        completed.append("keywords")
    if state.get("search_results") or any(step.get("action") == "search" for step in steps):
        completed.append("search")
    if state.get("content_results") or any(step.get("action") == "visit" for step in steps):
        completed.append("visit")
    if any(step.get("action") == "reflect" for step in steps):
        completed.append("reflect")
    elif (state.get("final_answer") or state.get("answer")) and "visit" in completed:
        completed.append("reflect")
    if state.get("final_answer") or state.get("answer"):
        completed.append("answer")
    return completed

def active_stage_from_state(state: dict) -> str:
    if state.get("final_answer"):
        return ""
    next_action = state.get("next_action")
    if next_action == "search":
        return "search"
    if next_action == "visit":
        return "visit"
    if next_action == "reflect":
        return "reflect"
    if next_action == "answer":
        return "answer"
    if state.get("keywords"):
        return "search"
    if state.get("extracted_question"):
        return "keywords"
    return "extract"

def result_from_state(question: str, state: dict, processing_time: float, cached: bool = False) -> RAGResult:
    return RAGResult(
        question=question,
        extracted_question=state.get("extracted_question", ""),
        keywords=state.get("keywords", ""),
        answer=state.get("final_answer", ""),
        confidence=state.get("confidence", 0.0),
        processing_time=processing_time,
        cached=cached,
        search_results=state.get("search_results", []),
        content_results=state.get("content_results", []),
        steps_log=state.get("steps_log", []),
        observability_events=state.get("events", []),
        total_steps=state.get("current_step", 0),
        success=state.get("success", bool(state.get("final_answer")))
    )

def async_route(f):
    """异步路由装饰器"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = f.__name__
    return wrapper

@api_bp.route('/ask', methods=['POST'])
@async_route
async def ask_question():
    """核心提问接口"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': '缺少问题参数'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        use_cache = data.get('use_cache', True)
        
        # 执行深度研究
        result = await rag_system.ask(
            question=question,
            use_cache=use_cache
        )
        
        return jsonify({
            'success': True,
            'data': result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"提问失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'success': True,
        'status': 'healthy'
    })

@api_bp.route('/ask/stream', methods=['GET'])
def ask_question_stream():
    """流式研究接口"""
    question = request.args.get('question', '').strip()
    if not question:
        return jsonify({'success': False, 'error': '问题不能为空'}), 400

    use_cache = parse_bool(request.args.get('use_cache'), True)
    request_id = uuid.uuid4().hex[:8]

    async def stream_async():
        start_time = time.time()
        await rag_system.initialize()
        logger.info(f"SSE研究请求开始: {question}")

        if use_cache and rag_system.cache:
            cached_result = rag_system.cache.get_by_namespace("answer", question)
            if cached_result:
                logger.info("SSE研究请求命中缓存")
                cached_result.cached = True
                cached_result.processing_time = time.time() - start_time
                yield sse_event("started", {"question": question, "cached": True})
                cached_data = cached_result.to_dict()
                completed = completed_stages_from_state(cached_data)
                yield sse_event("stage", {"active": "", "completed": completed})
                logger.info(f"SSE研究请求完成，耗时: {cached_result.processing_time:.2f}s")
                yield sse_event("final", {"data": cached_result.to_dict()})
                return

        yield sse_event("started", {"question": question, "cached": False})

        previous_completed = []
        previous_active = None
        previous_step_count = 0
        previous_extracted_question = ""
        previous_keywords = ""
        final_state = None

        async for state in rag_system.workflow.stream(question, use_cache=use_cache):
            if "error" in state:
                yield sse_event("failed", {"error": state["error"]})
                return

            final_state = state
            completed = completed_stages_from_state(state)
            active = active_stage_from_state(state)
            if completed != previous_completed or active != previous_active:
                yield sse_event("stage", {"active": active, "completed": completed})
                previous_completed = completed
                previous_active = active

            extracted_question = state.get("extracted_question", "")
            if extracted_question and extracted_question != previous_extracted_question:
                yield sse_event("activity", {
                    "type": "done",
                    "title": "核心问题",
                    "primary": extracted_question,
                    "detail": ""
                })
                previous_extracted_question = extracted_question

            keywords = state.get("keywords", "")
            if keywords and keywords != previous_keywords:
                yield sse_event("activity", {
                    "type": "done",
                    "title": "检索关键词",
                    "primary": keywords,
                    "detail": ""
                })
                previous_keywords = keywords

            steps = state.get("steps_log", [])
            for step in steps[previous_step_count:]:
                yield sse_event("activity", {"type": "step", "step": step})
            previous_step_count = len(steps)

        if not final_state:
            yield sse_event("failed", {"error": "研究流程未返回结果"})
            return

        processing_time = time.time() - start_time
        result = result_from_state(question, final_state, processing_time)
        if use_cache and rag_system.cache:
            rag_system.cache.set_by_namespace("answer", question, result)
        logger.info(f"SSE研究请求完成，耗时: {processing_time:.2f}s")
        yield sse_event("final", {"data": result.to_dict()})

    def stream_sync():
        token = set_log_request_id(request_id)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        generator = stream_async()
        context = contextvars.copy_context()
        try:
            while True:
                try:
                    yield context.run(loop.run_until_complete, generator.__anext__())
                except StopAsyncIteration:
                    break
        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            logger.info("SSE客户端连接已断开")
        except Exception as e:
            logger.error(f"SSE流式提问失败: {e}")
            yield sse_event("failed", {"error": str(e)})
        finally:
            try:
                context.run(loop.run_until_complete, generator.aclose())
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    context.run(loop.run_until_complete, asyncio.gather(*pending, return_exceptions=True))
                context.run(loop.run_until_complete, loop.shutdown_asyncgens())
            finally:
                loop.close()
                reset_log_request_id(token)

    return Response(
        stream_with_context(stream_sync()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )
