#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web API接口 - 简化版
"""

import asyncio
import logging
from flask import Blueprint, request, jsonify
from ..rag.system import rag_system

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

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
