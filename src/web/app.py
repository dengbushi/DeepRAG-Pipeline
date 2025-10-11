#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web应用 - 简化版
"""

import logging
from flask import Flask, render_template, jsonify, redirect
from flask_cors import CORS
from .api import api_bp

logger = logging.getLogger(__name__)

def create_app():
    """创建Flask应用"""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # 配置
    app.config['SECRET_KEY'] = 'deepresearch-pipeline-secret'
    app.config['JSON_AS_ASCII'] = False
    
    # 启用CORS
    CORS(app)
    
    # 注册API蓝图
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        """主页 - 直接跳转到聊天"""
        return redirect('/chat')
    
    @app.route('/chat')
    def chat():
        """聊天页面"""
        return render_template('chat.html')
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': '页面未找到'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': '服务器内部错误'}), 500
    
    logger.info("="*50)
    logger.info("DeepRAG Pipeline Web服务器")
    logger.info("="*50)
    
    return app
