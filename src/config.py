#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
支持多种配置源：环境变量、配置文件、命令行参数
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field, fields

@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"
    max_tokens: int = 1024
    temperature: float = 0.1
    timeout: int = 30

@dataclass
class SearchConfig:
    """搜索配置"""
    engine: str = "deep_search"
    max_results_per_query: int = 10
    retrievers: List[str] = field(default_factory=lambda: ["serper"])
    serper_api_key: Optional[str] = None
    max_rounds: int = 2  # 最多搜索轮次（每轮包含SEARCH→VISIT→REFLECT）
    max_total_queries: int = 6
    max_urls_to_scrape_per_query: int = 3
    token_budget: int = 50000
    context_top_k: int = 8
    context_max_chars: int = 12000
    report_context_max_chars: int = 9000
    evidence_similarity_threshold: float = 0.35
    chunk_size: int = 900
    chunk_overlap: int = 120
    max_planned_queries: int = 3
    # 步骤问题配置
    min_step_questions: int = 0  # 最少步骤问题数（0表示可以提前结束）
    max_step_questions: int = 5  # 最多步骤问题数
    allow_early_termination: bool = True  # 允许LLM判断提前结束

@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    ttl: int = 3600  # 1小时
    max_size: int = 1000
    search_ttl: int = 1800
    content_ttl: int = 3600
    planning_ttl: int = 1800
    answer_ttl: int = 3600

@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "logs/rag_system.log"

@dataclass
class AppConfig:
    """应用配置"""
    llm: LLMConfig
    search: SearchConfig
    cache: CacheConfig
    log: LogConfig
    debug: bool = False
    report: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """从字典创建配置"""
        search_data = cls._normalize_search_config(data.get('search', {}))
        return cls(
            llm=LLMConfig(**cls._filter_dataclass_kwargs(LLMConfig, data.get('llm', {}))),
            search=SearchConfig(**search_data),
            cache=CacheConfig(**cls._filter_dataclass_kwargs(CacheConfig, data.get('cache', {}))),
            log=LogConfig(**cls._filter_dataclass_kwargs(LogConfig, data.get('log', {}))),
            debug=data.get('debug', False),
            report=data.get('report', {})
        )

    @staticmethod
    def _filter_dataclass_kwargs(dataclass_type, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {item.name for item in fields(dataclass_type)}
        return {key: value for key, value in (values or {}).items() if key in allowed}

    @classmethod
    def _normalize_search_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(values or {})

        if 'max_results' in normalized and 'max_results_per_query' not in normalized:
            normalized['max_results_per_query'] = normalized['max_results']
        if 'max_urls_per_step' in normalized and 'max_urls_to_scrape_per_query' not in normalized:
            normalized['max_urls_to_scrape_per_query'] = normalized['max_urls_per_step']
        if 'context_max_chars' in normalized and 'report_context_max_chars' not in normalized:
            normalized['report_context_max_chars'] = min(int(normalized['context_max_chars']), 9000)

        normalized.pop('min_rounds', None)
        normalized.pop('jina_api_key', None)

        return cls._filter_dataclass_kwargs(SearchConfig, normalized)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> AppConfig:
        """加载配置"""
        # 默认配置
        config_data = {}
        
        # 从配置文件加载
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    config_data.update(file_config)
                logging.info(f"已加载配置文件: {self.config_file}")
            except Exception as e:
                logging.warning(f"加载配置文件失败: {e}")
        
        # 从环境变量覆盖
        env_overrides = self._load_from_env()
        if env_overrides:
            config_data.update(env_overrides)
            logging.info("已应用环境变量配置")
        
        return AppConfig.from_dict(config_data)
    
    def _load_from_env(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_config = {}
        
        # LLM配置
        if os.getenv('DEEPSEEK_API_KEY'):
            env_config['llm'] = {
                'api_key': os.getenv('DEEPSEEK_API_KEY'),
                'provider': os.getenv('LLM_PROVIDER', 'deepseek'),
                'model_name': os.getenv('LLM_MODEL', 'deepseek-chat'),
                'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '1024')),
                'temperature': float(os.getenv('LLM_TEMPERATURE', '0.1'))
            }
        
        search_overrides = {}
        if os.getenv('SEARCH_RETRIEVERS'):
            search_overrides['retrievers'] = [item.strip() for item in os.getenv('SEARCH_RETRIEVERS', '').split(',') if item.strip()]
        if os.getenv('SEARCH_CONTEXT_TOP_K'):
            search_overrides['context_top_k'] = int(os.getenv('SEARCH_CONTEXT_TOP_K', '8'))
        if os.getenv('SEARCH_CONTEXT_MAX_CHARS'):
            search_overrides['context_max_chars'] = int(os.getenv('SEARCH_CONTEXT_MAX_CHARS', '12000'))
        if os.getenv('SEARCH_REPORT_CONTEXT_MAX_CHARS'):
            search_overrides['report_context_max_chars'] = int(os.getenv('SEARCH_REPORT_CONTEXT_MAX_CHARS', '9000'))
        if os.getenv('SEARCH_MAX_PLANNED_QUERIES'):
            search_overrides['max_planned_queries'] = int(os.getenv('SEARCH_MAX_PLANNED_QUERIES', '3'))
        if os.getenv('SEARCH_MAX_TOTAL_QUERIES'):
            search_overrides['max_total_queries'] = int(os.getenv('SEARCH_MAX_TOTAL_QUERIES', '6'))
        if os.getenv('SEARCH_MAX_RESULTS_PER_QUERY'):
            search_overrides['max_results_per_query'] = int(os.getenv('SEARCH_MAX_RESULTS_PER_QUERY', '10'))
        if os.getenv('SEARCH_MAX_URLS_TO_SCRAPE_PER_QUERY'):
            search_overrides['max_urls_to_scrape_per_query'] = int(os.getenv('SEARCH_MAX_URLS_TO_SCRAPE_PER_QUERY', '3'))
        if os.getenv('SEARCH_EVIDENCE_SIMILARITY_THRESHOLD'):
            search_overrides['evidence_similarity_threshold'] = float(os.getenv('SEARCH_EVIDENCE_SIMILARITY_THRESHOLD', '0.35'))
        if search_overrides:
            env_config['search'] = {**env_config.get('search', {}), **search_overrides}

        # 其他环境变量
        if os.getenv('DEBUG'):
            env_config['debug'] = os.getenv('DEBUG').lower() == 'true'
        
        if os.getenv('LOG_LEVEL'):
            env_config['log'] = {'level': os.getenv('LOG_LEVEL')}
        
        return env_config
    
    def _setup_logging(self):
        """设置日志 - 直接操作 root logger 避免 basicConfig 的限制"""
        log_config = self.config.log
        
        # 创建日志目录
        if log_config.file:
            log_path = Path(log_config.file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取 root logger
        root_logger = logging.getLogger()
        
        # 设置日志级别
        root_logger.setLevel(getattr(logging, log_config.level.upper()))
        
        # 创建格式化器
        formatter = logging.Formatter(log_config.format)
        
        # 检查是否已有相同类型的 handler，避免重复添加
        has_console_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) 
                                  for h in root_logger.handlers)
        has_file_handler = any(isinstance(h, logging.FileHandler) 
                               for h in root_logger.handlers)
        
        # 添加控制台 handler（如果还没有）
        if not has_console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 添加文件 handler（如果还没有且配置了文件路径）
        if not has_file_handler and log_config.file:
            file_handler = logging.FileHandler(log_config.file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def save_config(self, config_file: Optional[str] = None):
        """保存配置到文件"""
        file_path = Path(config_file) if config_file else self.config_file
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        logging.info(f"配置已保存到: {file_path}")
    
    def get_config(self) -> AppConfig:
        """获取配置"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        current_dict = self.config.to_dict()
        current_dict.update(updates)
        self.config = AppConfig.from_dict(current_dict)
        logging.info("配置已更新")

# 全局配置实例
config_manager = ConfigManager()
config = config_manager.get_config()
