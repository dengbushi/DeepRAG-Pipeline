#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索缓存系统
"""

import time
import hashlib
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from ..config import CacheConfig

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """缓存条目"""
    data: Any
    timestamp: float
    ttl: int
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.timestamp > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SearchCache:
    """搜索缓存"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.access_times: Dict[str, float] = {}
        
        if not config.enabled:
            logger.info("缓存已禁用")
        else:
            logger.info(f"缓存已启用，TTL: {config.ttl}s, 最大大小: {config.max_size}")
    
    def _generate_key(self, query: str, **kwargs) -> str:
        """生成缓存键"""
        # 将查询和参数组合成字符串
        key_data = f"{query}_{str(sorted(kwargs.items()))}"
        # 使用MD5生成短键
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_by_namespace(self, namespace: str, query: str, **kwargs) -> Optional[Any]:
        """按命名空间获取缓存"""
        return self.get(f"{namespace}:{query}", **kwargs)

    def set_by_namespace(self, namespace: str, query: str, data: Any, ttl: Optional[int] = None, **kwargs):
        """按命名空间设置缓存"""
        self.set(f"{namespace}:{query}", data, ttl=ttl, **kwargs)
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """获取缓存"""
        if not self.config.enabled:
            return None
        
        key = self._generate_key(query, **kwargs)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # 检查是否过期
        if entry.is_expired():
            self.remove(key)
            logger.debug(f"缓存过期: {query}")
            return None
        
        # 更新访问时间
        self.access_times[key] = time.time()
        logger.debug(f"缓存命中: {query}")
        return entry.data
    
    def set(self, query: str, data: Any, ttl: Optional[int] = None, **kwargs):
        """设置缓存"""
        if not self.config.enabled:
            return
        
        key = self._generate_key(query, **kwargs)
        ttl = ttl or self.config.ttl
        
        # 检查缓存大小限制
        if len(self.cache) >= self.config.max_size:
            self._evict_oldest()
        
        # 添加缓存条目
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=ttl
        )
        
        self.cache[key] = entry
        self.access_times[key] = time.time()
        
        logger.debug(f"缓存设置: {query}")
    
    def remove(self, key: str):
        """移除缓存条目"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_oldest(self):
        """移除最旧的缓存条目"""
        if not self.access_times:
            return
        
        # 找到最旧的条目
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.remove(oldest_key)
        logger.debug(f"移除最旧缓存: {oldest_key}")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("缓存已清空")
    
    def cleanup_expired(self):
        """清理过期缓存"""
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self.remove(key)
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_entries = len(self.cache)
        expired_count = sum(1 for entry in self.cache.values() if entry.is_expired())
        
        return {
            "enabled": self.config.enabled,
            "total_entries": total_entries,
            "expired_entries": expired_count,
            "valid_entries": total_entries - expired_count,
            "max_size": self.config.max_size,
            "ttl": self.config.ttl
        }
    
    def get_cache_info(self) -> List[Dict[str, Any]]:
        """获取缓存详细信息"""
        info = []
        
        for key, entry in self.cache.items():
            info.append({
                "key": key,
                "timestamp": entry.timestamp,
                "ttl": entry.ttl,
                "expired": entry.is_expired(),
                "age": time.time() - entry.timestamp
            })
        
        return sorted(info, key=lambda x: x["timestamp"], reverse=True)
