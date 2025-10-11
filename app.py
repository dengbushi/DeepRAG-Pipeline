#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepRAG Pipeline - 主程序入口
Web服务器模式
"""

import sys
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.web.app import create_app
from src.config import config_manager

def run_web_server(host='127.0.0.1', port=5000, debug=False):
    """运行Web服务器"""
    app = create_app()
    
    print(f"🚀 启动Web服务器...")
    print(f"📍 访问地址: http://{host}:{port}")
    print(f"🎯 聊天页面: http://{host}:{port}/chat")
    print(f"⚙️ 管理页面: http://{host}:{port}/admin")
    print(f"🔧 调试模式: {'开启' if debug else '关闭'}")
    print("-" * 50)
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Web服务器已停止")
    except Exception as e:
        print(f"❌ Web服务器启动失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DeepRAG Pipeline - 智能深度研究系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python app.py                        # 启动Web界面 (默认端口5000)
  python app.py --host 0.0.0.0         # 允许外部访问
  python app.py --port 8080            # 指定端口
  python app.py --debug                # 启用调试模式
  python app.py --config custom.json   # 使用自定义配置
        """
    )
    
    parser.add_argument('--host', default='127.0.0.1', 
                       help='Web服务器主机地址 (默认: 127.0.0.1)')
    
    parser.add_argument('--port', type=int, default=5000, 
                       help='Web服务器端口 (默认: 5000)')
    
    parser.add_argument('--debug', action='store_true', 
                       help='启用调试模式')
    
    parser.add_argument('--config', 
                       help='指定配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config_manager.config_file = Path(args.config)
        config_manager.config = config_manager._load_config()
    
    # 更新调试配置
    if args.debug:
        config_manager.update_config({'debug': True})
    
    try:
        run_web_server(
            host=args.host, 
            port=args.port, 
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n👋 程序已退出")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        sys.exit(1)
    
    # 运行主程序
    main()
