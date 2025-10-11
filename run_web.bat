@echo off
chcp 65001 >nul
echo ====================================
echo    Agentic RAG System - Web Mode
echo ====================================
echo.

echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo 安装/更新依赖包...
pip install -r requirements.txt

echo.
echo 检查配置...
if not exist "config.json" (
    echo 警告: 未找到config.json，请先配置DeepSeek API密钥
    echo 你可以：
    echo 1. 复制config.json.template为config.json并编辑
    echo 2. 设置环境变量 DEEPSEEK_API_KEY
    echo.
    set /p choice="是否继续启动？(y/N): "
    if /i not "%choice%"=="y" exit /b 1
)

echo.
echo 🚀 启动Web服务器...
echo 📍 访问地址: http://127.0.0.1:5000
echo 🎯 聊天页面: http://127.0.0.1:5000/chat
echo ⚙️ 管理页面: http://127.0.0.1:5000/admin
echo.
echo 按 Ctrl+C 停止服务器
echo ====================================

python app.py web

echo.
echo 服务器已停止
pause
