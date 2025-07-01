# zhzAI - 智能终端大脑项目

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GitHub issues](https://img.shields.io/github/issues/geantendormi76/zhzAI)](https://github.com/geantendormi76/zhzAI/issues)
[![GitHub forks](https://img.shields.io/github/forks/geantendormi76/zhzAI)](https://github.com/geantendormi76/zhzAI/network)
[![GitHub stars](https://img.shields.io/github/stars/geantendormi76/zhzAI)](https://github.com/geantendormi76/zhzAI/stargazers)


---

## 🌟 项目简介 (Project Introduction)

**zhzAI** 是一个功能强大且高度集成的**智能终端大脑框架**，旨在为普通办公用户（尤其是非技术背景的“电脑小白”）提供一个开箱即用的、具备高级 RAG (Retrieval-Augmented Generation) 能力的本地AI助手。

本项目的核心理念是**将复杂的AI技术栈封装成一个易于部署和使用的桌面应用程序**，让每个人都能在自己的电脑上利用本地知识库和大型语言模型（LLM）来提升工作效率。

### ✨ 核心功能

*   **混合式RAG引擎**: 整合了向量检索、关键词检索 (BM25) 和知识图谱 (KG) 检索，并通过智能融合与重排序技术，为用户问题提供最相关的上下文。
*   **多格式文档处理**: 自动化的数据处理流水线 (`zhz_rag_pipeline_dagster`)，能够摄入和解析包括 `.txt`, `.md`, `.docx`, `.pdf`, `.xlsx`, `.html` 在内的多种文档格式。
*   **智能Agent架构**: 采用 Manager-Worker Agent 模式 (`agent_orchestrator_service.py`)，能够理解用户意图、自主规划任务并调用多种工具（如RAG查询、Web搜索、Excel分析、计算器等）。
*   **本地化模型支持**: 深度集成了本地LLM（如 Qwen3-GGUF）和嵌入模型，确保用户数据的隐私和安全。
*   **任务管理系统**: 内置了基于 `APScheduler` 的任务管理和提醒功能，可以将对话中的待办事项转化为实际的提醒。
*   **跨平台代理**: 设计了本地代理程序 (`local_agent_agent.py`)，旨在作为 Windows 系统上的执行端点，处理如 Excel 操作、发送桌面通知等任务。

## 🎯 项目愿景与当前挑战

**我的愿景**是让 **zhzAI** 成为一个真正服务于普通打工人的AI工具，它不需要复杂的命令行操作，用户只需通过简单的图形界面就能完成安装和使用，享受AI带来的便利。

**当前我遇到了一个巨大的挑战**：我并非专业的程序员，虽然我已经完成了项目绝大部分的核心逻辑和架构设计，但在将这个复杂的系统**打包成一个能在 Windows 上简单运行的应用程序**时遇到了重重困难。主要问题包括：

1.  **环境依赖复杂**：项目依赖 `torch`, `llama-cpp-python`, `duckdb` 等库，在 Windows 上的打包和环境隔离极具挑战。
2.  **跨平台通信**：项目设计的 Linux/WSL (主服务) 与 Windows (本地代理) 之间的通信机制，在实际部署时难以配置。
3.  **打包与分发**：我缺乏使用 `PyInstaller`, `Nuitka` 或其他工具将整个 Python 项目打包成单个可执行文件 (`.exe`) 的经验。

我坚信这个项目已经非常成熟和强大，但正被“最后一公里”的工程化问题所困。

## 寻求合作 (Seeking Collaboration)

**我在此真诚地邀请有共同兴趣和技术能力的开发者加入这个项目！**

我希望您是：
*   对 RAG、本地LLM、Agent 技术充满热情。
*   熟悉 Python 项目的工程化，尤其是在 **Windows 环境下的打包和部署**。
*   有 `PyInstaller`, `cx_Freeze`, `Nuitka` 或相关打包工具的使用经验。
*   了解如何处理复杂的第三方库依赖（特别是 `torch` 和 `llama-cpp-python`）。
*   或者，您只是单纯地觉得这个项目很有趣，愿意贡献您的任何一份力量！

我非常乐意与您分享项目的所有设计思路、架构细节和未来规划。让我们一起努力，将 **zhzAI** 带给成千上万需要它的普通用户！

## 🚀 如何开始 (Getting Started)

以下是在开发环境中运行本项目的基本步骤。

### 1. 环境要求
*   Python 3.10+
*   Git
*   (推荐) Linux 环境或 WSL2 (Windows Subsystem for Linux 2)

### 2. 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/geantendormi76/zhzAI.git
    cd zhzAI
    ```

2.  **创建并激活虚拟环境:**
    ```bash
    # For Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **安装依赖:**
    项目分为多个模块，请根据需要安装。核心 RAG 框架的依赖位于 `zhz_rag` 目录下。
    ```bash
    # 安装核心RAG模块
    pip install -e ./zhz_rag

    # 安装数据处理流水线模块
    pip install -e ./zhz_rag_pipeline_dagster
    ```
    *注意：您可能需要根据 `requirements.txt` (如果提供) 或各个 `setup.py` 中的 `install_requires` 来安装所有依赖。*

### 3. 配置
*   请复制项目根目录下的 `.env.example` 文件为 `.env`。
*   根据 `.env` 文件中的注释，配置您的本地模型路径、API密钥（如果需要）等信息。

### 4. 运行服务
本项目包含多个独立的服务，请根据需要启动：
*   **本地LLM服务**: `python local_llm_service.py`
*   **Agent编排服务**: `python agent_orchestrator_service.py` （预留的，目前只是一个RAG框架）
*   ... (请在此补充其他服务的启动命令)

## 🤝 如何贡献 (How to Contribute)

我们非常欢迎任何形式的贡献！

1.  **报告问题 (Issues)**: 如果您在使用中发现任何 Bug，或者有任何功能建议，请通过 [Issues](https://github.com/geantendormi76/zhzAI/issues) 页面提交。
2.  **提交代码 (Pull Requests)**:
    *   请先 Fork 本仓库。
    *   在您的 Fork 中创建一个新的分支 (`git checkout -b feature/your-feature-name`)。
    *   进行修改并提交 (`git commit -m 'Add some feature'`)。
    *   将您的分支推送到 GitHub (`git push origin feature/your-feature-name`)。
    *   创建一个 Pull Request。

**我们尤其欢迎在以下领域的贡献：**
*   **Windows 打包脚本和工作流。**
*   **简化安装和配置流程。**
*   **代码优化和性能提升。**
*   **前端图形界面 (GUI) 的开发。**

## 📝 开源许可证 (License)

本项目采用 **知识共享署名-非商业性使用-相同方式共享 4.0 国际 (CC BY-NC-SA 4.0) 许可协议**。

这意味着：
*   **署名 (BY)**: 您必须给出适当的署名，提供指向本许可协议的链接，并指出是否对原始内容进行了更改。
*   **非商业性使用 (NC)**: 您不得将本作品用于商业目的。
*   **相同方式共享 (SA)**: 如果您对本作品进行修改、转换或二次创作，您必须以与本作品相同的许可协议分发您的贡献。

**选择此许可证的核心原因，是我希望这个项目能保持其初心——服务于社区和个人用户，而不是被用于商业牟利。**

---

感谢您的关注与支持！让我们一起构建一个更智能、更易用的本地AI助手！