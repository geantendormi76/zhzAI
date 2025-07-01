**English** | [中文 (Chinese)](./README.md)
***
# zhzAI - Your Personal, Local AI Brain

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GitHub issues](https://img.shields.io/github/issues/geantendormi76/zhzAI)](https://github.com/geantendormi76/zhzAI/issues)
[![GitHub forks](https://img.shields.io/github/forks/geantendormi76/zhzAI)](https://github.com/geantendormi76/zhzAI/network)
[![GitHub stars](https://img.shields.io/github/stars/geantendormi76/zhzAI)](https://github.com/geantendormi76/zhzAI/stargazers)

---

> **"Have you ever been overwhelmed by a sea of local documents, struggling to find a single piece of information? Are you concerned about the privacy risks of uploading sensitive files to the cloud?"**
>
> **zhzAI Smart Assistant** was born to solve these pain points. It is a **secure, intelligent, and completely local "Personal AI Brain"** that runs entirely on your own computer. It can deeply read and understand all the documents in your designated folders and converse with you in the most natural way to help you quickly find information and complete your work tasks.

## Core Philosophy: Local-First, Secure & Reliable, Intelligence-Driven

**zhzAI** aims to build a next-generation knowledge management and interaction system that operates entirely within the user's local environment, free from cloud dependencies. Our architectural design adheres to three core principles:

*   🔒 **Data Sovereignty**: All your documents, queries, and interaction logs **never leave your local machine**, fundamentally eliminating the risk of cloud data breaches. **Fully offline capable**.
*   ⚡ **Peak Performance**: Through hardware-software co-optimization (HAL), we fully leverage the heterogeneous computing power of modern CPUs and GPUs to achieve low-latency, high-throughput real-time interaction. **It runs smoothly even on standard office computers**.
*   🧠 **Emergent Intelligence**: We believe that true intelligence is not the victory of a single algorithm, but the inevitable result of multiple specialized modules working in concert. Our system has evolved from "brute-force parallelism" to "strategic intelligence," striving for intelligence and efficiency in every step.

## ✨ Technical Highlights

zhzAI is not just a simple RAG application; it is a meticulously designed, end-to-end intelligent framework.

<details>
<summary><b>1. Offline Data Ingestion Pipeline</b> - <i>Click to expand</i></summary>

> This pipeline is automatically triggered by a background file monitoring service, responsible for transforming unstructured raw documents into structured, searchable knowledge.

*   **Intelligent Parsing**: A self-developed dispatch parser capable of handling various formats, including `.docx`, `.xlsx`, `.pdf`, `.md`, and `.html`. We don't just extract plain text; we perform deep analysis to preserve **structured information** (like heading levels, lists, and code blocks), with a special focus on **accurately converting tables into Markdown format**, laying a solid foundation for structured Q&A.
*   **High-Fidelity Semantic Chunking**: Abandoning traditional fixed-size chunking, we segment documents based on their **natural semantic boundaries**. Using a **dynamic merging and splitting** strategy, we ensure the contextual integrity of each knowledge chunk. Each chunk "knows" its chapter context (e.g., `Chapter 1 -> Section 1.2`), enabling precise, in-document filtering.
*   **Multi-vector Hybrid Indexing**: We build a dual index for every knowledge chunk:
    *   **Dense Vector Index**: Utilizes advanced text embedding models to convert the "semantics" of each chunk into high-dimensional vectors, stored in a local **ChromaDB** instance.
    *   **Sparse Keyword Index**: Employs the classic **BM25** algorithm to create an efficient inverted index of keywords for each chunk, ensuring absolute precision when searching for specific terms or names.

</details>

<details>
<summary><b>2. Online Real-time Query Engine</b> - <i>Click to expand</i></summary>

> When a user initiates a query, our engine launches a sophisticated, multi-stage, multi-path retrieval process designed to maximize the "breadth" and "precision" of the recall.

*   **LLM-Powered Intent Planner**: The user's natural language query is first sent to our **local Large Language Model (LLM) core**. The LLM acts as the "commander-in-chief," generating a structured JSON execution plan that shifts the strategy from "finding a needle in a haystack" to a "targeted strike."
*   **Multi-Path Parallel Retrieval**: Based on the LLM's plan, the system initiates parallel retrieval requests to three index stores: **vector retrieval**, **keyword retrieval**, and **graph-enhanced retrieval** (architecturally reserved, currently disabled to accommodate lightweight models).
*   **"Small-to-Big" Retrieval**: The system indexes semantically focused "small chunks" but returns the "large chunks" they belong to upon a hit. This strategy balances retrieval precision with the contextual completeness needed for generation.
*   **Multi-Stage Fusion & Re-ranking**:
    *   **Stage 1: RRF Fusion**: Uses the efficient **Reciprocal Rank Fusion (RRF)** algorithm to merge results from multiple retrieval paths into a single, diverse, and unbiased candidate list.
    *   **Stage 2: Cross-Encoder Re-ranking**: Employs a dedicated **Cross-Encoder** model to perform a deep semantic relevance calculation on the candidate list, outputting a highly accurate final ranking.
*   **Context-Aware Generation**: Finally, the highest-quality knowledge chunks, filtered through multiple layers, are submitted to our local LLM core. Our meticulously designed prompts instruct the LLM to generate answers **strictly and solely based on the provided context**, fundamentally suppressing model hallucinations.

</details>

<details>
<summary><b>3. Core Tech Stack</b> - <i>Click to expand</i></summary>

*   **Local LLM Core**: Based on the high-performance **GGUF** inference framework, running a deeply optimized open-source large language model (e.g., **Qwen 1.7B**).
*   **AI Service Framework**: FastAPI
*   **Vector Database & Metadata Filtering**: ChromaDB
*   **Keyword Retrieval**: BM25s
*   **Cross-Encoder & Embedding Models**: Based on SentenceTransformers
*   **Hardware Acceleration**: NVIDIA CUDA / cuBLAS

</details>

## 🎯 Vision & Call for Collaboration

My vision is to make **zhzAI** a truly useful AI tool for everyday office workers. It shouldn't require complex command-line operations; users should be able to install and use it through a simple graphical interface.

**However, I've hit a major roadblock:**

I only started learning about computers in mid-January 2025 and am not a professional programmer. Although I have used AI to complete the vast majority of the project's core logic and architecture, I'm facing significant difficulties in deploying this complex system to the Windows platform.

### Specifically, my core difficulties are:

*   **Extremely Complex Environment Dependencies**
    *   The project relies on libraries like `torch`, `llama-cpp-python`, and `duckdb`, which are extremely challenging to compile, package, and isolate on Windows.
    *   **My Implementation Details**: For performance, I'm not solely relying on `llama-cpp-python`. The LLM inference uses official `llama.cpp` compiled binaries, while the embedding model is implemented through my own compilation of `llama-cpp-python`. This adds to the packaging complexity.
    *   **Architectural Evolution**: The project has successfully been decoupled from `Dagster`, now featuring a pure Python automated data ingestion pipeline with smart directory monitoring.

*   **The Biggest Bottleneck: Packaging & Distribution**
    *   I lack the experience with tools like `PyInstaller` or `Nuitka` to package the entire Python project into a single executable file (`.exe`).
    *   **Solutions I've Tried**: I have attempted using `PyInstaller` and `Nuitka`, but they failed to resolve the low-level dependency issues of `llama.cpp`. I have abandoned this path for now and am considering an **embedded Python environment**, which also requires professional engineering experience.

*   **The "Last Mile" of Windows Platform Migration**
    *   **The Core Issue Today**: I don't have the programming expertise to efficiently debug and adapt the project for the Windows system.
    *   **AI Assistance Limitations**: Due to the **high degree of similarity** in the logic and structure of the project's internal modules (like the RAG service, Agent service, LLM interface, etc.), AI-powered tools often get confused by the context and suffer from a high rate of hallucination. They fail to achieve the "pixel-perfect" alignment needed for migration, leading to severely subpar test results on Windows.

*   **Code Engineering & Refactoring**
    *   I need experienced developers to help me **review and decouple some core modules** to improve maintainability and extensibility.
    *   We need to establish an **automated testing pipeline** to ensure that code changes do not break existing functionality.

I firmly believe this project is mature and powerful in its functionality and architecture, but it's currently stuck on these critical engineering challenges. I need a partner who can **deeply understand the entire project architecture**, not just modify isolated code snippets. If you are interested in solving these challenges, I eagerly await your help!

## 🔮 Future Roadmap

The goal for **zhzAI** extends far beyond being a powerful local RAG tool. We are committed to building it into an **extensible, small-model, end-to-end, offline personal intelligence hub**. We believe that with the community's joint effort, the future of zhzAI is limitless.

Our next steps will revolve around these four core areas:

### 1. **Memory Architecture**

This is our highest priority milestone. We will build an **end-to-end, offline long-term memory system** for zhzAI, evolving it from a passive knowledge query tool into a proactive personal assistant that truly understands the user.

*   **Core Capabilities**:
    *   **Proactive Memory**: Users can ask the assistant to remember any piece of information via natural language (e.g., "Remember that John owes me $20, due next Monday"), including but not limited to short texts (2-20 chars), images, and videos.
    *   **Intelligent Extraction & Task-ification**: The system will automatically extract key entities (people, events, times) and recognize intents that require reminders, creating tasks automatically.
    *   **Millisecond-level Feedback**: The memory system's queries and feedback will achieve millisecond-level response times, laying the performance foundation for future voice interaction.
*   **Open Architectural Challenge**: We are actively exploring whether to build a separate, highly-optimized database architecture for the memory system or to implement a unified but logically isolated solution within the existing RAG architecture. We welcome expert advice from the community on this.

### 2. **Voice Interaction**

To deliver a true "smart assistant" experience, we will integrate advanced local voice technologies to enable full-flow voice interaction.

*   **Technical Path**:
    *   Integrate an efficient **local Automatic Speech Recognition (ASR)** model.
    *   Integrate a natural **local Text-to-Speech (TTS)** model.
*   **Design Philosophy**: Responses must be **extremely concise**. For example, when asked "Who owes me money?", the assistant will only reply "John, Mary, and David," without any unnecessary pleasantries. This refined interaction is key to an excellent voice assistant experience.

### 3. **RAG Architecture Enhancement**

We will continue to iterate and enhance our core RAG capabilities to make them smarter and more powerful.

*   **Graph-Enhanced Retrieval**: Activate and optimize the knowledge graph retrieval path, which is already architecturally reserved.
*   **Interactive Verification & Clarification**: Activate the reserved "interactive verification" interface, allowing the LLM to proactively ask clarifying questions when faced with ambiguous results.
*   **Hybrid Data Source Support**: Explore the ability to seamlessly integrate structured data (like databases and APIs) with unstructured documents.

### 4. **Utilizing GBNF for Function Calling**

We will further leverage GBNF (GGML-based BNF) grammar constraints to give our small local model more reliable and powerful "hands and feet," enabling it to perform complex tasks stably.

*   **Reliable Tool Calls**: Force the LLM to generate perfectly formatted JSON for tool calls using GBNF, completely solving the instability issues of small models with function calling.
*   **Complex Task Flow Orchestration**: Design and implement more complex GBNF grammars to enable the small model to plan and output multi-step task flows with logical conditions (e.g., IF/ELSE).
*   **Empowering Developers**: We will polish this GBNF application framework into an easily extensible module, allowing community developers to effortlessly add new tools and capabilities to zhzAI.

---

## 📞 Contact Me

I am very excited to communicate and collaborate with developers interested in this project. If you have any ideas, suggestions, or are willing to contribute your skills, please feel free to contact me through the following channels:

*   **X :** [@geantendormi76]
*   **Email:** geantendormi76@gmail.com
*   **GitHub Issues:** For specific technical questions or feature suggestions, you are also welcome to start a discussion on our [Issues page](https://github.com/geantendormi76/zhzAI/issues).

Let's build a smarter, more accessible local AI assistant together!

---

## 🚀 Getting Started

### 1. System Requirements
*   **Basic Configuration (Works for Everyone)**:
    *   OS: Windows 10 / 11 (64-bit) or Linux
    *   RAM: 8 GB or more
    *   CPU: A mainstream processor from the last 5 years.
*   **Hardware Accelerated Configuration (For a Blazing-Fast Experience)**:
    *   **NVIDIA GPU** with **≥ 4 GB of VRAM** (e.g., GTX 1660, RTX 2060/3060/4060 or higher).

### 2. Running the Services
This project requires three core services to be running for Q&A functionality:
1.  **Embedding Model Service**: Responsible for converting text into vectors.
2.  **Local LLM Service**: Provides large language model inference capabilities.
3.  **RAG API Service**: Receives queries and orchestrates the entire RAG pipeline.

*Detailed installation and startup instructions will be provided in the contributor's guide (to be written).*

## 🤝 How to Contribute

We warmly welcome contributions of all kinds!

1.  **Report Issues**: If you find any bugs or have feature suggestions, please submit them through the [Issues](https://github.com/geantendormi76/zhzAI/issues) page.
2.  **Submit Pull Requests**:
    *   Fork the repository.
    *   Create a new branch in your fork (`git checkout -b feature/your-feature-name`).
    *   Make your changes and commit them (`git commit -m 'Add some feature'`).
    *   Push your branch to GitHub (`git push origin feature/your-feature-name`).
    *   Create a Pull Request.

## 📝 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License**.

This means:
*   **Attribution (BY)**: You must give appropriate credit.
*   **NonCommercial (NC)**: You may not use the material for commercial purposes.
*   **ShareAlike (SA)**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

**The core reason for choosing this license is my hope that this project maintains its original spirit—to serve the community and individual users, not to be exploited for commercial profit.**

---

Thank you for your interest and support!