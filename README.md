# Agentic RAG System (FAISS, Multi-Source Retrieval)

## 📌 Overview
This project implements an **Agentic Retrieval-Augmented Generation (RAG) system** designed to handle complex queries with intelligent rewriting, grading, and retrieval.  
The system integrates **FAISS vector stores** for efficient similarity search, supports **multi-turn conversations**, and incorporates external tools such as **arXiv paper search, YouTube transcripts, and web search**.  

Unlike traditional RAG pipelines, this implementation demonstrates **agentic AI capabilities** by dynamically routing queries through a graph-based workflow, ensuring accurate and context-aware responses while optimizing efficiency.  

---

## ✨ Features
- **FAISS-based retrieval** from multiple sources (PDFs, arXiv papers, YouTube transcripts, web pages).  
- **Agentic query processing**:
  - Query rewriting for better retrieval.
  - Query grading to decide if refinement is needed.
  - Tool routing for external lookups.  
- **Multi-turn persistence** with automated state management.  
- **Dynamic tool integration** via retrievers and APIs.  
- **Hallucination reduction** by limiting query rewrites (max 2 iterations).  

---

## 🛠️ Tech Stack
- **LLM:** OpenAI GPT (gpt-4o-mini)  
- **Frameworks:** LangGraph, LangChain  
- **Vector Search:** FAISS + OpenAIEmbeddings  
- **Tools & APIs:**  
  - `ArxivLoader` – research papers  
  - `PyMuPDFLoader` – PDFs  
  - `WebBaseLoader` – websites  
  - `YouTubeTranscriptAPI` – transcripts  
- **Utilities:** RecursiveCharacterTextSplitter, MemorySaver (checkpointing), dotenv  

---

