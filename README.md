# Agentic RAG with LangGraph

An advanced Retrieval-Augmented Generation (RAG) system that uses intelligent agents to dynamically select from multiple knowledge bases and validate document relevance before generating responses.

## Overview

This project implements an agentic RAG system using LangGraph, combining multiple specialized vector stores with intelligent routing and document validation. The system automatically selects the appropriate knowledge base for each query and validates retrieved documents before generating answers.

## Key Features

- **Multi-Vector Store Architecture**: Two specialized FAISS vector stores for different document types
  - Foundational Transformer Models (Attention is All You Need, BERT, RoBERTa)
  - Efficient Transformers and RAG Models (ALBERT, DistilBERT, RAG)

- **Intelligent Tool Selection**: Agentic workflow that automatically routes queries to the most relevant retriever

- **Document Relevance Validation**: Built-in grading system to assess whether retrieved documents are relevant to the query

- **Automatic Query Rewriting**: When documents fail relevance checks, queries are automatically rewritten for better results (up to 2 rewrites)

- **State Management**: Uses LangGraph's state management to track conversation flow and rewrite attempts

## Architecture

### Workflow States

1. **Agent**: Decides which retriever tool to use based on the query
2. **Retrieve**: Fetches relevant documents from the selected vector store
3. **Grade Documents**: Validates document relevance using LLM-based scoring
4. **Rewrite**: Transforms queries that didn't yield relevant results
5. **Generate**: Produces final answers using RAG prompt engineering

### Components

- **LLM**: OpenAI GPT-4o-mini for both reasoning and generation
- **Embeddings**: OpenAI embeddings for vector similarity search
- **Vector Store**: FAISS for efficient similarity search
- **State Graph**: LangGraph for orchestrating the agentic workflow
- **Tools**: Custom retriever tools with specialized descriptions for better routing

## Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see dependencies below)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies:
```bash
pip install langgraph langchain langchain-openai langchain-community
pip install faiss-cpu openai python-dotenv pydantic
pip install pymupdf youtube-transcript-api
```

3. Set up environment variables:

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Query

```python
# Initialize the graph
response = graph.invoke({
    "messages": "What are different types of BERT?",
    "rewrites": 0
})["messages"]

# Display the conversation
for msg in response:
    msg.pretty_print()
```

### How It Works

1. **Question**: User asks a question about transformer models
2. **Agent Decision**: The agent selects the appropriate retriever tool
3. **Document Retrieval**: Top 4 most similar documents are retrieved
4. **Relevance Check**: Documents are scored for relevance
5. **Generate or Rewrite**:
   - If relevant → Generate answer
   - If not relevant → Rewrite query and retry (max 2 times)
6. **Final Answer**: User receives a well-informed response

## Configuration

### Adjustable Parameters

- **MAX_REWRITES**: Maximum query rewrite attempts (default: 2)
- **chunk_size**: Document chunk size for splitting (default: 1000)
- **chunk_overlap**: Overlap between chunks (default: 200)
- **search_kwargs["k"]**: Number of documents to retrieve (default: 4)

### Customizing Vector Stores

To add your own documents:

```python
# Load your documents
pdf_paths = [
    "path/to/document1.pdf",
    "path/to/document2.pdf"
]
docs = [PyMuPDFLoader(file_path=path).load()[0] for path in pdf_paths]

# Split and create vector store
split_docs = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Create retriever tool
retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
    "tool_name",
    "Tool description for agent routing"
)
```

## Example Queries

### Direct Match
```python
response = graph.invoke({
    "messages": "What are different types of BERT?",
    "rewrites": 0
})
```
Output: Information about DistilBERT, ALBERT, and other BERT variants

### Ambiguous Query
```python
response = graph.invoke({
    "messages": "How did things change after the big model everyone used in 2018?",
    "rewrites": 0
})
```
The system will:
1. Retrieve initial documents
2. Detect low relevance
3. Rewrite query for clarity
4. Retrieve again with improved query
5. Generate answer about transformer evolution

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI Embeddings |
| Vector Store | FAISS |
| Document Loading | PyMuPDF, LangChain Loaders |
| Framework | LangChain |

## Project Structure

```
.
├── 2.Agentic_RAG_LG.ipynb    # Main Jupyter notebook implementation
├── .env                       # Environment variables (API keys)
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Advanced Features

### Structured Output Validation

The system uses Pydantic models for structured LLM outputs:

```python
class grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")
```

### Message State Management

Uses LangGraph's `add_messages` reducer for proper conversation flow:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrites: int
```

### Conditional Edge Routing

Dynamic workflow routing based on agent decisions and document relevance:

```python
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "retrieve", END: END}
)
```

## Limitations

- Maximum 2 query rewrites before fallback
- Requires OpenAI API access
- Local file paths need to be updated for your system
- Single-turn conversations (no memory checkpointing in current version)

## Future Enhancements

- [ ] Add conversation memory with MemorySaver checkpointing
- [ ] Support for additional document loaders (web, arXiv, YouTube)
- [ ] Multi-turn conversation support
- [ ] Streaming responses
- [ ] Custom evaluation metrics for retrieval quality
- [ ] Support for local LLMs

## Troubleshooting

### Common Issues

**ImportError for langgraph or langchain**
```bash
pip install --upgrade langgraph langchain langchain-openai
```

**FAISS installation issues**
```bash
# For CPU-only environments
pip install faiss-cpu

# For GPU support
pip install faiss-gpu
```

**OpenAI API errors**
- Verify your API key in `.env`
- Check your OpenAI account has sufficient credits
- Ensure you're using a supported model

## Contributing

Contributions are welcome! Please ensure your code follows these guidelines:
- Add docstrings to all functions
- Include type hints where appropriate
- Test with multiple query types
- Update documentation for new features

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Implements concepts from various transformer research papers (BERT, ALBERT, DistilBERT, RAG)
- Uses OpenAI's GPT models for reasoning and generation

---

**Note**: This is a research and educational project demonstrating advanced RAG techniques with agentic workflows. For production use, consider adding error handling, logging, monitoring, and security measures.
