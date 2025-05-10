# Clickindex

A Python package for indexing documents using LlamaIndex and FAISS with various chunking strategies (fixed, semantic, hierarchical) and embedding methods (OpenAI, HuggingFace, Gemini).

## Installation

Install the package via pip:

```bash
pip install clickindex
```

## Usage

```python
from clickindex import DataIndexing
from langchain_openai import ChatOpenAI

# Initialize the indexer
indexer = DataIndexing()

# Configure parameters
input_data_path = "Input data path"
embedding_method = "huggingface"
chunk_strategy = "hierarchical"
parent_chunk_size = 1024
child_chunk_size = 512
parsing_flag = True
fiass_index_path="Vector store path"
llm = ChatOpenAI()

# Start indexing
index = indexer.start_data_indexing(
    input_data_path=directory_path,
    embedding_method=embedding_method,
    chunk_strategy=chunk_strategy,
    parent_chunk_size=parent_chunk_size,
    child_chunk_size=child_chunk_size,
    parsing_flag=parsing_flag,
    fiass_index_path=fiass_index_path,
    llm=llm
)


## Features

- Supports multiple embedding methods: OpenAI, HuggingFace, Gemini.
- Implements fixed, semantic, and hierarchical chunking strategies.
- Uses FAISS with HNSW for efficient vector storage and search.
- LLM-based parsing for extracting content from tabular and image data in documents.
- Uses FAISS with HNSW for efficient vector storage and search

## Requirements

- Python >= 3.8
- See `requirements.txt` for dependencies.

## License

MIT License. See `LICENSE` for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Contact

For questions, contact [iamprasadsandeep.email@example.com](mailto:iamprasadsandeep.email@example.com).