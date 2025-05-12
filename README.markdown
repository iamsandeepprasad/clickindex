# Clickindex

A Python package for indexing documents using LlamaIndex and FAISS with various chunking strategies (fixed, semantic, hierarchical) and embedding methods (OpenAI, HuggingFace, Gemini).

## Installation

Install the package via pip:

```bash
pip install -i https://test.pypi.org/simple/ clickindex```

## Usage

```python
from clickindex import DataIndexing
from langchain_openai import ChatOpenAI

# Initialize the indexer
indexer = DataIndexing()

# Configure parameters
input_data_path = "Your Input Path"
embedding_method = "huggingface"
chunk_strategy = "hierarchical"
buffer_size =1
breakpoint_threshold=95
parent_chunk_size = 1024
child_chunk_size = 512
parsing_flag = False
database="faiss"
fiass_index_path = "vectore store path"
    # Start indexing
indexer=DataIndexing()
index = indexer.start_data_indexing(
        fiass_index_output_path=fiass_index_path,
        input_data_path=input_data_path,
        buffer_size=buffer_size,
        breakpoint_threshold=breakpoint_threshold,
        embedding_method=embedding_method,
        chunk_strategy=chunk_strategy,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        parsing_flag=parsing_flag,
        database="faiss",
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
