from indexing import DataIndexing
di = DataIndexing()
# Initialize the indexer
indexer = DataIndexing()

# Configure parameters
directory_path = "./data"
embedding_method = "huggingface"
chunk_strategy = "hierarchical"
parent_chunk_size = 1024
child_chunk_size = 512
parsing_flag = True

# Start indexing
index = indexer.start_data_indexing(
    directory_path=directory_path,
    embedding_method=embedding_method,
    chunk_strategy=chunk_strategy,
    parent_chunk_size=parent_chunk_size,
    child_chunk_size=child_chunk_size,
    parsing_flag=parsing_flag,
)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic of the documents?")
print(response)