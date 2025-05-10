from indexing import DataIndexing
import os
import warnings
from dotenv import load_dotenv
import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.schema import Document
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import numpy as np
# Initialize the indexer
indexer = DataIndexing()
load_dotenv()
# Configure parameters
input_data_path = "C:\\Users\\sandprasad\\Desktop\\Data Science EXP\\GENAI_APP\\GENAI_Application\\data"
embedding_method = "openai"
chunk_strategy = "hierarchical"
parent_chunk_size = 1024
child_chunk_size = 512
parsing_flag = False
database="faiss"
fiass_index_path = "C:\\Users\\sandprasad\\Desktop\\Data Science EXP\\GENAI_APP"
# Start indexing
index = indexer.start_data_indexing(
    fiass_index_output_path=fiass_index_path,
    input_data_path=input_data_path,
    embedding_method=embedding_method,
    chunk_strategy=chunk_strategy,
    parent_chunk_size=parent_chunk_size,
    child_chunk_size=child_chunk_size,
    parsing_flag=parsing_flag,
    database="faiss"
)
# embedding_method="openai"
index_path="C:\\Users\\sandprasad\\Desktop\\Data Science EXP\\storage2"

def get_embedding_obj( embedding_method, model_name=None):
        """
        Initialize the embedding model based on the specified method.
        
        Args:
            embedding_method (str): The embedding method ('openai', 'huggingface', 'gemini').
            model_name (str, optional): Specific model name for the embedding method.
        
        Returns:
            Embedding model instance.
        
        Raises:
            ValueError: If an unsupported embedding method is provided.
        """
        embedding_method = embedding_method.lower()
        if embedding_method == "openai":
            embedding = OpenAIEmbedding()
        elif embedding_method == "gemini":
            embedding = GeminiEmbedding()
        elif embedding_method == "huggingface":
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            embedding = HuggingFaceEmbedding(model_name=model_name)
        else:
            raise ValueError(
                f"Unsupported embedding method: {embedding_method}. "
                "Supported methods: 'openai', 'huggingface', 'gemini'."
            )
        return embedding
    
# def load_faiss_index(index_path, embedding_method, model_name=None):
#         """
#         Load a FAISS index from a .index file.
        
#         Args:
#             index_path (str): Path to the FAISS index file (e.g., 'faiss_index.index').
#             embedding_method (str): The embedding method used for the index.
#             model_name (str, optional): Specific model name for the embedding method.
        
#         Returns:
#             VectorStoreIndex: Loaded FAISS vector index.
        
#         Raises:
#             FileNotFoundError: If the index file does not exist.
#         """
#         if not os.path.exists(index_path):
#             raise FileNotFoundError(f"FAISS index file {index_path} does not exist.")
        
#         embedding = get_embedding_obj(embedding_method, model_name)
#         faiss_index = faiss.read_index(index_path)
#         vector_store = FaissVectorStore(faiss_index=faiss_index)
#         storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
#         index = VectorStoreIndex.from_vector_store(
#             vector_store=vector_store,
#             storage_context=storage_context,
#             embed_model=embedding
#         )
#         print(f"FAISS index loaded from {index_path}")
#         return index

# embedding_method="openai"
# index_path="C:\\Users\\sandprasad\\Desktop\\Data Science EXP\\storage"
# index = load_faiss_index(index_path, embedding_method)
embedding = get_embedding_obj(embedding_method)
def get_query_vector(question, embedding_model):
    # Convert the question into a vector using the embedding model
    query_vector = embedding_model.encode([question])
    return np.array(query_vector).astype('float32')


def search_faiss_index(faiss_index, query_vector, k=5):
    # Search the index for the k nearest neighbors
    distances, indices = faiss_index.search(query_vector, k)
    return distances, indices

question = "What is the Document about?"
query_vector = get_query_vector(question, embedding)
print(query_vector)
faiss_index=index
# Search the index
distances, indices = search_faiss_index(faiss_index, query_vector)

# Step 3: Retrieve and Interpret Results
# Use the indices to retrieve the corresponding documents or data
for idx in indices[0]:
    print(f"Document ID: {idx}, Distance: {distances[0][idx]}")
    # Retrieve and print the document or data associated with this index
