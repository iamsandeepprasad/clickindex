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

class DataIndexing:
    """A class for indexing documents using LlamaIndex and FAISS with various chunking strategies."""
    
    def __init__(self):
        """Initialize the DataIndexing class by loading environment variables."""
        load_dotenv()

    def get_embedding_obj(self, embedding_method, model_name=None):
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
    
    def load_data(self, directory_path):
        """
        Load documents from the specified directory.
        
        Args:
            directory_path (str): Path to the directory containing documents.
        
        Returns:
            List[Document]: List of loaded documents.
        """
        reader = SimpleDirectoryReader(input_dir=directory_path)
        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def text_parser(self, llm, documents, parser_prompt=None):
        """
        Parse documents using an LLM to extract content in Markdown format.
        
        Args:
            llm: Language model for parsing.
            documents: List of documents to parse.
            parser_prompt (str, optional): Custom prompt for parsing.
        
        Returns:
            List[Document]: List of parsed documents in Markdown format.
        """
        response_arr = []
        default_parser_prompt = """
            I am giving you content of pdf {pdf_cont},
            Extract the content from an image page and output in Markdown syntax. Enclose the content in the <markdown></markdown> tag and do not use code blocks. If the image is empty then output a <markdown></markdown> without anything in it.
            Make sure you STRICTLY do not make up any number or details while extracting the number if it is not present in the text provided, strictly parse the text which is provided to you dont make up the data.

            Follow these steps:

            1. Examine the provided page carefully.
            2. Identify all elements present in the page, including headers, body text, footnotes, tables, images, captions, and page numbers, etc.
            3. Use markdown syntax to format your output:
                - Headings: # for main, ## for sections, ### for subsections, etc.
                - Lists: * or - for bulleted, 1. 2. 3. for numbered
                - Do not repeat yourself
            4. If the element is an image (not table):
                - If the information in the image can be represented by a table, generate the table containing the information of the image
                - Otherwise provide a detailed description about the information in image
                - Classify the element as one of: Chart, Diagram, Logo, Icon, Natural Image, Screenshot, Other. Enclose the class in <figure_type></figure_type>
                - Enclose <figure_type></figure_type>, the table or description, and the figure title or caption (if available), in <figure></figure> tags
                - Do not transcribe text in the image after providing the table or description
            5. If the element is a table:
                - Create a markdown table, ensuring every row has the same number of columns
                - Maintain cell alignment as closely as possible
                - Do not split a table into multiple tables
                - If a merged cell spans multiple rows or columns, place the text in the top-left cell and output ' ' for other
                - Use | for column separators, |-|-| for header row separators
                - If a cell has multiple items, list them in separate rows
                - If the table contains sub-headers, separate the sub-headers from the headers in another row
            6. If the element is a paragraph:
                - Transcribe each text element precisely as it appears
            7. If the element is a header, footer, footnote, page number:
                - Transcribe each text element precisely as it appears

            Output Example:
            <markdown>
            <figure>
            <figure_type>Chart</figure_type>
            Figure 3: This chart shows annual sales in millions. The year 2020 was significantly down due to the COVID-19 pandemic.
            A bar chart showing annual sales figures, with the y-axis labeled "Sales ($Million)" and the x-axis labeled "Year". The chart has bars for 2018 ($12M), 2019 ($18M), 2020 ($8M), and 2021 ($22M).
            </figure>

            <figure>
            <figure_type>Chart</figure_type>
            Figure 3: This chart shows annual sales in millions. The year 2020 was significantly down due to the COVID-19 pandemic.
            | Year | Sales ($Million) |
            |-|-|
            | 2018 | $12M |
            | 2019 | $18M |
            | 2020 | $8M |
            | 2021 | $22M |
            </figure>

            # Annual Report

            ## Financial Highlights

            <figure>
            <figure_type>Logo</figure_type>
            The logo of Apple Inc.
            </figure>

            * Revenue: $40M
            * Profit: $12M
            * EPS: $1.25

            | | Year Ended December 31, | |
            | | 2021 | 2022 |
            |-|-|-|
            | Cash provided by (used in): | | |
            | Operating activities | $ 46,327 | $ 46,752 |
            | Investing activities | (58,154) | (37,601) |
            | Financing activities | 6,291 | 9,718 |
            </markdown>
        """
        parser_prompt = parser_prompt or default_parser_prompt
        parser_prompt = PromptTemplate(template=parser_prompt, input_variables=["pdf_cont"])
        llmchain = LLMChain(llm=llm, prompt=parser_prompt)
        for doc in documents:
            text = doc.get_content()
            response = llmchain.invoke({"pdf_cont": text})
            response_document = Document(text=response['text'])
            response_arr.append(response_document)
        return response_arr
    
    def create_faiss_hnsw_index(self, nodes, embedding_method, ef_construction=200, ef_search=40, m=32, model_name=None):
        """
        Create a FAISS vector index with HNSW from nodes.
        
        Args:
            nodes: List of nodes to index.
            embedding_method (str): The embedding method to use.
            ef_construction (int): HNSW parameter for index construction.
            ef_search (int): HNSW parameter for search efficiency.
            m (int): HNSW parameter for number of bidirectional links.
            model_name (str, optional): Specific model name for the embedding method.
        
        Returns:
            VectorStoreIndex: FAISS HNSW vector index.
        """
        embedding = self.get_embedding_obj(embedding_method, model_name)
        sample_text = "This is a sample text to determine embedding dimension."
        text_embedding = embedding.get_text_embedding(sample_text)
        d = len(text_embedding)  # Dimension of the embedding

        # Initialize FAISS with HNSW index
        faiss_index = faiss.IndexHNSWFlat(d, m)
        faiss_index.hnsw.efConstruction = ef_construction
        faiss_index.hnsw.efSearch = ef_search

        # Initialize FAISS vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes,
            embed_model=embedding,
            storage_context=storage_context
        )
        print("FAISS HNSW vector index created successfully")
        return index

    def start_data_indexing(
        self,
        directory_path,
        embedding_method,
        chunk_strategy,
        model_name=None,
        chunk_size=None,
        chunk_overlap=None,
        buffer_size=None,
        breakpoint_threshold=None,
        parent_chunk_size=None,
        child_chunk_size=None,
        parsing_flag=False,
        parser_prompt=None,
        llm=None
    ):
        """
        Start the data indexing process with specified chunking strategy.
        
        Args:
            directory_path (str): Path to the directory containing documents.
            embedding_method (str): The embedding method to use.
            chunk_strategy (str): Chunking strategy ('fixed', 'semantic', 'hierarchical').
            model_name (str, optional): Specific model name for the embedding method.
            chunk_size (int, optional): Size of chunks for fixed chunking.
            chunk_overlap (int, optional): Overlap between chunks for fixed chunking.
            buffer_size (int, optional): Buffer size for semantic chunking.
            breakpoint_threshold (int, optional): Threshold for semantic chunking.
            parent_chunk_size (int, optional): Parent chunk size for hierarchical chunking.
            child_chunk_size (int, optional): Child chunk size for hierarchical chunking.
            parsing_flag (bool): Whether to parse documents using LLM.
            parser_prompt (str, optional): Custom prompt for parsing.
            llm: Language model for parsing (required if parsing_flag is True).
        
        Returns:
            VectorStoreIndex: Indexed data with specified chunking strategy.
        
        Raises:
            ValueError: If required parameters for the chunking strategy are missing.
        """
        # Get embedding object
        embedding = self.get_embedding_obj(embedding_method, model_name)
        # Load documents
        loaded_documents = self.load_data(directory_path)
        if parsing_flag:
            if llm is None:
                raise ValueError("To parse the document for text, table, and images, please provide a Large Language Model")
            else:
                parsed_text_document = self.text_parser(llm, loaded_documents, parser_prompt)
                documents = parsed_text_document
        else:
            documents = loaded_documents

        print("Started")
        if chunk_strategy == "fixed":
            print("Starting Fixed Chunking")
            if chunk_size is None:
                raise ValueError("For fixed chunking strategy, chunk size cannot be None")
            if chunk_overlap is None:
                raise ValueError("For fixed chunking strategy, chunk overlap cannot be None")
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif chunk_strategy == "semantic":
            print("Starting Semantic Chunking")
            if buffer_size is None:
                raise ValueError("For semantic chunking strategy, buffer size cannot be None")
            if breakpoint_threshold is None:
                raise ValueError("For semantic chunking strategy, breakpoint threshold cannot be None")
            splitter = SemanticSplitterNodeParser(
                buffer_size=buffer_size,
                breakpoint_percentile_threshold=breakpoint_threshold,
                embed_model=embedding
            )
        elif chunk_strategy == "hierarchical":
            print("Starting Hierarchical Chunking")
            if parent_chunk_size is None:
                raise ValueError("For hierarchical chunking strategy, parent chunk size cannot be None")
            if child_chunk_size is None:
                raise ValueError("For hierarchical chunking strategy, child chunk size cannot be None")
            chunk_sizes = [parent_chunk_size, child_chunk_size]
            splitter = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        else:
            raise ValueError(
                f"Unsupported chunk strategy: {chunk_strategy}. "
                "Supported strategies: 'fixed', 'semantic', 'hierarchical'."
            )

        # Split documents into chunks
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} nodes")
        index = self.create_faiss_hnsw_index(
            nodes=nodes,
            embedding_method=embedding_method,
            model_name=model_name
        )
        index.storage_context.persist()
        return index