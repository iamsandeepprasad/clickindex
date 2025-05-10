from setuptools import setup,find_packages
setup(
    name="clickindex",
    version='0.1.1',
    packages=find_packages(),
    install_requires = [
    'pandas==2.2.3',
    'langchain==0.3.25',
    'langchain-community==0.3.23',
    'langchain_experimental==0.3.4',
    'llama-index==0.12.34',
    'llama-index-embeddings-gemini==0.3.2',
    'llama-index-embeddings-huggingface==0.5.3',
    'llama-index-vector-stores-faiss==0.4.0',
    'faiss-cpu==1.11.0',
    'langchain_openai==0.3.16',
    'faiss-cpu==1.11.0']



    )
