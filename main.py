"""
This is a simple text-RAG question and answering script using Langchain, Ollama, and ChromaDB.

1) Install Ollama: https://ollama.com/
2) Keep Ollama running in the background
3) Pull desired model from Ollama. In the terminal: ollama pull <model_name>
4) Create a virtual environment and activate it:
    python -m venv .venv
    source .venv/bin/activate
5) Install the necessary packages:
    pip install --upgrade pip
    pip install langchain langchain-community langchain-huggingface sentence_transformers chromadb unstructured
6) Create a text file named "text.txt" with the text you want to use as a knowledge base
7) Run the code below: python3 main.py

Based on: https://blog.futuresmart.ai/using-langchain-and-open-source-vector-db-chroma-for-semantic-search-with-openais-llm
"""

# Load the text file
from langchain_community.document_loaders import TextLoader
loader = TextLoader('text.txt')
documents = loader.load()

# Split the text into chunks
# Adjust the chunk_size and chunk_overlap parameters to fit your text and model context window
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Instanciate the embeddings model from Sentence Transformers
# Check which models are available at: https://huggingface.co/sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Instanciate the vector store
from langchain_community.vectorstores import Chroma
vectordb = Chroma.from_documents(docs, embeddings)

# Instanciate the LLM model using Ollama
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")

# Load the question answering chain
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# Start the interactive loop
print("="*50)
print("Ctrl+C to shut down the program.")
while True:
    question = input("\nQuestion: ")

    # Get the 6 most similar documents to the question (default k=4)
    matching_docs = vectordb.similarity_search(question, k=6)

    answer = chain.invoke(
        input= {
            "input_documents": matching_docs,
            "question": question
        },
        return_only_outputs=True
    )
    print("\nAnswer:", answer['output_text'])
    print("="*50)