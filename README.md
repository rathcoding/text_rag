## Simple text-RAG question and answering script using Langchain, Ollama, and ChromaDB.

To run this code, you need to have Ollama installed and running in the background. You also need to have a text file named "text.txt" with the text you want to use as a knowledge base.

1) Install Ollama: https://ollama.com/
2) Keep Ollama running in the background
3) Pull desired model from Ollama. In the terminal:
    ```
    $ ollama pull <model_name>
    ```
4) Create a virtual environment and activate it:
    ```
    $ python -m venv .venv
    ```
    ```
    $ source .venv/bin/activate
    ```
5) Install the necessary packages:
    ```
    $ pip install --upgrade pip
    ```
    ```
    $ pip install langchain langchain-community langchain-huggingface sentence_transformers chromadb unstructured
    ```
6) Create a text file named "text.txt" with the text you want to use as a knowledge base. In this example, the text.txt file contains the 3 volumes from "The Lord of the Rings" by J.R.R. Tolkien.
7) Run the code below:
    ```
    python3 main.py
    ```

> *Stack:* Python, LangChain, Ollama, ChromaDB


* Based on the article [Using Langchain and Open Source Vector DB Chroma for Semantic Search with OpenAI's LLM](https://blog.futuresmart.ai/using-langchain-and-open-source-vector-db-chroma-for-semantic-search-with-openais-llm) from Pradip Nichite.
