# Rag Assistant 🤖🛠   
RAG application with contextual retrieval. Targeted to agriculture needs.

# Context 🗺    
This solution is built to help small business owners providing customer support for their solutions.     

With this tool business owner will be partially relieved of the task of answer to common user questions that could be solved by just reading the manual.     

As is, this solution has nothing to do necesary with agriculture specifically, and code will be as general as posible. 

Yet this development is relevant as it will grow targeting the needs of the agriculture users that arise.

# Features ✨

1. Access through FastAPI. 

2. Support for: 
    - Addition of new files.
    - Updates of preexisiting files.
    - Deletion of deprecated files.

3. Support for this file types:
    - PDFs and Markdown.

4. Custom catching mechaninsm for user questions.

5. In memory vector database with persistance. _(As developed with langchain this could easily be changed)_.

6. Context embedded into the chunks of the document to improve results. Inspired from contextual retrieval from Anthropic.

7. Focus on leveraging Small Language Models (SML) as they are less hardware intensive, and could be used more widely. And as they are less smart being more secure to use.     
      

# Get started as dev 🚀

1. Clone the repository.
2. Install the dependencies.

    ```bash
    cd rag-assistant
    uv sync
    ```

    This will install the dependencies required for doing the paragraph extraction from the documents, by using tesseract. If you are on a linux machine you will need to install the following dependencies:

    ```bash
    sudo apt-get install libleptonica-dev tesseract-ocr tesseract-ocr-eng tesseract-ocr-script-latn
    ```


3. Change configuration file if needed, the default is `config.yaml`.
    ```yaml
    # config.yaml
    backend:
        llm: 
            default:
                provider: "ollama"
                model: "llama3.2:3b"
        embeddings:
            provider: "ollama"
            model: "nomic-embed-text"
        vector_db:
            provider: "opensearch"
            host: "localhost"
            port: 9200
            index: "rag-assistant"
    ```

    The `llm` provider is the one that will be used to generate the summaries, entity types identification, entity extraction, and relation generation  for the documents. Supported providers are:

    - `aws`: AWS through Bedrock.
    - `ollama`: Ollama is an open-source project that serves as a powerful and user-friendly platform for running LLMs on your local machine.
