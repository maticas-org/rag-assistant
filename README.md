# RAG Assistant 🤖🛠   
RAG application with Contextual Retrieval and GrapRAG. Targeted to agriculture needs.

# Context 🗺    
This solution is built to help small business owners providing customer support for their solutions.     

With this tool business owner will be partially relieved of the task of answer to common user questions that could be solved by just reading the manual.     

As is, this solution has nothing to do necesary with agriculture specifically, and **code will be as general as posible**.

Yet this development is relevant as *it will grow targeting the needs of the agriculture users that arise.*

# Features ✨

1. Access through FastAPI. 

2. Support for: 
    - Addition of new files.
    - Updates of preexisiting files.
    - Deletion of deprecated files.

3. Support for this file types:
    - PDFs and Markdown.

4. Two different strategies for the retrieval of the information:
    - [Contextual retrieval RAG](https://www.anthropic.com/news/contextual-retrieval). *(Useful for answering specific questions)*.
    - [GraphRag retrieval](https://arxiv.org/html/2404.16130v1). *(Useful for answering general questions that require general knowledge from different documents)*.

5. Focus on leveraging Small Language Models (SML) as they are less hardware intensive, and could be used more widely. And as they are less smart being more secure to use.
      

# Get started as dev 🚀

1. Clone the repository.

    ```bash
    git clone https://github.com/maticas-org/rag-assistant.git
    ```
2. Install the dependencies.

    ```bash
    cd rag-assistant
    uv sync
    ```

    This will install the dependencies required for doing the paragraph extraction from the documents, by using tesseract. If you are on a linux machine you will need to install the following dependencies:

    ```bash
    sudo apt-get install libleptonica-dev tesseract-ocr tesseract-ocr-eng tesseract-ocr-script-latn
    ```

    Get Ollama as well as the model you want to use with the embeddings model. For example, to get the `llama3.2:3b` model, and the `nomic-embed-text` model, you can run the following commands:
    
    ```bash 
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    ```

3. Change configuration file if needed, the file looks like this `config-local.yaml`, there is also `config-aws.yaml` for the AWS configuration.
    ```yaml
    # config-local.yaml
    backend:
        llm: 
            default:
                provider: "ollama"
                model_name: "llama3.2:3b"
                parameters:
                    temperature: 0.1
            semantic_grouping:
                provider: "ollama"
                model_name: "llama3.2:1b"
                parameters:
                    temperature: 0.1
        embeddings:
            provider: "ollama"
            model_name: "nomic-embed-text"
        vector_db:
            provider: "opensearch"
            host: "localhost"
    ```

    The `llm` provider is the one that will be used to generate the summaries, entity types identification, entity extraction, and relation generation  for the documents. Supported providers are:

    - `aws`: AWS through Bedrock.
    - `ollama`: Ollama is an open-source project that serves as a powerful and user-friendly platform for running LLMs on your local machine.

4. Currently we are under development, so you can run the following command to start the pipeline that will start processing the files in the `data` folder. Make sure to commenout certain lines in the `main.py` file to avoid reprocessing the files.

    ```bash
    uv run main.py
    ``` 