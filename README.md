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

