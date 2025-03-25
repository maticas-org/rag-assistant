from langchain_core.prompts import ChatPromptTemplate

contextualize_chunk_prompt = ChatPromptTemplate.from_template(
    """
    You are a contextualization assistant. Your task is to read the following document and produce a concise, high-quality preamble that captures how the text 
    bellow (chunk) is situated within all the document. Please follow these instructions carefully:

    1. **Conciseness:** Limit your preamble to no more than 5 sentences.
    2. **Clarity:** Use simple and clear language that is easy to understand.
    3. **Fidelity:** Stick strictly to the content of the document—do not introduce any new details or opinions.
    4. **Focus:** Emphasize the primary message and the most important information, avoiding unnecessary technical details.
    5. **Structure:** If appropriate, you may use bullet points for clarity, but ensure the overall contextualization remains coherent.

    Use <think> </think> tags to indicate your thought process.
    However, please ensure that your internal reasoning is not included in your final response.  
    
    Use <context> </context> tags to indicate the preamble that situates the chunk within the document.

    Document:
    {document}

    Chunk:
    {chunk}
    
    Your response: 
    """
)