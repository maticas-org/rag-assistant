from langchain_core.prompts import ChatPromptTemplate

summarize_prompt = ChatPromptTemplate.from_template(
    """
    You are a summarization assistant. Your task is to read the following document and produce a concise, high-quality summary that captures the main ideas and key points. Please follow these instructions carefully:

    1. **Conciseness:** Limit your summary to no more than 5 sentences.
    2. **Clarity:** Use simple and clear language that is easy to understand.
    3. **Fidelity:** Stick strictly to the content of the document—do not introduce any new details or opinions.
    4. **Focus:** Emphasize the primary message and the most important information, avoiding unnecessary technical details.
    5. **Structure:** If appropriate, you may use bullet points for clarity, but ensure the overall summary remains coherent.

    Use <think> </think> tags to indicate your thought process.
    However, please ensure that your internal reasoning is not included in your final response.  
    
    Use <summary> </summary> tags to indicate the summary of the document.

    Document:
    {document}
    """
)

summary_of_summaries_prompt = ChatPromptTemplate.from_template(
    """
    You are a summarization assistant tasked with combining several chunk summaries into one coherent final summary.
    Before providing your final answer, take a brief moment to reflect on the overall content and relationships between the summaries.
    Use <think> </think> tags to indicate your thought process.
    However, please ensure that your internal reasoning is not included in your final response.

    Summaries:
    {summaries}

    Use <summary> </summary> tags to indicate the summary of the document.
    """
)