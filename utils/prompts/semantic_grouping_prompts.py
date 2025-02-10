from langchain_core.prompts import ChatPromptTemplate

similarity_prompt = ChatPromptTemplate.from_template(
    """Analyze these text segments from an agriculture document:
    Context: {context}
    Current chunk: {joined_current_chunk}
    New Paragraph: {new_para}
    Next Paragraph (for context): {next_para}
    Should the new paragraph be grouped with Current chunk? Consider:
    1. Narrative continuity
    2. Topic consistency
    3. Presence of gibberish/non-sense
    Respond ONLY with: true/false, longer explanations are not needed and 
    are penalized. You can also use <think> tags to indicate your thought process.
    """
)