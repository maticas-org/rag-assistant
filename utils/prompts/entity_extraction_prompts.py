from langchain_core.prompts import ChatPromptTemplate

# Initial entity extraction prompt with escaped curly braces.
entity_extraction_prompt = ChatPromptTemplate.from_template(
    """
    You are an entity extraction assistant.
    Your goal is to identify all major relevant entities in the text below and output them in valid JSON. 

    Follow these rules:
    1. For each entity, create a JSON object with three fields:
       - "name": The entity’s name or label as mentioned in the text.
       - "type": One of the following categories (use your best judgment):
         for example ["plant", "bug", "disease", "chemical", "tool", ...]
       - "context": A brief (one-sentence) explanation of how this entity is referenced or what it is, based on the paragraph.
    2. Return the entities as a JSON array.
    3. If no entities are found, return an empty array `[]`.
    4. Do not include any additional commentary or chain-of-thought in the final answer. 
    5. Only include information that is explicitly in the paragraph; do not hallucinate.

    Paragraph:
    {paragraph_text}

    Use <think> </think> tags to indicate your thought process.
    Finally, use <entities> </entities> tags to indicate the entities extracted from the text.
    For example:
    <entities>
    [
        {{
            "name": "mold",
            "type": "disease",
            "context": "Mold is a common disease that affects many plants."
        }},
        {{
            "name": "garlic-pepper spray",
            "type": "chemical",
            "context": "Garlic and pepper are natural chemicals that can repel insects."
        }}
    ]
    </entities>

    Now extract the entities and return them in valid JSON:
    """
)

# Prompt to check if any entities are missing.
missing_entity_check_prompt = ChatPromptTemplate.from_template(
    """
    You are an entity extraction assistant.
    Given the following paragraph and the list of entities already extracted, please check if there are any additional entities that might be missing.

    Paragraph:
    {paragraph_text}

    Already extracted entities (names only):
    {entity_names}

    Answer with either <response>yes</response> if you believe there are missing entities, or <response>no</response> if all relevant entities have been detected.
    """
)

# Prompt to extract additional (missing) entities with escaped curly braces.
additional_entity_extraction_prompt = ChatPromptTemplate.from_template(
    """
    You are an entity extraction assistant.
    Based on the following paragraph, and excluding the already extracted entities below, please extract any additional entities that have not been mentioned.
    
    Paragraph:
    {paragraph_text}

    Already extracted entities (names only):
    {entity_names}

    Follow the same rules as before:
    1. Return the new entities as a JSON array.
    2. Each entity should be a JSON object with the fields "name", "type", and "context".
    3. If no new entities are found, return an empty array `[]`.

    Please output the result between <entities> and </entities> tags.
    For example:
    <entities>
    [
        {{
            "name": "example entity",
            "type": "example type",
            "context": "Example context explaining the entity."
        }}
    ]
    </entities>

    Now, extract any additional entities:
    """
)