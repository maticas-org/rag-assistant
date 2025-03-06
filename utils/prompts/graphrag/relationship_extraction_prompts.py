from langchain_core.prompts import ChatPromptTemplate

# Initial entity extraction prompt with escaped curly braces.
relationship_extraction_prompt = ChatPromptTemplate.from_template(
    """
    You are a relationship extraction assistant.
    Your goal is to identify all major relevant relationships among entities in the text below and output them in valid JSON. 

    Follow these rules:
    1. For each entity, create a JSON object with three fields:
       - "source_entity": The entity that is the source of the relationship.
       - "target_entity": The entity that is the target or object of the relationship.
       - "relation_detail": A brief description of the relationship between the two entities, could be a verb or a phrase.
    2. Return the entities as a JSON array.
    3. Make sure the "source_entity" and "target_entity" fields belong to the named entities found in the text.
    4. If no entities are found, return an empty array `[]`.
    5. Do not include any additional commentary or chain-of-thought in the final answer. 
    6. Only include information that is explicitly in the paragraph; do not hallucinate.

    Paragraph:
    {paragraph_text}

    Found entities:
    {found_entities}

    Use <think> </think> tags to indicate your thought process.
    Finally, use <relationships> </relationship> tags to indicate the entities extracted from the text.
    For example:
    <relationships>
    [
        json object 1,
        json object 2,
    ]
    </relationships>

    Now extract the relationships and return them in valid JSON:
    """
)

# Prompt to check if any relationships are missing.
missing_relations_check_prompt = ChatPromptTemplate.from_template(
    """
    You are a relationship extraction assistant.
    Given the following paragraph and the list of entities as well as relationships already extracted,
    please check if there are any additional relationships that might be missing.

    Paragraph:
    {paragraph_text}

    Already extracted entities (names only):
    {entity_names}

    Already extracted relationships:
    {extracted_relations}

    Answer with either <response>yes</response> if you believe there are missing relationships, or <response>no</response>
    if all relevant entities have been correctly linked.
    """
)

# Prompt to extract additional (missing) relationships with escaped curly braces.
additional_relations_extraction_prompt = ChatPromptTemplate.from_template(
    """
    You are an entity extraction assistant.
    Based on the following paragraph, and excluding the already extracted relationships below,
    please extract any additional relationships that have not been mentioned.
    
    Paragraph:
    {paragraph_text}

    Already extracted entities (names only):
    {entity_names}

    Already extracted relationships:
    {extracted_relations}

    Follow the same rules as before:
    1. Return the new entities as a JSON array.
    2. Each entity should be a JSON object with the fields "source_entity", "target_entity", and "relation_detail".
    3. If no new relationships are found, return an empty array `[]`.

    Please output the result between <relationships> and </relationships> tags.
    For example:
    <relationships>
    [
        json object 1,
        json object 2,
    ]
    </relationships>

    Now, extract any additional relationships:
    """
)