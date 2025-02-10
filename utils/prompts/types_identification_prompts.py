from langchain_core.prompts import ChatPromptTemplate

usecase_description = """This tool empowers farmers by turning a wide range of agricultural guides, case studies, 
and best practices into clear, actionable advice. It not only helps farmers find easy-to-follow instructions for
crop care but also provides tailored insights—whether it’s detecting early signs of disease, optimizing irrigation
and fertilization, or managing post-harvest challenges. By merging general expertise with specific, on-the-ground
recommendations, the tool supports smarter decision-making and promotes more sustainable, efficient farming practices."""

general_ent_type_extraction_prompt = ChatPromptTemplate.from_template(
    """
    You are an entity classification assistant. Analyze the document below and identify BROAD CATEGORIES of entities mentioned. 

    Rules:
    1. Focus only on explicit mentions in the text
    2. Use simple 1-3 word noun phrases
    3. Avoid speculative or inferred types
    4. Maximum 5 main categories

    Format Requirements:
    - Think step-by-step between <think> tags
    - Final answer ONLY as comma-separated values between <types> tags
    - No markdown or other formatting

    Example:
    <think>Document discusses cities and transportation infrastructure...</think>
    <types>city, transportation system, government</types>

    Document:
    {document}

    Response:
    """
)

specific_ent_type_extraction_prompt = ChatPromptTemplate.from_template(
    """
    You are a specialized classification assistant. Based on these general categories: {general_types}
    and the specific use case of {specific_usecase}, identify SPECIFIC SUBTYPES from the document.

    Rules:
    1. Create 3-7 specific subtypes
    2. Must be more specific than general categories
    3. Combine similar mentions
    4. Prioritize types relevant to the use case

    Format Requirements:
    - Think step-by-step between <think> tags
    - Final answer ONLY as comma-separated values between <types> tags
    - No explanations outside tags

    Example:
    <think>General types include cities... Use case needs urban planning details...</think>
    <types>public transit system, zoning district, municipal government</types>

    Document:
    {document}
    """
)

merge_entities_types_prompt = ChatPromptTemplate.from_template(
    """
    Refine and generalize entity types using these rules:
    1. GROUP specific terms under broader categories:
       - "tomato bacterial spot disease" → "plant diseases"
       - "whitefly life cycle stages" → "pest life cycles"
    2. SPLIT combined concepts into separate types:
       - "growing conditions/problems" → "growing conditions", "plant problems"
       - "blossom-end rot/blossom drop" → "fruit disorders", "flower abnormalities"
    3. APPLY consistent naming:
       - Use Title Case
       - Avoid hyphens/underscores ("container-grown" → "container plants")
    4. FILTER by relevance to {specific_usecase}:
       - Keep farmer-actionable types
       - Remove academic/species-level details

    Current Types: {all_types}

    Format Requirements:
    - Final types between <types> tags
    - MAX 15 merged types
    - Prefer 1-2 word categories
    - Include parent categories where needed:
      "pest management" NOT "integrated pest management practices for tomatoes"

    Bad Examples: ❌
    <types>eretomocerus californicus, blossom-end rot/blossom drop</types>

    Good Examples: ✅
    <types>biological pest control, tomato disorders, container gardening</types>

    Never include:
    - Specific species names (e.g., "Encarsia formosa")
    - Combined concepts with slashes (e.g., "blossom-end rot/blossom drop")
    - Overly technical terms (e.g., "eretomocerus californicus control")

    <think>Analyzing current types...</think>
    """
)