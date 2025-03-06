import re
from langchain_aws import ChatBedrockConverse
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from tqdm   import tqdm  # For cool terminal progress bars
from typing import List, Dict, Optional, Union

from utils.prompts.graphrag.types_identification_prompts import (
    usecase_description,
    general_ent_type_extraction_prompt,
    specific_ent_type_extraction_prompt,
    merge_entities_types_prompt
)


def clean_response(response: str, verbose: bool = False) -> str:
    """
    Clean the response by removing the think tags.
    """
    open_think_tag, close_think_tag = "<think>", "</think>"
    pattern_think = re.compile(f"{open_think_tag}([\s\S]*?){close_think_tag}")
    
    if open_think_tag in response:
        try:
            # Remove the think tags from the response
            think_str = pattern_think.search(response).group(1)
            response = response.replace(think_str, "")
        except Exception as e:
            if verbose:
                print(f"Error extracting the think tags from the response: {e}")
    
    return response

def extract_entity_types_from_response(response: str, verbose: bool = False) -> List[str]:
    """
    Extract entity types from the response.
    """
    # Set up some variables
    all_types: List[str] = []

    # Define the pattern to extract the types from the response.
    open_type_tag, close_type_tag = "<types>", "</types>"
    pattern = re.compile(f"{open_type_tag}([\s\S]*?){close_type_tag}")

    # clean the response
    response = clean_response(response, verbose)

    # Extract the types from the response
    if open_type_tag in response:
        try:
            types_str = pattern.search(response).group(1)
            all_types = [t.strip() for t in types_str.split(",") if t.strip()]
        except Exception as e:
            if verbose:
                print(f"Error extracting types from the response: {e}")
    
    # Split further by commas and slashes or \n
    all_types = [t.strip() for t in all_types for t in re.split(r"[,/\n]", t) if t.strip()]
    
    # Clean up the extracted types
    all_types = [re.sub(r"[-_/]", " ", t) for t in all_types]

    # Remove duplicates
    all_types = list(set(all_types))

    return all_types


def extract_entity_types(
                         llm: Union[OllamaLLM, ChatBedrockConverse],
                         summaries: List[str],
                         verbose: bool = False
                         ) -> Dict[str, List[str]]:
    """
    Identify general types of entities from a list of summaries with cool terminal progress output.

    This is done in 3 steps:
    1. Initial extraction of general entity types.
    2. Specific extraction of entity subtypes.
    3. Merging general and specific entities. Into a more broad and concise list.
    """
    all_types: List[str] = []
    general_types: List[str] = []
    specific_types: List[str] = []
    
    if verbose:
        print("\nStarting entity extraction...\n")
        print("Step 1: Extracting general entity types...")
    
    # --- Step 1: Initial extraction ---
    for summary in tqdm(summaries, desc="General Extraction", unit="doc", leave=False):
        response = (general_ent_type_extraction_prompt | llm).invoke({
            "document": summary
        }).content.strip()

        # Verbose output for debugging the response
        if verbose:
            print("Response from general extraction:", response)
        
        # Extract the types from the response
        extracted_types = extract_entity_types_from_response(response, verbose)
        
        # Add the extracted types to the general and all types
        general_types.extend(extracted_types)
        all_types.extend(extracted_types)
    
    # --- Step 2: Specific extraction ---
    if verbose:
        print("Step 1 complete. General Entities extracted:", general_types)
        print("\nStep 2: Extracting specific entity types...")
    
    for summary in tqdm(summaries, desc="Specific Extraction", unit="doc", leave=False):
        response = (specific_ent_type_extraction_prompt | llm).invoke({
            "document": summary,
            "general_types": ", ".join(general_types),
            "specific_usecase": usecase_description
        }).content.strip()
        if verbose:
            print("Response from specific extraction:", response)

        # Extract the types from the response
        extracted_types = extract_entity_types_from_response(response, verbose)

        # Add the extracted types to the specific and all types
        specific_types.extend(extracted_types)
        all_types.extend(extracted_types)
    
    # --- Step 3: Merge entities ---
    if verbose:
        print("Step 2 complete. Specific Entities extracted:", specific_types)
        print("\nStep 3: Merging general and specific entities...")
    
    response = (merge_entities_types_prompt | llm).invoke({
        "all_types": ", ".join(all_types),
        "specific_usecase": usecase_description
    }).content.strip()

    # Extract the types from the response
    extracted_types = extract_entity_types_from_response(response, verbose)
    all_types.extend(extracted_types)
    
    if verbose:
        print("Response from merging:", response)
        print("\nEntity extraction complete. Final merged entities:", all_types, "\n")
    
    return {
        "general_types": general_types,
        "specific_types": specific_types,
        "merged_types": extracted_types,
        "all_types": list(set(all_types))
    }