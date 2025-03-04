import re
from langchain_aws import ChatBedrockConverse
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from tqdm   import tqdm  # For cool terminal progress bars
from typing import List, Dict, Optional, Union

from utils.prompts.types_identification_prompts import (
    usecase_description,
    general_ent_type_extraction_prompt,
    specific_ent_type_extraction_prompt,
    merge_entities_types_prompt
)


def extract_entity_types(
                         llm: Union[OllamaLLM, ChatBedrockConverse],
                         summaries: List[str],
                         verbose: bool = False
                         ) -> Dict[str, List[str]]:
    """
    Identify general types of entities from a list of summaries with cool terminal progress output.
    """
    all_types: List[str] = []
    general_types: List[str] = []
    specific_types: List[str] = []
    
    # Define the pattern to extract the types from the response.
    open_type_tag, close_type_tag = "<types>", "</types>"
    open_think_tag, close_think_tag = "<think>", "</think>"
    pattern = re.compile(f"{open_type_tag}([\s\S]*?){close_type_tag}")
    pattern_think = re.compile(f"{open_think_tag}([\s\S]*?){close_think_tag}")

    print("\nStarting entity extraction...\n")
    
    # --- Step 1: Initial extraction ---
    print("Step 1: Extracting general entity types...")
    for summary in tqdm(summaries, desc="General Extraction", unit="doc", leave=False):
        response = (general_ent_type_extraction_prompt | llm).invoke({
            "document": summary
        }).strip()
        # Verbose output for debugging the response
        if verbose:
            print("Response from general extraction:", response)
        if open_type_tag in response:
            try:
                # Remove the think tags from the response
                think_str = pattern_think.search(response).group(1)
                response = response.replace(think_str, "")

                types_str = pattern.search(response).group(1)
                extracted_types = [t.strip() for t in types_str.split(",") if t.strip()]
            except Exception as e:
                print("Error parsing initial entities:", e)
                extracted_types = []
        else:
            extracted_types = []
    
        if extracted_types:
            # Clean up the extracted types and add them to the list of all entities
            # remove -, _, / from the extracted types
            extracted_types = [re.sub(r"[-_/]", " ", t) for t in extracted_types]
            all_types.extend(extracted_types)
            general_types.extend(extracted_types)
    
    # Remove duplicates
    general_types = list(set(general_types))
    all_types = list(set(all_types))
    print("Step 1 complete. General Entities extracted:", general_types)
    
    # --- Step 2: Specific extraction ---
    print("\nStep 2: Extracting specific entity types...")
    for summary in tqdm(summaries, desc="Specific Extraction", unit="doc", leave=False):
        response = (specific_ent_type_extraction_prompt | llm).invoke({
            "document": summary,
            "general_types": ", ".join(general_types),
            "specific_usecase": usecase_description
        }).strip()
        if verbose:
            print("Response from specific extraction:", response)
        if open_type_tag in response:
            try:
                # Remove the think tags from the response
                think_str = pattern_think.search(response).group(1)
                response = response.replace(think_str, "")
                
                types_str = pattern.search(response).group(1)
                extracted_types = [t.strip() for t in types_str.split(",") if t.strip()]
            except Exception as e:
                print("Error parsing specific entities:", e)
                extracted_types = []
        else:
            extracted_types = []
    
        if extracted_types:
            # Clean up the extracted types and add them to the list of all entities
            # remove -, _, / from the extracted types
            extracted_types = [re.sub(r"[-_/]", " ", t) for t in extracted_types]
            all_types.extend(extracted_types)
            specific_types.extend(extracted_types)
    
    # Remove duplicates
    all_types = list(set(all_types))
    specific_types = list(set(specific_types))
    print("Step 2 complete. Specific Entities extracted:", specific_types)
    
    # --- Step 3: Merge entities ---
    print("\nStep 3: Merging general and specific entities...")
    
    response = (merge_entities_types_prompt | llm).invoke({
        "all_types": ", ".join(all_types),
        "specific_usecase": usecase_description
    }).strip()
    
    if verbose:
        print("Response from merging:", response)
    
    if open_type_tag in response:
        try:
            # Remove the think tags from the response
            think_str = pattern_think.search(response).group(1)
            response = response.replace(think_str, "")
            
            types_str = pattern.search(response).group(1)
            all_types = [t.strip() for t in types_str.split(",") if t.strip()]
            
            #clean up the extracted types
            all_types = [re.sub(r"[-_/]", " ", t) for t in all_types]
            
            
        except Exception as e:
            print("Error parsing all entities:", e)
    
    print("\nEntity extraction complete. Final merged entities:", all_types, "\n")
    
    return {
        "general_types": all_types,
        "specific_types": specific_types,
        "all_types": all_types
    }