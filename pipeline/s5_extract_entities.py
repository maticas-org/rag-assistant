import os
import json

from os                 import path
from typing             import List, Union
from langchain_ollama   import ChatOllama
from langchain_aws      import ChatBedrockConverse

# Local imports
from utils.base_operations.file_search          import get_files_paths_local
from utils.base_operations.entity_extraction    import extract_entities_from_paragraphs


def extract_entities_(
                        llm:                        Union[ChatOllama, ChatBedrockConverse],
                        data_dir:                   str,
                        extensions:                 List[str] = ["json"],
                        base_file_prefix:           str = "summarized-grouped-",
                        base_typology_file_path:    str = "fourth-data-extraction/all-entity-types.json",
                        output_dir:                 str = "fifth-data-extraction",
                        verbose:                    bool = True,  
                    ) -> None:

    """
    Load JSON files that start with "summarized-grouped-", extract entities from their grouped paragraphs,
    and save the output with a new key "entities" in a "fifth-data-extraction" folder.
    """

    # Create the output directory path.
    output_dir = path.join(data_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load the entity types from the previous step.
    entity_types_file = path.join(data_dir, base_typology_file_path)

    with open(entity_types_file, "r", encoding="utf-8") as f:
        entity_types = json.load(f)
        all_entity_types = entity_types.get("all_types", [])
        
    
    # Load the JSON files that start with "summarized-grouped-".
    json_files = get_files_paths_local(data_dir, extensions=extensions)

    for json_file in json_files:

        if os.path.basename(json_file).startswith(base_file_prefix):

            # Load the grouped paragraphs from the JSON file.
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                grouped_paragraphs = data.get("grouped_paragraphs", [])
                print(f"Grouped paragraphs: {len(grouped_paragraphs)}")
    
            # extract entities from the grouped paragraphs 
            entities = extract_entities_from_paragraphs(
                                                            llm                     = llm,
                                                            paragraphs              = grouped_paragraphs,
                                                            relevant_entity_types   = all_entity_types,
                                                            verbose                 = False
                                                        )
            data["entities"] = entities
                               
            # Save the entities to a JSON file.
            output_file = path.join(output_dir, os.path.basename(json_file))
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)    
        