import os
import json

from os                 import path
from typing             import List, Union
from langchain_ollama   import ChatOllama
from langchain_aws      import ChatBedrockConverse

# Local imports
from utils.base_operations.file_search          import get_files_paths_local
from utils.base_operations.types_identification import extract_entity_types


def extract_entity_types_(
                            llm:                  Union[ChatOllama, ChatBedrockConverse],
                            data_dir:             str,
                            extensions:           List[str] = ["json"],
                            base_file_prefix:     str = "summarized-",
                            output_dir:           str = "fourth-data-extraction",
                            output_file_name:     str = "all-entity-types.json",
                            verbose:              bool = False,
                        ) -> None:

    """
    Load JSON files that start with "summarized-", extract entity types from their summaries,
    and save the output with a new key "entity_types" in a "fourth-data-extraction" folder.

    Args:
        - data_dir: str - The directory containing the summarized JSON files.
        - extensions: List[str] - The file extensions to consider.
        - base_file_prefix: str - The prefix of the summarized JSON files.
        - output_dir: str - The directory to save the entity types JSON file.
        - output_file_name: str - The name of the output JSON file.
        - verbose: bool - Whether to print the progress.
    """
    json_files = get_files_paths_local(data_dir, extensions=extensions)
    output_dir = path.join(data_dir, output_dir)

    all_summaries = []

    for json_file in json_files:
        if os.path.basename(json_file).startswith(base_file_prefix):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary = data.get("summary", [])
                all_summaries.append(summary)
                
    types = extract_entity_types(llm, all_summaries, verbose=True)

    # Save the entity types to a JSON file.
    output_file = path.join(data_dir, output_dir, output_file_name)
    os.makedirs(path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(types, f, ensure_ascii=False, indent=4)
    
    if verbose:
        print(f"Saved entity types file: {output_file}")