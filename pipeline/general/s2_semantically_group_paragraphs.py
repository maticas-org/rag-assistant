import os   
import json
import tqdm
import hashlib 

from datetime               import datetime
from typing                 import List, Union
from langchain_ollama       import ChatOllama
from langchain_aws          import ChatBedrockConverse

# Local imports
from utils.base_operations.file_search          import get_files_paths_local
from utils.base_operations.semantic_grouping    import semantic_grouping

def process_semantic_grouping(
                              llm:                  Union[ChatOllama, ChatBedrockConverse],
                              data_dir:             str,
                              extensions:           List[str] = ["json"], 
                              base_file_prefix:     str = "processed-",
                              max_merged_chunk_len: int = 4000,
                              output_file_prefix:   str = "grouped-",
                              output_dir:           str = "second-data-extraction",
                              verbose:              bool = False,
                              ) -> None:
    """
    Apply semantic grouping on processed JSON files whose filenames start with "processed-".

    Args:   
        - llm: Union[ChatOllama, ChatBedrockConverse] - The LLM API to use for semantic grouping.
        - data_dir: str - The directory containing the processed JSON files.
        - extensions: List[str] - The file extensions to consider.
        - base_file_prefix: str - The prefix of the processed JSON files.
        - max_merged_chunk_len: int - The maximum length of the merged chunks. Character wise.
        - output_file_prefix: str - The prefix of the output JSON files.
        - output_dir: str - The directory to save the grouped JSON files.
    """
    # Create the output directory
    output_dir = os.path.join(data_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    json_files = get_files_paths_local(data_dir,
                                       extensions   =   extensions,
                                       file_prefix  =   base_file_prefix,
                                       verbose      =   verbose)

    # Load the JSON files
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data.append(json.load(f))
    

    # Apply semantic grouping to each JSON file
    for partially_chunked_file, json_file in tqdm.tqdm(zip(data, json_files), total=len(data), desc="Semantic Grouping"):

        # Apply semantic grouping
        grouped = semantic_grouping(llm, partially_chunked_file, max_merged_chunk_len, verbose)
        if verbose:
            print(f"Semantically grouped: {json_file}")
        
        # Add metadata to the grouped paragraphs
        grouped["metadata"] = {
            "source": json_file,
            "grouped_timestamp": datetime.now().isoformat(),
            "grouped_by": "semantic_grouping",
        }
        grouped["metadata"]["file_hash"] = hashlib.md5(json.dumps(grouped).encode()).hexdigest()

        # Save the grouped paragraphs to a new JSON file
        output_file = os.path.join(output_dir, f"{output_file_prefix}{os.path.basename(json_file)}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(grouped, f, ensure_ascii=False, indent = 4)

        # Only process the first file for now
        break


