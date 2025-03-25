import os
import json
import tqdm
import hashlib

from datetime           import datetime
from os                 import path
from typing             import List, Union
from langchain_ollama   import ChatOllama
from langchain_aws      import ChatBedrockConverse

# Local imports
from utils.base_operations.file_search          import get_files_paths_local
from utils.base_operations.document_summary     import summarize_document

def summarize_grouped_files(
                            llm:                  Union[ChatOllama, ChatBedrockConverse],
                            data_dir:             str,
                            extensions:           List[str] = ["json"],
                            base_file_prefix:     str = "grouped-",
                            output_file_prefix:   str = "summarized-",
                            output_dir:           str = "third-data-extraction",
                            max_chunk_len:        int = 6000,
                            verbose:              bool = False,
                            ) -> None:
    """
    Load JSON files that start with "grouped-", summarize their grouped paragraphs,
    and save the output with a new key "summary" in a "second-data-extraction" folder.
    
    Args:
        - data_dir: str - The directory containing the grouped JSON files.
        - extensions: List[str] - The file extensions to consider.
        - base_file_prefix: str - The prefix of the grouped JSON files.
        - output_file_prefix: str - The prefix of the output JSON files.
        - output_dir: str - The directory to save the summarized JSON files.
        - max_chunk_len: int - The maximum length of the document chunks for summarization.
        - verbose: bool - Whether to print the progress.
    """

    json_files = get_files_paths_local(data_dir, extensions=extensions, file_prefix=base_file_prefix)
    output_dir = path.join(data_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON files
    data = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            loaded_file = json.load(f)
        data.append(loaded_file) 

    for json_file, grouped_doc in tqdm.tqdm(zip(json_files, data),
                                            total=len(data),
                                            desc="Summarizing grouped files"):

        if verbose:
            print("-" * 60)
            print(f"Summarizing: {json_file}")

        summary = summarize_document(llm=llm,
                                     document=grouped_doc,
                                     verbose=verbose,
                                     max_document_len=max_chunk_len)
        grouped_doc["summary"]  = summary
        
        # Add metadata to the summarized document
        grouped_doc["metadata"].update({
            "summarized_timestamp": datetime.now().isoformat(),
            "summarized_by": "summarize_grouped_files"
        })
        grouped_doc["metadata"].update({
            "file_hash": hashlib.md5(json.dumps(data).encode()).hexdigest(),
        })

        # Save the summarized JSON file
        file_name   = path.splitext(path.basename(json_file))[0]
        output_file = path.join(output_dir, f"{output_file_prefix}{file_name}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(grouped_doc, f, ensure_ascii=False, indent=4)
            
        if verbose:
            print(f"Saved summarized file: {output_file}")