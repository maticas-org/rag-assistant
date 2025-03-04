import os
import json

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
                            verbose:              bool = True,
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
        - verbose: bool - Whether to print the progress.
    """
    json_files = get_files_paths_local(data_dir, extensions=extensions)
    output_dir = path.join(data_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for json_file in json_files:

        if os.path.basename(json_file).startswith(base_file_prefix):

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            grouped_paragraphs = data.get("grouped_paragraphs", [])

            if verbose:
                print("-" * 60)
                print(f"Summarizing: {json_file}")
                print(f"Grouped paragraphs: {len(grouped_paragraphs)}")

            summary = summarize_document(grouped_paragraphs)
            data["summary"] = summary

            file_name = path.splitext(path.basename(json_file))[0]
            output_file = path.join(output_dir, f"{output_file_prefix}{file_name}.json")

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            if verbose:
                print(f"Saved summarized file: {output_file}")