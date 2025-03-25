import os
import json
import hashlib

from tqdm               import tqdm
from datetime           import datetime
from os                 import path
from typing             import List, Union
from langchain_ollama   import ChatOllama
from langchain_aws      import ChatBedrockConverse

# Local imports
from utils.base_operations.file_search                  import get_files_paths_local
from utils.base_operations.contextually_place_chunk     import contextualize_doc

def contextualize_chunks(
                            llm:                  Union[ChatOllama, ChatBedrockConverse],
                            data_dir:             str,
                            extensions:           List[str] = ["json"],
                            strategy:             str = "full_document",
                            base_file_prefix:     str = "summarized-",
                            output_file_prefix:   str = "contextualized-chunks-",
                            output_dir:           str = "contextual/first-data-extraction",
                            max_chunk_len:        int = 6000,
                            verbose:              bool = False,
                        ) -> None:
    """
    Load JSON files that start with "summarized-" because in the case we want to contextualize,
    the chunks based on the summary instead of the original text, we will need the summary.
     
    Independent of the strategy, we will contextualize the chunks by writing a preamble context,
    with a prompt and an LLM.
    
    The output is stored with a new key "context" for each chunk, and files go to the output_dir.
    
    Args:
        data_dir:             str - The directory containing the grouped JSON files.
        extensions:           List[str] - The file extensions to consider.
        strategy:             str - The contextualization strategy to use. Options: "full_document", "document_summary".
        base_file_prefix:     str - The prefix of the JSON files to load and contextualize.
        output_file_prefix:   str - The prefix of the output JSON files.
        output_dir:           str - The directory to save the summarized JSON files.
        max_chunk_len:        int - The maximum length of the document chunks for summarization.
        verbose:              bool - Whether to print the progress.
    """

    json_files = get_files_paths_local(data_dir, extensions=extensions, file_prefix=base_file_prefix)
    output_dir = path.join(data_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file_path in tqdm(json_files, desc="Contextualizing chunks", disable=not verbose):
        # Load the JSON file
        with open(file_path, "r") as f:
            data = json.load(f, encoding="utf-8")

        # Contextualize the chunks
        data = contextualize_doc(llm, data, strategy=strategy, verbose=verbose)