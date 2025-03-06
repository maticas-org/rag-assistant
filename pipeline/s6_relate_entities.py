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
from utils.base_operations.file_search          import get_files_paths_local
from utils.base_operations.relate_entities      import extract_relations_from_paragraphs

def relate_entities_(
                        llm:                        Union[ChatOllama, ChatBedrockConverse],
                        data_dir:                   str,
                        extensions:                 List[str] = ["json"],
                        base_file_prefix:           str = "entities-",
                        output_dir:                 str = "sixth-data-extraction",
                        output_file_prefix:         str = "related-",
                        verbose:                    bool = False,  
                    ) -> None:

    """
    Load JSON files that start with "entities-", based on the previous step, and relate the entities to each other.
    and save the output by adding a new field for each entity called "related_entities" in a "sixth-data-extraction" folder.
    """

    # Create the output directory path.
    output_dir = path.join(data_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON files that start with "summarized-grouped-".
    json_files = get_files_paths_local(data_dir, extensions=extensions, file_prefix=base_file_prefix)

    # Overall progress bar
    json_files = tqdm(json_files, desc="Relating entities", unit="file")

    for json_file in json_files:

        # Load the grouped paragraphs from the JSON file.
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            grouped_paragraphs = data.get("grouped_paragraphs", [])
    

        # Add the relationships information to the grouped paragraphs.
        related_grouped_paragraphs = extract_relations_from_paragraphs(llm = llm,
                                                                    paragraphs = grouped_paragraphs,
                                                                    verbose = verbose)
        data["grouped_paragraphs"] = related_grouped_paragraphs

        # Add metadata to the summarized document
        data["metadata"].update({
            "relationships_extraction_timestamp": datetime.isoformat(datetime.now()),
        })
        data["metadata"].update({
            "file_hash": hashlib.md5(json.dumps(data).encode()).hexdigest(),
        })
        
                               
        # Save the entities to a JSON file.
        filename = path.basename(json_file).replace(base_file_prefix, "")
        output_file = path.join(output_dir, f"{output_file_prefix}{filename}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)    
