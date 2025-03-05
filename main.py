import os
import json
import yaml

from os         import path
from datetime   import datetime

# Pipeline steps
from init                                       import get_default_llm, get_semantic_grouping_llm
from pipeline.s1_read_and_extract_text          import process_pdf_files
from pipeline.s2_semantically_group_paragraphs  import process_semantic_grouping
from pipeline.s3_summarize_grouped_files        import summarize_grouped_files
from pipeline.s4_entity_type_identification     import extract_entity_types_
from pipeline.s5_extract_entities               import extract_entities_



# Define the data directory relative to this file.
BASE_DIR = path.dirname(path.realpath(__file__))
DATA_DIR = path.join(BASE_DIR, "data")
DEFAULT_LLM = get_default_llm()

if __name__ == "__main__":
    # Uncomment the steps below as needed for your pipeline.

    # Step 1: Process PDF files.
    # process_pdf_files(data_dir = DATA_DIR)
    
    # Step 2: Apply semantic grouping to processed JSON files.
    #process_semantic_grouping(
    #    llm                     = get_semantic_grouping_llm(),
    #    data_dir                = DATA_DIR,
    #    max_merged_chunk_len    = 4000,
    #)

    
    # Step 3: Summarize grouped JSON files.
    # summarize_grouped_files()

    # Step 4: Extract entity types from the summarized documents.
    #extract_entity_types_()

    # Step 5: Extract entities from the grouped paragraphs.
    #extract_entities_()
    pass