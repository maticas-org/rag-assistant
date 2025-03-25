import os
import json
import yaml

from os             import path
from datetime       import datetime
from nano_graphrag  import GraphRAG, QueryParam

# Pipeline steps
from utils.models.document_processing                   import BaseFile
from utils.init                                         import get_llm
from pipeline.general.s1_read_and_extract_text          import process_pdf_files
from pipeline.general.s2_semantically_group_paragraphs  import semantic_grouping_
from pipeline.general.s3_summarize_grouped_files        import summarize_grouped_files


# Define the data directory relative to this file.
BASE_DIR = path.dirname(path.realpath(__file__))
DATA_DIR = path.join(BASE_DIR, "data")

if __name__ == "__main__":

    # Uncomment the steps below as needed for your pipeline.
    # Step 1: Process PDF files.
    #process_pdf_files(data_dir              = DATA_DIR, 
    #                  output_dir_full_path  = "data/ocr",
    #)
    
    # Step 2: Apply semantic grouping to processed JSON files.
    #semantic_grouping_(
    #    llm                     = get_llm(usecase="semantic_grouping"),
    #    data_dir                = DATA_DIR,
    #    output_dir_full_path    = "data/semantic_grouping",
    #    max_merged_chunk_len    = 4000,
    #    verbose                 = True,       
    #)


    graph_func = GraphRAG(working_dir           =   "./data/graphrag",
                          using_amazon_bedrock  =   True)
    

    with open('data/semantic_grouping/grouped-growing-tomatoes-successfully-on-the-texas-high-plains.json', 'r') as f:
        grouped_file = json.load(f)
        grouped_file = BaseFile(**grouped_file)

    all_paragraphs = [para.text for para in grouped_file.paragraphs]
    graph_func.insert(all_paragraphs)


    # Perform global graphrag search
    prompt: str = "How to grow tomatoes in Texas?"
    print(graph_func.query(prompt, param=QueryParam(mode="global")))

    # Perform local graphrag search
    prompt: str = "How to grow tomatoes in Texas?"
    print(graph_func.query(prompt, param=QueryParam(mode="local")))
    pass