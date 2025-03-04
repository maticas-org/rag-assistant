import os
import json
from os import path, getenv
from datetime import datetime
from langchain_aws import ChatBedrockConverse
from pdf_document_parser import extract_paragraphs_and_tables

from utils.file_search          import get_files_paths_local
from utils.semantic_grouping    import semantic_grouping
from utils.document_summary     import summarize_document
from utils.types_identification import extract_entity_types
from utils.entity_extraction    import extract_entities_from_paragraphs

# Load the environment variables from the .env file.
from dotenv import load_dotenv
load_dotenv()

if getenv("AWS_ACCESS_KEY_ID") and getenv("AWS_SECRET_ACCESS_KEY") and getenv("MODE") == "aws":
    llm = ChatBedrockConverse(model = getenv("LLM_MODEL_NAME_CLOUD"),
                              temperature = 0.1,
                              max_tokens = None,)
                
    

# Define the data directory relative to this file.
BASE_DIR = path.dirname(path.realpath(__file__))
DATA_DIR = path.join(BASE_DIR, "data")



def process_pdf_files():
    """
    Process PDF files by extracting paragraphs and tables,
    then save the results as JSON files in a "first-data-extraction" directory.
    """
    files = get_files_paths_local(DATA_DIR, extensions=["pdf"])
    print(f"Found {len(files)} PDF files in {DATA_DIR}")

    for file in files:
        original_dir = path.dirname(file)
        file_name = path.splitext(path.basename(file))[0]
        output_dir = path.join(original_dir, "first-data-extraction")
        os.makedirs(output_dir, exist_ok=True)

        # Extract paragraphs and tables from the PDF.
        paragraphs, tables = extract_paragraphs_and_tables(
            file,
            image_output_dir_path=output_dir
        )

        output_data = {
            "paragraphs": paragraphs,
            "tables": tables,
            "metadata": {
                "original_file": file,
                "processing_date": datetime.now().isoformat()
            }
        }

        output_file = path.join(output_dir, f"processed-{file_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"Processed: {file}")
        print(f"Paragraphs: {len(paragraphs)}")
        print(f"Tables: {len(tables)}")
        print("-" * 60)


def process_semantic_grouping():
    """
    Apply semantic grouping on processed JSON files whose filenames start with "processed-".
    """
    json_files = get_files_paths_local(DATA_DIR, extensions=["json"])
    for json_file in json_files:
        if os.path.basename(json_file).startswith("processed-"):
            semantic_grouping(json_file)
            print(f"Semantically grouped: {json_file}")


def summarize_grouped_files():
    """
    Load JSON files that start with "grouped-", summarize their grouped paragraphs,
    and save the output with a new key "summary" in a "second-data-extraction" folder.
    """
    json_files = get_files_paths_local(DATA_DIR, extensions=["json"])
    for json_file in json_files:
        if os.path.basename(json_file).startswith("grouped-"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            print("-" * 60)
            print(f"Summarizing: {json_file}")
            grouped_paragraphs = data.get("grouped_paragraphs", [])
            print(f"Grouped paragraphs: {len(grouped_paragraphs)}")

            summary = summarize_document(grouped_paragraphs)
            data["summary"] = summary

            # Save the summarized JSON output to a new folder "second-data-extraction"
            output_dir = path.join(BASE_DIR, "second-data-extraction")
            os.makedirs(output_dir, exist_ok=True)
            file_name = path.splitext(path.basename(json_file))[0]
            output_file = path.join(output_dir, f"summarized-{file_name}.json")

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Saved summarized file: {output_file}")

def extract_entity_types_():
    """
    Load JSON files that start with "summarized-", extract entity types from their summaries,
    and save the output with a new key "entity_types" in a "third-data-extraction" folder.
    """
    json_files = get_files_paths_local(DATA_DIR, extensions=["json"])
    all_summaries = []

    for json_file in json_files:
        if os.path.basename(json_file).startswith("summarized-"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary = data.get("summary", [])
                all_summaries.append(summary)
                
    types = extract_entity_types(all_summaries, verbose=True)

    # Save the entity types to a JSON file.
    filename = "all-entity-types.json"
    output_file = path.join(DATA_DIR, "third-data-extraction", filename)
    os.makedirs(path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(types, f, ensure_ascii=False, indent=4)
    print(f"Saved entity types file: {output_file}")
    

def extract_entities_():
    """
    Load JSON files that start with "summarized-grouped-", extract entities from their grouped paragraphs,
    and save the output with a new key "entities" in a "third-data-extraction" folder.
    """
    # Load the entity types from the previous step.
    entity_types_file = path.join(DATA_DIR, "third-data-extraction", "all-entity-types.json")

    with open(entity_types_file, "r", encoding="utf-8") as f:
        entity_types = json.load(f)
        all_entity_types = entity_types.get("all_types", [])
        
    # Load the grouped JSON files. And get the grouped paragraphs.
    paragraphs = []
    
    json_files = get_files_paths_local(DATA_DIR, extensions=["json"])
    for json_file in json_files:
        if os.path.basename(json_file).startswith("summarized-grouped-"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                grouped_paragraphs = data.get("grouped_paragraphs", [])
                print(f"Grouped paragraphs: {len(grouped_paragraphs)}")
                paragraphs.extend(grouped_paragraphs)
    
    # Build the graph documents
    graph_docs = extract_entities_from_paragraphs(paragraphs=paragraphs,
                                                  relevant_entity_types=all_entity_types,
                                                  model_name="llama3.2:3b",
                                                  verbose=False)
                               
    graph_docs = [doc.model_dump() for doc in graph_docs]

    # Save the graph documents to a JSON file.
    filename = "all-entities.json"
    output_file = path.join(DATA_DIR, "third-data-extraction", filename)
    os.makedirs(path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph_docs, f, ensure_ascii=False, indent=4)
    print(f"Saved entities file: {output_file}")
            


if __name__ == "__main__":
    # Uncomment the steps below as needed for your pipeline.

    # Step 1: Process PDF files.
    # process_pdf_files()
    
    # Step 2: Apply semantic grouping to processed JSON files.
    # process_semantic_grouping()
    
    # Step 3: Summarize grouped JSON files.
    # summarize_grouped_files()

    # Step 4: Extract entity types from the summarized documents.
    #extract_entity_types_()

    # Step 5: Extract entities from the grouped paragraphs.
    extract_entities_()
