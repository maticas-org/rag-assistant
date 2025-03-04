import os
import json
from os         import path
from datetime   import datetime
from typing     import Dict, List

from pdf_document_parser                import extract_paragraphs_and_tables
from utils.base_operations.file_search  import get_files_paths_local

def process_pdf_files(
                        data_dir:             str,
                        file_extensions:      List[str] = ["pdf"],
                        output_dir_name:      str = "first-data-extraction",
                        output_files_prefix:  str = "processed-",
                        verbose:              bool = True,
                      ) -> None:
    """
    Process PDF files by extracting paragraphs and tables,
    then save the results as JSON files in a "first-data-extraction" directory.

    Args:
        - data_dir (str): The directory containing the PDF files.
        - file_extensions (List[str]): The extensions of the files to process.
        - output_dir_name (str): The name of the output directory.
        - output_files_prefix (str): The prefix for the output JSON files.
        - verbose (bool): Whether to print progress information
    """

    files = get_files_paths_local(data_dir, extensions=file_extensions)
    if verbose:
        print(f"Found {len(files)} PDF files in {data_dir}")

    for file in files:
        original_dir = path.dirname(file)
        file_name = path.splitext(path.basename(file))[0]
        output_dir = path.join(original_dir, output_dir_name)
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

        output_file = path.join(output_dir, f"{output_files_prefix}{file_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"Processed: {file}")
            print(f"Paragraphs: {len(paragraphs)}")
            print(f"Tables: {len(tables)}")
            print("-" * 60)