import os
import json
import hashlib
import tqdm 
from os         import path
from datetime   import datetime
from typing     import Dict, List

from utils.models.document_processing       import BaseFile, Metadata, Paragraph
from utils.pdf_document_parser              import extract_paragraphs_and_tables
from utils.base_operations.file_search      import get_files_paths_local

def process_pdf_files(
                        data_dir:               str,
                        file_extensions:        List[str] = ["pdf"],
                        output_dir_full_path:   str = "first-data-extraction",
                        output_files_prefix:    str = "processed-",
                        verbose:                bool = True,
                        extract_images:         bool = False
                      ) -> None:
    """
    Process PDF files by extracting paragraphs and tables,
    then save the results as JSON files in a output_dir directory.

    Args:
        - data_dir (str): The directory containing the PDF files.
        - file_extensions (List[str]): The extensions of the files to process.
        - output_dir_name (str): The name of the output directory.
        - output_files_prefix (str): The prefix for the output JSON files.
        - verbose (bool): Whether to print progress information
        - extract_images (bool): Whether to extract images from the PDF.
    """

    files = get_files_paths_local(data_dir, extensions=file_extensions)

    if verbose:
        print(f"Found {len(files)} {file_extensions} files in {data_dir}")
    
    # Create the output directory if it does not exist.
    os.makedirs(output_dir_full_path, exist_ok=True)

    for file in tqdm.tqdm(files, desc=f"Processing {file_extensions} files", disable=not verbose):
        file_name = path.splitext(path.basename(file))[0]

        # Extract paragraphs and tables from the PDF.
        paragraphs, tables = extract_paragraphs_and_tables(
            file,
            extract_images = extract_images,
            image_output_dir_path = path.join(output_dir_full_path, "images"),
        )

        # Format the output data.
        metadata = Metadata(
            file_hash               = file_name,
            extraction_date         = datetime.now().isoformat(),
            original_full_file_path = file,
            grouping_info           = None,
            summary_info            = None,
            relation_info           = None,
        )

        base_file = BaseFile(
            paragraphs  =   paragraphs,
            metadata    =   metadata,
            tables      =   tables,
            summary     =   None,
        )

        output_data = base_file.model_dump()
        output_data["metadata"]["file_hash"] = hashlib.md5(json.dumps(output_data).encode()).hexdigest()

        output_file = path.join(output_dir_full_path, f"{output_files_prefix}{file_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)