from typing import List, Tuple
from unstructured.partition.pdf             import partition_pdf
from unstructured.documents.elements        import Text, Table, Image, Title, FigureCaption, NarrativeText, ListItem, Header, Footer
from utils.models.document_processing       import Paragraph, Table as uTable

def extract_paragraphs_and_tables(
                                    file_path:              str, 
                                    image_output_dir_path:  str,
                                    extract_images:         bool = False
                                ) -> Tuple[Paragraph, uTable]:
    """
    Extracts paragraphs with metadata and tables from a PDF document
    
    Args:
        file_path (str): The path to the PDF file to process.
        image_output_dir_path (str): The path to the directory where images will be saved.
        extract_images (bool): Whether to extract images from the PDF.

    Returns:
        Tuple[List[dict], List[str]]: A tuple containing a list of paragraphs and a list of tables.
    """
    elements = partition_pdf(
        filename                    =   file_path,
        strategy                    =   "auto",
        infer_table_structure       =   True,
        extract_image_block_types   =   ["Table"],
        extract_images              =   extract_images,
        languages                   =   ["en", "es"],
        image_output_dir_path       =   image_output_dir_path,
        chunking_strategy           =   "by_title",
        combine_text_under_n_chars  =    500,
    )

    paragraphs = []
    tables = []

    for element in elements:
        if isinstance(element, Table):
            tables.append({"text": element.metadata.text_as_html})

        elif isinstance(element, Image):
            continue

        elif isinstance(element, Title):
            element_type = 'Title'
            paragraphs.append({
                "text": element.text.strip(),
                "type": element_type,
                "chunk_id": None,  # Default to None
                "entities": None,  # Default to None
                "relations": None,  # Default to None
            })

        elif isinstance(element, Header):
            element_type = 'Header'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
                "chunk_id": None,  # Default to None
                "entities": None,  # Default to None
                "relations": None,  # Default to None
            })
        
        elif isinstance(element, Footer):
            element_type = 'Footer'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
                "chunk_id": None,  # Default to None
                "entities": None,  # Default to None
                "relations": None,  # Default to None
            })
            
        elif isinstance(element, FigureCaption):
            element_type = 'FigureCaption'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
                "chunk_id": None,  # Default to None
                "entities": None,  # Default to None
                "relations": None,  # Default to None
            })
            
        elif isinstance(element, NarrativeText):
            element_type = 'NarrativeText'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
                "chunk_id": None,  # Default to None
                "entities": None,  # Default to None
                "relations": None,  # Default to None
            })
            
        elif isinstance(element, ListItem):
            element_type = 'ListElement'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
                "chunk_id": None,  # Default to None
                "entities": None,  # Default to None
                "relations": None,  # Default to None
            })
            
        elif isinstance(element, Text):
            # Extract element type from metadata
            element_type = 'Text'
            clean_text = element.text.strip()

            if len(clean_text) > 5:
                paragraphs.append({
                    "text": clean_text,
                    "type": element_type,
                    "chunk_id": None,  # Default to None
                    "entities": None,  # Default to None
                    "relations": None,  # Default to None
                })


    paragraphs = [Paragraph(**p) for p in paragraphs]
    tables = [uTable(**t) for t in tables]

    return paragraphs, tables