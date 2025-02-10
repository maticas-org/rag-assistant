from typing import List, Tuple
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, Table, Image, Title, FigureCaption, NarrativeText, ListItem, Header, Footer

def extract_paragraphs_and_tables(file_path: str, 
                                  image_output_dir_path: str) -> Tuple[List[dict], List[str]]:
    """
    Extracts paragraphs with metadata and tables from a PDF document
    """
    elements = partition_pdf(
        filename=file_path,
        #strategy="hi_res",
        strategy="auto",
        infer_table_structure=True,
        extract_image_block_types=["Table"],
        languages=["en", "es"],
        image_output_dir_path=image_output_dir_path,
        chunking_strategy="by_title",
        combine_text_under_n_chars = 500,
    )

    paragraphs = []
    tables = []

    for element in elements:
        if isinstance(element, Table):
            tables.append(element.metadata.text_as_html)

        elif isinstance(element, Image):
            continue

        elif isinstance(element, Title):
            element_type = 'Title'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
            })

        elif isinstance(element, Header):
            element_type = 'Header'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
            })
        
        elif isinstance(element, Footer):
            element_type = 'Footer'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
            })
            
        elif isinstance(element, FigureCaption):
            element_type = 'FigureCaption'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
            })
            
        elif isinstance(element, NarrativeText):
            element_type = 'NarrativeText'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
            })
            
        elif isinstance(element, ListItem):
            element_type = 'ListElement'
            clean_text = element.text.strip()
            paragraphs.append({
                "text": clean_text,
                "type": element_type,
            })
            
        elif isinstance(element, Text):
            # Extract element type from metadata
            element_type = 'Text'
            clean_text = element.text.strip()

            if len(clean_text) > 5:
                paragraphs.append({
                    "text": clean_text,
                    "type": element_type,
                })

    return paragraphs, tables