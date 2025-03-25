import re
import hashlib
import tqdm

from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrockConverse
from typing import List, Dict, Union

from utils.models.document_processing import BaseFile, Paragraph
from utils.prompts.general.semantic_grouping_prompts import similarity_prompt

def semantic_grouping(
    llm:                    Union[ChatOllama, ChatBedrockConverse],
    partially_chunked_file: BaseFile,
    max_chunk:              int = 4000,
    verbose:                bool = False,
) -> BaseFile:
    """
    Groups paragraphs semantically using Ollama LLM with non-sense detection
    and context window maintenance, returning a BaseFile.
    """
    paragraphs = partially_chunked_file.paragraphs
    grouped = []
    current_chunk = []

    for i, para in tqdm.tqdm(enumerate(paragraphs), total=len(paragraphs), desc="Semantic Grouping [File level]"):
        # Clean paragraph and check length
        clean_para = para.text.strip()

        # Get LLM judgment
        response = (similarity_prompt | llm).invoke({
            "context": grouped[-1] if grouped else "None. Start of document.",
            "joined_current_chunk": " ".join(current_chunk) if current_chunk else "None. Start of chunk.",
            "new_para": clean_para,
            "next_para": paragraphs[i + 1] if i + 1 < len(paragraphs) else "None. End of document."
        }).content.strip().lower()

        if verbose:
            print("---------------------------------------------------------")
            print(f"Context: {grouped[-1] if grouped else 'None. Start of document.'}")
            print(f"Current chunk: {' '.join(current_chunk) if current_chunk else 'None. Start of chunk.'}")
            print(f"New Paragraph: {clean_para}")
        
        # Remove the text within <think> </think> tags from the response and extract only the remaining text
        if verbose:
            response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip().lower()
            
            # If only the initial <think> tag is present remove all the text before it
            # and try to keep the last sentence
            if response.startswith("<think>"):
                response = response[7:]
                candidate = response[response.rfind(".") + 1:]
                if len(candidate) > 3:
                    response = candidate

            # If only the final </think> tag is present remove all the text before it
            if response.endswith("</think>"):
                response = response[:-8]

            
            print(f"Response: {response}")
            print("---------------------------------------------------------")

        # Parse response robustly
        if any(kw in response for kw in ["true", "merge", "group", "yes", "join", "combine"]):
            # Check chunk limits
            new_length = sum(len(p) for p in current_chunk) + len(clean_para)
            if new_length <= max_chunk:
                current_chunk.append(clean_para)
            else:
                grouped.append(" ".join(current_chunk))
                current_chunk = [clean_para]
        else:
            # Finalize current chunk
            if current_chunk:
                grouped.append(" ".join(current_chunk))
            current_chunk = [clean_para]

    # Add final chunk
    if current_chunk:
        grouped.append(" ".join(current_chunk))

    # Generate chunk ids and create new Paragraphs
    grouped_paragraphs = [
        Paragraph(
            text        =   chunk,
            type        =   None,
            chunk_id    =   hashlib.md5(chunk.encode()).hexdigest(),
            entities    =   None,
            relations   =   None,
        )
        for chunk in grouped
    ]

    # Calculate average chunk length for metadata
    avg_chunk_len = sum(len(p.text) for p in grouped_paragraphs) // len(grouped_paragraphs) if grouped_paragraphs else 0

    # Build and return the updated BaseFile
    updated_base_file = BaseFile(
        paragraphs  =   grouped_paragraphs,
        tables      =   partially_chunked_file.tables,  # Assuming tables are unchanged
        metadata    =   partially_chunked_file.metadata.model_copy(
            update  = {
            "grouping_info": {
                "strategy": "semantic_grouping",
                "timestamp": str(datetime.now()),
                "avg_chunk_len": avg_chunk_len,
            }
        }),
        summary=partially_chunked_file.summary,
    )

    return updated_base_file