import re
import tqdm

from typing                 import List, Dict, Union
from langchain_ollama       import ChatOllama
from langchain_aws          import ChatBedrockConverse

from utils.prompts.contextual_retrieval.contextualize_chunks_prompts import contextualize_chunk_prompt


def contextualize_doc(
                        llm:                    Union[ChatOllama, ChatBedrockConverse],
                        document:               List[Dict[str, str]],
                        strategy:               str = "full_document",
                        verbose:                bool = False,
                     ):
    """
       Adds a preamble paragraph, placing the chunk within the context of the document using an LLM.
       This can increase the retrieval accuracy of the chunk.

       Args:
         llm (Union[ChatOllama, ChatBedrockConverse]): The LLM model to use for summarization.
         document (List[str]): The document to contextualize.
         based_on (str): The type of context to use for contextualization. Default: "document". 
                        Options: "full_document", "document_summary". 

    """

    preambles = []

    if strategy == "full_document":

        full_document   = " ".join([para["text"] for para in document["grouped_paragraphs"]])
        paragraphs      = [para["text"] for para in document["grouped_paragraphs"]]

        for p in tqdm.tqdm(paragraphs, desc="Summarizing document chunks [File level]"):
            response = (contextualize_chunk_prompt | llm).invoke({
                "document": full_document,
                "chunk": p,
            }).content.strip()

            response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip().lower()
            response = re.sub(r'<context>', '', response).strip()
            response = re.sub(r'</context>', '', response).strip()
            preambles.append(response)
        
        return preambles

    elif strategy == "document_summary":
        
        summary         = document["summary"]
        paragraphs      = [para["text"] for para in document["grouped_paragraphs"]]

        for p in tqdm.tqdm(paragraphs, desc="Summarizing document chunks [Summary level]"):
            response = (contextualize_chunk_prompt | llm).invoke({
                "document": summary,
                "chunk": p,
            }).content.strip()

            response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip().lower()
            response = re.sub(r'<context>', '', response).strip()
            response = re.sub(r'</context>', '', response).strip()
            preambles.append(response)
        
        return preambles

    else:
        raise ValueError("Invalid value for 'based_on'. Options: 'full_document', 'document_summary'.")