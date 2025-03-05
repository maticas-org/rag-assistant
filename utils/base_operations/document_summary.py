import re
import tqdm

from typing                 import List, Dict, Union
from langchain_ollama       import ChatOllama
from langchain_aws          import ChatBedrockConverse

from utils.prompts.summary_prompts import summarize_prompt, summary_of_summaries_prompt


def summarize_document(
                        llm:                    Union[ChatOllama, ChatBedrockConverse],
                        document:               List[str],
                        max_document_len:       int = 5000,
                        verbose:                bool = False,
                       ):
    """
       Summerizes the document using an LLM, with a maximum document length of 4000 characters.
       If the concatenated document exceeds 'max_document_len' characters, it will be split into multiple chunks.
       And we will summarize each chunk separately, so that finally, 
       merge the summaries of all chunks into a single summary, again built using the LLM.

       Args:
         document (List[str]): List of paragraphs from the document.
         max_document_length (int): Maximum length of the context window for the LLM.
         model_name (str): Name of the LLM model to use.
    """
    
    # Fist, we will split the document into chunks of max_document_length at most
    document_splits = [document[0], ]
    
    for para in document[1:]:
        if len(document_splits[-1]) + len(para) < max_document_len:
            document_splits[-1] += " " + para
        else:
            document_splits.append(para)
    
    if verbose:
        print(f"Number of document splits for summarization task: {len(document_splits)}")
    
    # Now, we will summarize each chunk separately
    summaries = []
    for doc in tqdm.tqdm(document_splits, desc="Summarizing document chunks [File level]"):
        response = (summarize_prompt | llm).invoke({
            "document": doc
        }).content.strip()
        response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip().lower()
        response = re.sub(r'<summary>', '', response).strip()
        response = re.sub(r'</summary>', '', response).strip()
        summaries.append(response)
    
    # Finally, we will merge the summaries of all chunks into a single summary
    response = (summary_of_summaries_prompt | llm).invoke({
        "summaries": "\n".join(summaries)
    }).content.strip()
         
    return response