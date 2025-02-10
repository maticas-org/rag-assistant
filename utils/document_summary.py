import re
from langchain_ollama.llms import OllamaLLM
from typing import List, Dict 

from utils.prompts.summary_prompts import summarize_prompt, summary_of_summaries_prompt


def summarize_document(document: List[str],
                       max_document_length=5000,
                       model_name = "llama3.2:1b"):
    """
       Summerizes the document using an LLM, with a maximum document length of 4000 characters.
       If the concatenated document exceeds 4000 characters, it will be split into multiple chunks.
       And we will summarize each chunk separately, so that finally, 
       merge the summaries of all chunks into a single summary, again built using the LLM.

       Args:
         document (List[str]): List of paragraphs from the document.
         max_document_length (int): Maximum length of the context window for the LLM.
         model_name (str): Name of the LLM model to use.
    """

    llm = OllamaLLM(model=model_name, temperature=0.1)
    
    # Fist, we will split the document into chunks of max_document_length at most
    document_splits = [document[0], ]
    
    for para in document[1:]:
        if len(document_splits[-1]) + len(para) < max_document_length:
            document_splits[-1] += " " + para
        else:
            document_splits.append(para)
    
    print(f"Number of document splits for summarization task: {len(document_splits)}")
    
    # Now, we will summarize each chunk separately
    summaries = []
    for doc in document_splits:
        response = (summarize_prompt | llm).invoke({
            "document": doc
        }).strip()
        response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip().lower()
        response = re.sub(r'<summary>', '', response).strip()
        response = re.sub(r'</summary>', '', response).strip()
        summaries.append(response)
    
    # Finally, we will merge the summaries of all chunks into a single summary
    response = (summary_of_summaries_prompt | llm).invoke({
        "summaries": "\n".join(summaries)
    }).strip()
         
    return response