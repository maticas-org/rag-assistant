import os
import re
import json
from langchain_ollama.llms import OllamaLLM

from utils.prompts.semantic_grouping_prompts import similarity_prompt

# --- Semantic Grouping Function ---

def semantic_grouping(json_path: str,
                      max_chunk: int = 4000,
                      model_name: str = "llama3.2:1b",
                      #model_name: str = "deepseek-r1:1.5b"
                      ) -> list:
    """
    Groups paragraphs semantically using Ollama LLM with non-sense detection
    and context window maintenance.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    paragraphs = data['paragraphs']
    grouped = []
    current_chunk = []
    llm = OllamaLLM(model=model_name, temperature=0.1)
    

    for i, para in enumerate(paragraphs):
        # Clean paragraph and check length
        clean_para = para["text"].strip()
            
        # Get LLM judgment
        response = (similarity_prompt | llm).invoke({
            "context": grouped[-1] if grouped else "None. Start of document.",
            "joined_current_chunk": " ".join(current_chunk) if current_chunk else "None. Start of chunk.",
            "new_para": clean_para,
            "next_para": paragraphs[i+1] if i+1 < len(paragraphs) else "None. End of document."
        }).strip().lower()

        print("---------------------------------------------------------")
        print(f"Context: {grouped[-1] if grouped else 'None. Start of document.'}")
        print(f"Current chunk: {' '.join(current_chunk) if current_chunk else 'None. Start of chunk.'}")
        print(f"New Paragraph: {clean_para}")
        
        # remove the text within the <think> </think> tags from the response
        # and extract only the remaining text
        #response = re.sub(r'<think>.*?</think>', '', response).strip()
        response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip().lower()
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

    # Save results
    output_path = os.path.join(
        os.path.dirname(json_path),
        f"grouped-{os.path.basename(json_path)}"
    )
    with open(output_path, 'w') as f:
        json.dump({
            "grouped_paragraphs": grouped,
            "processing_metadata": {
                "average_chunk_length": sum(len(c) for c in grouped)//len(grouped) if grouped else 0
            }
        }, f, indent=4, ensure_ascii=False)
    
    return grouped