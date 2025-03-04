from typing import List, Dict
from langchain_ollama.llms import OllamaLLM
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document


def extract_graph(
        documents:          List[str],
        #model_name: str = "deepseek-r1:7b",
        model_name:        str = "deepseek-r1:1.5b",
        #model_name:         str = "llama3.2:3b",
        #model_name:         str = "llama3.2:1b",
        #model_name:         str = "smollm2:1.7b",
        allowed_types:      List[str] = [],
        node_properties:    List[str] = ["context_description"],
        verbose:            bool = False,
) -> dict:

    llm       = OllamaLLM(model=model_name, temperature=0.1)
    documents = [Document(page_content=document) for document in documents]
    documents = documents[:3]  # For testing purposes.
    print(f"Loaded {len(documents)} documents.")

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_types,
        node_properties=False,
        relationship_properties=False
    )

    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print(f"Converted {len(documents)} documents to {len(graph_documents)} graph documents.")
    print(f"Graph document: {graph_documents[0].dict()}")

    return graph_documents