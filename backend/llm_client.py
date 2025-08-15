from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from vector_db import search_context
import os
API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_response(query: str, report_content: str = None, original_text: str = None) -> str:
    """
    Fetches a response from Groq Cloud API using LangChain.
    
    - Uses `original_text` for summarization.
    - Uses `report_content` for report generation.
    - Otherwise, defaults to normal LLM query.
    """
    
    chat = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=API_KEY)
    
    # Construct prompt
    if original_text:
        prompt = f"{query}\n\nOriginal text:\n{original_text}"
    elif report_content:
        prompt = f"{query}\n\nContent to base report on:\n{report_content}"
    else:
        context = search_context(query)
        prompt = f"{query}\n\nContext of previous chats similar to given query:\n{context}";

    try:
        response = chat.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

