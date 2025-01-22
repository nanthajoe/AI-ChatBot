from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
import logging

# FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Llama 3 model setup
MAX_RESPONSE_TOKENS = 2000  # Maximum tokens for the model's response
model = OllamaLLM(model="llama3", max_tokens=MAX_RESPONSE_TOKENS)  # Configuring the model

# Custom prompt template
general_query_template = """
You are a helpful medical AI assistant specialized in answering user queries. Please provide concise, informative, and user-friendly responses. Always remember to answer politely in Bahasa Indonesia. 
Introduce yourself as "AI Customer Service Rumah Sakit".

Tell them you can answer the question from below list if they greet you (even in Bahasa Indonesia e.g. halo, etc). 
- doctor's schedule search by its name
- doctor's schedule search by its specialty
- doctor's schedule search by its disease
- asking about health tips and general disease

Question: {query}

Answer:
"""

# Request/Response Models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# Microservices check
@app.get("/check")   
def health_check():
    return {"status": "ok"}

@app.post("/general-query", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    """
    Handle user queries and generate responses using Llama3.
    """
    try:
        # Format the custom prompt with the user's query
        formatted_prompt = general_query_template.format(query=request.query)

        # Generate a response using the LLM
        result = model.invoke(formatted_prompt)
        logging.info("Model response generated successfully.")

        return ChatResponse(response=result)

    except TimeoutError:
        logging.error("Model processing timed out.")
        raise HTTPException(
            status_code=504, detail="The request timed out. Please try again with a shorter query."
        )
    except Exception as e:
        logging.error("Unexpected error occurred: %s", str(e))
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
