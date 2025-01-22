from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from difflib import get_close_matches
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Simplified prompt template
intent_classification_template = """
Instruction:
- Classify as "General query" if only there is no medical term in the query .
- The user query may be in Bahasa Indonesia, please handle that. Only provide the intent name in English, nothing else.
- For "doctor's availability" intent (by its name, by specialization, or by disease) the keyword is "menemui", "bertemu"
- For the "doctor's availability search by its specialization" the specialization may be in Bahasa Indonesia (e.g. Praktisi Umum, Ahli Bedah, Dokter Bedah, Ahli Pencernaan, Dokter Pencernaan, Spesialis THT, Ahli Alergi, etc).
- If you encounter "THT", that's doctor's specialization, (in English it is same as ENT), so if there is "dokter tht" it means "ENT doctor".
- If the user asked about some diseases, health tips, or how to prevent some diseases, consider its intent as "Asking about health tips and general disease."

Classify the following user query into one of these intents:
- General query.
- Doctor's availability search by its name.
- Doctor's availability search by its specialization.
- Doctor's availability search by its disease.
- Asking about health tips and general disease.
- Unanswerable question.

Query: {question}
"""

# Initialize the model
model = OllamaLLM(model="llama3")
intent_classifier = ChatPromptTemplate.from_template(intent_classification_template)
intent_chain = intent_classifier | model

# Request/Response Models
class IntentRequest(BaseModel):
    query: str

class IntentResponse(BaseModel):
    intent: str

# Function to classify intent
def classify_intent(query):
    try:
        # Get the intent from the model
        result = intent_chain.invoke({"question": query})
        intent = result.lower().strip()

        # Valid intents list
        valid_intents = [
            "general query",
            "doctor's availability search by its name",
            "doctor's availability search by its specialization",
            "doctor's availability search by its disease",
            "asking about health tips and general disease",
            "unanswerable question"
        ]

        # Fuzzy matching to handle slight variations
        closest_match = get_close_matches(intent, valid_intents, n=1, cutoff=0.6)
        return closest_match[0] if closest_match else "unanswerable question"

    except Exception as e:
        logging.error(f"Error while classifying intent: {e}")
        return "unanswerable question"

# Microservices check
@app.get("/check")   
def health_check():
    return {"status": "ok"}

@app.post("/classify-intent", response_model=IntentResponse)
async def classify_intent_api(request: IntentRequest):
    try:
        detected_intent = classify_intent(request.query)
        return IntentResponse(intent=detected_intent)
    except Exception as e:
        logging.error(f"Unexpected error in classify-intent API: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
