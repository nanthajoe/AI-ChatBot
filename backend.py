from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Service URLs (update the URLs as needed)
INTENT_CLASSIFICATION_URL = "http://localhost:8001/classify-intent"
RAG_URL = "http://localhost:8002/rag"
DOCTOR_NAME_URL = "http://localhost:8003/doctor-availability-by-name"
DOCTOR_DISEASE_URL = "http://localhost:8004/doctor-availability-by-disease"
DOCTOR_SPECIALIZATION_URL = "http://localhost:8005/doctor-availability-by-specialty"
GENERAL_QUERY_URL = "http://localhost:8006/general-query"

# Request/Response Models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    intent: str
    response: str

# Microservices check
@app.get("/check")   
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Intent Classification
        intent_response = requests.post(
            INTENT_CLASSIFICATION_URL,
            json={"query": request.query},
            timeout=180
        )
        if intent_response.status_code != 200:
            logging.error(f"Intent classification failed: {intent_response.text}")
            return ChatResponse(intent="error", response="Sedang terjadi kesalahan.")
        
        intent_data = intent_response.json()
        intent = intent_data["intent"]
        logging.info(f"Detected intent: {intent}")

        # Intent Handling
        if intent == "asking about health tips and general disease":
            rag_response = requests.post(
                RAG_URL,
                json={"query": request.query},
                timeout=180
            )
            if rag_response.status_code != 200:
                logging.error(f"RAG service failed: {rag_response.text}")
                return ChatResponse(intent=intent, response="Sedang terjadi kesalahan.")
            
            rag_data = rag_response.json()
            response_text = rag_data.get("response", "Maaf, saya tidak bisa menjawab pertanyaan Anda.")

        elif intent == "doctor's availability search by its name":
            doctor_name_response = requests.post(
                DOCTOR_NAME_URL,
                json={"query": request.query},
                timeout=180
            )
            if doctor_name_response.status_code != 200:
                logging.error(f"Doctor Name Failed: {doctor_name_response.text}")
                return ChatResponse(intent=intent, response="Sedang terjadi kesalahan.")
            
            doctor_data = doctor_name_response.json()
            doctor_name = doctor_data.get("doctor", "Nama Dokter tidak diketahui.")
            availability = doctor_data.get("availability", "")

            if not availability.strip():
                response_text = f"{doctor_name} tidak memiliki jadwal tersedia."
            else:
                response_text = f"Jadwal {doctor_name}:\n\n{availability}"

        elif intent == "doctor's availability search by its disease":
            doctor_disease_response = requests.post(
                DOCTOR_DISEASE_URL,
                json={"query": request.query},
                timeout=180
            )
            if doctor_disease_response.status_code != 200:
                logging.error(f"Doctor Disease Failed: {doctor_disease_response.text}")
                return ChatResponse(intent=intent, response="Sedang terjadi kesalahan.")
            
            doctor_data = doctor_disease_response.json()
            doctor_name = doctor_data.get("doctor_name", "Nama Dokter tidak diketahui.")
            availability = doctor_data.get("availability", "")

            if not availability.strip():
                response_text = f"{doctor_name} tidak memiliki jadwal tersedia.."
            else:
                response_text = f"Jadwal {doctor_name}:\n\n{availability}"
        
        elif intent == "doctor's availability search by its specialization":
            doctor_specialty_response = requests.post(
                DOCTOR_SPECIALIZATION_URL,
                json={"query": request.query},
                timeout=180
            )
            if doctor_specialty_response.status_code != 200:
                logging.error(f"Doctor Specialty Failed: {doctor_specialty_response.text}")
                return ChatResponse(intent=intent, response="Sedang terjadi kesalahan.")
            
            doctor_data = doctor_specialty_response.json()
            specialty = doctor_data.get("specialty", "Spesialisasi tidak ditemukan.")
            availability = doctor_data.get("availability", "")

            if not availability.strip():
                response_text = f"Tidak ada dokter dengan sepsialisasi: {specialty}."
            else:
                response_text = f"Jadwal dokter dengan spesialisasi {specialty}:\n\n{availability}"

        elif intent == "general query":
            llm_response = requests.post(
                GENERAL_QUERY_URL,
                json={"query": request.query},
                timeout=180
            )
            if llm_response.status_code != 200:
                logging.error(f"LLM service failed: {llm_response.text}")
                return ChatResponse(intent=intent, response="Sedang terjadi kesalahan.")
            
            llm_data = llm_response.json()
            response_text = llm_data.get("response", "Maaf, saya tidak bisa menjawab pertanyaan Anda.")

        else:
            responses = {
                "unanswerable question": "Maaf, saya tidak bisa menjawab pertanyaan ini."
            }
            response_text = responses.get(intent, "Unexpected intent detected.")
        
        return ChatResponse(intent=intent, response=response_text)

    except Exception as e:
        logging.error(f"Unexpected error in chat API: {e}")
        return ChatResponse(intent="error", response="Sedang terjadi kesalahan.")
