from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from Levenshtein import ratio
import sqlite3

# Initialize the FastAPI app
app = FastAPI()

# Initialize the model
model = OllamaLLM(model="llama3")

# Request and Response Models
class QueryRequest(BaseModel):
    query: str

class DoctorDiseaseResponse(BaseModel):
    doctor_name: str
    availability: str


def extract_disease_or_symptom(query):
    """
    Extracts the disease or symptom from the user's query using the LLM.
    """
    disease_extraction_template = """
    Please extract the disease or symptom mentioned in the user's query.

    Query: {question}

    - If the user query is in Bahasa Indonesia, handle it accordingly.
    - Provide only the disease or symptom as it is (dont translate to English), nothing else.
    - Use proper capitalization for the extracted disease or symptom.
    """
    prompt = ChatPromptTemplate.from_template(disease_extraction_template)
    extraction_chain = prompt | model

    extracted_disease = extraction_chain.invoke({"question": query})
    return extracted_disease.strip()


def get_specialization_from_disease_or_symptom(disease_or_symptom):
    """
    Maps the extracted disease or symptom to a specialization using fuzzy matching.
    """
    conn = sqlite3.connect('doctors.db')
    cursor = conn.cursor()

    # Fetch all diseases and their specializations
    cursor.execute("""
        SELECT disease_or_symptom, specialization
        FROM disease_to_specialization
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    # Use fuzzy matching to find the best match
    best_match = None
    best_ratio = 0.0
    for disease, specialization in rows:
        similarity = ratio(disease_or_symptom.lower(), disease.lower())
        if similarity > best_ratio:
            best_match = specialization
            best_ratio = similarity

    # Apply a threshold to decide if the match is good enough
    threshold = 0.7  # Adjust as needed
    if best_ratio >= threshold:
        return best_match
    else:
        return None


def fetch_doctor_availability_by_specialty(specialty):
    """
    Fetches and returns the doctor's availability and name based on the specialization.
    """
    conn = sqlite3.connect('doctors.db')
    cursor = conn.cursor()

    availability_text = f"Jadwal dokter dengan spesialisasi {specialty}:"
    doctor_name = None

    # Select doctors with the given specialty
    cursor.execute("""
        SELECT id, name
        FROM doctors
        WHERE specialization = ?
    """, (specialty,))
    doctors = cursor.fetchall()

    if doctors:  # If doctors are found for the given specialty
        for doctor in doctors:
            doctor_id, doctor_name = doctor

            # Fetch the doctor's availability
            cursor.execute("""
                SELECT day, time_start, time_end
                FROM availability
                WHERE doctor_id = ?
            """, (doctor_id,))
            availability = cursor.fetchall()

            if availability:  # If availability data exists
                availability_text += f"\nDokter: {doctor_name}"
                for entry in availability:
                    day, time_start, time_end = entry
                    availability_text += f"\nHari: {day}, Pukul: {time_start} - {time_end}"
            else:
                availability_text += f"\nDokter {doctor_name} tidak memiliki jadwal yang tersedia."
    else:
        availability_text += f"\nTidak ditemukan dokter dengan spesialisasi {specialty}."

    conn.close()
    return doctor_name, availability_text

# Microservices check
@app.get("/check")   
def health_check():
    return {"status": "ok"}

@app.post("/doctor-availability-by-disease", response_model=DoctorDiseaseResponse)
async def get_doctor_availability(request: QueryRequest):
    """
    Process the user query, extract the disease or symptom, map it to a specialization, 
    and fetch the doctor's availability for the mapped specialization.
    """
    # Step 1: Extract disease or symptom from user query using LLM
    extracted_disease = extract_disease_or_symptom(request.query)
    
    if not extracted_disease:
        raise HTTPException(status_code=400, detail="Gejala atau penyakit tidak dapat diidentifikasi.")
    
    # Step 2: Map the extracted disease or symptom to a specialization
    specialization = get_specialization_from_disease_or_symptom(extracted_disease)
    
    if not specialization:
        raise HTTPException(status_code=400, detail="Gejala atau penyakit tersebut tidak ditemukan dalam database.")
    
    # Step 3: Fetch and display doctor availability for the mapped specialization
    doctor_name, availability_text = fetch_doctor_availability_by_specialty(specialization)
    
    if not doctor_name:
        raise HTTPException(status_code=404, detail="Tidak ditemukan dokter untuk spesialisasi tersebut.")
    
    return DoctorDiseaseResponse(doctor_name=doctor_name, availability=availability_text)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
