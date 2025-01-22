from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from Levenshtein import distance
import sqlite3

# Initialize the FastAPI app
app = FastAPI()

# Initialize the model
model = OllamaLLM(model="llama3")

# Predefined specialties
specialties = [
    'Kardiolog', 'Neurolog', 'Dermatolog', 'Pediater', 'Ortopedi', 'Praktisi Umum', 
    'Androlog', 'Endokrin', 'Ahli Bedah', 'Obstetri', 'Onkologi', 'Psikiater', 
    'Ahli Pencernaan', 'Pulmonologi', 'Reumatologi', 'Nephrologi', 'Spesialis THT', 
    'Ahli Alergi', 'Fisioterapis', 'Chiropractor'
]

# Request and Response Models
class SpecialtyRequest(BaseModel):
    query: str

class SpecialtyResponse(BaseModel):
    specialty: str
    availability: str


# Helper function to match specialties using Levenshtein distance
def get_best_match(extracted_specialty, specialties):
    extracted_specialty = extracted_specialty.lower()
    normalized_specialties = {specialty.lower(): specialty for specialty in specialties}

    substring_matches = [
        specialty for specialty_normalized, specialty in normalized_specialties.items()
        if extracted_specialty in specialty_normalized
    ]
    if substring_matches:
        return substring_matches[0]

    matches = {
        specialty: distance(extracted_specialty, specialty_normalized)
        for specialty_normalized, specialty in normalized_specialties.items()
    }

    best_match, best_distance = min(matches.items(), key=lambda x: x[1])
    threshold = 3  # Adjust as needed
    return best_match if best_distance <= threshold else None


# Function to extract the specialty using LLM
def specialty_extraction(query):
    specialty_extraction_template = """
    Please extract the specialty from the query.

    Query: {question}

    - The user query may be in Bahasa Indonesia, please handle that.
    - Provide only the specialty name, nothing else (one word only).
    - Some specialty names are two words (e.g. Praktisi Umum, Ahli Bedah, Ahli Pencernaan, Spesialis THT). You can return two words if included in these specialties.
    - "Praktisi" and "Spesialis" also known as "Dokter".
    - "Ahli" can also refer to "Bedah" (e.g. Dokter Bedah is also known as Dokter Bedah, Ahli Pencernaan also known as Dokter Pencernaan).
    - Don't translate it to English.
    - Capitalize the first letter.
    """
    prompt = ChatPromptTemplate.from_template(specialty_extraction_template)
    specialty_extraction_chain = prompt | model

    extracted_specialty = specialty_extraction_chain.invoke({"question": query}).strip()
    corrected_specialty = get_best_match(extracted_specialty, specialties)
    return corrected_specialty


# Function to fetch doctor availability
def fetch_doctor_availability_by_specialty(specialty):
    conn = sqlite3.connect('doctors.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name FROM doctors WHERE specialization = ?
    """, (specialty,))
    doctors = cursor.fetchall()

    if not doctors:
        conn.close()
        return None, f"Tidak ditemukan dokter dengan spesialisasi {specialty}."

    availability_text = ""
    for doctor_id, doctor_name in doctors:
        cursor.execute("""
            SELECT day, time_start, time_end FROM availability WHERE doctor_id = ?
        """, (doctor_id,))
        availability = cursor.fetchall()

        if availability:
            availability_text += f"\n\nDokter: {doctor_name}"
            for day, time_start, time_end in availability:
                availability_text += f"\nHari: {day}, Pukul: {time_start} - {time_end}"
        else:
            availability_text += f"\n\nDokter {doctor_name} tidak memiliki jadwal yang tersedia."

    conn.close()
    return doctors[0][1], availability_text

# Microservices check
@app.get("/check")   
def health_check():
    return {"status": "ok"}

@app.post("/doctor-availability-by-specialty", response_model=SpecialtyResponse)
async def get_doctor_availability_by_specialty(request: SpecialtyRequest):
    try:
        # Step 1: Extract the specialty
        extracted_specialty = specialty_extraction(request.query)
        if not extracted_specialty:
            raise HTTPException(status_code=400, detail="Spesialisasi tidak dapat diidentifikasi dari query.")

        # Step 2: Fetch doctor availability
        doctor_name, availability_text = fetch_doctor_availability_by_specialty(extracted_specialty)
        if not doctor_name:
            raise HTTPException(status_code=404, detail=availability_text)

        return SpecialtyResponse(specialty=extracted_specialty, availability=availability_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
