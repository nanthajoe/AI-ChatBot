from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import sqlite3

app = FastAPI()

# Initialize the model
model = OllamaLLM(model="llama3")

# Define the request and response models
class QueryRequest(BaseModel):
    query: str

class DoctorNameResponse(BaseModel):
    doctor: str
    availability: str

def doctor_extraction(query):
    # Define the prompt template
    doctor_extraction_template = """
    Please extract the doctor name from the query.

    Query: {question}

    - The user query may be in Bahasa Indonesia, please handle that. 
    - If you only got the name add "Dr." in front of the name. 
    - Only provide with format Dr. <name>, nothing else.
    - Don't explain anything just return the doctor's name with above format.
    """

    # Define the prompt and chain
    doctor_extractor = ChatPromptTemplate.from_template(doctor_extraction_template)
    doctor_extraction_chain = doctor_extractor | model
    
    # Invoke the chain and get the result
    extracted_doctor = doctor_extraction_chain.invoke({"question": query}).strip()
    return extracted_doctor  # Return the extracted doctor name

# Microservices check
@app.get("/check")   
def health_check():
    return {"status": "ok"}

@app.post("/doctor-availability-by-name", response_model=DoctorNameResponse)
async def get_doctor_availability(request: QueryRequest):
    query = request.query
    
    # Step 1: Extract doctor's name from the query
    extracted_doctor = doctor_extraction(query)
    print(f"Extracted Doctor: {extracted_doctor}")  # Debugging: Print the result

    conn = sqlite3.connect('doctors.db')
    cursor = conn.cursor()

    # Step 2: Fetch doctor's ID from the database
    cursor.execute(f"SELECT id FROM doctors WHERE name = '{extracted_doctor}'")
    doctor = cursor.fetchone()  # Fetch the result (single row)

    if doctor:  # Check if the doctor exists
        doctor_id = doctor[0]  # Get the doctor's ID from the first column

        # Step 3: Fetch doctor's availability
        cursor.execute(f"""
            SELECT day, time_start, time_end
            FROM availability
            WHERE doctor_id = {doctor_id}
        """)
        availability = cursor.fetchall()  # Fetch all availability rows

        # Format the availability response
        availability_text = ""
        for entry in availability:
            if len(entry) == 3:  # Ensure the tuple has 3 values (day, time_start, time_end)
                day, time_start, time_end = entry
                availability_text += f"Hari: {day}, Pukul: {time_start} - {time_end}\n"
            else:
                # Handle any unexpected data structure (optional logging)
                print(f"Data: {entry}, tidak bisa diakses.")

        if not availability_text:
            raise HTTPException(status_code=404, detail="Jadwal dokter tidak bisa ditemukan.")

        conn.close()

        if not availability:
            raise HTTPException(status_code=404, detail="Jadwal dokter tidak bisa ditemukan.")
        
        return DoctorNameResponse(doctor=extracted_doctor, availability=availability_text)
    else:
        conn.close()
        raise HTTPException(status_code=404, detail="Dokter tidak ditemukan.")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
