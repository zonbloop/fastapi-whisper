from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
import shutil
from typing import List
import subprocess
from audio2text import TranscriptionProcessor

app = FastAPI()
transcription_processor = TranscriptionProcessor()
@app.get("/")
def read_root():
    return {"message": "Estoy vivo!"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...),speakers: str = Query(..., title="Speaker Name", description="Name of the speaker"),
                             language: str = Query(..., title="Language", description="Language for speech-to-text")):
    try:
        if file.filename.endswith('.mp3'):
            with open(f"uploaded_files/{file.filename}", "wb") as mp3_file:
                shutil.copyfileobj(file.file, mp3_file)
            #subprocess.run(["python","audio2text.py",f"uploaded_files/{file.filename}",speakers,language])
            return JSONResponse(content=jsonable_encoder({"message": f"File uploaded successfully, speakers= {speakers}, language={language}"}), status_code=200)
        else:
            return JSONResponse(content=jsonable_encoder({"error": "Only MP3 files are allowed"}), status_code=400)
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=500)

@app.get("/get-transcript/")
async def get_text_file(file: str = Query(..., title="file to process", description="File to process")):
    audio_path = 'uploaded_files/' + file
    transcription_processor.run_transcription(audio_path)
    file_path = f"output_files/transcript.txt"
    return FileResponse(file_path, filename=f"transcript.txt", media_type="text/plain")