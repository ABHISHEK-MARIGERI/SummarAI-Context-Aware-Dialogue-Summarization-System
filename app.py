# fastapi

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import os

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# ------------------- CORS -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- MODEL -------------------
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------- INPUT MODEL -------------------
class DialogueInput(BaseModel):
    dialogue: str

# ------------------- CLEANING -------------------
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

# ------------------- SUMMARIZATION -------------------
def summarize_dialogue(dialogue: str) -> str:
    try:
        dialogue = clean_data(dialogue)

        inputs = tokenizer(
            dialogue,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        print("ERROR:", e)
        return f"Error: {str(e)}"

# ------------------- API -------------------
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

# ------------------- HOME (FIXED) -------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(file_path)