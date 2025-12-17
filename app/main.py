import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You help communities understand and draft Community Benefits Agreements (CBAs)."},
            {"role": "user", "content": req.question},
        ],
    )
    return {"answer": resp.choices[0].message.content}
