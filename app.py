"""FastAPI web backend for the Investing Chatbot.

It exposes one endpoint, POST /chat, which reuses the RAG logic in
query_data.py. It also serves the chat web page (static/index.html),
so you only need to run ONE server and there are no CORS issues.

Run it with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in your browser.
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Reuse your existing RAG logic. query_data.py calls load_dotenv() on
# import, so your OPENAI_API_KEY is loaded from .env here too.
from query_data import query_rag

app = FastAPI(title="Investing Chatbot")


# ---- Request/response shapes ----
# These tell FastAPI what JSON to expect and what JSON to send back.

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ---- The chat API endpoint ----

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # request.message is the user's question from the frontend.
    answer = query_rag(request.message)
    return ChatResponse(response=answer)


# ---- Serve the frontend ----
# Visiting http://127.0.0.1:8000/ returns the chat page.

@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


# Anything in the static/ folder is available under /static/...
app.mount("/static", StaticFiles(directory="static"), name="static")
