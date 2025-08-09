# backend/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Archivos estáticos (CSS, JS) Hola
app.mount("/static", StaticFiles(directory="static"), name="static")

# Motor de plantillas
templates = Jinja2Templates(directory="templates")

# Habilitar CORS para permitir llamadas desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODELO DE PETICIÓN
class ChatRequest(BaseModel):
    query: str

# Cargar documentos al iniciar
docs = []
for filename in os.listdir("txts"):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join("txts", filename), encoding="utf-8")
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Vectorizar
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vector_db",
    collection_name="txt_conocimiento"
)

# Chatbot con memoria
retriever = vectorstore.as_retriever()
memoria = ConversationBufferMemory()
memoria.chat_memory.add_ai_message("Hola, soy tu IA entrenada con base de conocimientos.")
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0.2)
chat = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, memory=memoria)

# Endpoint principal
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    respuesta = chat.invoke(request.query)
    return {"result": respuesta["result"]}

@app.get("/", response_class=HTMLResponse)
async def serve_chat(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})
