# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_400_BAD_REQUEST
from pydantic import BaseModel

# ---- LangChain / OpenAI / Vector ----
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# ---- PDF / OCR ----
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ---- HTTP cliente (RUC) ----
import httpx

# =========================
# Settings
# =========================
class Settings:
    BASE_DIR = Path(__file__).parent.resolve()
    UPLOAD_DIR = BASE_DIR / "uploads" / "pdf"
    KNOWLEDGE_DIR = BASE_DIR / "knowledge" / "pdf"   # ← tus PDFs base
    VECTOR_DIR = BASE_DIR / "vector_db"
    BASE_COLLECTION = "licitaciones_base"
    UPLOADS_COLLECTION = "licitaciones_uploads"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() != "false"

    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # p.ej. http://127.0.0.1:5500

    # SRI
    SRI_RUC_URL = (
        "https://srienlinea.sri.gob.ec/"
        "sri-catastro-sujeto-servicio-internet/rest/"
        "ConsolidadoContribuyente/obtenerPorNumerosRuc?&ruc={ruc}"
    )

S = Settings()

# =========================
# FastAPI (API-only)
# =========================
app = FastAPI(title="RAG Licitaciones", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[S.FRONTEND_ORIGIN] if S.FRONTEND_ORIGIN != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# solo montamos uploads para descargar/ver PDFs si hace falta
S.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
S.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
S.VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Utilidades PDF / OCR
# =========================
def _page_has_text(pdf_page: fitz.Page) -> bool:
    try:
        txt = pdf_page.get_text("text")
        return bool(txt and txt.strip())
    except Exception:
        return False

def extract_text_from_pdf(pdf_path: Path, enable_ocr: bool = S.ENABLE_OCR) -> str:
    """
    Extrae texto de PDF. Si una página no tiene capa de texto y enable_ocr=True,
    hace OCR con pytesseract.
    """
    doc = fitz.open(pdf_path)
    parts: List[str] = []
    for p in doc:
        if _page_has_text(p):
            parts.append(p.get_text("text"))
        else:
            if not enable_ocr:
                continue
            pix = p.get_pixmap(dpi=250)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="spa+eng")
            parts.append(text)
    return "\n".join(parts).strip()

def to_documents(text: str, meta: Dict[str, Any]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=S.CHUNK_SIZE, chunk_overlap=S.CHUNK_OVERLAP
    )
    return splitter.split_documents([Document(page_content=text, metadata=meta)])

# =========================
# Vector store helpers
# =========================
def _embeddings():
    return OpenAIEmbeddings(model=S.EMBEDDING_MODEL, openai_api_key=S.OPENAI_API_KEY)

def _ensure_chroma(collection_name: str) -> Chroma:
    return Chroma(
        embedding_function=_embeddings(),
        persist_directory=str(S.VECTOR_DIR),
        collection_name=collection_name
    )

def index_pdfs_in_dir(pdf_dir: Path, collection_name: str) -> int:
    """
    Lee todos los PDFs del directorio, extrae texto (+OCR si toca),
    chunquea e indexa en Chroma.
    """
    vs = _ensure_chroma(collection_name)
    added = 0
    for pdf in pdf_dir.glob("*.pdf"):
        text = extract_text_from_pdf(pdf)
        if not text:
            continue
        docs = to_documents(text, meta={"path": str(pdf), "source": pdf.name, "collection": collection_name})
        if not docs:
            continue
        vs.add_documents(docs)
        added += len(docs)
    vs.persist()
    return added

def index_uploaded_pdf(pdf_path: Path, collection_name: str) -> int:
    vs = _ensure_chroma(collection_name)
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return 0
    docs = to_documents(text, meta={"path": str(pdf_path), "source": pdf_path.name, "collection": collection_name})
    if docs:
        vs.add_documents(docs)
        vs.persist()
    return len(docs)

# =========================
# LLM / QA helpers
# =========================
def make_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model=S.CHAT_MODEL, temperature=temperature, openai_api_key=S.OPENAI_API_KEY)

def make_retriever(collection_name: str):
    vs = _ensure_chroma(collection_name)
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})

def build_rag_qa(retriever):
    mem = ConversationBufferMemory(memory_key="history", return_messages=True)
    llm = make_llm(temperature=0.2)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, memory=mem)

# =========================
# SRI / RUC
# =========================
def _normalize_ruc_payload(payload: Any) -> Dict[str, Any]:
    """
    La API devuelve una lista con un objeto. Normalizamos a:
    {
      ruc, razon_social, estado, actividad_principal, tipo, regimen,
      obligado_contabilidad, contribuyente_especial, agente_retencion,
      habilitado: bool|None
    }
    """
    if isinstance(payload, list) and payload:
        data = payload[0]
    elif isinstance(payload, dict):
        data = payload
    else:
        return {"raw": payload, "habilitado": None}

    estado = data.get("estadoContribuyenteRuc") or data.get("estado") or ""
    habilitado = None
    if isinstance(estado, str):
        # regla básica: ACTIVO => habilitado True
        habilitado = estado.strip().upper() == "ACTIVO"

    return {
        "ruc": data.get("numeroRuc"),
        "razon_social": data.get("razonSocial"),
        "estado": estado,
        "actividad_principal": data.get("actividadEconomicaPrincipal"),
        "tipo": data.get("tipoContribuyente"),
        "regimen": data.get("regimen"),
        "obligado_contabilidad": data.get("obligadoLlevarContabilidad"),
        "contribuyente_especial": data.get("contribuyenteEspecial"),
        "agente_retencion": data.get("agenteRetencion"),
        "fechas": data.get("informacionFechasContribuyente"),
        "habilitado": habilitado,
        "raw": data,
    }

async def fetch_ruc_info(ruc: str) -> Dict[str, Any]:
    url = S.SRI_RUC_URL.format(ruc=ruc)
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        try:
            payload = r.json()
        except Exception:
            return {"raw": r.text, "habilitado": None}
    return _normalize_ruc_payload(payload)

# =========================
# Análisis comparativo
# =========================
ANALYSIS_SYSTEM_PROMPT = """Eres un asistente experto en contratación pública y análisis de licitaciones.
Tu tarea:
1) Segmentar el/los documento(s) del oferente en secciones: Legales, Técnicas, Económicas, Otros.
2) Comparar contra los documentos base (pliegos/contratos de referencia) recuperados por RAG.
3) Producir:
  - Cumplimiento por requisito (OK / Parcial / NO) con breve evidencia (cita o extracto).
  - Riesgos y cláusulas sensibles (multas, garantías, plazos, ambigüedades).
  - Diferencias clave entre base y oferente (montos, plazos, condiciones).
  - Recomendaciones y cláusulas faltantes.
4) Si se provee información del RUC, incluir validaciones básicas de habilitación (si la razón social coincide con el giro declarado).
Devuelve SIEMPRE un JSON *válido* con este esquema:

{
  "resumen": "...",
  "secciones": {
    "legales": ["..."],
    "tecnicas": ["..."],
    "economicas": ["..."],
    "otros": ["..."]
  },
  "cumplimiento": [
    {"requisito": "...", "estado": "OK|PARCIAL|NO", "evidencia": "...", "fuente": "oferente|base"}
  ],
  "riesgos": [
    {"tipo": "legal|tecnico|economico", "detalle": "...", "critico": true|false}
  ],
  "diferencias_clave": [
    {"tema": "plazo|monto|garantias|otros", "base": "...", "oferente": "...", "impacto": "bajo|medio|alto"}
  ],
  "recomendaciones": ["..."],
  "ruc_validacion": {
    "ruc": "...",
    "nombre": "...",
    "habilitado": true|false|null,
    "notas": "..."
  }
}
Si algo no aplica, usa listas vacías o null. NO inventes datos si no hay evidencia.
"""

def json_guard(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"resumen": s.strip()[:800], "error": "El modelo no devolvió JSON puro."}

async def analyze_against_base(
    user_notes: str,
    ruc: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    RAG: consulta colección BASE y UPLOADS (oferente) y genera JSON de análisis.
    """
    base_r = make_retriever(S.BASE_COLLECTION)
    upld_r = make_retriever(S.UPLOADS_COLLECTION)

    base_docs = base_r.get_relevant_documents(user_notes)
    upld_docs = upld_r.get_relevant_documents(user_notes)

    context = ""
    for d in (base_docs + upld_docs):
        src = d.metadata.get("source")
        col = d.metadata.get("collection")
        context += f"\n### SOURCE: {src} ({col})\n{d.page_content}\n"

    ruc_obj = None
    if ruc:
        try:
            ruc_obj = await fetch_ruc_info(ruc)
        except Exception as e:
            ruc_obj = {"error": f"Fallo consultando RUC: {e}", "habilitado": None}

    llm = make_llm(temperature=0.0)
    prompt = (
        ANALYSIS_SYSTEM_PROMPT
        + "\n\n# Notas del solicitante (texto libre):\n"
        + (user_notes or "N/A")
        + "\n\n# Información RUC (normalizada, si existe):\n"
        + json.dumps(ruc_obj, ensure_ascii=False)
        + "\n\n# CONTEXTO RAG (extractos de base y oferente):\n"
        + context[:150000]
    )
    resp = await llm.ainvoke(prompt)
    return json_guard(resp.content)

# =========================
# Schemas / Endpoints
# =========================
class ChatRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    query: str
    ruc: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

@app.on_event("startup")
def startup_index_base():
    added = index_pdfs_in_dir(S.KNOWLEDGE_DIR, S.BASE_COLLECTION)
    print(f"🔎 Base indexada: {added} chunks")

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    retriever = make_retriever(S.BASE_COLLECTION)
    qa = build_rag_qa(retriever)
    res = qa.invoke(req.query)
    return {"result": res["result"]}

@app.post("/api/upload-pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No se recibieron archivos.")
    saved = []
    for f in files:
        if f.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"{f.filename} no es PDF.")
        safe = Path(f.filename).name
        dest = S.UPLOAD_DIR / safe
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        chunks = index_uploaded_pdf(dest, S.UPLOADS_COLLECTION)
        saved.append({"filename": safe, "chunks": chunks})
    return {"ok": True, "files": saved}

@app.get("/api/list-pdf")
async def list_pdf():
    files = []
    for p in S.UPLOAD_DIR.glob("*.pdf"):
        files.append({
            "filename": p.name,
            "disk_path": str(p.resolve()),
            "size_bytes": p.stat().st_size
        })
    return {"count": len(files), "files": files}

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    result = await analyze_against_base(user_notes=req.query, ruc=req.ruc, extra=req.extra)
    return {"ok": True, "analysis": result}

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="templates", html=True), name="front")
