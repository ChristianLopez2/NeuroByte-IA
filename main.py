# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, shutil, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_400_BAD_REQUEST
from pydantic import BaseModel

# ---- LangChain / OpenAI / Vector ---- (SE MANTIENEN TUS LIBRERÍAS)
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import asyncio

# ---- HTTP / RUC ----
import httpx

# ---- PDF / OCR ----
import fitz  # PyMuPDF
from PIL import Image
import easyocr, numpy as np
import pytesseract

# ---- YAML (NUEVO)
import yaml

# =========================
# Settings
# =========================
class Settings:
    BASE_DIR = Path(__file__).parent.resolve()
    UPLOAD_DIR = BASE_DIR / "uploads" / "pdf"
    KNOWLEDGE_DIR = BASE_DIR / "knowledge" / "pdf"   # PDFs base (árbol)
    VECTOR_DIR = BASE_DIR / "vector_db"
    BASE_COLLECTION = "licitaciones_base"
    UPLOADS_COLLECTION = "licitaciones_uploads"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() != "false"

    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # p.ej. http://127.0.0.1:5500

    # SRI
    SRI_RUC_URL = (
        "https://srienlinea.sri.gob.ec/"
        "sri-catastro-sujeto-servicio-internet/rest/"
        "ConsolidadoContribuyente/obtenerPorNumerosRuc?&ruc={ruc}"
    )

    ACTIONABLE_CHAT_RULES = (
    "Tu objetivo es entregar ACCIONES CONCRETAS para que la OFERTA cumpla el pliego.\n"
    "Prohibido responder con 'revisa el pliego', 'consulta el documento', 'ver anexo', ni frases equivalentes.\n"
    "No inventes requisitos: SOLO usa lo presente en el contexto RAG. Si no hay evidencia suficiente, dilo explícitamente.\n"
    "Formato de salida recomendado (en español):\n"
    "- Requisito: <texto corto del requisito detectado en el pliego>\n"
    "  Evidencia en oferta: <extracto o 'no encontrado'>\n"
    "  Brecha: OK | PARCIAL | NO\n"
    "  Acción concreta: <imperativa: qué debe añadir/modificar la oferta>\n"
    "  Documento base: <NOMBRE_SIN_PDF>\n"
    "Evita remitir a leer. Da instrucciones precisas: qué documento anexar, qué campo completar, qué índice calcular (y con qué fórmula si está explícita), etc.\n"
    "Si el requisito no está en el contexto, responde: 'No poseo información en la base para este punto.'\n"
    "Cuando cites documentos, usa SOLO el nombre tras '### DOC:' (sin .pdf ni colección).\n"
    )

    DOC_PATTERNS = [
        ("PLIEGO_COTIZACION", r"\bpliego\b.*(cotizaci[oó]n|bienes y/?o servicios)"),
        ("CONTRATO_CONDICIONES_PARTICULARES", r"condiciones\s+particulares\s+del\s+contrato"),
        ("CONTRATO_CONDICIONES_GENERALES", r"condiciones\s+generales\s+del\s+contrato"),
        ("FORMULARIO_OFERTA", r"\bformulario\b.*(único|oferta|SIE|cotizaci[oó]n)"),
        ("CONSORCIO_COMPROMISO", r"(compromiso).*(asociaci[oó]n|consorcio)"),
        ("SUBASTA_INVERSA_PLIEGO", r"subasta\s+inversa\s+electr[oó]nica"),
    ]

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

# ---- OCR backend (EasyOCR preferido; ya lo traías)
HAS_EASYOCR = False
try:
    EASY_READER = easyocr.Reader(['es','en'], gpu=False)  # descarga pesos la primera vez
    HAS_EASYOCR = True
except Exception:
    EASY_READER = None

def ocr_image(img: Image.Image) -> str:
    # EasyOCR (sin binarios externos)
    if HAS_EASYOCR:
        arr = np.array(img)  # PIL -> numpy
        lines = EASY_READER.readtext(arr, detail=0, paragraph=True)
        return "\n".join(lines)
    # Fallback suave a pytesseract si está instalado
    try:
        return pytesseract.image_to_string(img, lang="spa+eng")
    except Exception:
        return ""

def extract_text_from_pdf(pdf_path: Path, enable_ocr: bool = S.ENABLE_OCR) -> str:
    """
    Extrae texto de PDF. Si una página no tiene capa de texto y enable_ocr=True,
    hace OCR con easyocr/pytesseract (tu flujo original).
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
            parts.append(ocr_image(img))
    return "\n".join(parts).strip()

def extract_text_head(pdf_path: Path, enable_ocr: bool = S.ENABLE_OCR, max_chars: int = 4000) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for p in doc:
        if len(" ".join(parts)) > max_chars: break
        if _page_has_text(p):
            parts.append(p.get_text("text"))
        elif enable_ocr:
            pix = p.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            parts.append(ocr_image(img))
    return "\n".join(parts)[:max_chars]

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

# ======== YAML KB: facetas & bundles (NUEVO) ========
def _norm(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize("NFD", s or "")
        if unicodedata.category(c) != "Mn"
    ).lower()

class KBConfig:
    def __init__(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {}
        self.raw = data
        self.facets = data.get("facets", {})
        self.bundles = data.get("bundles", [])

        # compila regex
        self._compiled = {}
        for facet, items in self.facets.items():
            self._compiled[facet] = []
            for it in items:
                pats = [re.compile(p, re.I) for p in it.get("patterns", [])]
                self._compiled[facet].append((it.get("label"), pats))

    def match_facets(self, text: str) -> dict:
        text = _norm(text or "")
        res = {"familia": None, "procedimiento": None, "doc_role": None}
        for facet in ["familia", "procedimiento", "doc_role"]:
            for label, pats in self._compiled.get(facet, []):
                if any(p.search(text) for p in pats):
                    res[facet] = label
                    break
        return res

    def find_bundle(self, familia: str, procedimiento: str) -> dict | None:
        for b in self.bundles:
            w = b.get("when", {})
            if w.get("familia") == familia and w.get("procedimiento") == procedimiento:
                return b
        return None

CFG = KBConfig(S.KNOWLEDGE_DIR / "kb_config.yaml")

def derive_meta_from_kb_path_and_name(pdf_path: Path) -> Dict[str, Any]:
    # 1) facetas por ruta+nombre
    try:
        rel = pdf_path.relative_to(S.KNOWLEDGE_DIR)
    except Exception:
        rel = pdf_path
    path_text = " / ".join([p.name for p in rel.parents if p.name][::-1]) + " / " + pdf_path.name

    m = CFG.match_facets(path_text)
    if not m.get("doc_role"):
        # 2) si falta rol, intentamos con el head del PDF
        head = extract_text_head(pdf_path)
        m2 = CFG.match_facets(head)
        for k in ("familia","procedimiento","doc_role"):
            if not m.get(k) and m2.get(k): m[k] = m2[k]

    grupo = str(rel).split(os.sep, 1)[0] if isinstance(rel, Path) else None
    return {
        "kb_grupo": grupo,                   # "01_normativa", etc.
        "familia": m.get("familia"),
        "procedimiento": m.get("procedimiento"),
        "doc_role": m.get("doc_role"),
    }

def index_knowledge_tree(kb_root: Path, collection_name: str) -> int:
    """
    Indexa TODO el árbol de knowledge/pdf con metadatos de YAML.
    Usa 'source' = ruta relativa (para dedupe consistente).
    """
    vs = _ensure_chroma(collection_name)
    added = 0

    # existentes
    existing_sources = set()
    try:
        got = vs._collection.get(include=["metadatas"])
        for md in (got.get("metadatas") or []):
            if md and md.get("source"): existing_sources.add(md["source"])
    except Exception:
        pass

    for pdf in kb_root.rglob("*.pdf"):
        rel = pdf.relative_to(kb_root).as_posix()
        if rel in existing_sources:
            continue

        meta_extra = derive_meta_from_kb_path_and_name(pdf)
        text = extract_text_from_pdf(pdf)
        if not text:
            continue

        meta = {
            "path": str(pdf),
            "source": rel,
            "collection": collection_name,
            **meta_extra
        }
        docs = to_documents(text, meta=meta)
        if docs:
            vs.add_documents(docs)
            added += len(docs)

    try:
        vs.persist()
    except Exception:
        pass

    return added

# ======== uploads: clasificación con YAML (además del doctype heurístico) ========
def guess_doctype(filename: str, sample_text: str = "") -> str:
    name = filename.lower()
    text = (sample_text or "").lower()
    hay = name + "\n" + text[:3000]
    for label, pat in S.DOC_PATTERNS:
        if re.search(pat, hay, re.I):
            return label
    if "pliego" in name: return "PLIEGO_COTIZACION"
    if "formulario" in name: return "FORMULARIO_OFERTA"
    if "condiciones_particulares" in name and "contrato" in name: return "CONTRATO_CONDICIONES_PARTICULARES"
    if "condiciones_generales" in name and "contrato" in name: return "CONTRATO_CONDICIONES_GENERALES"
    if "consorcio" in name or "asociacion" in name or "asociación" in name: return "CONSORCIO_COMPROMISO"
    if "subasta" in name: return "SUBASTA_INVERSA_PLIEGO"
    return "OTROS"

def classify_upload(pdf_path: Path) -> Dict[str, Any]:
    head = extract_text_head(pdf_path)
    hay = f"{pdf_path.name}\n{head}"
    m = CFG.match_facets(hay)
    return {
        "familia": m.get("familia"),
        "procedimiento": m.get("procedimiento"),
        "doc_role": None,       # uploads no son "roles BASE"
        "upload_kind": "oferta"
    }

def index_uploaded_pdf(pdf_path: Path, collection_name: str) -> int:
    vs = _ensure_chroma(collection_name)
    # borrar previos de ese source
    try:
        vs._collection.delete(where={"source": {"$eq": pdf_path.name}})
    except Exception:
        pass

    head = extract_text_head(pdf_path)
    doctype = guess_doctype(pdf_path.name, head)
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return 0

    u = classify_upload(pdf_path)
    meta = {
        "path": str(pdf_path),
        "source": pdf_path.name,            # uploads: nombre simple
        "collection": collection_name,
        "doctype": doctype,
        **u
    }
    docs = to_documents(text, meta=meta)
    if docs:
        vs.add_documents(docs)
        try:
            vs.persist()
        except Exception:
            pass
    return len(docs)

# =========================
# LLM / QA helpers
# =========================
def make_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model=S.CHAT_MODEL, temperature=temperature, openai_api_key=S.OPENAI_API_KEY)

# =========================
# SRI / RUC
# =========================
def _normalize_ruc_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list) and payload:
        data = payload[0]
    elif isinstance(payload, dict):
        data = payload
    else:
        return {"raw": payload, "habilitado": None}

    estado = data.get("estadoContribuyenteRuc") or data.get("estado") or ""
    habilitado = None
    if isinstance(estado, str):
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
    intentos = 3

    for intento in range(1, intentos + 1):
        try:
            async with httpx.AsyncClient(timeout=50) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    try:
                        payload = r.json()
                        return _normalize_ruc_payload(payload)
                    except Exception:
                        return {
                            "ruc": ruc,
                            "razon_social": None,
                            "estado": "No se pudo procesar la respuesta",
                            "habilitado": None,
                            "raw": r.text
                        }
                else:
                    await asyncio.sleep(1)
        except Exception:
            await asyncio.sleep(1)

    return {
        "ruc": ruc,
        "razon_social": None,
        "estado": "Error al consultar",
        "habilitado": None,
        "raw": None
    }

# =========================
# Análisis comparativo
# =========================
ANALYSIS_SYSTEM_PROMPT = """Eres un experto en licitaciones y contratación pública.  
Analiza los documentos subidos por el usuario (OFERTA) comparándolos con los documentos de la base de conocimiento (BASE) que correspondan a su tipo.  
Tu objetivo es generar un análisis completo en formato JSON siguiendo exactamente el esquema proporcionado.

### Instrucciones:
1. **Segmenta** el contenido de la OFERTA en secciones:  
   - legales  
   - técnicas  
   - económicas  
   - otros  

2. **Compara** cada sección con los documentos base del mismo tipo de documento.  
   Usa las diferencias y similitudes para construir:
   - Cumplimiento por requisito (OK, PARCIAL, NO) con breve evidencia y fuente (oferente o base).
   - Riesgos y cláusulas sensibles: multas, garantías, plazos, ambigüedades.  
     Marca critico como true si el riesgo es alto.
   - Diferencias clave: indicar tema (plazo, monto, garantias, otros), contenido en base y en oferta, y su impacto (bajo, medio, alto).
   - Recomendaciones y cláusulas faltantes.

3. **RUC**: El sistema obtendrá esta información automáticamente para cada documento; debes incluir la clave "ruc_validacion" en tu salida, con el siguiente formato:
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

# =========================
# Schemas / Endpoints
# =========================
class ProcessHint(BaseModel):
    familia: Optional[str] = None
    procedimiento: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    focus_upload: Optional[str] = None
    process: Optional[ProcessHint] = None   # <-- NUEVO: hint desde el frontend

@app.on_event("startup")
def startup_index_base():
    # Indexar la BASE con facetas/bundles de YAML (árbol completo)
    index_knowledge_tree(S.KNOWLEDGE_DIR, S.BASE_COLLECTION)

    # Limpiar carpeta uploads y sincronizar índice
    if os.path.exists(S.UPLOAD_DIR):
        for filename in os.listdir(S.UPLOAD_DIR):
            filepath = os.path.join(S.UPLOAD_DIR, filename)
            try:
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception:
                print('')
    sync_uploads_index()

@app.get("/api/processes")
def list_processes():
    procs = []
    for b in CFG.bundles:
        w = b.get("when", {})
        if w.get("familia") and w.get("procedimiento"):
            procs.append(w)
    seen, out = set(), []
    for p in procs:
        key = (p["familia"], p["procedimiento"])
        if key not in seen:
            seen.add(key); out.append(p)
    return {"count": len(out), "items": out}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Un solo prompt para todo:
    - respuesta1: narrativa por archivo (texto humano) + comparativo en texto
    - respuesta2: JSON de decisión por archivo (procesoId, referencia, requisitos, oferentes, pesosCategoria)
    """
    user_q = (req.query or "").strip() or "Evalúa cumplimiento por oferta, detecta brechas y acciones de mejora concretas."
    process_hint = req.process.dict() if req.process else None

    # 1) Detectar ofertas vivas y construir contexto por oferta
    vs_up = _ensure_chroma(S.UPLOADS_COLLECTION)
    live_stems = sorted({Path(p.name).stem for p in S.UPLOAD_DIR.glob("*.pdf")})
    if not live_stems:
        return {
            "respuesta1": {"por_archivo": [], "comparativo_texto": "Sin datos."},
            "respuesta2": {"por_archivo": []}
        }

    async def build_ctx_for_stem(stem: str, k_base: int = 8, k_up: int = 8) -> Tuple[str, dict, dict]:
        # metadatos de la oferta en índice
        live = list(S.UPLOAD_DIR.glob(f"{stem}*.pdf"))
        meta = vs_up._collection.get(where={"source": {"$in": [p.name for p in live]}}, include=["metadatas"])
        up_meta = (meta.get("metadatas") or [None])[0] or {}

        # Oferta (chunks del mismo stem)
        up_ret = vs_up.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_up, "fetch_k": 40, "filter": {"source": {"$in": [p.name for p in live]}}}
        )
        up_docs = up_ret.get_relevant_documents(user_q)

        # Base (picker por YAML/hint)
        base_docs = pick_base_docs_for_upload(user_q, up_meta, k_per_role=max(1, k_base // 3), process_hint=process_hint)

        parts = []
        for d in base_docs:
            parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (BASE)\n{d.page_content}\n")
        for d in up_docs:
            parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (OFERTA)\n{d.page_content}\n")

        ctx = "".join(parts)
        # Candidatos "anti-fantasía" + RUC externo
        cands = extract_candidates(ctx, stem)
        ruc_info = await validate_ruc_for_offer(stem)
        return ctx, cands, ruc_info

    tasks = [build_ctx_for_stem(stem) for stem in live_stems]
    ctx_pack = await asyncio.gather(*tasks)

    # 2) Armar BLOQUES para el prompt único
    bloques_txt = []
    for stem, (ctx, cands, ruc) in zip(live_stems, ctx_pack):
        bloques_txt.append(
            f"\n=== OFERTA: {stem} ===\n"
            f"# CANDIDATOS_CONFIABLES\n{json.dumps(cands, ensure_ascii=False)}\n"
            f"# RUC_VALIDACION\n{json.dumps(ruc, ensure_ascii=False)}\n"
            f"# CONTEXTO RAG (recortado)\n{ctx[:150000]}\n"
        )
    bloques_joined = "\n".join(bloques_txt)

    # 3) Un (1) llamado al LLM
    llm = make_llm(temperature=0.0)
    prompt = UNIFIED_MASTER_PROMPT.replace("{USER_Q}", user_q).replace("{BLOQUES}", bloques_joined)
    resp = await llm.ainvoke(prompt)
    data = json_guard(resp.content)

    # 4) Validaciones mínimas y rellenos seguros
    # Esperamos exactamente "respuesta1" y "respuesta2"
    if not isinstance(data, dict):
        return {
            "respuesta1": {"por_archivo": [], "comparativo_texto": "No se pudo estructurar el análisis."},
            "respuesta2": {"por_archivo": []},
            "raw": resp.content
        }

    # Asegurar estructura esperada
    data.setdefault("respuesta1", {})
    data.setdefault("respuesta2", {})
    data["respuesta1"].setdefault("por_archivo", [])
    data["respuesta1"].setdefault("comparativo_texto", "")
    data["respuesta2"].setdefault("por_archivo", [])

    # 5) Ordenar por archivo para consistencia visual
    try:
        data["respuesta1"]["por_archivo"] = sorted(
            data["respuesta1"]["por_archivo"],
            key=lambda x: (x or {}).get("archivo","")
        )
        data["respuesta2"]["por_archivo"] = sorted(
            data["respuesta2"]["por_archivo"],
            key=lambda x: (x or {}).get("archivo","")
        )
    except Exception:
        pass

    return data


    # === Análisis por oferta: contexto más afinado con picker YAML ===
    async def build_context_for_offer_config(stem: str, k_base: int = 8, k_up: int = 8) -> str:
        vs_up = _ensure_chroma(S.UPLOADS_COLLECTION)
        live = list(S.UPLOAD_DIR.glob(f"{stem}*.pdf"))

        meta = vs_up._collection.get(where={"source": {"$in": [p.name for p in live]}}, include=["metadatas"])
        m = (meta.get("metadatas") or [None])[0] or {}

        # OFERTA (chunks del mismo stem)
        up_ret = vs_up.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_up, "fetch_k": 40, "filter": {"source": {"$in": [p.name for p in live]}}}
        )
        up_docs = up_ret.get_relevant_documents(user_q)

        # BASE (picker por YAML)
        base_docs = pick_base_docs_for_upload(user_q, m, k_per_role=max(1, k_base // 3), process_hint=process_hint)

        parts = []
        for d in base_docs:
            parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (BASE)\n{d.page_content}\n")
        for d in up_docs:
            parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (OFERTA)\n{d.page_content}\n")
        return "".join(parts)

    async def analyze_offer_json(stem: str, ruc_info: dict) -> dict:
        ctx_offer = await build_context_for_offer_config(stem, k_base=8, k_up=8)
        anal_prompt = (
            ANALYSIS_SYSTEM_PROMPT
            + "\n\n# CONTEXTO RAG por OFERTA (BASE y OFERTA pertinentes):\n"
            + ctx_offer[:150000]
            + "\n\nDevuelve SOLO el JSON solicitado, sin texto adicional."
        )
        resp = await make_llm(temperature=0.0).ainvoke(anal_prompt)
        anal = json_guard(resp.content)
        anal["ruc_validacion"] = {
            "ruc": ruc_info.get("ruc"),
            "nombre": ruc_info.get("razon_social"),
            "habilitado": ruc_info.get("habilitado"),
            "notas": ruc_info.get("notas") or ruc_info.get("estado") or None
        }
        anal["_archivo"] = stem
        return anal

    per_offer_tasks = []
    for item in por_oferta:
        stem = Path((item.get("archivo") or "oferta")).stem
        ruc_info = item.get("ruc_validacion") or {}
        per_offer_tasks.append(analyze_offer_json(stem, ruc_info))

    per_offer_analysis = await asyncio.gather(*per_offer_tasks) if per_offer_tasks else []

    return {
        "result": result_text,
        "analysis": data,
        "per_offer_analysis": per_offer_analysis
    }

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

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="templates", html=True), name="front")

def current_upload_sources() -> set[str]:
    return {p.name for p in S.UPLOAD_DIR.glob("*.pdf")}

def delete_upload_docs_where(where: Dict[str, Any]) -> int:
    vs = _ensure_chroma(S.UPLOADS_COLLECTION)
    vs._collection.delete(where=where)
    return 1

def sync_uploads_index() -> dict:
    deleted = 0
    vs = _ensure_chroma(S.UPLOADS_COLLECTION)
    keep = sorted(list(current_upload_sources()))

    if keep:
        vs._collection.delete(where={"source": {"$nin": keep}})
    else:
        got = vs._collection.get(include=[])
        ids = got.get("ids") or []
        if ids:
            vs._collection.delete(ids=ids)
            deleted = len(ids)
    try:
        vs.persist()
    except Exception:
        pass
    return {"kept": keep, "deleted": deleted}

@app.post("/api/sync-uploads")
async def api_sync_uploads():
    info = sync_uploads_index()
    return {"ok": True, "synced": info}

# === RUC helpers por oferta ===
RUC_RE = re.compile(r"\b(\d{13})\b")  # RUC Ecuador: 13 dígitos

def extract_ruc_from_offer_stem(stem: str) -> Optional[str]:
    pdf_path = S.UPLOAD_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        candidates = list(S.UPLOAD_DIR.glob(f"{stem}*.pdf"))
        if not candidates:
            return None
        pdf_path = candidates[0]
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None
    m = RUC_RE.search(text)
    return m.group(1) if m else None

async def validate_ruc_for_offer(stem: str) -> Dict[str, Any]:
    ruc = extract_ruc_from_offer_stem(stem)
    if not ruc:
        return {"ruc": None, "habilitado": None, "notas": "RUC no encontrado en el documento"}
    try:
        info = await fetch_ruc_info(ruc)
        return {
            "ruc": info.get("ruc") or ruc,
            "habilitado": info.get("habilitado"),
            "razon_social": info.get("razon_social") or (info.get("raw", {}) or {}).get("razonSocial"),
            "estado": info.get("estado"),
            "actividad_principal": info.get("actividad_principal"),
            "tipo": info.get("tipo"),
            "regimen": info.get("regimen"),
            "obligado_contabilidad": info.get("obligado_contabilidad"),
            "contribuyente_especial": info.get("contribuyente_especial"),
            "agente_retencion": info.get("agente_retencion"),
            "notas": "OK" if info.get("habilitado") is not None else "Sin confirmación de habilitación"
        }
    except Exception as e:
        return {"ruc": ruc, "habilitado": None, "notas": f"Error consultando SRI: {e}"}

GROUPED_ACTIONABLE_SCHEMA = """
Devuelve SIEMPRE un JSON VÁLIDO con este esquema:

{
  "resumen_general": "<texto corto>",
  "por_oferta": [
    {
      "archivo": "<NOMBRE_DE_LA_OFERTA_SIN_PDF>",
      "estado": "SIN_INCONSISTENCIAS|CON_INCONSISTENCIAS|SIN_EVIDENCIA",
      "ruc_validacion": {
        "ruc": "<13 dígitos o null>",
        "habilitado": true|false|null,
        "razon_social": "<string|null>",
        "estado": "<string|null>",
        "actividad_principal": "<string|null>",
        "tipo": "<string|null>",
        "regimen": "<string|null>",
        "notas": "<string>"
      },
      "inconsistencias": [
        {
          "requisito": "<texto corto del requisito detectado en la BASE>",
          "evidencia_oferta": "<extracto o 'no encontrado'>",
          "accion_concreta": "<qué debe añadir/modificar>",
          "documento_base": "<NOMBRE_DOC_BASE_SIN_PDF|null>",
          "tipo": "legal|tecnico|economico|otro"
        }
      ],
      "cumplimientos": [
        {
          "requisito": "<texto corto del requisito de la BASE que sí se cumple>",
          "evidencia_oferta": "<extracto que demuestra el cumplimiento>",
          "documento_base": "<NOMBRE_DOC_BASE_SIN_PDF|null>",
          "tipo": "legal|tecnico|economico|otro"
        }
      ],
      "semaforo": "aprobado|faltan_requisitos|no_cumple"
    }
  ],
  "comparativo": {
    "mejor_cumplimiento": "<archivo o 'empate' o 'sin_datos'>",
    "diferencias_relevantes": [
      {"tema": "plazo|monto|garantias|otros", "detalle": "<texto>", "impacto": "bajo|medio|alto", "base": "<opcional>", "oferta": "<opcional>"}
    ]
  }
}

REGLAS:
- Prohibido responder con “revisa el pliego”, “consulta el documento” o equivalentes.
- No inventes requisitos: SOLO usa lo presente en el CONTEXTO RAG.
- Si no hay evidencia suficiente en la base para evaluar un punto, marca el archivo como "SIN_EVIDENCIA" o la inconsistencia con "evidencia_oferta": "no encontrado" y "documento_base": null.
- Cuando cites documentos (base u oferta), usa SOLO el nombre tras '### DOC:' (sin .pdf ni colección).
- Calcula 'semaforo' así: aprobado=0 inconsistencias; faltan_requisitos=1-2 inconsistencias no críticas; no_cumple=3+ o inconsistencias críticas si se deduce del contexto.
- Si una oferta no presenta inconsistencias, rellena 'cumplimientos' con 3–10 puntos concretos que SÍ cumple (cada uno con evidencia corta y referencia al documento base).
"""

def _and_filter(*clauses: dict) -> dict:
    clauses = [c for c in clauses if c]
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def pick_base_docs_for_upload(
    query: str,
    upload_meta: dict,
    k_per_role: int = 3,
    process_hint: Optional[dict] = None
) -> list:
    """
    Selecciona documentos BASE relevantes según:
    - facetas YAML (familia/procedimiento/doc_role, bundles)
    - hint opcional del front
    """
    vs_base = _ensure_chroma(S.BASE_COLLECTION)

    familia = (process_hint or {}).get("familia") or (upload_meta or {}).get("familia")
    proc    = (process_hint or {}).get("procedimiento") or (upload_meta or {}).get("procedimiento")

    chosen = []
    bundle = CFG.find_bundle(familia, proc)

    def fetch_by(where_filter, k=6):
        ret = vs_base.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 50, "filter": where_filter}
        )
        return ret.get_relevant_documents(query)

    if bundle:
        roles = bundle.get("include_roles", [])
        for role in roles:
            docs = fetch_by(
                _and_filter(
                    {"familia": {"$eq": familia}} if familia else None,
                    {"procedimiento": {"$eq": proc}} if proc else None,
                    {"doc_role": {"$eq": role}}
                ),
                k=k_per_role
            )
            if not docs and familia:
                docs = fetch_by(
                    _and_filter({"familia": {"$eq": familia}}, {"doc_role": {"$eq": role}}),
                    k=k_per_role
                )
            if not docs:
                docs = fetch_by({"doc_role": {"$eq": role}}, k=k_per_role)
            chosen.extend(docs)

        # anexos/plus
        add_roles = bundle.get("also_include", [])
        for role in add_roles:
            chosen.extend(fetch_by({"doc_role": {"$eq": role}}, k=1))
    else:
        base_docs = []
        if familia or proc:
            where_fp = _and_filter(
                {"familia": {"$eq": familia}} if familia else None,
                {"procedimiento": {"$eq": proc}} if proc else None,
            )
            if where_fp:
                base_docs = fetch_by(where_fp, k=10)

        if not base_docs:
            # fallback: normativa/modelos
            base_docs = fetch_by({"kb_grupo": {"$in": ["01_normativa", "02_modelos_pliegos"]}}, k=10)
        chosen.extend(base_docs)

    # dedup por 'source'
    seen, final = set(), []
    for d in chosen:
        s = (d.metadata or {}).get("source")
        if s and s not in seen:
            seen.add(s); final.append(d)
    return final

def build_multi_offer_context(
    query: str,
    k_base: int = 8,
    k_up: int = 8,
    process_hint: Optional[dict] = None
) -> Tuple[str, list[str]]:
    vs_up = _ensure_chroma(S.UPLOADS_COLLECTION)
    live = list(S.UPLOAD_DIR.glob("*.pdf"))
    if not live:
        return "", []

    up_meta = vs_up._collection.get(where={"source": {"$in": [p.name for p in live]}}, include=["metadatas"])
    metas = up_meta.get("metadatas") or []
    by_stem: Dict[str, Dict[str, Any]] = {}
    for m in metas:
        if not m: continue
        src = m.get("source") or ""
        by_stem[Path(src).stem] = m

    context_parts, offers = [], sorted(by_stem.keys())
    for stem in offers:
        # OFERTA (mismo stem)
        up_ret = vs_up.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_up, "fetch_k": 40, "filter": {"source": {"$in": [p.name for p in live if Path(p).stem == stem]}}}
        )
        up_docs = up_ret.get_relevant_documents(query)
        for d in up_docs:
            context_parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (OFERTA)\n{d.page_content}\n")

        # BASE (según YAML/hint)
        base_docs = pick_base_docs_for_upload(query, by_stem[stem], k_per_role=max(1, k_base // 3), process_hint=process_hint)
        for d in base_docs:
            context_parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (BASE)\n{d.page_content}\n")

    return "".join(context_parts), offers



# =========================
# Helpers para decisión "anti-fantasía"
# =========================
# =========================
# Helpers para decisión "anti-fantasía"
# =========================
PROC_ID_RE = re.compile(r'\b[A-Z]{2,}(?:-[A-Z0-9]{2,}){1,4}-\d{4}-\d{2,4}\b')
MONEY_RE   = re.compile(r'(?:(?:USD|US\$|\$)\s?)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d+)\b')
DAYS_RE    = re.compile(r'\b(\d{1,4})\s*d[ií]as\b', re.IGNORECASE)
CLAUSE_RE  = re.compile(r'cl[áa]usula\s+([0-9A-Za-z\.\-]+)', re.IGNORECASE)

# Requisitos: solo líneas con "verbo de exigencia" y NO títulos comunes
REQ_POS = (
    "deberá", "debe", "deben", "deberán", "presentar", "adjuntar", "entregar",
    "contar con", "acreditar", "vigencia", "garantía", "garantias", "garantías",
    "certificación", "certificacion", "iso", "ruc", "experiencia", "plazo", "cronograma"
)
REQ_NEG = ("conclusiones", "clausura", "análisis", "analisis", "índice", "indice", "marco", "glosario")

def _split_ctx_by_role(ctx: str) -> Tuple[str, str]:
    base_parts, ofer_parts = [], []
    for block in re.split(r'\n(?=### DOC: )', ctx or ""):
        head = (block.splitlines() or [""])[0].lower()
        if "(base)" in head:
            base_parts.append(block)
        elif "(oferta)" in head:
            ofer_parts.append(block)
    return "\n".join(base_parts), "\n".join(ofer_parts)

def _to_int_safe(s: str) -> Optional[int]:
    try:
        s = s.replace('.', '').replace(',', '')
        return int(float(s))
    except Exception:
        return None

def _harvest_requirement_lines(base_text: str, max_items: int = 25) -> list[str]:
    out = []
    for raw in (base_text or "").splitlines():
        line = raw.strip()
        if not (8 <= len(line) <= 300):
            continue
        low = line.lower()
        if any(bad in low for bad in REQ_NEG):
            continue
        if any(tok in low for tok in REQ_POS):
            out.append(line)
        if len(out) >= max_items:
            break
    # dedup conservando orden
    seen, final = set(), []
    for t in out:
        k = t.lower()
        if k not in seen:
            seen.add(k); final.append(t)
    return final

def _find_numeric_near_keywords(text: str, value_re: re.Pattern, keywords: tuple[str, ...], window: int = 120) -> Optional[int]:
    if not text:
        return None
    low = text.lower()
    for m in value_re.finditer(text):
        start, end = m.span()
        left = max(0, start - window)
        right = min(len(text), end + window)
        ctx = low[left:right]
        if any(k in ctx for k in keywords):
            val = _to_int_safe(m.group(1) if m.lastindex else m.group(0))
            if val is not None:
                return val
    return None

# Palabras clave (contexto)
BASE_COST_KW   = ("presupuesto", "referencial", "estimado", "base", "precio referencial", "monto referencial", "valor referencial")
BASE_PLAZO_KW  = ("plazo", "ejecución", "ejecucion", "entrega")
OF_COST_KW     = ("oferta", "ofertado", "propuesta", "cotización", "cotizacion", "valor de la oferta", "monto de la oferta")
OF_PLAZO_KW    = ("plazo", "oferta", "ejecución", "ejecucion", "entrega")

def extract_candidates(ctx_offer: str, archivo_stem: str) -> dict:
    base_text, oferta_text = _split_ctx_by_role(ctx_offer)

    # procesoId
    proc_ids  = list(dict.fromkeys(PROC_ID_RE.findall(base_text)))

    # costos/plazos del BASE: con contexto; si no hay, fallback muy conservador (None)
    base_cost = _find_numeric_near_keywords(base_text, MONEY_RE, BASE_COST_KW)  # uno bueno basta
    base_days = _find_numeric_near_keywords(base_text, DAYS_RE, BASE_PLAZO_KW)

    # costos/plazos de la OFERTA: con contexto
    oferta_cost = _find_numeric_near_keywords(oferta_text, MONEY_RE, OF_COST_KW)
    oferta_days = _find_numeric_near_keywords(oferta_text, DAYS_RE, OF_PLAZO_KW)

    # requisitos del BASE (frases literales de exigencia)
    req_lines = _harvest_requirement_lines(base_text)

    # posibles cláusulas
    clauses = list(dict.fromkeys(CLAUSE_RE.findall(base_text)))

    return {
        "archivo": archivo_stem,
        "base": {
            "procesoIds": proc_ids[:5],
            "costos": [base_cost] if base_cost is not None else [],
            "plazos_dias": [base_days] if base_days is not None else [],
            "req_samples": req_lines[:20],
            "clausulas": clauses[:20]
        },
        "oferta": {
            "costos": [oferta_cost] if oferta_cost is not None else [],
            "plazos_dias": [oferta_days] if oferta_days is not None else [],
            "nombres_posibles": [archivo_stem]  # fallback seguro
        }
    }


DECISION_PROMPT_BASE = """
Eres analista de licitaciones. Devuelve SOLO un JSON **válido** EXACTAMENTE con esta estructura:

{
  "procesoId": "<string|null>",
  "referencia": { "costo": <number|null>, "plazo_dias": <number|null> },
  "requisitos": [
    { "id": "R-001", "categoria": "Legal|Tecnica|Economica", "sub": "<string|null>", "texto": "<string>", "peso": <number>, "clausula": "<string|null>", "pagina": <number|null> }
  ],
  "oferentes": [
    {
      "id": "O1",
      "nombre": "<string|null>",
      "costo": <number|null>,
      "plazo_dias": <number|null>,
      "cumplimientos": [
        { "reqId": "R-001", "estado": "CUMPLE|PARCIAL|NO CUMPLE" }
      ],
      "riesgos": [
        { "nivel": "ALTO|MEDIO|BAJO", "clausula": "<string|null>", "pagina": <number|null>, "descripcion": "<string>" }
      ]
    }
  ],
  "pesosCategoria": { "Legal": <number>, "Tecnica": <number>, "Economica": <number> }
}

REGLAS ESTRICTAS (NO INVENTES):
- Usa SOLO los CANDIDATOS_CONFIABLES provistos abajo; si el dato no está en esa lista, devuelve null (o [] según corresponda).
- "procesoId" debe salir EXACTAMENTE de CANDIDATOS_CONFIABLES.base.procesoIds; si está vacía → null.
- "referencia": costo/plazo SOLO de CANDIDATOS_CONFIABLES.base.(costos|plazos_dias); si nada coincide → null.
- "oferentes[0]": "nombre" SOLO de CANDIDATOS_CONFIABLES.oferta.nombres_posibles; "costo"/"plazo_dias" SOLO de CANDIDATOS_CONFIABLES.oferta.(costos|plazos_dias).
- "requisitos": cada "texto" debe ser una **cita literal** de CANDIDATOS_CONFIABLES.base.req_samples. No parafrasear. No inventar. 5–20 ítems si hay suficientes.
- "clausula" intenta mapear usando CANDIDATOS_CONFIABLES.base.clausulas o pon null.
- "peso": si no hay criterio explícito, usa 1.0; rango permitido [0.5, 2.0].
- "cumplimientos": determina CUMPLE/PARCIAL/NO CUMPLE comparando la frase literal del requisito con el contenido OFERTA; si no hay evidencia → "NO CUMPLE".
- "pesosCategoria": si no hay información explícita, usa { "Legal":0.4, "Tecnica":0.4, "Economica":0.2 }.
"""

def _sanitize_decision_with_candidates(decision: dict, cands: dict) -> dict:
    d = dict(decision or {})
    base_c = cands.get("base", {})
    of_c   = cands.get("oferta", {})

    # procesoId
    if d.get("procesoId") not in (base_c.get("procesoIds") or []):
        d["procesoId"] = None

    # referencia
    ref = d.get("referencia") or {}
    if ref.get("costo") not in (base_c.get("costos") or []):
        ref["costo"] = None
    if ref.get("plazo_dias") not in (base_c.get("plazos_dias") or []):
        ref["plazo_dias"] = None
    d["referencia"] = ref

    # oferente 0
    ofs = d.get("oferentes") or []
    if ofs:
        of0 = dict(ofs[0])
        if of0.get("nombre") not in (of_c.get("nombres_posibles") or []):
            of0["nombre"] = (of_c.get("nombres_posibles") or [None])[0]
        if of0.get("costo") not in (of_c.get("costos") or []):
            of0["costo"] = None
        if of0.get("plazo_dias") not in (of_c.get("plazos_dias") or []):
            of0["plazo_dias"] = None
        ofs[0] = of0
        d["oferentes"] = ofs

    # requisitos: texto debe estar en req_samples
    allow = set((base_c.get("req_samples") or []))
    reqs = []
    for i, r in enumerate(d.get("requisitos") or [], start=1):
        txt = (r or {}).get("texto") or ""
        if txt in allow:
            r = dict(r)
            r["id"] = f"R-{i:03d}"
            try:
                w = float(r.get("peso", 1.0))
                if not (0.5 <= w <= 2.0): r["peso"] = 1.0
            except Exception:
                r["peso"] = 1.0
            reqs.append(r)
    d["requisitos"] = reqs

    # pesos por defecto
    d.setdefault("pesosCategoria", {"Legal": 0.4, "Tecnica": 0.4, "Economica": 0.2})
    return d

async def build_decision_json_for_offer(stem: str, ctx_offer: str) -> dict:
    cands = extract_candidates(ctx_offer, stem)
    prompt = (
        DECISION_PROMPT_BASE
        + "\n\n# CANDIDATOS_CONFIABLES (elige EXCLUSIVAMENTE de aquí; si no hay, usa null/[]):\n"
        + json.dumps(cands, ensure_ascii=False)
        + "\n\n# CONTEXTO RAG (BASE y OFERTA pertinentes):\n"
        + ctx_offer[:150000]
        + "\n\nDevuelve SOLO el JSON solicitado, sin texto adicional."
    )
    resp = await make_llm(temperature=0.0).ainvoke(prompt)
    raw = json_guard(resp.content) or {}
    clean = _sanitize_decision_with_candidates(raw, cands)
    clean["_archivo"] = stem
    return clean

# === PROMPT UNIFICADO (un solo llamado al LLM) ===
UNIFIED_MASTER_PROMPT = """
Eres un experto en licitaciones y contratación pública. Recibirás varios BLOQUES_DE_ENTRADA,
uno por cada OFERTA, y para todos ellos deberás producir **un único JSON final** con dos partes:

- respuesta1: narrativa (100% TEXTO humano) por archivo + un comparativo en TEXTO.
- respuesta2: JSON de decisión por archivo con el esquema exacto solicitado.

REGLAS GENERALES (NO INVENTES):
- Usa SOLO la información del CONTEXTO RAG y de CANDIDATOS_CONFIABLES provistos en cada bloque.
- Si un dato no está en esos insumos, debes dejarlo en null (o [] según corresponda).
- Cuando cites documentos, usa SOLO el nombre tras '### DOC:' (sin .pdf ni colección).
- No incluyas texto fuera del JSON final. Sin backticks, sin comentarios.

# ESTRUCTURA DE SALIDA (OBLIGATORIA)
Devuelve UN ÚNICO JSON **válido** exactamente con esta forma:

{
  "respuesta1": {
    "por_archivo": [
      { "archivo": "<stem>", "texto": "<narrativa legible y ordenada>" }
    ],
    "comparativo_texto": "<texto corto con el mejor cumplimiento y diferencias más relevantes>"
  },
  "respuesta2": {
    "por_archivo": [
      { "archivo": "<stem>", "json": {
          "secciones": {
            "legales":    ["..."],
            "tecnicas":   ["..."],
            "economicas": ["..."],
            "otros":      ["..."]
          },
          "procesoId": "<string|null>",
          "referencia": { "costo": <number|null>, "plazo_dias": <number|null> },
          "requisitos": [
            { "id": "R-001", "categoria": "Legal|Tecnica|Economica", "sub": "<string|null>",
              "texto": "<CITA LITERAL>", "peso": <number>, "clausula": "<string|null>", "pagina": <number|null> }
          ],
          "oferentes": [
            { "id": "O1", "nombre": "<string|null>", "costo": <number|null>, "plazo_dias": <number|null>,
              "cumplimientos": [ { "reqId": "R-001", "estado": "CUMPLE|PARCIAL|NO CUMPLE" } ],
              "riesgos": [ { "nivel": "ALTO|MEDIO|BAJO", "clausula": "<string|null>", "pagina": <number|null>, "descripcion": "<string>" } ]
            }
          ],
          "riesgos": [
            {"tipo": "legal|tecnico|economico", "detalle": "<texto>", "critico": true|false}
          ],
          "diferencias_clave": [
            {"tema": "plazo|monto|garantias|otros", "base": "...", "oferente": "...", "impacto": "bajo|medio|alto"}
          ],
          "recomendaciones": ["..."],
          "ruc_validacion": {
            "ruc": "<string|null>",
            "nombre": "<string|null>",
            "habilitado": true|false|null,
            "notas": "<string|null>"
          },
          "semaforo": "aprobado|faltan_requisitos|no_cumple",
          "pesosCategoria": { "Legal": <number>, "Tecnica": <number>, "Economica": <number> }
      } }
    ]
  }
}

# CRITERIOS ESPECÍFICOS PARA 'respuesta2.json'
- "procesoId": toma EXACTAMENTE de base.procesoIds (si no hay → null).
- "referencia.costo" y "referencia.plazo_dias": SOLO de base.costos / base.plazos_dias (si no hay → null).
- "oferentes[0].nombre": SOLO de oferta.nombres_posibles; "costo"/"plazo_dias": SOLO de oferta.costos / oferta.plazos_dias.
- "requisitos": cada "texto" debe ser **cita literal** de base.req_samples (sin parafrasear). Incluye 5–20 si hay suficientes.
  - "clausula": intenta mapear con base.clausulas; si no aplica → null.
  - "peso": si no hay criterio, usa 1.0; rango [0.5, 2.0].
- "cumplimientos": decide CUMPLE/PARCIAL/NO CUMPLE comparando literalmente el requisito con el contenido de la OFERTA; si no hay evidencia → "NO CUMPLE".
- "riesgos": marca "critico": true cuando afecte garantías, multas, plazos o pagos de forma material.
- "secciones": reparte frases/ítems detectados en cada ámbito (legales/técnicas/económicas/otros).
- "semaforo": aprobado=0 inconsistencias; faltan_requisitos=1–2 no críticas; no_cumple=3+ o críticas.
- "pesosCategoria": si no hay info explícita, usa { "Legal":0.4, "Tecnica":0.4, "Economica":0.2 }.

# CRITERIOS PARA 'respuesta1.texto' (NARRATIVA)
- Estructura por archivo:
  1) "Oferta: <nombre_archivo_sin_pdf>"
  2) Si hay procesoId, muéstralo.
  3) "Referencia BASE — costo: <N/D o número>, plazo: <N/D o número> días"
  4) "Oferente: <nombre> — costo: <N/D o número>, plazo: <N/D o número> días"
  5) "RUC <num> — HABILITADO/NO HABILITADO (Razón social: <...>, Estado: <...>)" cuando exista validación
  6) "Semáforo: APROBADO/FALTAN_REQUISITOS/NO_CUMPLE"
  7) "Cumplimientos relevantes:" 3–10 bullets (requisito, evidencia breve, doc base).
  8) "Brechas detectadas y acciones:" bullets (requisito, evidencia no encontrada o parcial, ACCIÓN CONCRETA, doc base).
  9) "Riesgos del oferente:" [nivel] descripción (cláusula, pág).
  10) "Pesos categoría: Legal X, Técnica Y, Económica Z" si están.

# COMPARATIVO (respuesta1.comparativo_texto)
- Resume en 1–3 líneas el "mejor cumplimiento" y 1–2 diferencias (plazo, monto, garantías).

# INSUMOS
- PREGUNTA_DEL_USUARIO:
{USER_Q}

- BLOQUES_DE_ENTRADA (repite para cada archivo):
{BLOQUES}
"""
