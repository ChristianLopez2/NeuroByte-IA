# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, shutil, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import Response

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
import hashlib

# ===== Estado del último análisis (/chat) =====
ANALYSIS_LOCK = asyncio.Lock()
LAST_ANALYSIS_DATA: Dict[str, Any] = None  # guarda el JSON de análisis final
ANALYSIS_DONE: bool = False                # indica si ya hay análisis para el set actual

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

    STRICT_BASE = os.getenv("STRICT_BASE", "false").lower() == "true"
    K_BASE = int(os.getenv("K_BASE", "8"))
    K_UP = int(os.getenv("K_UP", "8"))

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

    # 1) Limpiar carpeta uploads en disco
    try:
        if S.UPLOAD_DIR.exists():
            for filename in os.listdir(S.UPLOAD_DIR):
                filepath = S.UPLOAD_DIR / filename
                try:
                    if filepath.is_file() or filepath.is_symlink():
                        filepath.unlink()
                    elif filepath.is_dir():
                        shutil.rmtree(filepath)
                except Exception:
                    pass
        else:
            S.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo limpiar uploads: {e}")
    
    # 2) Limpiar colección de uploads en Chroma
    try:
        vs = _ensure_chroma(S.UPLOADS_COLLECTION)
        got = vs._collection.get(include=[])
        ids = got.get("ids") or []
        if ids:
            vs._collection.delete(ids=ids)
        try:
            vs.persist()
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo limpiar índice de uploads: {e}")
    
    # 4) Sincronizar índice por si acaso
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

# ===== Cache persistente del último análisis =====
LAST_ANALYSIS: Dict[str, Any] = {}   # {"sig": "...", "ts": 123.4, "data": {...}}
ANALYSIS_LOCK = asyncio.Lock()

def _file_snapshot_for_dir(dirpath: Path) -> list[dict]:
    items = []
    for p in sorted(dirpath.glob("*.pdf"), key=lambda x: x.name.lower()):
        try:
            st = p.stat()
            items.append({"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime)})
        except Exception:
            items.append({"name": p.name, "size": None, "mtime": None})
    return items

def _kb_config_mtime() -> int:
    cfg = S.KNOWLEDGE_DIR / "kb_config.yaml"
    try:
        return int(cfg.stat().st_mtime)
    except Exception:
        return 0

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _uploads_signature_only() -> str:
    """
    Firma basada SOLO en el estado de los PDFs + kb_config + modelos/prompt.
    Así, si ya hubo análisis para este estado, lo reutilizamos SIEMPRE,
    sin importar el prompt nuevo.
    """
    snapshot = {
        "uploads": _file_snapshot_for_dir(S.UPLOAD_DIR),
        "kb_config_mtime": _kb_config_mtime(),
        "chat_model": S.CHAT_MODEL,
        "embedding_model": S.EMBEDDING_MODEL,
        "strict_base": S.STRICT_BASE,
        "k_base": S.K_BASE,
        "k_up": S.K_UP,
        "prompt_hash": hashlib.sha256(UNIFIED_MASTER_PROMPT.encode("utf-8")).hexdigest(),
    }
    raw = _stable_json(snapshot)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Si aún no hay análisis para los archivos cargados, lo genera.
    Si ya existe, responde usando el análisis cacheado (no re-analiza).
    """
    user_q = (req.query or "").strip() or "Evalúa cumplimiento por oferta, detecta brechas y acciones de mejora concretas."
    process_hint = req.process.dict() if req.process else None

    global LAST_ANALYSIS_DATA, ANALYSIS_DONE

    # ¿Ya hay análisis vigente para este set de archivos?
    async with ANALYSIS_LOCK:
        cached_analysis = LAST_ANALYSIS_DATA if ANALYSIS_DONE and LAST_ANALYSIS_DATA else None

    if cached_analysis:
        llm = make_llm(temperature=0.1)
        followup_prompt = (
            "Eres un analista de licitaciones. Debes responder ÚNICAMENTE "
            "usando el análisis previo que te paso (ANALISIS_BASE). "
            "NO reanalices documentos, NO inventes datos.\n\n"
            "=== ANALISIS_BASE (JSON) ===\n"
            f"{json.dumps(cached_analysis, ensure_ascii=False)}\n\n"
            "=== PREGUNTA_DEL_USUARIO ===\n"
            f"{user_q}\n\n"
            "=== INSTRUCCIONES ===\n"
            "- Responde en español, claro y conciso.\n"
            "- Si algo NO está en ANALISIS_BASE, dilo explícitamente.\n"
            "- Puedes citar elementos de 'respuesta1.por_archivo', 'comparativo_texto' o 'respuesta2.por_archivo'.\n"
            "- No devuelvas JSON; responde en texto natural.\n"
        )
        resp = await llm.ainvoke(followup_prompt)

        # Devolvemos JSON para no tocar apiPost
        return {
            "respuesta_chat": resp.content,
            "_source": "cache-followup"
        }

    # Si no hay análisis, lo generamos ahora
    vs_up = _ensure_chroma(S.UPLOADS_COLLECTION)
    live_stems = sorted({Path(p.name).stem for p in S.UPLOAD_DIR.glob("*.pdf")})
    if not live_stems:
        result = {
            "respuesta1": {"por_archivo": [], "comparativo_texto": "Sin datos."},
            "respuesta2": {"por_archivo": []}
        }
        async with ANALYSIS_LOCK:
            LAST_ANALYSIS_DATA = result
            ANALYSIS_DONE = True
        return {**result, "_source": "fresh"}

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
    if not isinstance(data, dict):
        result = {
            "respuesta1": {"por_archivo": [], "comparativo_texto": "No se pudo estructurar el análisis."},
            "respuesta2": {"por_archivo": []},
            "raw": resp.content
        }
        async with ANALYSIS_LOCK:
            LAST_ANALYSIS_DATA = result
            ANALYSIS_DONE = True
        return {**result, "_source": "fresh"}

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

    # Guardar en cache como análisis vigente y devolver
    async with ANALYSIS_LOCK:
        LAST_ANALYSIS_DATA = data
        ANALYSIS_DONE = True

    return {**data, "_source": "fresh"}


@app.post("/api/upload-pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se recibieron archivos.")

    # 3) Guardar e indexar nuevos archivos
    saved = []
    for f in files:
        if f.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(status_code=400, detail=f"{f.filename} no es PDF.")
        safe = Path(f.filename).name
        dest = S.UPLOAD_DIR / safe
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        chunks = index_uploaded_pdf(dest, S.UPLOADS_COLLECTION)
        saved.append({"filename": safe, "chunks": chunks})

    # 4) Sincronizar índice por si acaso
    try:
        sync_uploads_index()
    except Exception:
        pass

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

@app.get("/api/cambioTipoDoc")
async def cambioTDoc():
    # 1) Limpiar carpeta uploads en disco
    try:
        if S.UPLOAD_DIR.exists():
            for filename in os.listdir(S.UPLOAD_DIR):
                filepath = S.UPLOAD_DIR / filename
                try:
                    if filepath.is_file() or filepath.is_symlink():
                        filepath.unlink()
                    elif filepath.is_dir():
                        shutil.rmtree(filepath)
                except Exception:
                    pass
        else:
            S.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo limpiar uploads: {e}")
    
    # 2) Limpiar colección de uploads en Chroma
    try:
        vs = _ensure_chroma(S.UPLOADS_COLLECTION)
        got = vs._collection.get(include=[])
        ids = got.get("ids") or []
        if ids:
            vs._collection.delete(ids=ids)
        try:
            vs.persist()
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo limpiar índice de uploads: {e}")
    
    # 4) Sincronizar índice por si acaso
    try:
        sync_uploads_index()
    except Exception:
        pass

    # 5) Resetear estado del análisis (nuevo set de archivos => nuevo análisis)
    global LAST_ANALYSIS_DATA, ANALYSIS_DONE
    async with ANALYSIS_LOCK:
        LAST_ANALYSIS_DATA = None
        ANALYSIS_DONE = False

    return Response(status_code=204)

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
    process_hint: Optional[dict] = None,
    strict: bool = S.STRICT_BASE
) -> list:
    """
    Selecciona documentos BASE relevantes:
    - Modo estricto: SOLO (familia, procedimiento, doc_role) según bundle. Sin fallbacks.
    - Modo normal: mantiene tus fallbacks actuales.
    """
    vs_base = _ensure_chroma(S.BASE_COLLECTION)

    familia = (process_hint or {}).get("familia") or (upload_meta or {}).get("familia")
    proc    = (process_hint or {}).get("procedimiento") or (upload_meta or {}).get("procedimiento")

    def fetch_by(where_filter, k=6):
        ret = vs_base.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 50, "filter": where_filter}
        )
        return ret.get_relevant_documents(query)

    chosen = []
    bundle = CFG.find_bundle(familia, proc)

    if strict:
        # ——— MODO ESTRICTO ———
        if not (familia and proc and bundle):
            return []  # sin facetas completas o sin bundle → nada

        roles = bundle.get("include_roles", [])
        for role in roles:
            where_fp_role = {
                "$and": [
                    {"familia": {"$eq": familia}},
                    {"procedimiento": {"$eq": proc}},
                    {"doc_role": {"$eq": role}}
                ]
            }
            chosen.extend(fetch_by(where_fp_role, k=k_per_role))

        # anexos opcionales estrictos
        for role in bundle.get("also_include", []):
            where_extra = {
                "$and": [
                    {"familia": {"$eq": familia}},
                    {"procedimiento": {"$eq": proc}},
                    {"doc_role": {"$eq": role}}
                ]
            }
            chosen.extend(fetch_by(where_extra, k=1))
    else:
        # ——— MODO NORMAL (tu lógica original con fallbacks) ———
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

            for role in bundle.get("also_include", []):
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
                base_docs = fetch_by({"kb_grupo": {"$in": ["01_normativa", "02_modelos_pliegos"]}}, k=10)
            chosen.extend(base_docs)

    # dedup por 'source'
    seen, final = set(), []
    for d in chosen:
        s = (d.metadata or {}).get("source")
        if s and s not in seen:
            seen.add(s)
            final.append(d)
    return final

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
          "pesosCategoria": { "Legal": <number>, "Tecnica": <number>, "Economica": <number> }
      } }
    ]
  }
}

# CRITERIOS ESPECÍFICOS PARA 'respuesta2.json' (DECISIÓN)
- "procesoId": toma EXACTAMENTE de base.procesoIds (si no hay → null).
- "referencia.costo" y "referencia.plazo_dias": SOLO de base.costos / base.plazos_dias respectivamente (si no hay → null).
- "oferentes[0].nombre": SOLO de oferta.nombres_posibles; "costo"/"plazo_dias": SOLO de oferta.costos / oferta.plazos_dias.
- "requisitos": cada "texto" debe ser **cita literal** de base.req_samples (sin parafrasear). Incluye de 5 a 20 si hay suficientes; si no, incluye las disponibles.
  - "clausula": intenta mapear con base.clausulas; si no aplica → null.
  - "peso": si no hay criterio, usa 1.0; rango permitido [0.5, 2.0].
- "cumplimientos": decide CUMPLE/PARCIAL/NO CUMPLE comparando literalmente el requisito con el contenido de la OFERTA en el CONTEXTO; si no hay evidencia → "NO CUMPLE".
- "pesosCategoria": si no hay info explícita, usa { "Legal":0.4, "Tecnica":0.4, "Economica":0.2 }.

# CRITERIOS PARA 'respuesta1.texto' (NARRATIVA)
- Estructura por archivo:
  1) "Oferta: <nombre_archivo_sin_pdf>"
  2) Si hay procesoId, muéstralo.
  3) "Referencia BASE — costo: <N/D o número>, plazo: <N/D o número> días"
  4) "Oferente: <nombre> — costo: <N/D o número>, plazo: <N/D o número> días"
  5) Si hay RUC_VALIDACION externo, muéstralo en una sola línea: "RUC <num> — HABILITADO/NO HABILITADO (Razón social: <...>, Estado: <...>)"
  6) "Semáforo: VERDE/AMARILLO/ROJO" (derivado de la calidad de cumplimiento según el CONTEXTO; si no puedes inferir, usa AMARILLO)
  7) "Cumplimientos relevantes:" con 3–10 bullets cortos si existen (requisito, evidencia breve, documento base).
  8) "Brechas detectadas y acciones:" con bullets (requisito, evidencia no encontrada o parcial, acción concreta, documento base).
  9) "Riesgos del oferente:" con bullets [nivel] descripción (cláusula, pág).
  10) "Pesos categoría (si aplica): Legal X, Técnica Y, Económica Z" si están en la decisión.

- Semáforo (heurística textual): verde=0 brechas; amarillo=1–2 no críticas; rojo=3+ o críticas.

# COMPARATIVO (respuesta1.comparativo_texto)
- Resume en 1–3 líneas el "mejor cumplimiento" entre archivos y una o dos diferencias relevantes (plazo, monto, garantías).
- Debe ser TEXTO, no JSON.

# INSUMOS
- PREGUNTA_DEL_USUARIO:
{USER_Q}

- BLOQUES_DE_ENTRADA (repite para cada archivo):
{BLOQUES}
"""