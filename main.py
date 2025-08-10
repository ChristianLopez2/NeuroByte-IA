# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, shutil, unicodedata, hashlib, asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# LangChain / OpenAI / Vector
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# HTTP para SRI
import httpx

# PDF / OCR
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import pytesseract
try:
    import easyocr
    EASY_READER = easyocr.Reader(['es','en'], gpu=False)
    HAS_EASYOCR = True
except Exception:
    EASY_READER = None
    HAS_EASYOCR = False

# YAML
import yaml

# =========================
# Settings
# =========================
class Settings:
    BASE_DIR = Path(__file__).parent.resolve()
    UPLOAD_DIR = BASE_DIR / "uploads" / "pdf"
    KNOWLEDGE_DIR = BASE_DIR / "knowledge" / "pdf"
    VECTOR_DIR = BASE_DIR / "vector_db"
    BASE_COLLECTION = "licitaciones_base"
    UPLOADS_COLLECTION = "licitaciones_uploads"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    # chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

    # recortes de contexto por documento
    CTX_MAX_CHARS_PER_DOC = int(os.getenv("CTX_MAX_CHARS_PER_DOC", "12000"))
    CTX_HEAD_PER_DOC      = int(os.getenv("CTX_HEAD_PER_DOC", "6000"))

    # RAG top-K
    K_BASE = int(os.getenv("K_BASE", "6"))
    K_UP   = int(os.getenv("K_UP", "6"))

    ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() != "false"
    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

    # SRI
    SRI_RUC_URL = (
        "https://srienlinea.sri.gob.ec/"
        "sri-catastro-sujeto-servicio-internet/rest/"
        "ConsolidadoContribuyente/obtenerPorNumerosRuc?&ruc={ruc}"
    )

    STRICT_BASE = os.getenv("STRICT_BASE", "false").lower() == "true"

S = Settings()

# ===== Caches / estado =====
ANALYSIS_BY_RUC: Dict[str, Dict[str, Any]] = {}    # {ruc: {"r1_items":[], "r2_items":[], "comparativos":[]}}
UPLOAD_RUC_INDEX: Dict[str, Optional[str]] = {}    # {filename.pdf: ruc}
ANALYSIS_LOCK = asyncio.Lock()

# ===== Master JSON persistente =====
MASTER_PATH = S.BASE_DIR / "data" / "master.json"
MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_master():
    if not MASTER_PATH.exists():
        return {"respuesta1":{"por_archivo":[],"comparativo_texto":""},
                "respuesta2":{"por_archivo":[]}}
    with open(MASTER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_master(obj):
    with open(MASTER_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def merge_into_master(partial):
    m = load_master()
    r1_new = (partial.get("respuesta1") or {}).get("por_archivo") or []
    r2_new = (partial.get("respuesta2") or {}).get("por_archivo") or []
    if r1_new:
        seen = {(x or {}).get("archivo") for x in m["respuesta1"]["por_archivo"]}
        for it in r1_new:
            if (it or {}).get("archivo") not in seen:
                m["respuesta1"]["por_archivo"].append(it)
    if r2_new:
        seen = {(x or {}).get("archivo") for x in m["respuesta2"]["por_archivo"]}
        for it in r2_new:
            if (it or {}).get("archivo") not in seen:
                m["respuesta2"]["por_archivo"].append(it)
    comp_new = (partial.get("respuesta1") or {}).get("comparativo_texto") or ""
    if comp_new:
        base = m["respuesta1"].get("comparativo_texto") or ""
        m["respuesta1"]["comparativo_texto"] = " | ".join(filter(None,[base,comp_new]))[:1200]
    save_master(m)
    return m

def _dedup_join_pipes(*texts: str, limit: int = 1200) -> str:
    """Une textos con '|', separa en frases por '|', hace trim + dedup y vuelve a unir."""
    seen, out = set(), []
    for t in texts:
        if not t:
            continue
        for part in t.split("|"):
            s = (part or "").strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    res = " | ".join(out)
    return res[:limit]

# =========================
# FastAPI
# =========================
app = FastAPI(title="RAG Licitaciones", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[S.FRONTEND_ORIGIN] if S.FRONTEND_ORIGIN != "*" else ["*"],
    allow_methods=["*"], allow_headers=["*"]
)
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

def ocr_image(img: Image.Image) -> str:
    if HAS_EASYOCR:
        arr = np.array(img)
        lines = EASY_READER.readtext(arr, detail=0, paragraph=True)
        return "\n".join(lines)
    try:
        return pytesseract.image_to_string(img, lang="spa+eng")
    except Exception:
        return ""

def extract_text_from_pdf(pdf_path: Path, enable_ocr: bool = S.ENABLE_OCR) -> str:
    doc = fitz.open(pdf_path)
    parts: List[str] = []
    for p in doc:
        if _page_has_text(p):
            parts.append(p.get_text("text"))
        else:
            if not enable_ocr: continue
            pix = p.get_pixmap(dpi=250)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            parts.append(ocr_image(img))
    return "\n".join(parts).strip()

def extract_text_head(pdf_path: Path, enable_ocr: bool = S.ENABLE_OCR, max_chars: int = S.CTX_HEAD_PER_DOC) -> str:
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

def extract_text_head_from_bytes(pdf_bytes: bytes, max_chars: int = 6000, enable_ocr: bool = False) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return ""
    parts = []
    for p in doc:
        if len(" ".join(parts)) > max_chars: break
        try:
            if _page_has_text(p):
                parts.append(p.get_text("text"))
            elif enable_ocr:
                pix = p.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                parts.append(ocr_image(img))
        except Exception:
            continue
    return "\n".join(parts)[:max_chars]

def to_documents(text: str, meta: Dict[str, Any]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=S.CHUNK_SIZE, chunk_overlap=S.CHUNK_OVERLAP)
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

# =========================
# YAML facetas/bundles para BASE
# =========================
def _norm(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn").lower()

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
    try:
        rel = pdf_path.relative_to(S.KNOWLEDGE_DIR)
    except Exception:
        rel = pdf_path
    path_text = " / ".join([p.name for p in rel.parents if p.name][::-1]) + " / " + pdf_path.name
    m = CFG.match_facets(path_text)
    if not m.get("doc_role"):
        head = extract_text_head(pdf_path)
        m2 = CFG.match_facets(head)
        for k in ("familia","procedimiento","doc_role"):
            if not m.get(k) and m2.get(k): m[k] = m2[k]
    grupo = str(rel).split(os.sep, 1)[0] if isinstance(rel, Path) else None
    return {"kb_grupo": grupo, "familia": m.get("familia"), "procedimiento": m.get("procedimiento"), "doc_role": m.get("doc_role")}

def index_knowledge_tree(kb_root: Path, collection_name: str) -> int:
    vs = _ensure_chroma(collection_name)
    added = 0
    existing_sources = set()
    try:
        got = vs._collection.get(include=["metadatas"])
        for md in (got.get("metadatas") or []):
            if md and md.get("source"): existing_sources.add(md["source"])
    except Exception:
        pass
    for pdf in kb_root.rglob("*.pdf"):
        rel = pdf.relative_to(kb_root).as_posix()
        if rel in existing_sources: continue
        meta_extra = derive_meta_from_kb_path_and_name(pdf)
        text = extract_text_from_pdf(pdf)
        if not text: continue
        meta = { "path": str(pdf), "source": rel, "collection": collection_name, **meta_extra }
        docs = to_documents(text, meta=meta)
        if docs:
            vs.add_documents(docs); added += len(docs)
    try: vs.persist()
    except Exception: pass
    return added

# =========================
# Uploads
# =========================
def classify_upload(pdf_path: Path) -> Dict[str, Any]:
    head = extract_text_head(pdf_path)
    m = CFG.match_facets(f"{pdf_path.name}\n{head}")
    return {"familia": m.get("familia"), "procedimiento": m.get("procedimiento"), "doc_role": None, "upload_kind": "oferta"}

def index_uploaded_pdf(pdf_path: Path, collection_name: str) -> int:
    vs = _ensure_chroma(collection_name)
    try: vs._collection.delete(where={"source": {"$eq": pdf_path.name}})
    except Exception: pass
    text = extract_text_from_pdf(pdf_path)
    if not text: return 0
    u = classify_upload(pdf_path)
    meta = {"path": str(pdf_path), "source": pdf_path.name, "collection": collection_name, **u}
    docs = to_documents(text, meta=meta)
    if docs:
        vs.add_documents(docs)
        try: vs.persist()
        except Exception: pass
    return len(docs)

# =========================
# LLM
# =========================
def make_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model=S.CHAT_MODEL, temperature=temperature, openai_api_key=S.OPENAI_API_KEY)

# =========================
# RUC / SRI
# =========================
RUC_RE = re.compile(r"\b(\d{13})\b")

def _normalize_ruc_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list) and payload:
        data = payload[0]
    elif isinstance(payload, dict):
        data = payload
    else:
        return {"raw": payload, "habilitado": None}
    estado = data.get("estadoContribuyenteRuc") or data.get("estado") or ""
    habilitado = isinstance(estado, str) and estado.strip().upper() == "ACTIVO"
    return {
        "ruc": data.get("numeroRuc"),
        "razon_social": data.get("razonSocial"),
        "estado": estado,
        "habilitado": habilitado,
        "raw": data,
    }

async def fetch_ruc_info(ruc: str) -> Dict[str, Any]:
    url = S.SRI_RUC_URL.format(ruc=ruc)
    for _ in range(3):
        try:
            async with httpx.AsyncClient(timeout=50) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    try: return _normalize_ruc_payload(r.json())
                    except Exception: return {"ruc": ruc, "razon_social": None, "estado":"No se pudo procesar", "habilitado": None, "raw": r.text}
        except Exception:
            await asyncio.sleep(1)
    return {"ruc": ruc, "razon_social": None, "estado":"Error al consultar", "habilitado": None, "raw": None}

def extract_ruc_from_offer_stem(stem: str) -> Optional[str]:
    pdf_path = S.UPLOAD_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        candidates = list(S.UPLOAD_DIR.glob(f"{stem}*.pdf"))
        if not candidates: return None
        pdf_path = candidates[0]
    text = extract_text_from_pdf(pdf_path)
    if not text: return None
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
            "razon_social": info.get("razon_social"),
            "estado": info.get("estado"),
            "notas": "OK" if info.get("habilitado") is not None else "Sin confirmación"
        }
    except Exception as e:
        return {"ruc": ruc, "habilitado": None, "notas": f"Error consultando SRI: {e}"}

async def extract_ruc_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    head = extract_text_head_from_bytes(pdf_bytes, max_chars=10000, enable_ocr=False)
    if not head: return None
    m = RUC_RE.search(head)
    return m.group(1) if m else None

# =========================
# Extractores anti-alucinación
# =========================
PROC_ID_RE = re.compile(r'\b[A-Z]{2,}(?:-[A-Z0-9]{2,}){1,4}-\d{4}-\d{2,4}\b')
MONEY_RE   = re.compile(r'(?:(?:USD|US\$|\$)\s?)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d+)\b')
DAYS_RE    = re.compile(r'\b(\d{1,4})\s*d[ií]as\b', re.IGNORECASE)
CLAUSE_RE  = re.compile(r'cl[áa]usula\s+([0-9A-Za-z\.\-]+)', re.IGNORECASE)

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
        if "(base)" in head: base_parts.append(block)
        elif "(oferta)" in head: ofer_parts.append(block)
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
        if not (8 <= len(line) <= 300): continue
        low = line.lower()
        if any(bad in low for bad in REQ_NEG): continue
        if any(tok in low for tok in REQ_POS): out.append(line)
        if len(out) >= max_items: break
    seen, final = set(), []
    for t in out:
        k = t.lower()
        if k not in seen: seen.add(k); final.append(t)
    return final

def _find_numeric_near_keywords(text: str, value_re: re.Pattern, keywords: tuple[str, ...], window: int = 120) -> Optional[int]:
    if not text: return None
    low = text.lower()
    for m in value_re.finditer(text):
        start, end = m.span()
        left = max(0, start - window)
        right = min(len(text), end + window)
        ctx = low[left:right]
        if any(k in ctx for k in keywords):
            val = _to_int_safe(m.group(1) if m.lastindex else m.group(0))
            if val is not None: return val
    return None

BASE_COST_KW   = ("presupuesto", "referencial", "estimado", "base", "precio referencial", "monto referencial", "valor referencial")
BASE_PLAZO_KW  = ("plazo", "ejecución", "ejecucion", "entrega")
OF_COST_KW     = ("oferta", "ofertado", "propuesta", "cotización", "cotizacion", "valor de la oferta", "monto de la oferta")
OF_PLAZO_KW    = ("plazo", "oferta", "ejecución", "ejecucion", "entrega")

def extract_candidates(ctx_offer: str, archivo_stem: str) -> dict:
    base_text, oferta_text = _split_ctx_by_role(ctx_offer)
    proc_ids  = list(dict.fromkeys(PROC_ID_RE.findall(base_text)))
    base_cost = _find_numeric_near_keywords(base_text, MONEY_RE, BASE_COST_KW)
    base_days = _find_numeric_near_keywords(base_text, DAYS_RE, BASE_PLAZO_KW)
    oferta_cost = _find_numeric_near_keywords(oferta_text, MONEY_RE, OF_COST_KW)
    oferta_days = _find_numeric_near_keywords(oferta_text, DAYS_RE, OF_PLAZO_KW)
    req_lines = _harvest_requirement_lines(base_text)
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
            "nombres_posibles": [archivo_stem]
        }
    }

# =========================
# PROMPT UNIFICADO
# =========================
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
              "cumplimientos": [ { "reqId": "R-001", "estado": "CUMPLE|PARCIAL|NO CUMPLE", "evidencia": "<string|null>" } ],
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
- No repitas datos ya mostrados.
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

- BLOQUES_DE_ENTRADA (solo 1 por llamada):
{BLOQUES}
"""

def json_guard(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {"resumen": s.strip()[:800], "error": "El modelo no devolvió JSON puro."}

# =========================
# Procesos (frontend combo)
# =========================
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

# =========================
# Uploads Endpoints
# =========================
@app.post("/api/upload-pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se recibieron archivos.")
    saved = []
    for f in files:
        if f.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(status_code=400, detail=f"{f.filename} no es PDF.")
        safe = Path(f.filename).name
        pdf_bytes = await f.read()
        try:
            ruc = await extract_ruc_from_pdf_bytes(pdf_bytes)
        except Exception:
            ruc = None
        # si ya hay análisis para ese RUC, no re-indexamos
        if ruc and ruc in ANALYSIS_BY_RUC:
            async with ANALYSIS_LOCK:
                UPLOAD_RUC_INDEX[safe] = ruc
            saved.append({"filename": safe, "chunks": 0, "skipped": True, "reason": "RUC ya analizado"})
            continue
        dest = S.UPLOAD_DIR / safe
        with dest.open("wb") as out:
            out.write(pdf_bytes)
        chunks = index_uploaded_pdf(dest, S.UPLOADS_COLLECTION)
        async with ANALYSIS_LOCK:
            UPLOAD_RUC_INDEX[safe] = ruc
        saved.append({"filename": safe, "chunks": chunks, "skipped": False, "ruc": ruc})
    try:
        sync_uploads_index()
    except Exception:
        pass
    return {"ok": True, "files": saved}

@app.get("/api/list-pdf")
async def list_pdf():
    files = []
    for p in S.UPLOAD_DIR.glob("*.pdf"):
        files.append({"filename": p.name, "disk_path": str(p.resolve()), "size_bytes": p.stat().st_size})
    return {"count": len(files), "files": files}

@app.get("/api/cambioTipoDoc")
async def cambio_tipo_doc():
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
    try:
        vs = _ensure_chroma(S.UPLOADS_COLLECTION)
        got = vs._collection.get(include=[])
        ids = got.get("ids") or []
        if ids:
            vs._collection.delete(ids=ids)
        try: vs.persist()
        except Exception: pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo limpiar índice de uploads: {e}")
    try: sync_uploads_index()
    except Exception: pass
    global ANALYSIS_BY_RUC, UPLOAD_RUC_INDEX
    async with ANALYSIS_LOCK:
        ANALYSIS_BY_RUC = {}; UPLOAD_RUC_INDEX = {}
    return Response(status_code=204)

def current_upload_sources() -> set[str]:
    return {p.name for p in S.UPLOAD_DIR.glob("*.pdf")}

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
            vs._collection.delete(ids=ids); deleted = len(ids)
    try: vs.persist()
    except Exception: pass
    return {"kept": keep, "deleted": deleted}

@app.post("/api/sync-uploads")
async def api_sync_uploads():
    info = sync_uploads_index()
    return {"ok": True, "synced": info}

# =========================
# RAG selección BASE
# =========================
def _and_filter(*clauses: dict) -> dict:
    clauses = [c for c in clauses if c]
    if not clauses: return {}
    if len(clauses) == 1: return clauses[0]
    return {"$and": clauses}

def _clean_filter(f):
    if isinstance(f, dict):
        out = {}
        for k, v in f.items():
            if k in ("$and", "$or") and isinstance(v, list):
                cleaned = [_clean_filter(x) for x in v if _clean_filter(x) is not None]
                if cleaned: out[k] = cleaned
            elif isinstance(v, dict) and "$eq" in v and v["$eq"] is None:
                continue
            else:
                cv = _clean_filter(v)
                if cv is not None and cv != {} and cv != []:
                    out[k] = cv
        return out or None
    elif isinstance(f, list):
        cleaned = [_clean_filter(x) for x in f if _clean_filter(x) is not None]
        return cleaned or None
    else:
        return f

def pick_base_docs_for_upload(
    query: str,
    upload_meta: dict,
    k_per_role: int = 2,
    process_hint: Optional[dict] = None,
    strict: bool = S.STRICT_BASE
) -> list:
    vs_base = _ensure_chroma(S.BASE_COLLECTION)
    try:
        if getattr(vs_base, "_collection", None):
            if (vs_base._collection.count() or 0) == 0:
                return []
    except Exception:
        pass

    familia = (process_hint or {}).get("familia") or (upload_meta or {}).get("familia")
    proc    = (process_hint or {}).get("procedimiento") or (upload_meta or {}).get("procedimiento")

    def fetch_by(where_filter, k=6):
        where_filter = _clean_filter(where_filter) or {}
        ret = vs_base.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(20, k * 5), "filter": where_filter}
        )
        try:
            return ret.invoke(query)
        except Exception:
            try: return vs_base.similarity_search(query, k=k, filter=where_filter)
            except Exception:
                try: return vs_base.similarity_search(query, k=k)
                except Exception: return []

    chosen = []
    bundle = CFG.find_bundle(familia, proc)

    if strict:
        if not (familia and proc and bundle): return []
        roles = [r for r in bundle.get("include_roles", []) if r]
        for role in roles:
            where_and = [{"familia": {"$eq": familia}}, {"procedimiento": {"$eq": proc}}]
            if role: where_and.append({"doc_role": {"$eq": role}})
            chosen.extend(fetch_by({"$and": where_and}, k=k_per_role))
        for role in [r for r in bundle.get("also_include", []) if r]:
            chosen.extend(fetch_by({"$and":[{"familia":{"$eq":familia}},{"procedimiento":{"$eq":proc}},{"doc_role":{"$eq":role}}]}, k=1))
    else:
        if bundle:
            roles = [r for r in bundle.get("include_roles", []) if r]
            for role in roles:
                base_filter = _and_filter(
                    {"familia": {"$eq": familia}} if familia else None,
                    {"procedimiento": {"$eq": proc}} if proc else None,
                    {"doc_role": {"$eq": role}} if role else None
                )
                docs = fetch_by(base_filter, k=k_per_role) or \
                       fetch_by(_and_filter({"familia":{"$eq":familia}} if familia else None, {"doc_role":{"$eq":role}} if role else None), k=k_per_role) or \
                       fetch_by({"doc_role":{"$eq":role}}, k=k_per_role)
                chosen.extend(docs)
            for role in [r for r in bundle.get("also_include", []) if r]:
                chosen.extend(fetch_by({"doc_role": {"$eq": role}}, k=1))
        else:
            base_docs = []
            if familia or proc:
                where_fp = _and_filter(
                    {"familia": {"$eq": familia}} if familia else None,
                    {"procedimiento": {"$eq": proc}} if proc else None,
                )
                if where_fp: base_docs = fetch_by(where_fp, k=10)
            if not base_docs:
                base_docs = fetch_by({"kb_grupo": {"$in": ["01_normativa", "02_modelos_pliegos"]}}, k=10)
            chosen.extend(base_docs)

    seen, final = set(), []
    for d in chosen:
        s = (d.metadata or {}).get("source")
        if s and s not in seen:
            seen.add(s); final.append(d)
    return final

# =========================
# Chat / análisis por archivo
# =========================
class ProcessHint(BaseModel):
    familia: Optional[str] = None
    procedimiento: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    focus_upload: Optional[str] = None
    process: Optional[ProcessHint] = None

@app.on_event("startup")
def startup_index_base():
    index_knowledge_tree(S.KNOWLEDGE_DIR, S.BASE_COLLECTION)
    # limpiar uploads en disco e índice
    try:
        if S.UPLOAD_DIR.exists():
            for filename in os.listdir(S.UPLOAD_DIR):
                filepath = S.UPLOAD_DIR / filename
                try:
                    if filepath.is_file() or filepath.is_symlink(): filepath.unlink()
                    elif filepath.is_dir(): shutil.rmtree(filepath)
                except Exception: pass
        else:
            S.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        vs = _ensure_chroma(S.UPLOADS_COLLECTION)
        got = vs._collection.get(include=[])
        ids = got.get("ids") or []
        if ids: vs._collection.delete(ids=ids)
        try: vs.persist()
        except Exception: pass
    except Exception:
        pass
    try: sync_uploads_index()
    except Exception: pass
    global ANALYSIS_BY_RUC, UPLOAD_RUC_INDEX
    ANALYSIS_BY_RUC = {}; UPLOAD_RUC_INDEX = {}

def _filename_for_stem(stem: str) -> Optional[str]:
    for p in S.UPLOAD_DIR.glob("*.pdf"):
        if Path(p.name).stem == stem:
            return p.name
    return None

def _get_ruc_for_stem(stem: str) -> Optional[str]:
    filename = _filename_for_stem(stem)
    if not filename: return None
    r = UPLOAD_RUC_INDEX.get(filename)
    if r: return r
    return extract_ruc_from_offer_stem(stem)

def _cache_store_analysis_for_ruc(ruc: Optional[str], data_json: dict):
    """
    Guarda en caché por RUC los ítems por archivo.
    Evita duplicar 'comparativos'.
    """
    if not isinstance(data_json, dict) or not ruc:
        return
    r1 = (data_json.get("respuesta1") or {}).get("por_archivo") or []
    r2 = (data_json.get("respuesta2") or {}).get("por_archivo") or []
    comp = (data_json.get("respuesta1") or {}).get("comparativo_texto") or ""

    bucket = ANALYSIS_BY_RUC.setdefault(ruc, {"r1_items": [], "r2_items": [], "comparativos": []})

    def _merge(lst: list, new_items: list, key="archivo"):
        seen = {(i or {}).get(key) for i in lst}
        for it in new_items:
            if (it or {}).get(key) not in seen:
                lst.append(it)

    _merge(bucket["r1_items"], r1)
    _merge(bucket["r2_items"], r2)
    if comp and comp not in bucket["comparativos"]:
        bucket["comparativos"].append(comp)

def _collect_cached_items_for_current_uploads() -> Tuple[list, list, str]:
    """
    Combina del caché SOLO lo que corresponde a los uploads vivos.
    Si no hay uploads ni caché en memoria, cae al master.json persistente.
    Devuelve (r1_items, r2_items, comparativo_deduplicado).
    """
    r1_all, r2_all, comparativos = [], [], []
    active_filenames = {p.name for p in S.UPLOAD_DIR.glob("*.pdf")}
    active_rucs = set()

    # Fallback directo a master.json si no hay uploads NI caché en memoria
    if not active_filenames and not ANALYSIS_BY_RUC:
        try:
            m = load_master()
            r1_all = (m.get("respuesta1") or {}).get("por_archivo") or []
            r2_all = (m.get("respuesta2") or {}).get("por_archivo") or []
            comp_text = _dedup_join_pipes((m.get("respuesta1") or {}).get("comparativo_texto") or "", limit=1200)
            return r1_all, r2_all, comp_text
        except Exception:
            return [], [], ""

    # Caso normal: hay uploads o hay caché en memoria
    if active_filenames:
        for fn in active_filenames:
            r = UPLOAD_RUC_INDEX.get(fn)
            if r:
                active_rucs.add(r)
    else:
        active_rucs = set(ANALYSIS_BY_RUC.keys())

    for ruc in active_rucs:
        bucket = ANALYSIS_BY_RUC.get(ruc)
        if not bucket:
            continue
        r1_all.extend(bucket.get("r1_items", []))
        r2_all.extend(bucket.get("r2_items", []))
        comparativos.extend(bucket.get("comparativos", []))

    comp_text = _dedup_join_pipes(*comparativos, limit=1200)
    return r1_all, r2_all, comp_text


# ========= CHAT ENDPOINT (REEMPLAZA COMPLETO) =========
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Analiza SOLO lo que no esté en caché por RUC, mezcla con lo cacheado,
    y devuelve comparativo deduplicado.
    """
    user_q = (req.query or "").strip() or "Evalúa cumplimiento por oferta, detecta brechas y acciones de mejora concretas."
    process_hint = req.process.dict() if req.process else None

    # Archivos actuales
    live_files = sorted([p.name for p in S.UPLOAD_DIR.glob("*.pdf")])
    live_stems = [Path(n).stem for n in live_files]
    if not live_stems:
        # Fallback a master.json si existe contenido previo
        try:
            m = load_master()
            r1 = (m.get("respuesta1") or {})
            r2 = (m.get("respuesta2") or {})
            if (r1.get("por_archivo") or r2.get("por_archivo")):
                # ordenar y deduplicar comparativo
                try:
                    r1["por_archivo"] = sorted(r1.get("por_archivo") or [], key=lambda x: (x or {}).get("archivo",""))
                except Exception:
                    pass
                try:
                    r2["por_archivo"] = sorted(r2.get("por_archivo") or [], key=lambda x: (x or {}).get("archivo",""))
                except Exception:
                    pass
                r1["comparativo_texto"] = _dedup_join_pipes(r1.get("comparativo_texto") or "", limit=1200)
                return {"respuesta1": r1, "respuesta2": r2, "_source": "master"}
        except Exception:
            pass

        # Si no hay nada en master, devolvemos vacío
        return {
            "respuesta1": {"por_archivo": [], "comparativo_texto": "Sin datos."},
            "respuesta2": {"por_archivo": []},
            "_source": "fresh"
        }


    # ¿Qué stems faltan por analizar?
    stems_to_analyze: list[str] = []
    for stem in live_stems:
        ruc = _get_ruc_for_stem(stem)
        fn = next((p.name for p in S.UPLOAD_DIR.glob(f"{stem}*.pdf")), None)
        if fn and ruc:
            async with ANALYSIS_LOCK:
                if UPLOAD_RUC_INDEX.get(fn) != ruc:
                    UPLOAD_RUC_INDEX[fn] = ruc
        if not ruc or ruc not in ANALYSIS_BY_RUC:
            stems_to_analyze.append(stem)

    # Si no hay nada nuevo, trabaja sobre el caché
    if not stems_to_analyze:
        cached_r1, cached_r2, comp = _collect_cached_items_for_current_uploads()

        if not cached_r1 and not cached_r2:
            return {
                "respuesta1": {"por_archivo": [], "comparativo_texto": "Sin datos en caché."},
                "respuesta2": {"por_archivo": []},
                "_source": "cache-empty"
            }

        # Si no hay pregunta: mensaje de estado
        if not req.query or not req.query.strip():
            return {
                "mensaje": "No hay documentos pendientes. Puedes volver a analizar o formular una pregunta.",
                "_source": "cache-status"
            }

        # Responder sobre el análisis ya hecho (sin reanalizar)
        llm = make_llm(temperature=0.1)
        followup_prompt = (
            "Eres un analista de licitaciones. Debes responder ÚNICAMENTE usando el análisis previo (ANALISIS_BASE). "
            "NO reanalices documentos ni inventes datos.\n\n"
            "=== ANALISIS_BASE (JSON) ===\n"
            f"{json.dumps({'respuesta1':{'por_archivo':cached_r1,'comparativo_texto':comp},'respuesta2':{'por_archivo':cached_r2}}, ensure_ascii=False)}\n\n"
            "=== PREGUNTA_DEL_USUARIO ===\n"
            f"{user_q}\n\n"
            "- Responde en español, claro y conciso. No devuelvas JSON; responde en texto natural."
        )
        resp = await llm.ainvoke(followup_prompt)
        return {"respuesta_chat": resp.content, "_source": "cache-followup"}

    # -------- Función para construir contexto por stem --------
    async def build_ctx_for_stem(stem: str, k_base: int = S.K_BASE, k_up: int = S.K_UP) -> Tuple[str, dict, dict, str]:
        vs_up = _ensure_chroma(S.UPLOADS_COLLECTION)
        live = list(S.UPLOAD_DIR.glob(f"{stem}*.pdf"))
        meta = vs_up._collection.get(where={"source": {"$in": [p.name for p in live]}}, include=["metadatas"])
        up_meta = (meta.get("metadatas") or [None])[0] or {}

        up_ret = vs_up.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_up, "fetch_k": 40, "filter": {"source": {"$in": [p.name for p in live]}}}
        )
        up_docs = up_ret.invoke(user_q)

        base_docs = pick_base_docs_for_upload(
            user_q, up_meta, k_per_role=max(1, k_base // 3), process_hint=process_hint
        )

        def clip(txt: str, cap: int = S.CTX_MAX_CHARS_PER_DOC) -> str:
            return (txt or "")[:cap]

        parts, total = [], 0
        for d in base_docs:
            c = clip(d.page_content)
            if not c:
                continue
            s = f"\n### DOC: {Path(d.metadata['source']).stem} (BASE)\n{c}\n"
            parts.append(s); total += len(s)
            if total >= S.CTX_MAX_CHARS_PER_DOC:
                break

        for d in up_docs:
            c = clip(d.page_content)
            if not c:
                continue
            s = f"\n### DOC: {Path(d.metadata['source']).stem} (OFERTA)\n{c}\n"
            parts.append(s); total += len(s)
            if total >= 2 * S.CTX_MAX_CHARS_PER_DOC:
                break

        ctx = "".join(parts)
        cands = extract_candidates(ctx, stem)
        ruc_info = await validate_ruc_for_offer(stem)
        ruc = ruc_info.get("ruc")
        return ctx, cands, ruc_info, ruc

    # -------- Analizar stems pendientes (uno por uno) --------
    llm = make_llm(temperature=0.0)
    mixed_r1, mixed_r2, comparativos = [], [], []

    for stem in stems_to_analyze:
        ctx, cands, ruc_info, ruc = await build_ctx_for_stem(stem)
        bloque = (
            f"\n=== OFERTA: {stem} ===\n"
            f"# CANDIDATOS_CONFIABLES\n{json.dumps(cands, ensure_ascii=False)}\n"
            f"# RUC_VALIDACION\n{json.dumps(ruc_info, ensure_ascii=False)}\n"
            f"# CONTEXTO RAG (recortado)\n{ctx}\n"
        )
        prompt = UNIFIED_MASTER_PROMPT.replace("{USER_Q}", user_q).replace("{BLOQUES}", bloque)
        resp = await llm.ainvoke(prompt)
        data_new = json_guard(resp.content)

        if isinstance(data_new, dict):
            mini = {
                "respuesta1": {
                    "por_archivo": (data_new.get("respuesta1") or {}).get("por_archivo") or [],
                    "comparativo_texto": (data_new.get("respuesta1") or {}).get("comparativo_texto") or ""
                },
                "respuesta2": {
                    "por_archivo": (data_new.get("respuesta2") or {}).get("por_archivo") or []
                }
            }
            async with ANALYSIS_LOCK:
                _cache_store_analysis_for_ruc(ruc, mini)
            # (opcional) persistir a master si usas esa función:
            try:
                merge_into_master(mini)  # ignórala si no la usas
            except Exception:
                pass

            mixed_r1.extend(mini["respuesta1"]["por_archivo"])
            mixed_r2.extend(mini["respuesta2"]["por_archivo"])
            if mini["respuesta1"]["comparativo_texto"]:
                comparativos.append(mini["respuesta1"]["comparativo_texto"])

    # -------- Mezclar con cache y DEDUP comparativo --------
    cached_r1, cached_r2, comp_cached = _collect_cached_items_for_current_uploads()

    seen = {(x or {}).get("archivo") for x in mixed_r1}
    for it in cached_r1:
        if (it or {}).get("archivo") not in seen:
            mixed_r1.append(it)

    seen2 = {(x or {}).get("archivo") for x in mixed_r2}
    for it in cached_r2:
        if (it or {}).get("archivo") not in seen2:
            mixed_r2.append(it)

    # ✅ DEDUP de comparativo final
    comparativo_final = _dedup_join_pipes(comp_cached, *comparativos) or "Análisis combinado (cache + nuevos)."

    # Orden estable por nombre de archivo
    try:
        mixed_r1 = sorted(mixed_r1, key=lambda x: (x or {}).get("archivo", ""))
        mixed_r2 = sorted(mixed_r2, key=lambda x: (x or {}).get("archivo", ""))
    except Exception:
        pass

    return {
        "respuesta1": {"por_archivo": mixed_r1, "comparativo_texto": comparativo_final[:1200]},
        "respuesta2": {"por_archivo": mixed_r2},
        "_source": "per-file"
    }
# ========= /CHAT ENDPOINT =========

# Static (front)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="templates", html=True), name="front")
