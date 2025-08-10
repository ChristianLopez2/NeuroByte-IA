# -*- coding: utf-8 -*-
"""
main.py — API RAG para análisis de licitaciones
------------------------------------------------
Este archivo expone endpoints FastAPI para:
- Indexar PDFs (base de conocimiento y subidos por el usuario) en ChromaDB.
- Extraer texto de PDFs (con OCR opcional).
- Recuperar contexto relevante (BASE + OFERTA) y consultar un LLM (OpenAI) para
  generar comparativos y acciones concretas.
- Validar RUC de oferentes con el servicio público del SRI.

Se añadieron docstrings y comentarios explicativos en todas las funciones.
El objetivo es facilitar el mantenimiento y la extensión del sistema.
"""

from __future__ import annotations
import os, re, json, shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
import asyncio

# ---- HTTP / RUC ----
import httpx

# ---- PDF / OCR ----
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# =========================
# Settings
# =========================
class Settings:
    """Parámetros de configuración de rutas, modelos y opciones.

    - BASE_DIR: ruta base del proyecto.
    - UPLOAD_DIR / KNOWLEDGE_DIR: ubicaciones de PDFs del usuario y base.
    - VECTOR_DIR: persistencia de ChromaDB.
    - BASE_COLLECTION / UPLOADS_COLLECTION: nombres de colecciones en la vector DB.
    - OPENAI_API_KEY / EMBEDDING_MODEL / CHAT_MODEL: configuración de OpenAI.
    - CHUNK_SIZE / CHUNK_OVERLAP: tamaños de split de texto.
    - ENABLE_OCR: activa OCR cuando una página no tiene capa de texto.
    - FRONTEND_ORIGIN: origen permitido para CORS.
    - SRI_RUC_URL: endpoint para validar RUC en línea.
    - ACTIONABLE_CHAT_RULES: reglas para forzar respuestas accionables del LLM.
    - DOC_PATTERNS: patrones regex para clasificar tipos de documento.
    """

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

# CORS para permitir al front conectarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=[S.FRONTEND_ORIGIN] if S.FRONTEND_ORIGIN != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear carpetas necesarias si no existen
S.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
S.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
S.VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Utilidades PDF / OCR
# =========================
def _page_has_text(pdf_page: fitz.Page) -> bool:
    """Devuelve True si la página PDF tiene capa de texto extraíble.
    
    Si la extracción falla, retorna False para permitir el fallback a OCR
    en funciones superiores.
    """
    try:
        txt = pdf_page.get_text("text")
        return bool(txt and txt.strip())
    except Exception:
        return False

def extract_text_from_pdf(pdf_path: Path, enable_ocr: bool = S.ENABLE_OCR) -> str:
    """Extrae texto completo de un PDF.

    - Recorre cada página y usa la capa de texto cuando está disponible.
    - Si la página no tiene texto y `enable_ocr=True`, rasteriza y aplica OCR
      con Tesseract usando `spa+eng`.
    - Devuelve el texto concatenado y limpio.
    """
    doc = fitz.open(pdf_path)
    parts: List[str] = []
    for p in doc:
        if _page_has_text(p):
            parts.append(p.get_text("text"))
        else:
            if not enable_ocr:
                continue
            # Render de página a imagen para OCR
            pix = p.get_pixmap(dpi=250)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="spa+eng")
            parts.append(text)
    return "\n".join(parts).strip()

def to_documents(text: str, meta: Dict[str, Any]) -> List[Document]:
    """Divide un texto grande en chunks y los envuelve como `Document` de LangChain.

    Args:
        text: texto completo extraído del PDF.
        meta: metadatos a inyectar en cada chunk (ruta, source, colección, etc.).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=S.CHUNK_SIZE, chunk_overlap=S.CHUNK_OVERLAP
    )
    return splitter.split_documents([Document(page_content=text, metadata=meta)])

# =========================
# Vector store helpers
# =========================
def _embeddings():
    """Crea la instancia de `OpenAIEmbeddings` con el modelo y API key configurados."""
    return OpenAIEmbeddings(model=S.EMBEDDING_MODEL, openai_api_key=S.OPENAI_API_KEY)

def _ensure_chroma(collection_name: str) -> Chroma:
    """Obtiene (o crea) una colección Chroma persistente para `collection_name`.

    Se reutiliza la misma carpeta `persist_directory` para múltiples colecciones.
    """
    return Chroma(
        embedding_function=_embeddings(),
        persist_directory=str(S.VECTOR_DIR),
        collection_name=collection_name
    )

def index_pdfs_in_dir(pdf_dir: Path, collection_name: str) -> int:
    """Indexa todos los PDFs de `pdf_dir` en la colección Chroma indicada.

    - Evita reindexar archivos ya presentes comparando `metadata.source`.
    - Aplica extracción de texto (head para clasificar y full para indexar).
    - Persiste la colección al final.

    Returns:
        Número de chunks añadidos.
    """
    vs = _ensure_chroma(collection_name)
    added = 0
    existing_sources = set(vs.get()["metadatas"] and [m["source"] for m in vs.get()["metadatas"] if m])

    for pdf in pdf_dir.glob("*.pdf"):
        if pdf.name in existing_sources:
            continue
        head = extract_text_head(pdf)
        doctype = guess_doctype(pdf.name, head)
        full_text = extract_text_from_pdf(pdf)  # texto completo
        if not full_text:
            continue
        meta = {"path": str(pdf), "source": pdf.name, "collection": collection_name, "doctype": doctype}
        docs = to_documents(full_text, meta=meta)
        if docs:
            vs.add_documents(docs)
            added += len(docs)
    vs.persist()
    return added

def index_uploaded_pdf(pdf_path: Path, collection_name: str) -> int:
    """(Re)indexa un PDF subido puntualmente en la colección `collection_name`.

    - Borra primero cualquier documento previo con el mismo `source` (nombre de archivo).
    - Clasifica el doctype a partir del encabezado de texto.
    - Extrae el texto completo y lo trocea a chunks `Document`.

    Returns:
        Cantidad de chunks añadidos.
    """
    vs = _ensure_chroma(collection_name)
    # Limpieza de entradas anteriores del mismo archivo
    vs._collection.delete(where={"source": {"$eq": pdf_path.name}})

    head = extract_text_head(pdf_path)
    doctype = guess_doctype(pdf_path.name, head)
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return 0
    meta = {"path": str(pdf_path), "source": pdf_path.name, "collection": collection_name, "doctype": doctype}
    docs = to_documents(text, meta=meta)
    if docs:
        vs.add_documents(docs)
        vs.persist()
    return len(docs)

# =========================
# LLM / QA helpers
# =========================
def make_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Crea un `ChatOpenAI` con el modelo y temperatura deseados.

    Nota: usa la API key definida en `Settings`.
    """
    return ChatOpenAI(model=S.CHAT_MODEL, temperature=temperature, openai_api_key=S.OPENAI_API_KEY)

# =========================
# SRI / RUC
# =========================
def _normalize_ruc_payload(payload: Any) -> Dict[str, Any]:
    """Normaliza la respuesta cruda del SRI a un diccionario compacto.

    La API puede devolver lista con un objeto o un dict. Se homogeniza y se añade
    una heurística simple para `habilitado`: True si `estado` es "ACTIVO".

    Returns:
      {
        ruc, razon_social, estado, actividad_principal, tipo, regimen,
        obligado_contabilidad, contribuyente_especial, agente_retencion,
        habilitado, fechas, raw
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
    """Consulta asíncrona al SRI para obtener información de un RUC.

    - Reintenta hasta 3 veces en caso de error/transitorio.
    - Si la respuesta es 200, intenta parsear JSON y normalizar.
    - Devuelve estructura estandarizada con campos relevantes.
    """
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
                    # Espera breve antes de reintentar
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
# Análisis comparativo (prompts y utilidades)
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
   - Cumplimiento por requisito (OK, PARCIAL, NO) con breve evidencia y fuente (`oferente` o `base`).
   - Riesgos y cláusulas sensibles: multas, garantías, plazos, ambigüedades.  
     Marca `critico` como `true` si el riesgo es alto.
   - Diferencias clave: indicar tema (`plazo`, `monto`, `garantias`, `otros`), contenido en base y en oferta, y su impacto (`bajo`, `medio`, `alto`).
   - Recomendaciones y cláusulas faltantes.

3. **RUC**: El sistema obtendrá esta información automáticamente para cada documento; debes incluir la clave `"ruc_validacion"` en tu salida, con el siguiente formato:
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
    """Intenta parsear un JSON del string devuelto por el LLM.

    - Si llega texto con otros elementos, busca el primer bloque `{...}` para
      rescatar un JSON válido.
    - Si no es posible, retorna un dict con un resumen del texto original.
    """
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
class ChatRequest(BaseModel):
    """Payload para el endpoint /api/chat.

    - query: instrucción o pregunta del usuario.
    - focus_upload: nombre de archivo a priorizar (opcional, gestionado por el front).
    """
    query: str
    focus_upload: Optional[str] = None

@app.on_event("startup")
def startup_index_base():
    """Evento de arranque del servicio.

    - Indexa PDFs de la base en la colección `BASE_COLLECTION`.
    - Limpia la carpeta de uploads en disco para iniciar “en blanco”.
    - Sincroniza la colección de uploads con el contenido en disco.
    """
    index_pdfs_in_dir(S.KNOWLEDGE_DIR, S.BASE_COLLECTION)

    # Limpieza de archivos previos en uploads
    if os.path.exists(S.UPLOAD_DIR):
        for filename in os.listdir(S.UPLOAD_DIR):
            filepath = os.path.join(S.UPLOAD_DIR, filename)
            try:
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception as e:
                # Silenciar errores no críticos en arranque
                print('')

    sync_uploads_index()

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Construye el contexto combinado BASE+OFERTA y consulta al LLM.

    Flujo:
      1) Construye el contexto con `build_multi_offer_context`.
      2) Arma un prompt con reglas de respuesta accionables y el esquema JSON.
      3) Invoca el LLM.
      4) Intenta convertir la salida a JSON y valida su estructura mínima.
      5) Enriquecer por-oferta con validación de RUC.
      6) Formatear una respuesta “humana” y lanzar análisis por oferta (paralelo).
    """
    user_q = (req.query or "").strip() or "Evalúa cumplimiento por oferta, detecta brechas y acciones de mejora concretas."
    context, offers = build_multi_offer_context(user_q, k_base=8, k_up=8)

    llm = make_llm(temperature=0.1)
    prompt = (
        "Eres un experto en licitaciones. Compara lo exigido por la BASE con lo presentado en cada OFERTA.\n"
        + S.ACTIONABLE_CHAT_RULES
        + GROUPED_ACTIONABLE_SCHEMA
        + "\n\n# Pregunta del usuario:\n"
        + user_q
        + "\n\n# CONTEXTO RAG (cada bloque marcado con '### DOC: <nombre> (BASE|OFERTA)'):\n"
        + context[:150000]
        + "\n\nDevuelve SOLO el JSON solicitado, sin texto adicional."
    )

    resp = await llm.ainvoke(prompt)
    data = json_guard(resp.content)

    if not isinstance(data, dict) or "por_oferta" not in data:
        # Fallback: devolver salida cruda y parse parcial si el modelo no cumplió.
        return {"result": f"Salida del modelo:\n\n{resp.content}", "raw": data}

    # === Agregar ruc_validacion por cada oferta (por nombre/archivo) ===
    por_oferta = data.get("por_oferta") or []
    for item in por_oferta:
        archivo = (item.get("archivo") or "").strip()
        if not archivo:
            continue
        # Validar RUC de este PDF (stem = nombre sin .pdf)
        stem = Path(archivo).stem
        ruc_info = await validate_ruc_for_offer(stem)
        item["ruc_validacion"] = ruc_info  # inyecta/actualiza

    # === Formateo de salida humano (resumen de inconsistencias/cumplimientos) ===
    lines = []
    por_oferta = sorted(por_oferta, key=lambda x: (x.get("archivo") or "").lower())

    for item in por_oferta:
        archivo = item.get("archivo") or "Oferta_sin_nombre"
        estado = item.get("estado") or "SIN_EVIDENCIA"
        inconsistencias = item.get("inconsistencias") or []
        cumplimientos = item.get("cumplimientos") or []
        ruc_v = item.get("ruc_validacion") or {}
        ruc_txt = "RUC: no encontrado"
        if ruc_v.get("ruc"):
            ok = ruc_v.get("habilitado")
            rz = ruc_v.get("razon_social") or "N/D"
            est = ruc_v.get("estado") or "N/D"
            ruc_txt = f"RUC {ruc_v['ruc']} — " + ("HABILITADO" if ok else "NO HABILITADO") + f" (Razón social: {rz}, Estado: {est})"
        elif ruc_v.get("notas"):
            ruc_txt = ruc_v["notas"]

        lines.append(f"Oferta: {archivo}")
        lines.append(f"- Verificación RUC: {ruc_txt}")

        if estado == "SIN_INCONSISTENCIAS" or len(inconsistencias) == 0:
            if cumplimientos:
                lines.append("Cumplimientos detectados:")
                for i, c in enumerate(cumplimientos, start=1):
                    req = c.get("requisito") or "(requisito no especificado)"
                    ev  = c.get("evidencia_oferta") or "—"
                    doc = c.get("documento_base") or "N/A"
                    lines.append(f"  {i}. {req}")
                    lines.append(f"     Evidencia: {ev}")
                    lines.append(f"     Documento base: {doc}")
            else:
                lines.append("Sin inconsistencias. (No se listaron cumplimientos específicos.)")
                lines.append("")
            continue

        for i, inc in enumerate(inconsistencias, start=1):
            req = inc.get("requisito") or "(requisito no especificado)"
            ev  = inc.get("evidencia_oferta") or "no encontrado"
            acc = inc.get("accion_concreta") or "Definir acción concreta"
            doc = inc.get("documento_base") or "N/A"
            lines.append(f"{i}. {req}")
            lines.append(f"   Evidencia en oferta: {ev}")
            lines.append(f"   Acción: {acc}")
            lines.append(f"   Documento base: {doc}")
        lines.append("")

    result_text = "\n".join(lines).strip()

    # Construimos contexto enfocado por oferta (OFERTA + BASE del mismo doctype)
    vs_base = _ensure_chroma(S.BASE_COLLECTION)
    vs_up   = _ensure_chroma(S.UPLOADS_COLLECTION)

    async def build_context_for_offer(stem: str, k_base: int = 8, k_up: int = 8) -> tuple[str, str]:
        """Devuelve (contexto, doctype_detectado) para una oferta concreta.

        - Detecta el doctype desde metadatos de la colección de uploads.
        - Recupera chunks relevantes tanto de la OFERTA como de la BASE del mismo tipo.
        - Formatea bloques etiquetados como `### DOC: <nombre> (BASE|OFERTA)`.
        """
        # Obtener el doctype del upload desde metadatos
        meta = vs_up._collection.get(where={"source": {"$in": [f.name for f in S.UPLOAD_DIR.glob(f"{stem}*.pdf")] }},
                                     include=["metadatas"])
        doctype = "OTROS"
        if meta and meta.get("metadatas"):
            for m in meta["metadatas"]:
                if m and m.get("doctype"):
                    doctype = m["doctype"]; break

        # Recuperar documentos de la OFERTA (por source = ese stem)
        up_ret = vs_up.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_up, "fetch_k": 20,
                "filter": {"source": {"$in": [p.name for p in S.UPLOAD_DIR.glob(f"{stem}*.pdf")]}}
            }
        )
        up_docs = up_ret.get_relevant_documents(user_q)

        # Recuperar documentos de la BASE del mismo doctype
        base_ret = vs_base.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_base, "fetch_k": 20, "filter": {"doctype": {"$eq": doctype}}}
        )
        base_docs = base_ret.get_relevant_documents(user_q)

        parts = []
        for d in base_docs:
            parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (BASE)\n{d.page_content}\n")
        for d in up_docs:
            parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (OFERTA)\n{d.page_content}\n")
        return ("".join(parts), doctype)

    async def analyze_offer_json(stem: str, ruc_info: dict) -> dict:
        """Ejecuta el análisis JSON detallado por cada oferta.

        - Construye contexto específico de la oferta.
        - Invoca un prompt de análisis (ANALYSIS_SYSTEM_PROMPT) y fuerza salida JSON.
        - Inyecta la validación de RUC ya consultada.
        """
        ctx_offer, dt = await build_context_for_offer(stem, k_base=8, k_up=8)
        anal_prompt = (
            ANALYSIS_SYSTEM_PROMPT
            + "\n\n# CONTEXTO RAG por OFERTA (BASE y OFERTA del mismo tipo):\n"
            + ctx_offer[:150000]
            + "\n\nDevuelve SOLO el JSON solicitado, sin texto adicional."
        )
        resp = await make_llm(temperature=0.0).ainvoke(anal_prompt)
        anal = json_guard(resp.content)
        # Inyecta RUC
        anal["ruc_validacion"] = {
            "ruc": ruc_info.get("ruc"),
            "nombre": ruc_info.get("razon_social"),
            "habilitado": ruc_info.get("habilitado"),
            "notas": ruc_info.get("notas") or ruc_info.get("estado") or None
        }
        anal["_archivo"] = stem
        anal["_doctype"] = dt
        return anal

    # Ejecutar análisis por oferta en paralelo
    per_offer_tasks = []
    for item in por_oferta:
        stem = Path((item.get("archivo") or "oferta")).stem
        ruc_info = item.get("ruc_validacion") or {}
        per_offer_tasks.append(analyze_offer_json(stem, ruc_info))

    per_offer_analysis = await asyncio.gather(*per_offer_tasks) if per_offer_tasks else []

    # Respuesta final combinada
    return {
        "result": result_text,
        "analysis": data,
        "per_offer_analysis": per_offer_analysis
    }

@app.post("/api/upload-pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    """Recibe PDFs, los guarda en `uploads/pdf` e indexa su contenido.

    Validaciones:
    - Asegura que sean PDFs por `content_type`.
    - Devuelve por archivo la cantidad de chunks generados.
    """
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
    """Lista los PDFs actualmente presentes en la carpeta de uploads.

    Útil para que el front muestre un selector de archivos disponibles.
    """
    files = []
    for p in S.UPLOAD_DIR.glob("*.pdf"):
        files.append({
            "filename": p.name,
            "disk_path": str(p.resolve()),
            "size_bytes": p.stat().st_size
        })
    return {"count": len(files), "files": files}

# Servir estáticos y front (plantillas)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="templates", html=True), name="front")

# =========================
# Sincronización y utilidades de uploads
# =========================

def current_upload_sources() -> set[str]:
    """Retorna el conjunto de nombres de archivo PDF presentes en `uploads/pdf`."""
    return {p.name for p in S.UPLOAD_DIR.glob("*.pdf")}


def delete_upload_docs_where(where: Dict[str, Any]) -> int:
    """Elimina documentos de la colección de uploads según un filtro `where` (Chroma).

    Nota: la API interna de Chroma para `delete` no devuelve conteo, por lo que
    se retorna 1 como confirmación simbólica.
    """
    vs = _ensure_chroma(S.UPLOADS_COLLECTION)
    # langchain->chroma admite filtros "where"
    vs._collection.delete(where=where)  # devuelve None
    return 1


def sync_uploads_index() -> dict:
    """Sincroniza la colección `licitaciones_uploads` con lo que existe en disco.

    - Si hay PDFs en disco: borra de Chroma lo que NO esté en `uploads/pdf`.
    - Si NO hay PDFs en disco: borra TODO lo que haya en la colección para evitar
      resultados fantasmas en búsquedas.
    - Llama a `persist()` (opcional en Chroma >=0.4) para asegurar consistencia.

    Returns:
        Dict con listas de `kept` (archivos mantenidos) y `deleted` (conteo borrado).
    """
    deleted = 0
    vs = _ensure_chroma(S.UPLOADS_COLLECTION)
    keep = sorted(list(current_upload_sources()))

    if keep:
        # borra todo lo que NO esté en keep
        vs._collection.delete(where={"source": {"$nin": keep}})
    else:
        # no hay PDFs en disco: borra TODO por ids
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
    """Endpoint para forzar una sincronización de la colección de uploads."""
    info = sync_uploads_index()
    return {"ok": True, "synced": info}

# === RUC helpers por oferta ===
RUC_RE = re.compile(r"\b(\d{13})\b")  # RUC Ecuador: 13 dígitos


def extract_ruc_from_offer_stem(stem: str) -> Optional[str]:
    """Intenta extraer un RUC (13 dígitos) del PDF cuyo nombre base es `stem`.

    - Localiza el PDF en `uploads/pdf` por `stem` (con o sin sufijos).
    - Extrae texto (incluyendo OCR si aplica) y busca la primera coincidencia.
    - Devuelve el RUC encontrado o `None` si no hay match.
    """
    pdf_path = S.UPLOAD_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        # intenta con nombres que incluyan la extensión u otros sufijos
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
    """Extrae RUC de la oferta (por `stem`) y lo valida contra el SRI.

    Returns un dict normalizado con campos clave (`habilitado`, `razon_social`, etc.).
    Si no se encuentra RUC, retorna un dict con `notas` explicando la situación.
    """
    ruc = extract_ruc_from_offer_stem(stem)
    if not ruc:
        return {"ruc": None, "habilitado": None, "notas": "RUC no encontrado en el documento"}

    try:
        info = await fetch_ruc_info(ruc)
        # Asegura campos clave y un mensaje compacto
        return {
            "ruc": info.get("ruc") or ruc,
            "habilitado": info.get("habilitado"),
            "razon_social": info.get("razon_social") or info.get("raw", {}).get("razonSocial"),
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
      "semaforo": "verde|amarillo|rojo"
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
- Calcula 'semaforo' así: verde=0 inconsistencias; amarillo=1-2 inconsistencias no críticas; rojo=3+ o inconsistencias críticas si se deduce del contexto.
- Si una oferta no presenta inconsistencias, rellena 'cumplimientos' con 3–10 puntos concretos que SÍ cumple (cada uno con evidencia corta y referencia al documento base).
"""


def build_multi_offer_context(query: str, k_base: int = 8, k_up: int = 8) -> Tuple[str, list[str]]:
    """Construye un contexto combinado por cada oferta detectada en `uploads/pdf`.

    Lógica:
      - Si no hay uploads, retorna sólo contexto BASE.
      - Si hay uploads: para cada `stem` detectado, obtiene chunks relevantes de la OFERTA
        y de la BASE que coincidan con su `doctype`.

    Returns:
      (contexto_formateado, lista_de_stems_de_ofertas)
    """
    vs_base = _ensure_chroma(S.BASE_COLLECTION)
    vs_up   = _ensure_chroma(S.UPLOADS_COLLECTION)

    # 1) detectar qué uploads están vivos y sus doctypes
    live = list(S.UPLOAD_DIR.glob("*.pdf"))
    if not live:
        # fallback: solo base sin filtro de doctype
        base_ret = vs_base.as_retriever(search_type="mmr", search_kwargs={"k": k_base, "fetch_k": 20})
        base_docs = base_ret.get_relevant_documents(query)
        parts = [f"\n### DOC: {Path(d.metadata['source']).stem} (BASE)\n{d.page_content}\n" for d in base_docs]
        return "".join(parts), []

    # 2) obtener los doctypes de esos uploads directamente desde Chroma
    up_meta = vs_up._collection.get(where={"source": {"$in": [p.name for p in live]}}, include=["metadatas","documents"])
    metadatas = up_meta.get("metadatas") or []
    ids = up_meta.get("ids") or []
    by_stem = {}
    for m in metadatas:
        if not m:
            continue
        src = m.get("source") or ""
        stem = Path(src).stem
        by_stem.setdefault(stem, m.get("doctype") or "OTROS")

    # 3) para cada doctype en uploads, traemos base del MISMO doctype
    context_parts = []
    offers_stem = sorted(by_stem.keys())

    for stem in offers_stem:
        dt = by_stem[stem]
        # uploads del mismo stem
        up_ret = vs_up.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_up, "fetch_k": 20, "filter": {"source": {"$in": [f.name for f in live if Path(f).stem == stem]}}}
        )
        up_docs = up_ret.get_relevant_documents(query)
        for d in up_docs:
            context_parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (OFERTA)\n{d.page_content}\n")

        # base del mismo doctype
        base_kwargs = {"k": k_base, "fetch_k": 20, "filter": {"doctype": {"$eq": dt}}}
        base_ret = vs_base.as_retriever(search_type="mmr", search_kwargs=base_kwargs)
        base_docs = base_ret.get_relevant_documents(query)
        for d in base_docs:
            context_parts.append(f"\n### DOC: {Path(d.metadata['source']).stem} (BASE)\n{d.page_content}\n")

    return "".join(context_parts), offers_stem


def guess_doctype(filename: str, sample_text: str = "") -> str:
    """Intenta clasificar el tipo de documento a partir del nombre y contenido inicial.

    - Usa patrones regex (`Settings.DOC_PATTERNS`).
    - Aplica heurísticas adicionales por nombre de archivo.
    - Devuelve una etiqueta como `PLIEGO_COTIZACION`, `FORMULARIO_OFERTA`, etc.
    """
    name = filename.lower()
    text = (sample_text or "").lower()
    hay = name + "\n" + text[:3000]  # miramos nombre + primeras líneas
    for label, pat in S.DOC_PATTERNS:
        if re.search(pat, hay, re.I):
            return label
    # heurísticas por nombre
    if "pliego" in name: return "PLIEGO_COTIZACION"
    if "formulario" in name: return "FORMULARIO_OFERTA"
    if "condiciones_particulares" in name and "contrato" in name: return "CONTRATO_CONDICIONES_PARTICULARES"
    if "condiciones_generales" in name and "contrato" in name: return "CONTRATO_CONDICIONES_GENERALES"
    if "consorcio" in name or "asociacion" in name or "asociación" in name: return "CONSORCIO_COMPROMISO"
    if "subasta" in name: return "SUBASTA_INVERSA_PLIEGO"
    return "OTROS"


def extract_text_head(pdf_path: Path, enable_ocr: bool = S.ENABLE_OCR, max_chars: int = 4000) -> str:
    """Extrae un “encabezado” de texto (hasta `max_chars`) para clasificación rápida.

    - Recorre páginas hasta acumular el límite.
    - Usa capa de texto o, si falta y `enable_ocr=True`, aplica OCR con dpi 200.
    - Retorna el texto inicial útil para `guess_doctype`.
    """
    doc = fitz.open(pdf_path)
    parts = []
    for p in doc:
        if len(" ".join(parts)) > max_chars:
            break
        if _page_has_text(p):
            parts.append(p.get_text("text"))
        elif enable_ocr:
            pix = p.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            parts.append(pytesseract.image_to_string(img, lang="spa+eng"))
    return "\n".join(parts)[:max_chars]

# (build_context_for_offer alternativo quedó comentado para referencia histórica)
# def build_context_for_offer(...):
#     ...
