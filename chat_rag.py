import os
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

# API Key de OpenAI. Se recomienda usar una variable de entorno en lugar de escribirla aquí.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def cargar_documentos_txt(ruta_archivos: str) -> List:
    """
    Carga todos los archivos `.txt` de una carpeta y devuelve una lista de documentos.

    Args:
        ruta_archivos (str): Ruta de la carpeta que contiene los archivos de texto.

    Returns:
        List: Lista de objetos Document cargados con `TextLoader`.
    """
    documentos = []
    for archivo in os.listdir(ruta_archivos):
        if archivo.endswith(".txt"):
            loader = TextLoader(os.path.join(ruta_archivos, archivo), encoding="utf-8")
            documentos.extend(loader.load())
    return documentos

def dividir_documentos(documentos: List, chunk_size: int = 500, chunk_overlap: int = 50) -> List:
    """
    Divide una lista de documentos en fragmentos (chunks) con solapamiento para optimizar el procesamiento.

    Args:
        documentos (List): Lista de documentos cargados.
        chunk_size (int, opcional): Tamaño máximo de cada fragmento. Por defecto 500.
        chunk_overlap (int, opcional): Número de caracteres que se solapan entre fragmentos. Por defecto 50.

    Returns:
        List: Lista de fragmentos de documentos.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documentos)

def crear_base_vectorial(chunks: List, persist_dir: str = "vector_db") -> Chroma:
    """
    Crea una base vectorial usando Chroma y la persiste en disco.

    Args:
        chunks (List): Lista de fragmentos de documentos.
        persist_dir (str, opcional): Carpeta donde se almacenará la base vectorial. Por defecto "vector_db".

    Returns:
        Chroma: Objeto de base vectorial con embeddings generados.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="txt_conocimiento"
    )
    vectorstore.persist()
    return vectorstore

def crear_chatbot(vectorstore: Chroma) -> RetrievalQA:
    """
    Crea un chatbot basado en recuperación (RetrievalQA) con memoria de conversación.

    Args:
        vectorstore (Chroma): Base vectorial que se usará para recuperar información.

    Returns:
        RetrievalQA: Objeto listo para recibir preguntas y responder usando IA.
    """
    retriever = vectorstore.as_retriever()
    memoria = ConversationBufferMemory()
    memoria.chat_memory.add_ai_message(
        "Hola, soy tu IA entrenada con archivos de texto 📚. Pregúntame lo que quieras sobre ellos."
    )
    llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0.2)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, memory=memoria)

def main() -> None:
    """
    Punto de entrada del script.

    - Carga documentos desde la carpeta `./txts`.
    - Divide el contenido en fragmentos.
    - Crea una base vectorial persistente.
    - Inicia un chat interactivo en consola.
    """
    ruta_txt = "./txts"  # Cambia si tus archivos están en otra carpeta
    print("📂 Cargando archivos de texto...")
    documentos = cargar_documentos_txt(ruta_txt)

    print("✂️ Dividiendo en chunks...")
    chunks = dividir_documentos(documentos)

    print("💾 Creando base de conocimiento...")
    vectorstore = crear_base_vectorial(chunks)

    print("🤖 ¡Chat listo para usarse!")
    chat = crear_chatbot(vectorstore)

    while True:
        pregunta = input("\n🧠 Tú: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("🚪 Saliendo del chat... ¡Hasta la próxima!")
            break
        respuesta = chat.invoke(pregunta)
        print("🤖 IA:", respuesta["result"])

if __name__ == "__main__":
    main()
