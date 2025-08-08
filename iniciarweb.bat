@echo off
title 🚀 Lanza Backend IA RAG
echo ================================
echo   Activando entorno virtual...
echo   (omite si no usas uno)
echo ================================

REM Si tienes un entorno virtual llamado ".venv", actívalo así:
REM call .venv\Scripts\activate

echo.
echo ================================
echo   Verificando dependencias...
echo ================================

set PACKAGES=fastapi uvicorn openai langchain langchain-openai langchain-community chromadb tiktoken jinja2

for %%p in (%PACKAGES%) do (
    pip show %%p >nul 2>&1
    if errorlevel 1 (
        echo Instalando %%p...
        pip install %%p
    ) else (
        echo Ya está instalado: %%p
    )
)

echo.
echo ================================
echo   ¡Listo!
echo   Llamando a la inteligencia artificial...
echo   Servidor corriendo en http://localhost:8000
echo ================================

python -m uvicorn main:app --reload

pause
