# ⚖️ Hack-Kognia: Asistente Legal Inteligente (RAG)

## 📄 Resumen del Proyecto
Este es un prototipo funcional (MVP) desarrollado para el reto **Hackathon Caldas 2025: Hack-Kognia 1.0**. 

Es un asistente legal basado en Inteligencia Artificial que utiliza la arquitectura **RAG (Recuperación Aumentada por Generación)** para interpretar documentos jurídicos complejos (contratos, leyes, sentencias) y responder preguntas ciudadanas con evidencia y sin alucinaciones.

🔗 **DEMO EN VIVO:** [PEGA AQUÍ TU ENLACE DE STREAMLIT]

## 🚀 Arquitectura Técnica (Enfoque Híbrido)
Para cumplir con los requisitos de eficiencia y privacidad, implementamos una arquitectura híbrida:

1.  **Ingesta de Datos:** Procesamiento de PDFs usando `pypdf`.
2.  **Indexación Local (Privacidad):** Utilizamos `HuggingFace Embeddings` (modelo `all-MiniLM-L6-v2`) para vectorizar el texto localmente en el servidor, eliminando dependencias de APIs de terceros para la búsqueda.
3.  **Recuperación (Retriever):** Motor de búsqueda semántica construido con `LlamaIndex`.
4.  **Generación (LLM):** Conexión directa con **Google Gemini 1.5 Flash** para la síntesis de respuestas, garantizando velocidad y alta ventana de contexto.

## 🛠️ Stack Tecnológico
* **Frontend:** Streamlit (Python)
* **Orquestación:** LlamaIndex
* **Embeddings:** Sentence-Transformers (HuggingFace)
* **LLM:** Google Gemini API (1.5 Flash)

## ⚙️ Instrucciones de Ejecución Local
1.  Clonar el repositorio.
2.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configurar la API Key de Google.
4.  Ejecutar la aplicación:
    ```bash
    streamlit run app.py
    ```

## 👥 Equipo
Participante del Reto Hack-Kognia 2025.
