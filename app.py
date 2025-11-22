import streamlit as st
import os
import tempfile
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Hack-Kognia Final", layout="wide")
st.title("⚖️ Asistente Legal - Conexión Directa")
st.markdown("Solución RAG híbrida: Indexación Local + Generación Google Directa")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Pega tu Google API Key", type="password")
    
    # Selector de seguridad por si un modelo falla
    model_choice = st.selectbox("Modelo de Google:", ["gemini-1.5-flash", "gemini-pro"])

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key.strip()
    genai.configure(api_key=api_key.strip())

    # --- 1. CONFIGURACIÓN DE INDEXACIÓN (HuggingFace - GRATIS Y LOCAL) ---
    # Esto evita errores de OpenAI y de Google Embeddings
    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Desactivamos el LLM de LlamaIndex para que no de error 404
        Settings.llm = None 
    except Exception as e:
        st.error(f"Error cargando modelos locales: {e}")

    # --- 2. SUBIDA DE ARCHIVO ---
    uploaded_file = st.file_uploader("Sube tu PDF Legal", type=['pdf'])

    if uploaded_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "temp.pdf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner("🧠 Analizando documento (Indexación Local)..."):
                try:
                    # Cargar y crear índice
                    documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
                    index = VectorStoreIndex.from_documents(documents)
                    # Creamos un "retriever" (buscador) en lugar de un motor de chat completo
                    retriever = index.as_retriever(similarity_top_k=3)
                    st.success("✅ Documento indexado correctamente.")
                except Exception as e:
                    st.error(f"Error al leer el PDF: {e}")

            # --- 3. CHAT HÍBRIDO ---
            st.divider()
            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Escribe tu pregunta..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Consultando documento y generando respuesta..."):
                        try:
                            # A. BUSCAR INFORMACIÓN (RAG)
                            nodes = retriever.retrieve(prompt)
                            contexto = "\n\n".join([n.get_content() for n in nodes])
                            
                            # B. GENERAR RESPUESTA (CONEXIÓN DIRECTA A GOOGLE)
                            # Aquí evitamos el error 404 de LlamaIndex llamando a Google directo
                            full_prompt = f"""
                            Actúa como un abogado experto y claro. 
                            Usa la siguiente INFORMACIÓN DEL CONTEXTO para responder la PREGUNTA del usuario.
                            Si la respuesta no está en el contexto, dilo.
                            
                            INFORMACIÓN DEL CONTEXTO (PDF):
                            {contexto}
                            
                            PREGUNTA DEL USUARIO:
                            {prompt}
                            """
                            
                            model = genai.GenerativeModel(model_choice)
                            response = model.generate_content(full_prompt)
                            
                            st.markdown(response.text)
                            
                            # Mostrar evidencia
                            with st.expander("🔍 Ver Evidencia del PDF"):
                                st.info(contexto)
                                
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                            
                        except Exception as e:
                            st.error(f"Error: {e}")

elif not api_key:
    st.warning("👈 Pega tu API Key para comenzar.")
