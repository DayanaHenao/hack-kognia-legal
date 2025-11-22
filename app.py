import streamlit as st
import os
import tempfile
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Asistente Legal Final", layout="wide")
st.title("⚖️ Asistente Legal - Hackathon Ready")
st.markdown("Solución RAG: Indexación Local (Gratis) + Respuesta Gemini (Rápida)")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Pega tu Google API Key", type="password")
    
    # CORRECCIÓN AQUÍ: Quitamos el prefijo "models/" que a veces causa el error 404
    model_choice = st.selectbox("Modelo de Google:", ["gemini-1.5-flash", "gemini-pro"])

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key.strip()
    genai.configure(api_key=api_key.strip())

    # --- 1. CONFIGURACIÓN DE INDEXACIÓN (LOCAL) ---
    try:
        # Usamos el modelo "all-MiniLM-L6-v2" que es ligero y rápido para Hackatons
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
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

            with st.spinner("🧠 Analizando documento..."):
                try:
                    documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
                    index = VectorStoreIndex.from_documents(documents)
                    retriever = index.as_retriever(similarity_top_k=5)
                    st.success("✅ Documento indexado correctamente.")
                except Exception as e:
                    st.error(f"Error al leer el PDF: {e}")

            # --- 3. CHAT ---
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
                    with st.spinner("Generando respuesta..."):
                        try:
                            # A. BUSCAR
                            nodes = retriever.retrieve(prompt)
                            contexto = "\n\n".join([n.get_content() for n in nodes])
                            
                            # B. RESPONDER (Conexión corregida)
                            full_prompt = f"""
                            Eres un asistente legal experto. Responde basándote EXCLUSIVAMENTE en el siguiente contexto.
                            Si la respuesta no está en el texto, di "No encuentro esa información en el documento proporcionado".
                            
                            CONTEXTO DEL PDF:
                            {contexto}
                            
                            PREGUNTA:
                            {prompt}
                            """
                            
                            # Corrección de llamada al modelo
                            model = genai.GenerativeModel(model_choice)
                            response = model.generate_content(full_prompt)
                            
                            st.markdown(response.text)
                            
                            with st.expander("🔍 Ver Evidencia del PDF"):
                                st.info(contexto)
                                
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                            
                        except Exception as e:
                            st.error(f"Error de conexión: {e}")
                            st.warning("Intenta cambiar de 'gemini-1.5-flash' a 'gemini-pro' en el menú de la izquierda.")

elif not api_key:
    st.warning("👈 Pega tu API Key para comenzar.")
