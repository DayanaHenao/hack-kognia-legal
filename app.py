import streamlit as st
import os
import tempfile
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Asistente Legal Final", layout="wide")
st.title("⚖️ Asistente Legal - Hackathon Ready")
st.markdown("Solución RAG: Indexación Local + Detector Automático de Modelos")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Pega tu Google API Key", type="password")
    
    model_choice = None
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key.strip()
        genai.configure(api_key=api_key.strip())
        
        try:
            # --- MAGIA: DETECTOR AUTOMÁTICO DE MODELOS ---
            # Preguntamos a Google qué modelos tiene tu llave
            modelos_disponibles = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    modelos_disponibles.append(m.name)
            
            # Filtramos los mejores para chat
            mejores_modelos = [m for m in modelos_disponibles if "gemini" in m and "vision" not in m]
            
            if not mejores_modelos:
                mejores_modelos = ["models/gemini-1.5-flash"] # Fallback
                
            st.success(f"¡Conectado! {len(mejores_modelos)} modelos encontrados.")
            model_choice = st.selectbox("Selecciona un modelo activo:", mejores_modelos)
            
        except Exception as e:
            st.error("Error de conexión con Google (Revisa tu API Key)")
            st.caption(e)

# --- LÓGICA PRINCIPAL ---
if api_key and model_choice:

    # --- 1. CONFIGURACIÓN DE INDEXACIÓN (LOCAL) ---
    try:
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
                            
                            # B. RESPONDER
                            full_prompt = f"""
                            Eres un asistente legal experto. Responde basándote EXCLUSIVAMENTE en el siguiente contexto.
                            Si la respuesta no está en el texto, dilo.
                            
                            CONTEXTO DEL PDF:
                            {contexto}
                            
                            PREGUNTA:
                            {prompt}
                            """
                            
                            # Usamos el modelo que detectamos automáticamente
                            model = genai.GenerativeModel(model_choice)
                            response = model.generate_content(full_prompt)
                            
                            st.markdown(response.text)
                            
                            with st.expander("🔍 Ver Evidencia del PDF"):
                                st.info(contexto)
                                
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                            
                        except Exception as e:
                            st.error(f"Error al generar respuesta: {e}")
                            st.warning("Prueba seleccionando OTRO modelo en la lista de la izquierda.")

elif not api_key:
    st.warning("👈 Pega tu API Key para comenzar.")
