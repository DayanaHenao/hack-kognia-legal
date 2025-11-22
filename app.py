import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Asistente Legal Kognia", layout="wide")

st.title("⚖️ Hack-Kognia: Asistente Legal IA")
st.markdown("""
**Sistema RAG:** Sube un documento y haz preguntas. La IA responderá basándose ÚNICAMENTE en el texto.
""")

# --- BARRA LATERAL DE CONFIGURACIÓN ---
with st.sidebar:
    st.header("🔧 Configuración Técnica")
    
    # 1. API KEY
    api_key_input = st.text_input("1. Pega tu Google API Key", type="password")
    
    # 2. SELECTOR DE MODELO (¡LA SOLUCIÓN AL ERROR 404!)
    st.divider()
    st.write("2. Selecciona el modelo (Si uno falla, prueba otro):")
    model_option = st.selectbox(
        "Modelo de IA:",
        (
            "models/gemini-1.5-flash",  # El más rápido y nuevo
            "models/gemini-pro",        # El clásico (a veces funciona mejor en cuentas viejas)
            "models/gemini-1.5-pro",    # El más inteligente (pero más lento)
            "models/gemini-1.0-pro"     # Versión legacy
        )
    )
    
    st.info(f"Intentando conectar con: {model_option}")

# --- LÓGICA PRINCIPAL ---
if api_key_input:
    try:
        # Limpieza y configuración
        os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
        
        # Configuración Dinámica (Usa lo que seleccionaste en el menú)
        try:
            Settings.llm = Gemini(model=model_option, temperature=0)
            Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
        except Exception as e:
            st.error(f"Error configurando el modelo {model_option}. Prueba seleccionar otro en la lista.")

        # --- CARGA DE ARCHIVOS ---
        uploaded_file = st.file_uploader("3. Sube tu PDF legal", type=['pdf'])

        if uploaded_file:
            # Crear archivo temporal
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "temp.pdf")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Procesamiento
                with st.spinner("⚙️ Indexando documento..."):
                    try:
                        documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
                        index = VectorStoreIndex.from_documents(documents)
                        query_engine = index.as_query_engine()
                        st.success("✅ Documento listo.")
                    except Exception as e:
                        st.error(f"Error leyendo el PDF: {e}")

                # --- CHAT ---
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
                        with st.spinner("Consultando documento..."):
                            try:
                                response = query_engine.query(prompt)
                                st.markdown(response.response)
                                
                                # Evidencia
                                with st.expander("🔍 Ver Evidencia (Fuente)"):
                                    if hasattr(response, 'source_nodes') and response.source_nodes:
                                        st.info(response.source_nodes[0].get_content())
                                    else:
                                        st.warning("Respuesta general basada en contexto.")
                                
                                st.session_state.messages.append({"role": "assistant", "content": response.response})
                            except Exception as e:
                                st.error(f"Error al generar respuesta. Intenta cambiar el modelo en el menú.")

    except Exception as e:
        st.error(f"Error general: {e}")

elif not api_key_input:
    st.warning("👈 Pega tu API Key para comenzar.")
