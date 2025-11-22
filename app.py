import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Asistente Legal Kognia", layout="wide")

st.title("⚖️ Hack-Kognia: Asistente Legal con IA")
st.markdown("""
**Sistema RAG (Retrieval-Augmented Generation):** Sube tu documento y la IA buscará la respuesta exacta dentro del texto.
*Modelo activo: Gemini 1.5 Flash (Google)*
""")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Paso 1: Configuración")
    # .strip() elimina espacios en blanco accidentales al copiar
    api_key_input = st.text_input("Pega tu Google API Key aquí", type="password")
    
    st.divider()
    st.info("Si sale error 404, verifica que tu API Key sea correcta y tenga permisos en Google AI Studio.")

# --- LÓGICA PRINCIPAL ---
if api_key_input:
    try:
        # Limpiamos la clave por seguridad
        my_api_key = api_key_input.strip()
        os.environ["GOOGLE_API_KEY"] = my_api_key
        
        # --- CONFIGURACIÓN DEL MODELO (LA SOLUCIÓN) ---
        # Usamos 'models/gemini-1.5-flash' que es el más compatible actualmente
        Settings.llm = Gemini(model="models/gemini-1.5-flash", temperature=0)
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

        # --- CARGA DE ARCHIVOS ---
        uploaded_file = st.file_uploader("Paso 2: Sube tu PDF legal", type=['pdf'])

        if uploaded_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "temp.pdf")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("⚙️ Indexando documento... (Creando base de conocimiento)"):
                    try:
                        # Cargar datos
                        documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
                        # Crear índice vectorial (RAG)
                        index = VectorStoreIndex.from_documents(documents)
                        query_engine = index.as_query_engine()
                        st.success("✅ ¡Documento indexado! Ya puedes preguntar.")
                    except Exception as e:
                        st.error(f"Error al leer el documento: {e}")

                # --- CHAT ---
                st.divider()
                st.subheader("💬 Paso 3: Pregúntale al documento")

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Escribe tu pregunta aquí..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analizando evidencia legal..."):
                            try:
                                response = query_engine.query(prompt)
                                st.markdown(response.response)
                                
                                # Mostrar evidencia (Requisito clave)
                                with st.expander("🔍 Ver fragmento original del texto"):
                                    if hasattr(response, 'source_nodes') and response.source_nodes:
                                        st.info(response.source_nodes[0].get_content())
                                    else:
                                        st.warning("No se encontró una cita exacta en el texto.")
                                
                                st.session_state.messages.append({"role": "assistant", "content": response.response})
                            except Exception as e:
                                st.error(f"Error al generar respuesta: {e}")

    except Exception as e:
        st.error(f"Error crítico de configuración: {e}")

elif not api_key_input:
    st.warning("👈 Para empezar, pega tu API Key en el menú de la izquierda.")
