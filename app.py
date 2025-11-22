import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Legal Kognia", layout="wide")

st.title("⚖️ Hack-Kognia: Asistente Legal Inteligente")
st.markdown("""
**Solución de Justicia Abierta:** Este sistema utiliza IA para democratizar el acceso a la información legal.
Sube un documento y obtén respuestas claras y fundamentadas.
""")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Google API Key", type="password")
    st.info("Estado: Listo para indexar documentos legales.")

# --- LÓGICA PRINCIPAL ---
if api_key:
    try:
        # Configurar API Key
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # --- CAMBIO CLAVE: Usamos modelos con nombres explícitos para evitar error 404 ---
        # Intentamos usar el modelo PRO que es el más estable para demos
        Settings.llm = Gemini(model_name="models/gemini-pro", temperature=0)
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

        # --- PASO 1: CARGA DE DOCUMENTOS ---
        uploaded_file = st.file_uploader("Sube tu documento legal (PDF)", type=['pdf'])

        if uploaded_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Guardar archivo temporalmente
                temp_path = os.path.join(temp_dir, "temp.pdf")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("🔍 Analizando documento legal..."):
                    try:
                        # Cargar y procesar
                        documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
                        index = VectorStoreIndex.from_documents(documents)
                        query_engine = index.as_query_engine()
                        st.success("✅ Documento procesado y listo para consultas.")
                    except Exception as e:
                        st.error(f"Error técnico al leer el PDF: {e}")

                # --- PASO 2: CHAT ---
                st.divider()
                
                # Inicializar historial
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Mostrar historial
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Input del usuario
                if prompt := st.chat_input("Pregunta algo sobre el documento (ej: ¿Qué vigencia tiene?):"):
                    # Mostrar pregunta usuario
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generar respuesta
                    with st.chat_message("assistant"):
                        with st.spinner("Consultando bases legales..."):
                            try:
                                response = query_engine.query(prompt)
                                st.markdown(response.response)
                                
                                # Evidencia (Requisito Hackathon)
                                with st.expander("Ver fragmento original (Evidencia)"):
                                    st.write(response.source_nodes[0].get_content())
                                
                                st.session_state.messages.append({"role": "assistant", "content": response.response})
                            except Exception as e:
                                st.error("No pude encontrar una respuesta exacta en el documento.")
                                st.caption(f"Detalle del error: {e}")

    except Exception as e:
        st.error(f"Error de conexión con Google: {e}")

elif not api_key:
    st.warning("👈 Por favor pega tu Google API Key en la barra lateral izquierda.")
