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
Este sistema utiliza **RAG (Retrieval-Augmented Generation)** con la tecnología más reciente 
de Google (Gemini 1.5 Flash) para analizar documentos legales.
""")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Google API Key", type="password")
    st.info("Usando modelo: gemini-1.5-flash (Más rápido y preciso)")

# --- LÓGICA PRINCIPAL ---
if api_key:
    try:
        # Configurar el cerebro de la IA (AQUÍ ESTABA EL ERROR, YA CORREGIDO)
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Usamos "gemini-1.5-flash" que es el modelo actual y rápido
        Settings.llm = Gemini(model="models/gemini-1.5-flash", temperature=0)
        
        # Usamos el modelo de embeddings más estable
        Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004")

        # --- PASO 1: CARGA DE DOCUMENTOS ---
        uploaded_file = st.file_uploader("Sube tu documento legal (PDF)", type=['pdf'])

        if uploaded_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "temp.pdf")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("Indexando documento... (Esto puede tardar unos segundos)"):
                    try:
                        documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
                        index = VectorStoreIndex.from_documents(documents)
                        query_engine = index.as_query_engine()
                        st.success("¡Documento procesado correctamente!")
                    except Exception as e:
                        st.error(f"Error procesando el PDF: {e}")

                # --- PASO 2: INTERFAZ DE CHAT ---
                st.divider()
                st.subheader("💬 Chat con el Documento")

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ej: ¿Cuáles son las cláusulas de rescisión?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analizando evidencia..."):
                            try:
                                response = query_engine.query(prompt)
                                st.markdown(response.response)
                                
                                # Mostrar fuentes (Requisito del reto)
                                with st.expander("🔍 Ver fuente exacta (Evidencia)"):
                                    # Verificación de seguridad por si no encuentra fuente
                                    if hasattr(response, 'source_nodes') and response.source_nodes:
                                        st.write(response.source_nodes[0].get_content())
                                    else:
                                        st.write("Respuesta general basada en el contexto.")
                                
                                st.session_state.messages.append({"role": "assistant", "content": response.response})
                            except Exception as e:
                                st.error(f"Ocurrió un error al generar la respuesta: {e}")

    except Exception as e:
        st.error(f"Error de configuración de API: {e}")

elif not api_key:
    st.warning("⚠️ Por favor ingresa tu API Key en la barra lateral.")
