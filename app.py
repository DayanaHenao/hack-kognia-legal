import streamlit as st
import os
import tempfile
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from gtts import gTTS
import io

# --- CONFIGURACIÓN VISUAL PRO ---
st.set_page_config(page_title="Kognia Legal AI", layout="wide", page_icon="⚖️")

# CSS para ocultar marcas de agua y mejorar estilo
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- TÍTULO CON ESTILO ---
st.markdown('<h1 class="main-header">⚖️ Kognia: Justicia Accesible</h1>', unsafe_allow_html=True)
st.markdown("---")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    api_key = st.text_input("🔑 Google API Key", type="password")
    
    # Selector de Modelo
    model_options = ["gemini-1.5-flash", "gemini-pro"]
    model_choice = st.selectbox("🧠 Modelo de IA:", model_options)
    
    st.info("💡 **Tip:** Usa 'gemini-1.5-flash' para mayor velocidad y respuestas ilimitadas.")
    st.markdown("---")
    st.caption("Desarrollado para Hack-Kognia 2025")

# --- FUNCIONES AUXILIARES ---
def texto_a_audio(texto):
    """Convierte texto a audio MP3 en memoria"""
    try:
        tts = gTTS(text=texto, lang='es')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        return None

# --- LÓGICA PRINCIPAL ---
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key.strip()
    genai.configure(api_key=api_key.strip())

    # Configuración Embeddings Locales
    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.llm = None 
    except Exception:
        pass

    # --- PESTAÑAS DE NAVEGACIÓN ---
    tab1, tab2 = st.tabs(["💬 Chat con el Documento", "🆚 Traductor de Jerga Legal"])

    # ========================================================
    # PESTAÑA 1: CHAT RAG (Lo que ya tenías mejorado)
    # ========================================================
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success("📂 **Paso 1: Carga tu Documento**")
            uploaded_file = st.file_uploader("Sube PDF (Leyes, Contratos, Fallos)", type=['pdf'])

        # Lógica de Indexación
        index = None
        retriever = None
        if uploaded_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "temp.pdf")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                with st.spinner("🔍 Leyendo y comprendiendo el documento..."):
                    try:
                        documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
                        index = VectorStoreIndex.from_documents(documents)
                        retriever = index.as_retriever(similarity_top_k=5)
                    except Exception:
                        st.error("Error leyendo el archivo.")

        # Chat
        with col2:
            if uploaded_file and retriever:
                st.info("✅ **Documento activo.** Pregunta lo que quieras.")
                
                # Historial
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        # Si hay audio guardado, mostrarlo (opcional, simple por ahora)

                # Input
                if prompt := st.chat_input("Ej: ¿Cuáles son mis obligaciones según este contrato?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analizando y redactando..."):
                            try:
                                # RAG
                                nodes = retriever.retrieve(prompt)
                                contexto = "\n\n".join([n.get_content() for n in nodes])
                                
                                full_prompt = f"""
                                Eres un abogado experto en lenguaje claro y accesible.
                                Usa el siguiente contexto para responder la pregunta.
                                
                                REGLAS:
                                1. Usa un tono empático y sencillo.
                                2. Si usas términos legales, explícalos entre paréntesis.
                                3. Sé conciso.

                                CONTEXTO: {contexto}
                                PREGUNTA: {prompt}
                                """
                                
                                model = genai.GenerativeModel(model_choice)
                                response = model.generate_content(full_prompt)
                                
                                st.markdown(response.text)
                                
                                # --- FUNCIÓN PRO: AUDIO ---
                                audio = texto_a_audio(response.text)
                                if audio:
                                    st.audio(audio, format='audio/mp3')
                                
                                with st.expander("🔍 Ver evidencia legal"):
                                    st.caption(contexto)
                                    
                                st.session_state.messages.append({"role": "assistant", "content": response.text})
                            except Exception as e:
                                st.error(f"Error: {e}")

            elif not uploaded_file:
                st.warning("👈 Sube un PDF a la izquierda para activar el chat.")

    # ========================================================
    # PESTAÑA 2: COMPARADOR (TRADUCTOR TÉCNICO -> NATURAL)
    # ========================================================
    with tab2:
        st.header("🆚 Traductor de Lenguaje Jurídico")
        st.markdown("Pega un párrafo complejo y mira cómo se transforma en lenguaje ciudadano.")
        
        texto_complejo = st.text_area("Pega aquí el texto legal difícil:", height=150, placeholder="Ej: El arrendatario se constituye en mora...")
        
        if st.button("✨ Traducir a Lenguaje Claro"):
            if texto_complejo and api_key:
                with st.spinner("Traduciendo..."):
                    prompt_traduccion = f"""
                    Actúa como un traductor experto. 
                    Toma el siguiente texto legal y crea una tabla comparativa Markdown con dos columnas:
                    Columna 1: "Texto Original" (El fragmento clave).
                    Columna 2: "Explicación Sencilla" (Lenguaje de 5to grado, muy claro).
                    
                    Texto a procesar:
                    {texto_complejo}
                    """
                    model = genai.GenerativeModel(model_choice)
                    res = model.generate_content(prompt_traduccion)
                    st.markdown(res.text)
                    
                    # Audio de la explicación
                    st.markdown("---")
                    st.caption("🎧 Escuchar explicación:")
                    audio_trad = texto_a_audio(res.text.replace("|", " ")) # Limpieza simple
                    if audio_trad:
                        st.audio(audio_trad, format='audio/mp3')
            else:
                st.warning("Pega un texto y asegúrate de tener la API Key configurada.")

elif not api_key:
    st.warning("🔒 Ingresa tu API Key en la barra lateral para desbloquear el sistema.")
