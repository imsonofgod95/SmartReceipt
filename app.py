import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import os
import google.generativeai as genai

# =============================
# STREAMLIT
# =============================
st.set_page_config(
    page_title="SmartReceipt AI",
    layout="wide",
    page_icon="🧾"
)

st.title("💸 SmartReceipt AI")
st.markdown("### Control de Gastos Personal")

# =============================
# API KEY
# =============================
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY no configurada en Secrets")
    st.stop()

genai.configure(api_key=API_KEY)
st.success("✅ API Key cargada")

# =============================
# DETECTAR MODELO DISPONIBLE
# =============================
@st.cache_resource
def obtener_modelo_disponible():
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            return m.name
    return None

MODELO = obtener_modelo_disponible()

if not MODELO:
    st.error(
        "❌ Tu API Key NO tiene acceso a ningún modelo Gemini.\n\n"
        "Solución: crea una API Key nueva en https://aistudio.google.com/app/apikey "
        "usando un proyecto con Gemini habilitado."
    )
    st.stop()

st.success(f"🤖 Usando modelo: {MODELO}")

# =============================
# SESIÓN
# =============================
if "historial" not in st.session_state:
    st.session_state.historial = []

# =============================
# OCR
# =============================
@st.cache_resource
def load_reader():
    return easyocr.Reader(["es"], gpu=False)

reader = load_reader()

def extraer_texto(archivo):
    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    texto = " ".join(reader.readtext(processed, detail=0))
    return re.sub(r"[^\w\s\$\.\:]", "", texto)

# =============================
# IA
# =============================
def analizar_con_ia(texto):
    prompt = f"""
Devuelve SOLO este formato:

Comercio:
Monto:
Categoria:

Texto:
{texto}
"""
    try:
        model = genai.GenerativeModel(MODELO)
        response = model.generate_content(
            prompt,
            request_options={"timeout": 15}
        )
        return response.text
    except Exception as e:
        return f"❌ Error IA: {e}"

# =============================
# UI
# =============================
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Subir Ticket")
    archivo = st.file_uploader(
        "Imagen del ticket",
        type=["jpg", "jpeg", "png"]
    )

    if archivo:
        st.image(archivo, width=280)

        if st.button("🧠 Analizar"):
            with st.spinner("Procesando..."):
                st.write("🔍 OCR…")
                texto = extraer_texto(archivo)
                st.code(texto[:1000])

                st.write("🤖 Analizando…")
                resultado = analizar_con_ia(texto)
                st.session_state.res = resultado

    if "res" in st.session_state:
        st.info(st.session_state.res)

        comercio, monto = "Desconocido", 0.0
        try:
            c = re.search(r"Comercio:\s*(.*)", st.session_state.res)
            m = re.search(r"Monto:\s*([\d\.]+)", st.session_state.res)
            if c: comercio = c.group(1).strip()
            if m: monto = float(m.group(1))
        except:
            pass

        st.subheader("📝 Confirmar")
        f_c = st.text_input("Comercio", comercio)
        f_m = st.number_input("Monto", value=monto, step=0.01)

        if st.button("💾 Guardar"):
            st.session_state.historial.append({
                "Comercio": f_c,
                "Monto": f_m,
                "Fecha": pd.Timestamp.now().strftime("%d/%m/%Y")
            })
            st.success("Guardado")
            del st.session_state.res

with col2:
    st.header("📊 Historial")
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.table(df)
        st.metric("Total", f"${df['Monto'].sum():.2f}")
    else:
        st.info("Sin registros aún")
