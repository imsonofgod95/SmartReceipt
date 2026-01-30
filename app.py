import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import os
import time
import google.generativeai as genai

# =============================
# CONFIGURACIÓN STREAMLIT
# =============================
st.set_page_config(
    page_title="SmartReceipt AI",
    layout="wide",
    page_icon="🧾"
)

st.title("💸 SmartReceipt AI")
st.markdown("### Control de Gastos Personal")

# =============================
# CONFIGURACIÓN GEMINI
# =============================
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY no está configurada en Secrets")
    st.stop()

genai.configure(api_key=API_KEY)
st.success("✅ API Key cargada correctamente")

# =============================
# MEMORIA DE SESIÓN
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

def extraer_texto_sucio(archivo):
    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    resultados = reader.readtext(processed, detail=0)
    texto = " ".join(resultados)

    return re.sub(r"[^\w\s\$\.\:]", "", texto)

# =============================
# GEMINI CON TIMEOUT
# =============================
def analizar_con_ia(texto):
    prompt = f"""
Devuelve ÚNICAMENTE este formato:

Comercio:
Monto:
Categoria: (Gasolina | Despensa | Juguetes | Comida)

Texto del ticket:
{texto}
"""

    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        response = model.generate_content(
            prompt,
            request_options={"timeout": 15}  # ⏱️ evita cuelgues
        )

        return response.text

    except Exception as e:
        return f"❌ Error Gemini: {e}"

# =============================
# INTERFAZ
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

        if st.button("🧠 Analizar con IA"):
            with st.spinner("Procesando ticket..."):
                st.write("🔍 Ejecutando OCR...")
                texto_ocr = extraer_texto_sucio(archivo)

                st.write("📄 Texto detectado:")
                st.code(texto_ocr[:1000])

                st.write("🤖 Llamando a Gemini...")
                resultado = analizar_con_ia(texto_ocr)

                st.write("✅ Respuesta recibida")
                st.session_state.res_ia = resultado

    if "res_ia" in st.session_state:
        st.info(st.session_state.res_ia)

        comercio, monto = "Desconocido", 0.0

        try:
            m = re.search(r"Monto:\s*([\d\.]+)", st.session_state.res_ia)
            if m:
                monto = float(m.group(1))

            c = re.search(r"Comercio:\s*(.*)", st.session_state.res_ia)
            if c:
                comercio = c.group(1).strip()
        except:
            pass

        st.subheader("📝 Confirmar datos")
        f_comercio = st.text_input("Comercio", comercio)
        f_monto = st.number_input("Monto ($)", value=monto, step=0.01)

        if st.button("💾 Guardar gasto"):
            st.session_state.historial.append({
                "Comercio": f_comercio,
                "Monto": f_monto,
                "Fecha": pd.Timestamp.now().strftime("%d/%m/%Y")
            })
            st.success("✅ Gasto registrado")
            del st.session_state.res_ia

with col2:
    st.header("📊 Historial de Gastos")

    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.table(df)
        st.metric(
            "💰 Total Acumulado",
            f"${df['Monto'].sum():.2f}"
        )
    else:
        st.info("Aún no hay gastos registrados")

# =============================
# TEST DIRECTO DE GEMINI
# =============================
st.divider()
st.subheader("🧪 Test rápido Gemini")

if st.button("Probar conexión Gemini"):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    r = model.generate_content(
        "Di hola y dime que Gemini funciona correctamente",
        request_options={"timeout": 10}
    )
    st.success(r.text)
