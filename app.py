import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import os
import google.generativeai as genai

# =============================
# CONFIGURACIÓN GEMINI (SEGURA)
# =============================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(
    page_title="SmartReceipt AI - Pro",
    layout="wide",
    page_icon="🤖"
)

st.title("💸 SmartReceipt AI")
st.markdown("### Control de Gastos Personal (Versión Estable)")

# =============================
# MEMORIA DE SESIÓN
# =============================
if "historial" not in st.session_state:
    st.session_state.historial = []

# =============================
# MOTOR OCR
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

    results = reader.readtext(processed, detail=0)
    texto = " ".join(results)

    return re.sub(r"[^\w\s\$\.\:]", "", texto)

# =============================
# IA (GEMINI)
# =============================
def analizar_con_ia(texto_sucio):
    modelos = [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro"
    ]

    prompt = f"""
Analiza el siguiente texto de un ticket de compra en México.

Devuelve ÚNICAMENTE este formato:

Comercio: <texto>
Monto: <numero decimal>
Categoria: <Gasolina | Despensa | Juguetes | Comida>

Texto del ticket:
{texto_sucio}
"""

    for modelo in modelos:
        try:
            model = genai.GenerativeModel(modelo)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Fallo modelo {modelo}: {e}")
            continue

    return "Error: Gemini no respondió"

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
            with st.spinner("Analizando ticket..."):
                texto_ocr = extraer_texto_sucio(archivo)
                resultado = analizar_con_ia(texto_ocr)
                st.session_state.res_ia = resultado

    if "res_ia" in st.session_state:
        st.info(st.session_state.res_ia)

        # =============================
        # PARSEO SEGURO
        # =============================
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
            st.success("¡Gasto registrado!")
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
        st.info("Aún no hay gastos registrados.")
