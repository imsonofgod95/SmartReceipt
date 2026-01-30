import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import google.generativeai as genai

# --- CONFIGURACIÓN DE IA ---
# Tu llave personal activa
genai.configure(api_key="AIzaSyCocUJXAY1D2b0P_52Kc8BmBatMZvHrhjQ")

st.set_page_config(page_title="SmartReceipt AI - MVP", layout="wide", page_icon="🤖")

st.title("💸 SmartReceipt AI")
st.markdown("### Control de Gastos Personal con IA")

# --- MEMORIA DE LA SESIÓN ---
if 'historial' not in st.session_state:
    st.session_state['historial'] = []

# --- MOTOR OCR ---
@st.cache_resource
def load_reader():
    return easyocr.Reader(['es'], gpu=False)

reader = load_reader()

def extraer_texto_sucio(archivo):
    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    results = reader.readtext(processed, detail=0)
    return " ".join(results)

# --- CEREBRO IA (Con protección de errores 404) ---
def analizar_con_ia(texto_sucio):
    # Intentamos con el modelo más nuevo, y si no, usamos el pro
    modelos_a_probar = ['gemini-1.5-flash', 'gemini-pro']
    
    prompt = f"Analiza este ticket y extrae exactamente este formato:\nComercio: [Nombre]\nMonto: [Solo numero]\nCategoria: [Tipo]\nUbicacion: [Zona]\n\nTEXTO: {texto_sucio}"
    
    for nombre_modelo in modelos_a_probar:
        try:
            model = genai.GenerativeModel(nombre_modelo)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Si el modelo no existe, intenta con el siguiente de la lista
            continue
            
    return "Error: No se pudo conectar con ningún modelo de IA."

# --- INTERFAZ ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 1. Cargar Ticket")
    archivo = st.file_uploader("Sube tu ticket", type=['jpg', 'jpeg', 'png'])
    
    if archivo:
        st.image(archivo, width=300)
        if st.button("🧠 Analizar con IA"):
            with st.spinner("Conectando con el cerebro de Google..."):
                texto_raw = extraer_texto_sucio(archivo)
                resultado_ia = analizar_con_ia(texto_raw)
                st.session_state['res_ia'] = resultado_ia

    if 'res_ia' in st.session_state:
        st.divider()
        res = st.session_state['res_ia']
        st.info(f"Análisis IA:\n{res}")
        
        try:
            c_sug = re.search(r"Comercio: (.*)", res).group(1)
            m_sug = float(re.search(r"Monto: ([\d.]+)", res).group(1))
        except:
            c_sug, m_sug = "Pendiente", 0.0

        final_c = st.text_input("Confirmar Comercio", c_sug)
        final_m = st.number_input("Confirmar Monto ($)", value=m_sug)
        
        if st.button("💾 Guardar"):
            st.session_state.historial.append({
                "Comercio": final_c, 
                "Monto": final_m,
                "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("¡Agregado!")
            del st.session_state['res_ia']

with col2:
    st.header("📊 Mis Gastos")
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.table(df)
        st.metric("TOTAL ACUMULADO", f"${df['Monto'].sum():.2f}")