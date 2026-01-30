import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import google.generativeai as genai

# --- CONFIGURACIÓN DE IA ---
genai.configure(api_key="AIzaSyCocUJXAY1D2b0P_52Kc8BmBatMZvHrhjQ")

st.set_page_config(page_title="SmartReceipt AI - MVP", layout="wide", page_icon="🤖")

st.title("💸 SmartReceipt AI")
st.markdown("### Control de Gastos Personal con Inteligencia Artificial")

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

# --- CEREBRO IA (Gemini corregido) ---
def analizar_con_ia(texto_sucio):
    # Cambiamos a 'gemini-1.5-flash' para evitar el error 404
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Eres un experto en tickets mexicanos. Analiza este texto de OCR:
    {texto_sucio}
    
    Extrae EXACTAMENTE:
    Comercio: [Nombre de la tienda o gasolinera]
    Monto: [Solo el número del total final]
    Categoria: [Despensa, Gasolina, Juguetes o Comida]
    Ubicacion: [Ciudad o zona]
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- INTERFAZ ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 1. Cargar Ticket")
    archivo = st.file_uploader("Sube tu ticket", type=['jpg', 'jpeg', 'png'])
    
    if archivo:
        st.image(archivo, width=300)
        if st.button("🧠 Analizar con IA"):
            with st.spinner("La IA está trabajando..."):
                try:
                    texto_raw = extraer_texto_sucio(archivo)
                    # Llamada a la IA
                    resultado_ia = analizar_con_ia(texto_raw)
                    st.session_state['res_ia'] = resultado_ia
                except Exception as e:
                    st.error(f"Error técnico: {e}")

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
        
        if st.button("💾 Guardar en Historial"):
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
        st.metric("TOTAL", f"${df['Monto'].sum():.2f}")
        
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar CSV", csv, "gastos.csv")
    else:
        st.write("Sube un ticket para ver tu resumen.")