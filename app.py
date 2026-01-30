import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions

# --- CONFIGURACIÓN DE IA ROBUSTA ---
# Tu llave activa
API_KEY = "AIzaSyCocUJXAY1D2b0P_52Kc8BmBatMZvHrhjQ"
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="SmartReceipt AI - Pro", layout="wide", page_icon="🤖")

st.title("💸 SmartReceipt AI")
st.markdown("### Control de Gastos Personal (Versión Estable)")

# --- MEMORIA DE SESIÓN ---
if 'historial' not in st.session_state:
    st.session_state['historial'] = []

# --- MOTOR OCR ---
@st.cache_resource
def load_reader():
    # Desactivamos GPU para estabilidad en la nube
    return easyocr.Reader(['es'], gpu=False)

reader = load_reader()

def extraer_texto_sucio(archivo):
    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filtro para tickets de gas o arrugados
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    results = reader.readtext(processed, detail=0)
    # Limpiamos caracteres que puedan confundir a la IA
    texto = " ".join(results)
    return re.sub(r'[^\w\s\$\.]', '', texto)

# --- CEREBRO IA CON PARCHES DE SEGURIDAD ---
def analizar_con_ia(texto_sucio):
    # Lista de modelos por orden de estabilidad
    modelos_disponibles = ['gemini-1.5-flash', 'gemini-pro']
    
    prompt = f"""
    Como experto financiero, analiza este texto de ticket mexicano:
    "{texto_sucio}"
    
    Extrae exactamente este formato:
    Comercio: [Nombre comercial]
    Monto: [Solo numero decimal]
    Categoria: [Gasolina, Despensa, Juguetes o Comida]
    """
    
    for nombre_modelo in modelos_disponibles:
        try:
            # Forzamos al modelo a inicializarse
            model = genai.GenerativeModel(model_name=nombre_modelo)
            response = model.generate_content(prompt)
            return response.text
        except exceptions.NotFound:
            # Si el 404 persiste, intenta con el siguiente modelo
            continue
        except Exception as e:
            return f"Error de sistema: {str(e)}"
            
    return "Error: No se pudo establecer conexión con los modelos de Google."

# --- INTERFAZ DE USUARIO ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Subir Ticket")
    archivo = st.file_uploader("Imagen del ticket", type=['jpg', 'jpeg', 'png'])
    
    if archivo:
        st.image(archivo, width=280)
        if st.button("🧠 Analizar con IA"):
            with st.spinner("Conectando con servidores de IA..."):
                raw = extraer_texto_sucio(archivo)
                res = analizar_con_ia(raw)
                st.session_state['res_ia'] = res

    if 'res_ia' in st.session_state:
        st.info(st.session_state['res_ia'])
        
        # Extracción segura de datos
        res_texto = st.session_state['res_ia']
        c_sug, m_sug = "Desconocido", 0.0
        
        try:
            # Regex robusto para buscar montos después de la palabra 'Monto:'
            m_match = re.search(r"Monto:\s*([\d.]+)", res_texto)
            if m_match: m_sug = float(m_match.group(1))
            
            c_match = re.search(r"Comercio:\s*(.*)", res_texto)
            if c_match: c_sug = c_match.group(1).strip()
        except:
            pass

        st.subheader("📝 Confirmación")
        f_c = st.text_input("Comercio", c_sug)
        f_m = st.number_input("Monto ($)", value=m_sug)
        
        if st.button("💾 Guardar Gasto"):
            st.session_state.historial.append({
                "Comercio": f_c, 
                "Monto": f_m, 
                "Fecha": pd.Timestamp.now().strftime("%d/%m/%Y")
            })
            st.success("¡Registrado!")
            del st.session_state['res_ia']

with col2:
    st.header("📊 Tabla de Gastos")
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.table(df)
        st.metric("Total Acumulado", f"${df['Monto'].sum():.2f}")
    else:
        st.info("Sube un ticket para generar tu reporte diario.")