import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import google.generativeai as genai

# --- CONFIGURACIÓN DE IA ---
# Tu llave activa
genai.configure(api_key="AIzaSyCocUJXAY1D2b0P_52Kc8BmBatMZvHrhjQ")

st.set_page_config(page_title="SmartReceipt AI", layout="wide", page_icon="🤖")

st.title("💸 SmartReceipt AI")
st.markdown("### Control de Gastos Personal (Fase 1)")

# --- MEMORIA ---
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

# --- CEREBRO IA (Versión Simplificada para Evitar Errores) ---
def analizar_con_ia(texto_sucio):
    try:
        # Usamos el modelo más estándar disponible actualmente
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Extrae de este ticket:
        1. Comercio
        2. Monto (solo numero)
        3. Categoria (Gasolina, Despensa, Juguetes)
        
        TEXTO: {texto_sucio}
        
        Formato:
        Comercio: [Nombre]
        Monto: [Numero]
        Categoria: [Tipo]
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error de conexión: {str(e)}"

# --- INTERFAZ ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Subir Ticket")
    archivo = st.file_uploader("Imagen del ticket", type=['jpg', 'jpeg', 'png'])
    
    if archivo:
        st.image(archivo, width=250)
        if st.button("🧠 Analizar Gasto"):
            with st.spinner("IA analizando..."):
                raw = extraer_texto_sucio(archivo)
                res = analizar_con_ia(raw)
                st.session_state['res_ia'] = res

    if 'res_ia' in st.session_state:
        st.info(st.session_state['res_ia'])
        
        # Lógica de extracción simple
        res_texto = st.session_state['res_ia']
        c_sug = "Pendiente"
        m_sug = 0.0
        
        try:
            # Buscamos el monto en el texto que regresó la IA
            montos_encontrados = re.findall(r"Monto: ([\d.]+)", res_texto)
            if montos_encontrados:
                m_sug = float(montos_encontrados[0])
            
            comercio_encontrado = re.findall(r"Comercio: (.*)", res_texto)
            if comercio_encontrado:
                c_sug = comercio_encontrado[0].strip()
        except:
            pass

        f_c = st.text_input("Confirmar Comercio", c_sug)
        f_m = st.number_input("Confirmar Monto", value=m_sug)
        
        if st.button("💾 Guardar"):
            st.session_state.historial.append({"Comercio": f_c, "Monto": f_m})
            st.success("¡Guardado!")
            del st.session_state['res_ia']

with col2:
    st.header("📊 Mis Gastos")
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.table(df)
        st.metric("Total Acumulado", f"${df['Monto'].sum():.2f}")
    else:
        st.write("Sube un ticket para ver tu tabla.")