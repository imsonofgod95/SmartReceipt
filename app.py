import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import google.generativeai as genai

# --- CONFIGURACIÓN DE IA ---
# Nota: Esta es la llave que obtuviste en AI Studio
genai.configure(api_key="AIzaSyCocUJXAY1D2b0P_52Kc8BmBatMZvHrhjQ")

st.set_page_config(page_title="SmartReceipt AI", layout="wide", page_icon="🤖")

# Títulos de la App
st.title("💸 SmartReceipt AI")
st.subheader("Control de Gastos con Inteligencia Artificial")

# --- MEMORIA DE LA SESIÓN ---
if 'historial' not in st.session_state:
    st.session_state['historial'] = []

# --- MOTOR OCR (Extrae texto sucio) ---
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

# --- CEREBRO IA (Limpia y entiende el ticket) ---
def analizar_con_ia(texto_sucio):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Eres un experto en finanzas. Analiza el siguiente texto extraído de un ticket de compra mexicano. 
    A veces el texto está mal escrito por el OCR o el ticket está arrugado. 
    Usa tu lógica para corregir nombres de tiendas y encontrar el monto real.
    
    TEXTO DEL TICKET:
    {texto_sucio}
    
    EXTRAE LOS SIGUIENTES DATOS EN ESTE FORMATO EXACTO:
    Comercio: [Nombre del establecimiento]
    Monto: [Solo el número del total]
    Categoria: [Despensa, Juguetes, Gasolina, Hogar o Comida]
    Ubicacion: [Ciudad o zona si aparece, si no pon Tlalnepantla/Satélite]
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- INTERFAZ DE USUARIO ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Paso 1: Sube tu Ticket")
    uploaded_file = st.file_uploader("Sube una imagen (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        st.info("🤖 La IA está analizando tu ticket... esto toma 2 segundos.")
        
        # 1. OCR saca el texto
        raw_text = extraer_texto_sucio(uploaded_file)
        
        # 2. Gemini lo entiende
        resultado_ia = analizar_con_ia(raw_text)
        
        # Mostramos lo que la IA entendió (para que el usuario lo valide)
        st.text_area("Análisis de la IA:", resultado_ia, height=150)
        
        # --- PARSEO DE DATOS (Para la tabla) ---
        # Extraemos los datos del texto de la IA usando regex simple
        try:
            comercio_ia = re.search(r"Comercio: (.*)", resultado_ia).group(1)
            monto_ia = float(re.search(r"Monto: ([\d.]+)", resultado_ia).group(1))
            cat_ia = re.search(r"Categoria: (.*)", resultado_ia).group(1)
            ubi_ia = re.search(r"Ubicacion: (.*)", resultado_ia).group(1)
        except:
            comercio_ia, monto_ia, cat_ia, ubi_ia = "Error", 0.0, "Otros", "N/A"

        st.subheader("📝 ¿Los datos son correctos?")
        c_final = st.text_input("Confirmar Comercio", comercio_ia)
        m_final = st.number_input("Confirmar Monto ($)", value=monto_ia)
        
        if st.button("💾 Guardar en mi Control"):
            st.session_state.historial.append({
                "Comercio": c_final, 
                "Monto": m_final, 
                "Categoría": cat_ia,
                "Ubicación": ubi_ia
            })
            st.success("¡Agregado exitosamente!")

with col2:
    st.header("📊 Mi Control Personal")
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.dataframe(df, use_container_width=True)
        
        st.metric("Gasto Total", f"${df['Monto'].sum():.2f}")
        
        # Exportar a CSV para el usuario
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar Reporte (CSV)", csv, "mis_gastos.csv", "text/csv")
    else:
        st.write("Tu tabla de gastos aparecerá aquí.")