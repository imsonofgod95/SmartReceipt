import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
from PIL import Image

# Configuración inicial
st.set_page_config(page_title="Mi Control de Gastos", layout="wide")
st.title("🚀 Sistema de Control de Gastos")

# Inicializar el historial en la sesión para que no se borre al subir otro ticket
if 'historial' not in st.session_state:
    st.session_state['historial'] = []

# 1. MOTOR DE LECTURA (Tu lógica ganadora)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['es'])

reader = load_reader()

def motor_ocr(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    results = reader.readtext(processed, detail=0)
    texto_unido = " ".join(results).upper().replace(",", "")
    
    # Extracción de Monto
    montos = re.findall(r'(\d+\.\d{2})', texto_unido)
    montos_num = [float(m) for m in montos if float(m) < 10000]
    total = max(montos_num) if montos_num else 0.0
    
    # Comercio y Ubicación
    comercio = results[0] if results else "DESCONOCIDO"
    ubicacion = "EDOMEX (ZONA GENERAL)"
    if "COSTCO" in texto_unido:
        comercio = "COSTCO WHOLESALE"
        ubicacion = "Arboledas, Tlalnepantla"
    elif "JUGUETRON" in texto_unido:
        comercio = "JUGUETRON"
        ubicacion = "Plaza Satélite, Naucalpan"
    elif "AL-MOS" in texto_unido:
        comercio = "COMERCIALIZADORA AL-MOS"
        ubicacion = "Tlalnepantla Centro"
        
    return total, comercio, ubicacion

# --- 2. INTERFAZ DE USUARIO (UX) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Paso 1: Sube tu ticket")
    archivo = st.file_uploader("Cargar imagen", type=['jpg', 'jpeg', 'png'])
    
    if archivo:
        st.image(archivo, caption="Ticket a procesar", width=300)
        if st.button("Procesar Ticket"):
            monto, tienda, lugar = motor_ocr(archivo)
            st.session_state['temp_data'] = {'Tienda': tienda, 'Monto': monto, 'Lugar': lugar}

    # Sección de validación/edición (Equivalente al input M del código anterior)
    if 'temp_data' in st.session_state:
        st.divider()
        st.subheader("📝 Validar Datos")
        t_edit = st.text_input("Comercio", st.session_state['temp_data']['Tienda'])
        l_edit = st.text_input("Ubicación", st.session_state['temp_data']['Lugar'])
        m_edit = st.number_input("Monto Real", value=st.session_state['temp_data']['Monto'])
        
        if st.button("💾 Guardar en mi Control"):
            st.session_state.historial.append({'Comercio': t_edit, 'Lugar': l_edit, 'Monto': m_edit})
            st.success("¡Guardado!")
            del st.session_state['temp_data'] # Limpiar para el siguiente

with col2:
    st.header("📊 Resumen de Gastos")
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.table(df)
        
        total_acumulado = df['Monto'].sum()
        st.metric("TOTAL ACUMULADO", f"${total_acumulado:.2f}")
        
        # Botón para que el cliente se lleve su CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar mi reporte (CSV)", csv, "mis_gastos.csv", "text/csv")
    else:
        st.info("Aún no has agregado tickets.")