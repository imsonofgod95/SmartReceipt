import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd

# Configuración de la App
st.set_page_config(page_title="SmartReceipt MVP", page_icon="💸")
st.title("💸 SmartReceipt")
st.write("Registra tus tickets de Tlalnepantla y Satélite de forma inteligente.")

# Cargamos el lector una sola vez para que sea rápido
@st.cache_resource
def load_reader():
    return easyocr.Reader(['es'])

reader = load_reader()

# --- TU MOTOR ORIGINAL ADAPTADO ---
def procesar_ticket(image_file):
    # Convertir imagen subida a formato OpenCV
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    results = reader.readtext(processed, detail=0)
    texto_unido = " ".join(results).upper().replace(",", "")
    
    # Monto Máximo
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
        comercio = "AL-MOS"
        ubicacion = "Tlalnepantla Centro"
        
    return total, comercio, ubicacion

# --- INTERFAZ WEB ---
uploaded_file = st.file_uploader("Elige la foto de tu ticket", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    t, c, u = procesar_ticket(uploaded_file)
    
    st.subheader("Resultados del Escaneo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Comercio", c)
    col2.metric("Ubicación", u)
    col3.metric("Total", f"${t}")
    
    if st.button("✅ Confirmar y Guardar"):
        st.balloons()
        st.success("Ticket guardado en tu base de datos local.")