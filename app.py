import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd

st.set_page_config(page_title="SmartReceipt - Control de Gastos", layout="wide")
st.title("💸 Mi Control de Gastos Personal")

# --- MEMORIA DE LA APP (Fase 1) ---
if 'historial' not in st.session_state:
    st.session_state['historial'] = []

@st.cache_resource
def load_reader():
    # Cargamos sin GPU para que Streamlit no se trabe
    return easyocr.Reader(['es'], gpu=False)

reader = load_reader()

def motor_ocr(archivo):
    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mejoramos el contraste para tickets arrugados o de gas
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    results = reader.readtext(processed, detail=0)
    texto = " ".join(results).upper().replace(",", "")
    
    # Buscador de Monto Robusto
    montos = re.findall(r'(\d+\.\d{2})', texto)
    total = max([float(m) for m in montos]) if montos else 0.0
    
    # Identificador de Comercio (Tu lógica de Tlalnepantla/Satélite)
    tienda = "DESCONOCIDO"
    if "COSTCO" in texto: tienda = "COSTCO"
    elif "JUGUETRON" in texto: tienda = "JUGUETRON"
    elif "AL-MOS" in texto: tienda = "AL-MOS"
    elif "G500" in texto or "GAS" in texto: tienda = "GASOLINERA"
    
    return total, tienda

# --- INTERFAZ ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Sube tu Ticket")
    uploaded_file = st.file_uploader("Formatos: JPG, PNG", type=['jpg','png','jpeg'])
    
    if uploaded_file:
        monto_det, tienda_det = motor_ocr(uploaded_file)
        
        st.subheader("📝 Validar y Editar")
        # Aquí el cliente puede corregir si el OCR falló (Fase 1)
        c_final = st.text_input("Establecimiento", tienda_det)
        m_final = st.number_input("Monto Total ($)", value=monto_det)
        
        if st.button("💾 Guardar en mi lista"):
            st.session_state.historial.append({"Comercio": c_final, "Monto": m_final})
            st.success("¡Ticket agregado!")

with col2:
    st.header("📊 Mi Resumen del Día")
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.dataframe(df, use_container_width=True)
        st.metric("Total Acumulado", f"${df['Monto'].sum():.2f}")
        
        # Botón para descargar el Excel/CSV (Tu Fase 1 completa)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar mi Control (CSV)", csv, "mis_gastos.csv")
    else:
        st.write("Aún no hay gastos registrados.")