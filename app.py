import streamlit as st
import easyocr
import cv2
import numpy as np
import re
import pandas as pd
import google.generativeai as genai

# --- CONFIGURACIÓN DE IA ---
# Tu llave de Google AI Studio
genai.configure(api_key="AIzaSyCocUJXAY1D2b0P_52Kc8BmBatMZvHrhjQ")

# Configuración de página de Streamlit
st.set_page_config(page_title="SmartReceipt AI - MVP", layout="wide", page_icon="🤖")

st.title("💸 SmartReceipt AI")
st.markdown("### Control de Gastos Personal (Fase 1: Tlalnepantla / Satélite)")

# --- MEMORIA DE LA SESIÓN ---
# Esto permite que la tabla de gastos no se borre al subir un nuevo ticket
if 'historial' not in st.session_state:
    st.session_state['historial'] = []

# --- MOTOR OCR ---
@st.cache_resource
def load_reader():
    # Usamos gpu=False para evitar errores de memoria en Streamlit Cloud
    return easyocr.Reader(['es'], gpu=False)

reader = load_reader()

def extraer_texto_sucio(archivo):
    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # Procesamiento básico para mejorar lectura
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    results = reader.readtext(processed, detail=0)
    return " ".join(results)

# --- CEREBRO IA (Gemini) ---
def analizar_con_ia(texto_sucio):
    # Usamos la versión latest para evitar el error de 'NotFound'
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
    Eres un experto en contabilidad mexicana y análisis de datos. 
    Analiza este texto de un ticket extraído por OCR (puede tener errores).
    
    TEXTO:
    {texto_sucio}
    
    INSTRUCCIONES:
    1. Identifica el nombre comercial del establecimiento.
    2. Encuentra el MONTO TOTAL (el número más alto que parezca el pago final).
    3. Clasifícalo en: Despensa, Gasolina, Juguetes, Restaurante u Otros.
    4. Identifica la ubicación (ej. Tlalnepantla, Naucalpan, Satélite).

    RESPONDE EXACTAMENTE EN ESTE FORMATO:
    Comercio: [Nombre]
    Monto: [Número sin signos]
    Categoria: [Categoría]
    Ubicacion: [Ubicación]
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- INTERFAZ WEB ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 1. Cargar Ticket")
    archivo = st.file_uploader("Sube la foto de tu ticket", type=['jpg', 'jpeg', 'png'])
    
    if archivo:
        st.image(archivo, caption="Ticket cargado", width=300)
        
        if st.button("🧠 Analizar con IA"):
            with st.spinner("La IA está procesando el ticket..."):
                try:
                    # 1. Extraer texto con EasyOCR
                    texto_raw = extraer_texto_sucio(archivo)
                    
                    # 2. Interpretar con Gemini
                    resultado_ia = analizar_con_ia(texto_raw)
                    
                    # Guardamos el resultado en la sesión para que no desaparezca
                    st.session_state['res_ia'] = resultado_ia
                except Exception as e:
                    st.error(f"Hubo un error: {e}")

    # Mostrar resultados para validación
    if 'res_ia' in st.session_state:
        st.divider()
        st.subheader("📝 Validar Datos")
        res = st.session_state['res_ia']
        st.text(res) # Muestra el texto tal cual lo dio la IA
        
        # Intentamos extraer los datos para los campos de texto
        try:
            c_sug = re.search(r"Comercio: (.*)", res).group(1)
            m_sug = float(re.search(r"Monto: ([\d.]+)", res).group(1))
        except:
            c_sug, m_sug = "Desconocido", 0.0

        # Campos editables (Fase 1: Control Manual)
        final_c = st.text_input("Comercio Confirmado", c_sug)
        final_m = st.number_input("Monto Confirmado ($)", value=m_sug)
        
        if st.button("💾 Guardar en mi historial"):
            st.session_state.historial.append({
                "Comercio": final_c, 
                "Monto": final_m,
                "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("¡Gasto guardado!")
            del st.session_state['res_ia'] # Limpiamos para el siguiente

with col2:
    st.header("📊 Mis Gastos del Día")
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.table(df)
        
        total = df['Monto'].sum()
        st.metric("GASTO TOTAL ACUMULADO", f"${total:.2f}")
        
        # Descarga de datos
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar Reporte CSV", csv, "mis_gastos.csv", "text/csv")
        
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.historial = []
            st.rerun()
    else:
        st.info("Sube un ticket para empezar tu control de gastos.")