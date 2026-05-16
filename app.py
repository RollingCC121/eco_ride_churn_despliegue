import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Churn", layout="centered")

# Carga de modelos
@st.cache_resource
def load_assets():
    modelo = joblib.load('modelo_churn.pkl')
    pipeline = joblib.load('pipeline_preproc.pkl')
    return modelo, pipeline

try:
    modelo_churn, pipeline_preproc = load_assets()
    
    st.title("🚀 Sistema de Predicción de Churn")
    st.markdown("Introduce los datos del cliente para evaluar el riesgo de abandono.")

    # Formulario de entrada
    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
            plan = st.selectbox("Plan", ["basico", "estandar", "premium", "elite"])
            uso = st.number_input("Uso Mensual (Km)", min_value=0.0, value=50.0)
            tickets = st.number_input("Soporte Tickets", min_value=0, max_value=20, value=1)
        
        with col2:
            antiguedad = st.number_input("Días de Antigüedad", min_value=0.0, value=365.0)
            region = st.selectbox("Región", ["Norte", "Centro", "Sur", "Este", "Oeste"])
            gasto = st.number_input("Gasto Promedio", min_value=0, value=1000)
        
        submit = st.form_submit_button("Realizar Predicción")

    if submit:
        # Crear DataFrame
        df_input = pd.DataFrame([[edad, plan, uso, tickets, antiguedad, region, gasto]], 
                                columns=['Edad', 'Plan', 'Uso_Mensual_Km', 'Soporte_Tickets', 'Dias_Antiguedad', 'Region', 'Gasto_Promedio'])
        
        # Procesamiento y Predicción
        preprocessed = pipeline_preproc.transform(df_input)
        pred = modelo_churn.predict(preprocessed)[0]
        prob = modelo_churn.predict_proba(preprocessed)[0]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ ALTA PROBABILIDAD DE CHURN ({prob[1]*100:.1f}%)")
        else:
            st.success(f"✅ CLIENTE FIDELIZADO ({prob[0]*100:.1f}% de probabilidad de quedarse)")

except Exception as e:
    st.error(f"Error al cargar la aplicación: {e}")
