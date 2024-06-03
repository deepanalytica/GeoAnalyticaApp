# geoquimica_minera/app.py

import streamlit as st
from pages import inicio, carga_datos, resumen_datos
from modules import exploratorio, estadistico, pca, clustering, correlaciones, machine_learning, predicciones, exportar_resultados, explorador_interactivo, analisis_avanzados

# ... otros imports

def main():
    st.set_page_config(page_title="Geoquímica Minera", layout="wide", page_icon=":bar_chart:")

    # Menú lateral
    st.sidebar.title("Menú")
    opcion = st.sidebar.radio(
        "Seleccione una opción:",
        [
            "Inicio 🏠",
            "Cargar Datos 📂",
            "Resumen de Datos 📊",
            "Análisis Exploratorio 🔍",
            "Análisis Estadísticos 📈",
            "Análisis de Componentes Principales (PCA) 🧭",
            "Análisis de Clustering 🧬",
            "Análisis de Correlaciones 🔗",
            "Machine Learning 🤖",
            "Predicciones 🔮",
            "Exportar Resultados 📤",
            "Explorador Interactivo 🔎",
            "Análisis Avanzados 🧪"
        ],
        horizontal=False
    )

    # Inicializar el estado de sesión para datos
    if 'datos' not in st.session_state:
        st.session_state['datos'] = pd.DataFrame()

    # Rutas de las páginas
    if opcion == "Inicio 🏠":
        inicio.mostrar_inicio()
    elif opcion == "Cargar Datos 📂":
        carga_datos.cargar_datos()
    elif opcion == "Resumen de Datos 📊":
        resumen_datos.resumen_datos()
    elif opcion == "Análisis Exploratorio 🔍":
        exploratorio.analisis_exploratorio()
    elif opcion == "Análisis Estadísticos 📈":
        estadistico.analisis_estadistico()
    elif opcion == "Análisis de Componentes Principales (PCA) 🧭":
        pca.analisis_pca()
    elif opcion == "Análisis de Clustering 🧬":
        clustering.analisis_clustering()
    elif opcion == "Análisis de Correlaciones 🔗":
        correlaciones.analisis_correlaciones()
    elif opcion == "Machine Learning 🤖":
        machine_learning.machine_learning()
    elif opcion == "Predicciones 🔮":
        predicciones.predicciones()
    elif opcion == "Exportar Resultados 📤":
        exportar_resultados.exportar_resultados()
    elif opcion == "Explorador Interactivo 🔎":
        explorador_interactivo.explorador_datos()
    elif opcion == "Análisis Avanzados 🧪":
        analisis_avanzados.analisis_avanzados()

if __name__ == "__main__":
    main()
