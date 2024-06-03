import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from fpdf import FPDF
from PIL import Image
from io import BytesIO
import base64
from sklearn.impute import SimpleImputer
import streamlit.components.v1 as components
import geopandas as gpd  # Para análisis geoespacial
import folium  # Para visualización de mapas
import holoviews as hv
from holoviews.plotting.plotly.dash import to_dash
from holoviews.operation.datashader import datashade
from plotly.data import carshare
from plotly.colors import sequential
import statsmodels.formula.api as sm  # Importa statsmodels para la regresión
import panel as pn  # Importa la biblioteca Panel
import altair as alt  # Importa Altair para gráficos interactivos
import seaborn as sns  # Importa Seaborn para visualizaciones
from streamlit_option_menu import option_menu  # Importa el menú de opciones
import matplotlib.pyplot as plt  # Importa Matplotlib

# Configuración de la página
st.set_page_config(page_title="Geoquímica Minera", layout="wide", page_icon=":bar_chart:")

# Estilos CSS para una interfaz de usuario más atractiva
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
        padding: 20px;
    }
    .st-h1 {
        font-size: 3em;
        font-weight: bold;
        color: #333;
    }
    .st-h2 {
        font-size: 2em;
        font-weight: bold;
        margin-top: 30px;
        color: #333;
    }
    .st-h3 {
        font-size: 1.5em;
        font-weight: bold;
        color: #333;
    }
    .main-content {
        padding: 20px;
    }
    .st-expanderHeader {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 1.2em;
    }
    .st-expanderContent {
        padding: 10px;
    }
    .st-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 5px;
        cursor: pointer;
    }
    .st-button:hover {
        background-color: #45a049;
    }
    .st-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }
    .st-selectbox {
        margin-bottom: 10px;
    }
    .st-multiselect {
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Menú Lateral
st.sidebar.title("Menú")
opcion = option_menu(
    "Geoquímica Minera",
    ["Inicio 🏠", "Cargar Datos 📂", "Resumen de Datos 📊", "Análisis Exploratorio 🔍", "Análisis Estadísticos 📈",
     "Análisis de Componentes Principales (PCA) 🧭", "Análisis de Clustering 🧬", "Análisis de Correlaciones 🔗",
     "Machine Learning 🤖", "Predicciones 🔮", "Exportar Resultados 📤", "Explorador Interactivo 🔎"],
    icons=["house", "file-earmark", "bar-chart-fill", "search", "graph-up", "compass", "dna", "link-45deg", "robot", "hourglass-split", "file-earmark-arrow-down", "eye"],
    menu_icon="cast",
    default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
)

# Inicializar el estado de sesión para datos
if 'datos' not in st.session_state:
    st.session_state['datos'] = pd.DataFrame()

# Función para corregir tipos de datos
def corregir_tipos(datos):
    datos_corregidos = datos.copy()
    for columna in datos_corregidos.columns:
        try:
            datos_corregidos[columna] = pd.to_numeric(datos_corregidos[columna], errors='coerce')
        except ValueError:
            continue
    return datos_corregidos

# Función para extraer la unidad de una columna
def obtener_unidad(nombre_columna):
    partes = nombre_columna.split('_')
    if len(partes) > 1:
        return partes[-1]
    else:
        return ""

# Función para guardar un DataFrame a un archivo
def guardar_dataframe(datos, formato="csv"):
    if formato == "csv":
        csv = datos.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="datos.{formato}">Descargar como CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    elif formato == "excel":
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        datos.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
        processed_data = output.getvalue()
        b64 = base64.b64encode(processed_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="datos.{formato}">Descargar como Excel</a>'
        st.markdown(href, unsafe_allow_html=True)

# Función de Inicio
def mostrar_inicio():
    st.title("Bienvenido a la Aplicación de Geoquímica Minera")
    st.write("Esta aplicación le permite analizar y visualizar datos geoquímicos de manera avanzada y profesional.")

    # Mostrar KPI's en tarjetas
    datos = st.session_state['datos']
    if not datos.empty:
        st.subheader("KPIs")
        # Obtener columnas numéricas (elementos químicos)
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        # Crear una tarjeta para cada elemento químico
        for columna in columnas_numericas:
            if columna != "Sample_ID":
                with st.container():
                    st.metric(f"{columna}", datos[columna].mean(), help=f"Valor medio de {columna}")
                    
# Función de Cargar Datos
def cargar_datos():
    st.title("Cargar Datos")
    with st.container():
        st.subheader("Cargar desde el Equipo")
        archivo = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
        if archivo is not None:
            try:
                if archivo.name.endswith('.csv'):
                    st.session_state['datos'] = pd.read_csv(archivo)
                else:
                    # Lee el archivo Excel con las dos primeras filas como encabezados
                    st.session_state['datos'] = pd.read_excel(archivo, header=[2, 3], skiprows=4) 
                    
                    # Verifica que las columnas tengan nombres únicos:
                    if st.session_state['datos'].columns.nlevels > 1:
                        st.session_state['datos'].columns = ['_'.join(col).strip() for col in st.session_state['datos'].columns.values]
                    
                    # Agrega la columna 'Unidades' si es necesario
                    if 'Unidades' not in st.session_state['datos'].columns:
                        st.session_state['datos']['Unidades'] = st.session_state['datos'].columns.str.split('_').str[-1]

                st.session_state['datos'] = corregir_tipos(st.session_state['datos'])

                # Obtener la columna de muestras
                columna_muestras = st.session_state['datos'].columns[0]

                # Asignar ID a las muestras automáticamente
                st.session_state['datos'].index = st.session_state['datos'][columna_muestras] 
                st.session_state['datos'] = st.session_state['datos'].set_index('SAMPLE')

                st.write("Vista previa de los datos:", st.session_state['datos'].head())
                guardar_dataframe(st.session_state['datos'], formato="csv")
                guardar_dataframe(st.session_state['datos'], formato="excel")
            except Exception as e:
                st.error(f"Error al cargar los datos: {e}")

# Función de Resumen de Datos
def resumen_datos():
    st.title("Resumen de Datos")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return

    # Asignar ID a las muestras
    datos['Sample_ID'] = range(1, len(datos) + 1)

    # Procesar valores bajo el límite de detección
    for columna in datos.columns:
        if isinstance(columna, str) and columna.startswith("Sample"):
            continue
        if datos[columna].dtype == np.number:
            datos[columna] = datos[columna].replace("<", "", regex=True)
            datos[columna] = pd.to_numeric(datos[columna], errors='coerce')
            datos[columna] = datos[columna].fillna(0)

    st.write("Vista previa de los datos:", datos.head())
    st.write("Resumen estadístico:", datos.describe())

# Función de Análisis Exploratorio
def analisis_exploratorio():
    st.title("Análisis Exploratorio")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return

    # Obtener las columnas numéricas
    columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()

    # Menú para seleccionar la columna a analizar
    columna_seleccionada = st.selectbox("Selecciona una columna para el análisis exploratorio:", columnas_numericas)
    
    # Analizar la columna seleccionada
    with st.expander("Histograma"):
        fig = px.histogram(datos, x=columna_seleccionada, marginal="box", title=f"Histograma de {columna_seleccionada}")
        st.plotly_chart(fig)

    with st.expander("Diagrama de Cajas y Bigotes"):
        fig = px.box(datos, x=columna_seleccionada, title=f"Diagrama de Cajas y Bigotes de {columna_seleccionada}")
        st.plotly_chart(fig)

    with st.expander("Diagrama de Dispersión"):
        columnas_seleccionadas = st.multiselect("Selecciona una segunda columna para el diagrama de dispersión", columnas_numericas)
        if columnas_seleccionadas:
            fig = px.scatter(data_frame=datos, x=columna_seleccionada, y=columnas_seleccionadas[0], title=f"Dispersión de {columna_seleccionada} vs {columnas_seleccionadas[0]}")
            st.plotly_chart(fig)

    with st.expander("Gráfico de Violin"):
        fig = px.violin(datos, y=columna_seleccionada, box=True, points="all", title=f"Gráfico de Violin de {columna_seleccionada}")
        st.plotly_chart(fig)

    # Visualizaciones interactivas con Seaborn
    st.subheader("Visualizaciones Interactivas con Seaborn")
    # Obtener columnas numéricas para Seaborn
    columnas_numericas_seaborn = [col for col in datos.columns if datos[col].dtype == np.number and col != "Sample_ID"]
    
    # Seleccionar variables para la visualización de Seaborn
    variable_x = st.selectbox("Selecciona la variable X", columnas_numericas_seaborn)
    variable_y = st.selectbox("Selecciona la variable Y", columnas_numericas_seaborn)
    
    # Mostrar visualizaciones de Seaborn en recuadros
    with st.expander("Análisis de Densidad de Kernel"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=datos, x=variable_x, y=variable_y, ax=ax, fill=True, cmap="viridis")
        st.pyplot(fig)
        
    with st.expander("Diagrama de Dispersión con Marcadores"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=datos, x=variable_x, y=variable_y, hue="Sample_ID", ax=ax, s=50, alpha=0.7)
        st.pyplot(fig)
        
    with st.expander("Diagrama de Caja y Bigotes para cada Muestra"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=datos, x="Sample_ID", y=variable_y, ax=ax, showmeans=True, color="skyblue")
        st.pyplot(fig)
        
    with st.expander("Histograma con Densidad de Kernel"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=datos, x=variable_x, kde=True, ax=ax, color="purple")
        st.pyplot(fig)
        
    with st.expander("Mapa de Calor de Correlación"):
        # Correlación entre todas las variables
        corr = datos.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Función de Análisis Estadísticos
def analisis_estadisticos():
    st.title("Análisis Estadísticos")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    datos_numericos = datos.select_dtypes(include=[np.number])
    if datos_numericos.empty:
        st.warning("No hay suficientes columnas numéricas para el análisis.")
        return
    with st.container():
        st.subheader("Análisis Descriptivo")
        if st.checkbox("Mostrar análisis descriptivo"):
            st.write(datos_numericos.describe())

    with st.container():
        st.subheader("Matriz de Correlación")
        if st.checkbox("Mostrar matriz de correlación"):
            corr = datos_numericos.corr()
            fig = px.imshow(corr, color_continuous_scale="RdBu", labels=dict(x="Variable", y="Variable", color="Correlación"))
            st.plotly_chart(fig)

    with st.container():
        st.subheader("Regresión Lineal")
        if st.checkbox("Realizar Regresión Lineal"):
            columnas_numericas = datos_numericos.columns.tolist()
            x_col = st.selectbox("Variable Independiente (X)", columnas_numericas)
            y_col = st.selectbox("Variable Dependiente (Y)", columnas_numericas)
            try:
                # Utiliza statsmodels para la regresión lineal
                modelo = sm.ols(f"{y_col} ~ {x_col}", data=datos)  
                resultados = modelo.fit()
                st.write("Resumen del modelo:")
                st.write(resultados.summary())
                st.write(f"Pendiente: {resultados.params[x_col]}")
                st.write(f"Intersección: {resultados.params['Intercept']}")
                fig = px.scatter(datos, x=x_col, y=y_col, trendline="ols", title=f"Regresión Lineal: {x_col} vs {y_col}")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error en la Regresión Lineal: {e}")

# Función de Análisis de Componentes Principales (PCA)
def analisis_pca():
    st.title("Análisis de Componentes Principales (PCA)")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    datos_numericos = datos.select_dtypes(include=[np.number])
    if datos_numericos.empty:
        st.warning("No hay suficientes columnas numéricas para el análisis.")
        return

    # Imputar valores faltantes (NaN) con la media
    imputer = SimpleImputer(strategy='mean')
    datos_numericos = imputer.fit_transform(datos_numericos)

    with st.container():
        st.subheader("Opciones de Transformación y Rotación")
        tipo_transformacion = st.selectbox("Transformación de datos", ["Ninguna", "Normalización", "Estandarización"], index=0)
        tipo_rotacion = st.selectbox("Tipo de Rotación", ["Ninguna", "Varimax", "Quartimax"], index=0)
        if tipo_transformacion == "Normalización":
            datos_numericos = MinMaxScaler().fit_transform(datos_numericos)
        elif tipo_transformacion == "Estandarización":
            datos_numericos = StandardScaler().fit_transform(datos_numericos)
        pca = PCA()
        pca.fit(datos_numericos)
        st.write("Varianza Explicada por Componente:", pca.explained_variance_ratio_)
        st.write("Valores Propios:", pca.explained_variance_)
        st.write("Cargas Factoriales:", pca.components_)
        scores = pca.transform(datos_numericos)
        st.write("Scores de los Componentes:", scores)
        fig = px.scatter_matrix(pd.DataFrame(scores), labels={col: f"PC{col+1}" for col in range(scores.shape[1])}, title="Matriz de Dispersión de Componentes Principales")
        st.plotly_chart(fig)

# Función de Análisis de Clustering
def analisis_clustering():
    st.title("Análisis de Clustering")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    datos_numericos = datos.select_dtypes(include=[np.number])
    if datos_numericos.empty:
        st.warning("No hay suficientes columnas numéricas para el análisis.")
        return
    with st.container():
        st.subheader("Configuración del Clustering")
        n_clusters = st.slider("Número de Clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(datos_numericos)
        st.write("Centroides:", kmeans.cluster_centers_)
        st.write("Etiquetas de los Clusters:", kmeans.labels_)
        fig = px.scatter(x=datos_numericos.iloc[:, 0], y=datos_numericos.iloc[:, 1], color=kmeans.labels_, title="Clustering K-Means")
        st.plotly_chart(fig)

# Function de Análisis de Correlaciones
def analisis_correlaciones():
    st.title("Análisis de Correlaciones")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    datos_numericos = datos.select_dtypes(include=[np.number])
    if datos_numericos.empty:
        st.warning("No hay suficientes columnas numéricas para el análisis.")
        return
    with st.container():
        st.subheader("Selección de Variables")
        columnas_analisis = [col for col in datos_numericos.columns if col != 'Sample_ID']  # Excluye Sample_ID
        seleccionadas = st.multiselect("Selecciona las Variables para Analizar Correlaciones", columnas_analisis)
        if len(seleccionadas) > 1:
            correlaciones = {}
            for col1 in seleccionadas:
                for col2 in seleccionadas:
                    if col1 != col2:
                        correlacion = datos_numericos[col1].corr(datos_numericos[col2])
                        correlaciones[f"{col1}_{col2}"] = correlacion  # Concatenar nombres de columnas
            st.write("Correlaciones Calculadas:")
            st.write(correlaciones)
            
            # Create corr_df correctly
            corr_df = pd.DataFrame(correlaciones, index=[0]).T.reset_index()
            corr_df.columns = ["Variable 1_Variable 2", "Correlación"]
            st.write(corr_df)
            
            # Pivot and create the plot
            fig = px.imshow(corr_df.pivot(index='Variable 1_Variable 2', 
                                     columns='Correlación', 
                                     values='Correlación'), 
                       color_continuous_scale="RdBu", 
                       labels=dict(x="Variable 1", y="Variable 2", color="Correlación"), 
                       title="Matriz de Correlación")
            st.plotly_chart(fig)
        else:
            st.warning("Seleccione al menos dos variables para analizar las correlaciones.")

# Función de Machine Learning
def machine_learning():
    st.title("Machine Learning")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    with st.container():
        st.subheader("Predicción con Random Forest")
        if st.checkbox("Realizar Predicción con Random Forest"):
            columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
            target_col = st.selectbox("Selecciona la Variable Objetivo (Y)", columnas_numericas)
            feature_cols = st.multiselect("Selecciona las Variables Predictoras (X)", columnas_numericas, default=columnas_numericas)
            if target_col and feature_cols:
                X = datos[feature_cols]
                y = datos[target_col]
                modelo = RandomForestRegressor()
                modelo.fit(X, y)
                predicciones = modelo.predict(X)
                st.write("MSE:", mean_squared_error(y, predicciones))
                st.write("R2:", r2_score(y, predicciones))
                fig = px.scatter(x=y, y=predicciones, labels={'x': 'Valores Reales', 'y': 'Predicciones'}, title="Predicciones de Random Forest")
                fig.add_trace(go.Scatter(x=[min(y), max(y)], y=[min(y), max(y)], mode='lines', name='Ideal'))
                st.plotly_chart(fig)
                st.write("Importancias de las características:")
                importancias = pd.Series(modelo.feature_importances_, index=feature_cols).sort_values(ascending=False)
                fig = px.bar(x=importancias.index, y=importancias.values, title="Importancia de las Características")
                st.plotly_chart(fig)

# Función de Predicciones
def predicciones():
    st.title("Predicciones")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    st.write("Análisis de predicciones en progreso.")
    # Implementar más análisis según los requerimientos específicos.

# Función de Exportar Resultados
def exportar_resultados():
    st.title("Exportar Resultados")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    with st.container():
        st.subheader("Opciones de Exportación")
        guardar_dataframe(datos, formato="csv")
        guardar_dataframe(datos, formato="excel")

# Function para crear el Explorador de Datos Interactivo
def explorador_datos():
    st.title("Explorador de Datos Interactivo")
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return

    # Define las columnas a mostrar en el explorador
    columnas_mostrar = st.multiselect("Selecciona las columnas a mostrar", datos.columns)

    # Crea el explorador interactivo de datos con Altair
    interactive_explorer = alt.Chart(datos[columnas_mostrar]).mark_circle().encode(
        alt.X(alt.repeat("column"), type="quantitative"),
        alt.Y(alt.repeat("row"), type="quantitative"),
        color="Sample_ID",
        tooltip=[alt.Tooltip(column, title=column) for column in columnas_mostrar]
    ).properties(
        width=300,
        height=300
    ).repeat(
        row=columnas_mostrar,
        column=columnas_mostrar
    )

    # Print the chart dictionary to debug if needed:
    # print(interactive_explorer.to_dict()) 
    
    st.altair_chart(interactive_explorer, use_container_width=True)

# Mostrar contenido según selección del menú
if __name__ == "__main__":
    if opcion == "Inicio 🏠":
        mostrar_inicio()
    elif opcion == "Cargar Datos 📂":
        cargar_datos()
    elif opcion == "Resumen de Datos 📊":
        resumen_datos()
    elif opcion == "Análisis Exploratorio 🔍":
        analisis_exploratorio()
    elif opcion == "Análisis Estadísticos 📈":
        analisis_estadisticos()
    elif opcion == "Análisis de Componentes Principales (PCA) 🧭":
        analisis_pca()
    elif opcion == "Análisis de Clustering 🧬":
        analisis_clustering()
    elif opcion == "Análisis de Correlaciones 🔗":
        analisis_correlaciones()
    elif opcion == "Machine Learning 🤖":
        machine_learning()
    elif opcion == "Predicciones 🔮":
        predicciones()
    elif opcion == "Exportar Resultados 📤":
        exportar_resultados()
    elif opcion == "Explorador Interactivo 🔎":
        explorador_datos()
