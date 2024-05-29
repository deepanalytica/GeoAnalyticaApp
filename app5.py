import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import base64
import altair as alt  # Para gráficos interactivos
import pydeck as pdk  # Para visualizaciones geográficas 3D

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
    }
    .st-h2 {
        font-size: 2em;
        font-weight: bold;
        margin-top: 30px;
    }
    .st-h3 {
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Barra de Navegación
st.sidebar.title("Navegación")
opciones = st.sidebar.radio("Ir a", ["Inicio", "Cargar Datos", "Análisis", "Visualizaciones", "Generación de Informes", "Ayuda/Documentación"], index=0)

# Inicializar el estado de sesión para datos
if 'datos' not in st.session_state:
    st.session_state['datos'] = pd.DataFrame()

# Función para corregir tipos de datos
def corregir_tipos(datos):
    """Corrige los tipos de datos de las columnas de un DataFrame.

    Args:
        datos (pd.DataFrame): El DataFrame de entrada.

    Returns:
        pd.DataFrame: El DataFrame con tipos de datos corregidos.
    """
    datos_corregidos = datos.copy()
    for columna in datos_corregidos.columns:
        try:
            datos_corregidos[columna] = pd.to_numeric(datos_corregidos[columna], errors='coerce')
        except ValueError:
            continue
    return datos_corregidos

# Función para extraer la unidad de una columna
def obtener_unidad(nombre_columna):
    """Extrae la unidad de medida de una columna."""
    partes = nombre_columna.split('_')
    if len(partes) > 1:
        return partes[-1]
    else:
        return ""  # Si no se encuentra la unidad, retorna una cadena vacía

# Función para guardar un DataFrame a un archivo
def guardar_dataframe(datos, formato="csv"):
    """Guarda un DataFrame en un archivo."""
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
    """Muestra la pantalla de bienvenida."""
    st.title("Bienvenido a la Aplicación de Geoquímica Minera")
    st.write("""
    Esta aplicación le permite analizar y visualizar datos geoquímicos de manera avanzada y profesional.
    """)
    imagen = Image.open("logo_GeoAnaytica.png")  # Reemplace con la ruta a su imagen
    st.image(imagen)

# Función de Cargar Datos
def cargar_datos():
    """Permite cargar datos desde un archivo CSV o Excel."""
    st.title("Cargar Datos")
    archivo = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
    if archivo is not None:
        try:
            if archivo.name.endswith('.csv'):
                st.session_state['datos'] = pd.read_csv(archivo)
            else:
                # Leer archivo Excel y procesar encabezados
                st.session_state['datos'] = pd.read_excel(archivo, header=[2, 3], skiprows=4)
                st.session_state['datos'].columns = ['_'.join(col).strip() for col in st.session_state['datos'].columns.values]
                # Agregar columna de unidades
                st.session_state['datos']['Unidades'] = st.session_state['datos'].columns.str.split('_').str[-1]
            st.session_state['datos'] = corregir_tipos(st.session_state['datos'])
            st.write("Vista previa de los datos:", st.session_state['datos'].head())
            # Opciones para guardar el DataFrame
            st.markdown("Guardar datos:")
            guardar_dataframe(st.session_state['datos'], formato="csv")
            guardar_dataframe(st.session_state['datos'], formato="excel")

            # Mostrar la tabla interactiva con Altair
            st.markdown("## Tabla Interactiva")
            st.write(alt.Chart(st.session_state['datos']).mark_text().encode(
                alt.X('Sample:N', sort=None),
                alt.Y('Au:Q', sort=None),
                alt.Text('Au:Q'),
            ).properties(title="Datos de Oro (Au)"))

        except Exception as e:
            st.error(f"Error al cargar los datos: {e}")

# Función de Análisis
def analisis():
    """Realiza análisis estadísticos de los datos."""
    st.title("Análisis Estadísticos")
    
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    
    datos_numericos = datos.select_dtypes(include=[np.number])
    
    if datos_numericos.empty:
        st.warning("No hay suficientes columnas numéricas para el análisis.")
        return
    
    # Análisis descriptivo
    if st.checkbox("Mostrar análisis descriptivo"):
        st.write(datos_numericos.describe())
    
    # Análisis de correlación
    if st.checkbox("Mostrar matriz de correlación"):
        corr = datos_numericos.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Análisis PCA
    if st.checkbox("Realizar Análisis de Componentes Principales (PCA)"):
        # Opciones de transformación de datos
        tipo_transformacion = st.selectbox("Transformación de datos", ["Ninguna", "Normalización", "Estandarización"], index=0)
        
        # Selección de rotación
        tipo_rotacion = st.selectbox("Tipo de Rotación", ["Ninguna", "Varimax", "Quartimax"], index=0)
        
        # Realizar la transformación de datos si se seleccionó
        if tipo_transformacion == "Normalización":
            datos_numericos = (datos_numericos - datos_numericos.min()) / (datos_numericos.max() - datos_numericos.min())
        elif tipo_transformacion == "Estandarización":
            datos_numericos = (datos_numericos - datos_numericos.mean()) / datos_numericos.std()

        # Realizar PCA
        num_componentes = st.number_input("Número de componentes:", min_value=1, max_value=min(len(datos_numericos.columns), 10), value=2)
        pca = PCA(n_components=num_componentes)
        componentes = pca.fit_transform(datos_numericos)

        # Mostrar resultados del PCA
        st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
        st.write("Valores propios:", pca.singular_values_**2)  # Valores propios
        st.write("Cargas factoriales:", pca.components_)  # Cargas factoriales
        st.write("Scores:", componentes)  # Scores
        st.write("Tipo de Rotación:", tipo_rotacion)  # Tipo de Rotación

    # Clustering
    if st.checkbox("Realizar Clustering (KMeans)"):
        num_clusters = st.number_input("Número de clusters:", min_value=1, max_value=10, value=3)
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(datos_numericos)
        datos['Cluster'] = clusters
        fig, ax = plt.subplots()
        sns.scatterplot(x=componentes[:, 0], y=componentes[:, 1], hue=clusters, palette="viridis", ax=ax)
        st.pyplot(fig)

    # Regresión Lineal (Ejemplo)
    if st.checkbox("Realizar Regresión Lineal"):
        columnas_numericas = datos_numericos.columns.tolist()
        x_col = st.selectbox("Variable Independiente (X)", columnas_numericas)
        y_col = st.selectbox("Variable Dependiente (Y)", columnas_numericas)
        try:
            modelo = LinearRegression()
            modelo.fit(datos[[x_col]], datos[[y_col]])
            st.write(f"Pendiente: {modelo.coef_[0][0]}")
            st.write(f"Intersección: {modelo.intercept_[0]}")
            # Visualización de la regresión con Plotly
            fig = px.scatter(datos, x=x_col, y=y_col, trendline="ols")
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error en la Regresión Lineal: {e}")

# Función de Visualizaciones
def visualizaciones():
    """Genera diferentes tipos de visualizaciones de datos."""
    st.title("Visualizaciones")
    
    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return
    
    tipo_grafico = st.selectbox("Seleccione el tipo de gráfico", ["Dispersión", "Histograma", "Boxplot", "Mapa", "Gráfico Interactivo (Plotly)", "Gráfico Personalizado (Plotly)", "Gráfico de Barras", "Mapa de Calor", "Gráfico 3D (PyDeck)"])
    
    if tipo_grafico == "Dispersión":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if len(columnas_numericas) < 2:
            st.warning("Se requieren al menos dos columnas numéricas para el gráfico de dispersión.")
            return
        x = st.selectbox("Eje X", columnas_numericas)
        y = st.selectbox("Eje Y", columnas_numericas)
        fig, ax = plt.subplots()
        sns.scatterplot(data=datos, x=x, y=y, ax=ax)
        st.pyplot(fig)

    elif tipo_grafico == "Histograma":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if not columnas_numericas:
            st.warning("No hay columnas numéricas disponibles para el histograma.")
            return
        columna = st.selectbox("Seleccione la columna", columnas_numericas)
        bins = st.slider("Número de bins", 10, 100, 30)
        fig, ax = plt.subplots()
        sns.histplot(datos[columna], bins=bins, ax=ax)
        st.pyplot(fig)

    elif tipo_grafico == "Boxplot":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if not columnas_numericas:
            st.warning("No hay columnas numéricas disponibles para el boxplot.")
            return
        columna = st.selectbox("Seleccione la columna", columnas_numericas)
        fig, ax = plt.subplots()
        sns.boxplot(data=datos[columna], ax=ax)
        st.pyplot(fig)

    elif tipo_grafico == "Mapa":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if len(columnas_numericas) < 2:
            st.warning("Se requieren columnas de latitud y longitud para el mapa.")
            return
        lat = st.selectbox("Columna de latitud", columnas_numericas)
        lon = st.selectbox("Columna de longitud", columnas_numericas)
        if st.button("Mostrar Mapa"):
            mapa = folium.Map(location=[datos[lat].mean(), datos[lon].mean()], zoom_start=5)
            for i, row in datos.iterrows():
                folium.Marker([row[lat], row[lon]], popup=row.to_string()).add_to(mapa)
            st_data = st_folium(mapa, width=700, height=500)

    elif tipo_grafico == "Gráfico Interactivo (Plotly)":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if len(columnas_numericas) < 2:
            st.warning("Se requieren al menos dos columnas numéricas para el gráfico interactivo.")
            return
        x = st.selectbox("Eje X", columnas_numericas)
        y = st.selectbox("Eje Y", columnas_numericas)
        try:
            fig = px.scatter(datos, x=x, y=y, color="Cluster" if "Cluster" in datos.columns else None,
                            title="Gráfico Interactivo",
                            labels={x: obtener_unidad(x), y: obtener_unidad(y)})
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error al generar el gráfico interactivo: {e}")

    elif tipo_grafico == "Gráfico Personalizado (Plotly)":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if len(columnas_numericas) < 2:
            st.warning("Se requieren al menos dos columnas numéricas para el gráfico personalizado.")
            return
        x = st.selectbox("Eje X", columnas_numericas)
        y = st.selectbox("Eje Y", columnas_numericas)
        try:
            # Crear un gráfico de dispersión con Plotly
            fig = go.Figure(data=go.Scatter(x=datos[x], y=datos[y], mode='markers',
                                          marker=dict(size=10, color='blue')))
            # Personalizar el gráfico
            fig.update_layout(title="Gráfico Personalizado",
                              xaxis_title=f"{x} ({obtener_unidad(x)})",
                              yaxis_title=f"{y} ({obtener_unidad(y)})",
                              showlegend=False)
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error al generar el gráfico personalizado: {e}")

    elif tipo_grafico == "Gráfico de Barras":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if not columnas_numericas:
            st.warning("No hay columnas numéricas disponibles para el gráfico de barras.")
            return
        columna = st.selectbox("Seleccione la columna", columnas_numericas)
        try:
            fig = px.bar(datos, x=datos.index, y=columna, title="Gráfico de Barras",
                         labels={columna: obtener_unidad(columna)})
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error al generar el gráfico de barras: {e}")

    elif tipo_grafico == "Mapa de Calor":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if len(columnas_numericas) < 2:
            st.warning("Se requieren al menos dos columnas numéricas para el mapa de calor.")
            return
        x = st.selectbox("Eje X", columnas_numericas)
        y = st.selectbox("Eje Y", columnas_numericas)
        try:
            fig = px.imshow(datos[[x, y]].corr(), title="Mapa de Calor",
                           labels={"x": x, "y": y})
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error al generar el mapa de calor: {e}")

    elif tipo_grafico == "Gráfico 3D (PyDeck)":
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
        if len(columnas_numericas) < 3:
            st.warning("Se requieren al menos tres columnas numéricas para el gráfico 3D.")
            return
        x = st.selectbox("Eje X", columnas_numericas)
        y = st.selectbox("Eje Y", columnas_numericas)
        z = st.selectbox("Eje Z", columnas_numericas)
        try:
            # Crear un gráfico 3D con PyDeck
            view_state = pdk.ViewState(latitude=datos[y].mean(), longitude=datos[x].mean(), zoom=10, 
                                        pitch=40, bearing=0)
            layer = pdk.Layer("ScatterplotLayer", data=datos, get_position=[x, y, z], get_color=[200, 30, 0, 160], 
                               auto_highlight=True, pickable=True)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
        except Exception as e:
            st.error(f"Error al generar el gráfico 3D: {e}")

# Función de Generación de Informes
def generacion_informes():
    """Genera un informe PDF con los resultados del análisis."""
    st.title("Generación de Informes")

    datos = st.session_state['datos']
    if datos.empty:
        st.warning("Por favor, cargue los datos primero.")
        return

    secciones = st.multiselect("Seleccione las secciones para el informe", ["Análisis Descriptivo", "Matriz de Correlación", "PCA", "Clustering", "Visualizaciones"])
    if st.button("Generar Informe"):
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Informe de Análisis Geoquímico", ln=True, align="C")
        
        datos_numericos = datos.select_dtypes(include=[np.number])
        
        if "Análisis Descriptivo" in secciones:
            pdf.add_page()
            pdf.cell(200, 10, txt="Análisis Descriptivo", ln=True, align="C")
            desc = datos_numericos.describe().to_string()
            pdf.multi_cell(0, 10, desc)
        
        if "Matriz de Correlación" in secciones:
            pdf.add_page()
            pdf.cell(200, 10, txt="Matriz de Correlación", ln=True, align="C")
            corr = datos_numericos.corr().to_string()
            pdf.multi_cell(0, 10, corr)
        
        if "PCA" in secciones:
            pdf.add_page()
            pdf.cell(200, 10, txt="Análisis de Componentes Principales (PCA)", ln=True, align="C")
            pca = PCA(n_components=2)
            componentes = pca.fit_transform(datos_numericos)
            pdf.multi_cell(0, 10, f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
            pdf.multi_cell(0, 10, f"Valores propios:\n{pca.singular_values_**2}")
            pdf.multi_cell(0, 10, f"Cargas factoriales:\n{pca.components_}")
            pdf.multi_cell(0, 10, f"Scores:\n{componentes}")

        if "Clustering" in secciones:
            pdf.add_page()
            pdf.cell(200, 10, txt="Clustering (KMeans)", ln=True, align="C")
            kmeans = KMeans(n_clusters=3)
            clusters = kmeans.fit_predict(datos_numericos)
            datos['Cluster'] = clusters
            pdf.multi_cell(0, 10, f"Clusters asignados:\n{clusters}")

        if "Visualizaciones" in secciones:
            pdf.add_page()
            pdf.cell(200, 10, txt="Visualizaciones", ln=True, align="C")
            # Puedes agregar código para incluir las visualizaciones en el informe
            # Por ejemplo:
            # pdf.image("scatter_plot.png", x=10, y=10, w=190, h=100)

        pdf.output("informe_geoquimico.pdf")
        st.success("Informe generado exitosamente. Puede descargarlo [aquí](informe_geoquimico.pdf)")

# Función de Ayuda/Documentación
def ayuda():
    """Muestra información de ayuda y documentación."""
    st.title("Ayuda/Documentación")
    st.write("""
    ### Preguntas Frecuentes
    1. **¿Cómo cargo mis datos?**
       Suba su archivo CSV o Excel en la sección "Cargar Datos".

    2. **¿Qué tipos de análisis puedo realizar?**
       Puede realizar análisis descriptivo, análisis de correlación, PCA, clustering y regresión lineal.

    3. **¿Cómo genero un informe?**
       Seleccione las secciones en "Generación de Informes" y haga clic en "Generar Informe".
    """)

# Enrutamiento de las opciones del menú
if opciones == "Inicio":
    mostrar_inicio()
elif opciones == "Cargar Datos":
    cargar_datos()
elif opciones == "Análisis":
    analisis()
elif opciones == "Visualizaciones":
    visualizaciones()
elif opciones == "Generación de Informes":
    generacion_informes()
elif opciones == "Ayuda/Documentación":
    ayuda()
