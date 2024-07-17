import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pydeck as pdk

st.set_page_config(layout="wide", page_title="GeoMiner - Mapa de Prospectividad")
st.title("GeoMiner: Exploración y Planificación Minera")

# Datos globales (inicialmente vacíos)
data = None
gdf = None
modelo = None

# Funciones para cada sección del menú

def cargar_datos():
    """Permite al usuario cargar datos desde un archivo CSV, GeoJSON o Shapefile."""
    global data, gdf
    st.header("Cargar Datos")
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona tu archivo (CSV, GeoJSON, Shapefile)",
        type=["csv", "geojson", "shp"],
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith((".geojson", ".shp")):
                gdf = gpd.read_file(uploaded_file)
                data = pd.DataFrame(gdf)
            else:
                data = pd.read_csv(uploaded_file)
                # Intentar convertir a GeoDataFrame si hay columnas de coordenadas
                try:
                    gdf = gpd.GeoDataFrame(
                        data, geometry=gpd.points_from_xy(data.Longitude, data.Latitude)
                    )
                except:
                    st.warning(
                        "No se encontraron columnas de coordenadas. Asegúrate de que tu archivo CSV tenga columnas nombradas como 'Longitude' y 'Latitude' si deseas visualizar los datos en un mapa."
                    )
            st.success("Datos cargados con éxito!")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error al cargar: {e}")


def analisis_exploratorio():
    """Muestra estadísticas descriptivas e histogramas de las variables numéricas."""
    st.header("Análisis Exploratorio")
    if data is not None:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(data.describe())

        st.subheader("Histogramas")
        columnas_numericas = data.select_dtypes(include=['number']).columns
        for col in columnas_numericas:
            fig = px.histogram(data, x=col, nbins=30)
            st.plotly_chart(fig)
    else:
        st.warning("Carga los datos primero.")


def modelado():
    """Permite al usuario aplicar clustering (KMeans) a los datos."""
    global data, modelo
    st.header("Modelado de Prospectividad")
    if data is not None:
        columnas_numericas = data.select_dtypes(include=['number']).columns.tolist()
        if columnas_numericas:
            st.subheader("Selección de Variables")
            columnas_seleccionadas = st.multiselect(
                "Elige las variables para el clustering", columnas_numericas
            )
            if columnas_seleccionadas:
                # Preprocesamiento
                st.subheader("Preprocesamiento")
                escalar = st.checkbox("Estandarizar datos (recomendado)")
                if escalar:
                    scaler = StandardScaler()
                    data_procesada = scaler.fit_transform(
                        data[columnas_seleccionadas]
                    )
                else:
                    data_procesada = data[columnas_seleccionadas]

                # Algoritmo de Clustering
                st.subheader("Configuración del Modelo")
                n_clusters = st.slider("Número de Clusters", 2, 10, 3)
                algoritmo = st.selectbox(
                    "Algoritmo de Clustering", ["KMeans"]
                )  # Agregar más opciones

                if algoritmo == "KMeans":
                    modelo = KMeans(n_clusters=n_clusters, random_state=42)
                    modelo.fit(data_procesada)

                # Evaluación
                st.subheader("Evaluación del Modelo")
                labels = modelo.labels_
                silhouette_avg = silhouette_score(data_procesada, labels)
                st.write(f"Puntuación de Silueta: {silhouette_avg:.2f}")

                # Asignar clusters al DataFrame original
                data['Cluster'] = modelo.labels_

                st.success("Modelo entrenado y clusters asignados!")
            else:
                st.warning("Selecciona al menos una variable para el clustering.")
        else:
            st.warning("No hay columnas numéricas en el conjunto de datos.")
    else:
        st.warning("Carga los datos y asegúrate de tener variables numéricas.")


def visualizacion():
    """Muestra un mapa interactivo con los resultados del clustering."""
    global gdf
    st.header("Visualización de Resultados")
    if gdf is not None and 'Cluster' in gdf.columns:
        st.subheader("Mapa de Clusters")
        # Definir una paleta de colores para los clusters
        colores = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], 
            [255, 255, 0], [0, 255, 255], [255, 0, 255],
            [128, 0, 0], [0, 128, 0], [0, 0, 128]  # Agrega más colores si necesitas más clusters
        ]

        # Capa de mapa base
        base_layer = pdk.Layer(
            "Tiles3DLayer",
            style=pdk.DeckStyle.DARK,
            elevation_decoder={"rScaler": 100, "isReversed": True},
        )

        # Capa de clusters (puntos)
        clusters_layer = pdk.Layer(
            "ScatterplotLayer",
            data=gdf,
            get_position=["Longitud", "Latitud"],
            get_color=[f"colores[{row['Cluster']}]" for row in gdf.to_dict('records')],  # Asignar colores basados en el cluster
            get_radius=500,
            pickable=True,
            auto_highlight=True,
        )

        # Configuración de la vista inicial
        view_state = pdk.ViewState(
            latitude=gdf.geometry.centroid.y.mean(),
            longitude=gdf.geometry.centroid.x.mean(),
            zoom=10,
            pitch=45,
        )

        # Crear el mapa con Pydeck
        st.pydeck_chart(
            pdk.Deck(
                layers=[base_layer, clusters_layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>Latitud:</b> {Latitud}<br/>"
                            "<b>Longitud:</b> {Longitud}<br/>"
                            "<b>Cluster:</b> {Cluster}",
                    "style": {"backgroundColor": "steelblue", "color": "white"}
                }
            )
        )
    else:
        st.warning("Entrena un modelo y asegúrate de tener datos geoespaciales válidos.")


# Menú principal en la barra lateral
menu = st.sidebar.selectbox(
    "Menú Principal", ("Cargar Datos", "Análisis Exploratorio", "Modelado", "Visualización")
)

# Ejecutar la función seleccionada en el menú
if menu == "Cargar Datos":
    cargar_datos()
elif menu == "Análisis Exploratorio":
    analisis_exploratorio()
elif menu == "Modelado":
    modelado()
elif menu == "Visualización":
    visualizacion()
