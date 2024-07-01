import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import griddata

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Sondajes Mineros",
    page_icon="⛏️",
    layout="wide",
)

# Función para cargar datos con manejo de errores
@st.cache_data
def cargar_datos(archivo_cargado):
    """
    Carga los datos del archivo Excel o CSV. 
    Asegura que las columnas numéricas se carguen correctamente
    y maneja posibles errores durante la carga.
    """
    if archivo_cargado is not None:
        try:
            if archivo_cargado.type == "text/csv":
                df = pd.read_csv(archivo_cargado)
            elif archivo_cargado.type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]:
                df = pd.read_excel(archivo_cargado)
            else:
                st.error(
                    "Formato de archivo no soportado. Carga un .csv o .xlsx"
                )
                return None

            # Asegura que las columnas numéricas se carguen correctamente
            columnas_numericas = [
                "Este",
                "Norte",
                "Elevación",
                "Elemento",
            ]  
            df[columnas_numericas] = df[columnas_numericas].apply(
                pd.to_numeric, errors="coerce"
            )
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return None


# ------------------------------------------------------------------------
#                             SIDEBAR
# ------------------------------------------------------------------------
st.sidebar.title("Panel de Control")

archivo_cargado = st.sidebar.file_uploader(
    "Cargar datos de sondajes:", type=["csv", "xlsx"]
)

if archivo_cargado:
    df = cargar_datos(archivo_cargado)

    if df is not None:
        elemento_x = st.sidebar.selectbox(
            "Elemento X (Eje X):", df.columns
        )
        elemento_y = st.sidebar.selectbox(
            "Elemento Y (Eje Y):", df.columns
        )

        # Filtro para la visualización 3D
        valor_minimo = st.sidebar.slider(
            f"Filtrar valores mínimos de {elemento_x}:",
            float(df[elemento_x].min()),
            float(df[elemento_x].max()),
            float(df[elemento_x].min()),
        )
        df_filtrado = df[df[elemento_x] >= valor_minimo]
    else:
        df_filtrado = pd.DataFrame() 

# ------------------------------------------------------------------------
#                             CONTENIDO PRINCIPAL
# ------------------------------------------------------------------------
st.title("Visualización de Datos de Sondajes ⛏️")

# Diseño de tres columnas
col1, col2, col3 = st.columns([1, 0.8, 1])

# Variable para almacenar los índices de los puntos seleccionados
selected_indices = []

# ------------------------------------------------------------------------
#                             COLUMNA 1: GRÁFICO 3D
# ------------------------------------------------------------------------
with col1:
    st.subheader("Visualización 3D de Sondajes")
    if not df_filtrado.empty:
        # Diccionario de colores para la litología
        litologia_colores = {
            "Litologia1": "red",
            "Litologia2": "blue",
            "Litologia3": "green",
            # Agrega más litologías y colores según sea necesario
        }

        # Obtener colores para los puntos
        colores_puntos = [
            litologia_colores.get(lito, "grey")
            for lito in df_filtrado["Litologia"]
        ]
        simbolos_puntos = [
            "circle" if i not in selected_indices else "diamond"
            for i in range(len(df_filtrado))
        ]

        # Crear figura 3D con Plotly
        fig_3d = go.Figure()

        # Añadir traza para todos los puntos
        fig_3d.add_trace(
            go.Scatter3d(
                x=df_filtrado["Este"],
                y=df_filtrado["Norte"],
                z=df_filtrado["Elevación"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=colores_puntos,
                    symbol=simbolos_puntos,  # color=df_filtrado['Elemento']
                ),
                text=[
                    f"Profundidad: {row['Elevación']}<br>{elemento_x}: {row[elemento_x]}<br>Litología: {row['Litologia']}"
                    for index, row in df_filtrado.iterrows()
                ],
                hoverinfo="text",
                name="Sondajes",
            )
        )

        # Actualizar el diseño del gráfico
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Este (m)",
                yaxis_title="Norte (m)",
                zaxis_title="Elevación (m)",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False,  # Ocultar la leyenda por ahora
        )
        st.plotly_chart(fig_3d)
    else:
        st.info(
            "Carga un archivo y ajusta el filtro para visualizar los sondajes en 3D."
        )

# ------------------------------------------------------------------------
#                             COLUMNA 2: GRÁFICO DE DISPERSIÓN
# ------------------------------------------------------------------------
with col2:
    st.subheader("Gráfico de Dispersión")
    if not df_filtrado.empty:
        fig_scatter = px.scatter(
            df_filtrado,
            x=elemento_x,
            y=elemento_y,
            trendline="ols",
            color=elemento_x,
            color_continuous_scale="Viridis",
        )

        # Habilitar selección en el gráfico 2D
        fig_scatter.update_traces(
            selectedpoints=df_filtrado.index,
            selector=dict(type="lasso", mode="select"),
            unselected=dict(marker=dict(color="lightgrey")),
        )

        # Obtener los puntos seleccionados
        selected_points = fig_scatter.data[0].selectedpoints
        if selected_points is not None:
            selected_indices = selected_points

        st.plotly_chart(fig_scatter)
    else:
        st.info("Carga un archivo para visualizar el gráfico de dispersión.")

# ------------------------------------------------------------------------
#                             COLUMNA 3: OTROS GRÁFICOS
# ------------------------------------------------------------------------
with col3:
    if not df_filtrado.empty:
        # ----------------------------------------------------------------
        #                  GRÁFICO DE TESELACIÓN WAVELET
        # ----------------------------------------------------------------
        st.subheader("Teselación Wavelet")

        # Seleccionar columna para la teselación wavelet
        columna_wavelet = st.selectbox(
            "Seleccionar columna para Wavelet:", df.columns
        )

        # Aplicar transformada wavelet (ejemplo básico)
        signal = df_filtrado[columna_wavelet].values
        coeffs = pywt.dwt(signal, "db4")
        cA, cD = coeffs

        # Crear gráfico de la teselación wavelet
        fig_wavelet = make_subplots(rows=2, cols=1)
        fig_wavelet.add_trace(
            go.Scatter(y=cA, mode="lines", name="Aproximación"),
            row=1,
            col=1,
        )
        fig_wavelet.add_trace(
            go.Scatter(y=cD, mode="lines", name="Detalle"), row=2, col=1
        )
        fig_wavelet.update_layout(height=400)
        st.plotly_chart(fig_wavelet)

        # ----------------------------------------------------------------
        #                  GRÁFICOS INTERACTIVOS Y DINÁMICOS
        # ----------------------------------------------------------------
        st.subheader("Gráficos Interactivos")

        # ----------------------------------------------------------------
        #                             HISTOGRAMA
        # ----------------------------------------------------------------
        st.subheader("Histograma de " + elemento_x)
        fig_hist = px.histogram(
            df_filtrado.iloc[selected_indices], x=elemento_x
        )
        st.plotly_chart(fig_hist)

        # ----------------------------------------------------------------
        #                           DIAGRAMA DE CAJA
        # ----------------------------------------------------------------
        st.subheader("Diagrama de Caja de " + elemento_x)
        fig_box = px.box(
            df_filtrado.iloc[selected_indices], y=elemento_x
        )
        st.plotly_chart(fig_box)

        # ----------------------------------------------------------------
        #                   CÁLCULO DE LA SILUETA
        # ----------------------------------------------------------------
        if len(selected_indices) >= 4:
            st.subheader("Silueta del Cuerpo Mineralizado")

            # Obtener las coordenadas de los puntos seleccionados
            puntos_seleccionados = df_filtrado[
                ["Este", "Norte", "Elevación"]
            ].iloc[selected_indices]

            # Calcular la envoltura convexa en 3D
            hull = ConvexHull(puntos_seleccionados)

            # Visualizar la silueta (envoltura convexa) en 3D
            fig_silueta = go.Figure(
                data=[
                    go.Mesh3d(
                        x=puntos_seleccionados["Este"],
                        y=puntos_seleccionados["Norte"],
                        z=puntos_seleccionados["Elevación"],
                        i=hull.simplices,
                        opacity=0.5,
                        color="gold",
                        name="Silueta",
                    ),
                    go.Scatter3d(
                        x=puntos_seleccionados["Este"],
                        y=puntos_seleccionados["Norte"],
                        z=puntos_seleccionados["Elevación"],
                        mode="markers",
                        marker=dict(size=4, color="blue"),
                        name="Puntos Seleccionados",
                    ),
                ]
            )
            fig_silueta.update_layout(
                scene=dict(
                    xaxis_title="Este (m)",
                    yaxis_title="Norte (m)",
                    zaxis_title="Elevación (m)",
                ),
                margin=dict(l=0, r=0, b=0, t=0),
            )
            st.plotly_chart(fig_silueta)
        else:
            st.info(
                "Selecciona al menos 4 puntos en el gráfico de dispersión para visualizar la silueta."
            )

        # ----------------------------------------------------------------
        #                       CÁLCULO DEL VOLUMEN
        # ----------------------------------------------------------------
        st.subheader("Cálculo del Volumen")

        if len(selected_indices) >= 4:
            # 1. Crear una malla regular de puntos en 3D
            resolucion = 1  # Ajusta la resolución según sea necesario
            x_min, x_max = (
                df_filtrado["Este"].min(),
                df_filtrado["Este"].max(),
            )
            y_min, y_max = (
                df_filtrado["Norte"].min(),
                df_filtrado["Norte"].max(),
            )
            z_min, z_max = (
                df_filtrado["Elevación"].min(),
                df_filtrado["Elevación"].max(),
            )
            x_grid, y_grid, z_grid = np.mgrid[
                x_min:x_max:resolucion,
                y_min:y_max:resolucion,
                z_min:z_max:resolucion,
            ]
            xyz_grid = np.vstack(
                (x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
            ).T

            # Seleccionar método de interpolación
            metodo_interpolacion = st.selectbox(
                "Selecciona el método de interpolación:",
                ["IDW", "Lineal", "Cúbica"],
            )

            # 2. Interpolar valores del elemento en la malla 3D
            if metodo_interpolacion == "IDW":
                modelo_idw = KNeighborsRegressor(
                    n_neighbors=5, weights="distance"
                )
                modelo_idw.fit(
                    df_filtrado[["Este", "Norte", "Elevación"]].iloc[
                        selected_indices
                    ],
                    df_filtrado[elemento_x].iloc[selected_indices],
                )
                valores_interpolados = modelo_idw.predict(xyz_grid)
            elif metodo_interpolacion == "Lineal":
                valores_interpolados = griddata(
                    df_filtrado[["Este", "Norte", "Elevación"]].iloc[
                        selected_indices
                    ],
                    df_filtrado[elemento_x].iloc[selected_indices],
                    xyz_grid,
                    method="linear",
                )
            else:  # Cúbica
                valores_interpolados = griddata(
                    df_filtrado[["Este", "Norte", "Elevación"]].iloc[
                        selected_indices
                    ],
                    df_filtrado[elemento_x].iloc[selected_indices],
                    xyz_grid,
                    method="cubic",
                )

            # 3. Calcular el volumen
            volumen_estimado = np.sum(valores_interpolados) * resolucion**3

            st.write(
                f"Volumen estimado: {volumen_estimado:.2f} unidades cúbicas"
            )
        else:
            st.info("Selecciona al menos 4 puntos para calcular el volumen.")

    else:
        st.info(
            "Carga un archivo para visualizar la teselación wavelet y otros gráficos."
        )
