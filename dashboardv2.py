import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pywt
from scipy.interpolate import griddata
from sklearn.cluster import KMeans

# --- Parámetros de la Simulación (Ajustables) ---
NUM_SONDAJES = 20
PROFUNDIDAD_SONDAJE = 200  # Metros
LEY_MEDIA = {"Cu": 0.7, "Au": 0.2, "Mo": 0.01}  # %Cu, g/t Au, % Mo
DESVIACION_ESTANDAR = {"Cu": 0.4, "Au": 0.1, "Mo": 0.005}
NPTS = 50  # Número de puntos para el grid en la visualización 3D

# --- Datos de los Sondajes (Ajustables) ---
datos_sondajes = {
    "Sondaje": [f"DH-{i+1}" for i in range(NUM_SONDAJES)],
    "Este (m)": np.random.uniform(0, 100, NUM_SONDAJES),
    "Norte (m)": np.random.uniform(0, 100, NUM_SONDAJES),
    "Elevación (m)": 1000 + np.random.uniform(-10, 10, NUM_SONDAJES),
    "Azimut (°)": np.random.uniform(0, 360, NUM_SONDAJES),
    "Inclinación (°)": np.random.uniform(-70, -30, NUM_SONDAJES),
    "Profundidad (m)": [PROFUNDIDAD_SONDAJE] * NUM_SONDAJES,
}
df_sondajes = pd.DataFrame(datos_sondajes)

# --- Funciones ---
def generar_leyes(profundidad, ley_media, desviacion_estandar, factor_zonificacion=1):
    """Genera leyes con tendencia y zonificación."""
    tendencia = np.random.normal(0, 0.005) * profundidad
    return np.maximum(
        0,
        np.random.normal(
            (ley_media - tendencia) * factor_zonificacion,
            desviacion_estandar,
            profundidad,
        ),
    )

def generar_datos_sondaje(sondaje_data):
    """Genera datos de puntos de muestra con leyes y alteración."""
    datos = []
    for j in range(sondaje_data["Profundidad (m)"]):
        x = sondaje_data["Este (m)"] + j * np.sin(np.deg2rad(sondaje_data["Inclinación (°)"])) * np.cos(
            np.deg2rad(sondaje_data["Azimut (°)"]))
        y = sondaje_data["Norte (m)"] + j * np.sin(np.deg2rad(sondaje_data["Inclinación (°)"])) * np.sin(
            np.deg2rad(sondaje_data["Azimut (°)"]))
        z = sondaje_data["Elevación (m)"] - j * np.cos(np.deg2rad(sondaje_data["Inclinación (°)"]))

        # Zonificación simple (distancia al centro)
        dist_centro = np.sqrt(x ** 2 + y ** 2)
        factor_cu = max(0.1, 1 - (dist_centro / 50))  # Mayor ley de Cu hacia el centro
        factor_au = max(0.1, (dist_centro / 70))  # Mayor ley de Au en la periferia

        # Simular alteraciones (probabilidad según la profundidad)
        prob_silice = 1 / (1 + np.exp(-(z - 970) / 5))  # Sílice en profundidad
        prob_potasica = 1 / (1 + np.exp(-(z - 985) / 3))  # Potásica más arriba
        alteracion = (
            "Sílice"
            if np.random.rand() < prob_silice
            else "Potásica"
            if np.random.rand() < prob_potasica
            else "Sin alteración"
        )

        datos.append(
            {
                "Sondaje": sondaje_data["Sondaje"],
                "Profundidad": j + 1,
                "X": x,
                "Y": y,
                "Z": z,
                "Cu (%)": generar_leyes(
                    j + 1, LEY_MEDIA["Cu"], DESVIACION_ESTANDAR["Cu"], factor_cu
                )[j],
                "Au (g/t)": generar_leyes(
                    j + 1, LEY_MEDIA["Au"], DESVIACION_ESTANDAR["Au"], factor_au
                )[j],
                "Mo (%)": generar_leyes(
                    j + 1, LEY_MEDIA["Mo"], DESVIACION_ESTANDAR["Mo"]
                )[j],
                "Alteración": alteracion,
            }
        )
    return datos

def calcular_wavelet_transform(df):
    """Calcula la transformada wavelet para la ley de Cu."""
    cu_ley = df["Cu (%)"].values
    coeffs = pywt.wavedec(cu_ley, 'haar', level=4)
    reconstructed = pywt.waverec(coeffs, 'haar')
    return np.arange(len(cu_ley)), cu_ley, reconstructed

def clustering_geoquimico(df):
    """Agrupa los datos en función de las características geoquímicas usando KMeans."""
    X = df[["Cu (%)", "Au (g/t)", "Mo (%)"]].values
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    df['Cluster'] = kmeans.labels_
    return df

# --- Generar datos (una sola vez) ---
@st.cache_data
def cargar_datos():
    """Genera y almacena en caché los datos de los sondajes."""
    datos_sondajes_3d = []
    for i in range(len(df_sondajes)):
        datos_sondajes_3d.extend(generar_datos_sondaje(df_sondajes.iloc[i]))
    df_sondajes_3d = pd.DataFrame(datos_sondajes_3d)
    df_sondajes_3d = clustering_geoquimico(df_sondajes_3d)
    return df_sondajes_3d

# --- Cargar datos usando la función cacheada ---
df_sondajes_3d = cargar_datos()

# --- Interfaz de usuario de Streamlit ---
st.set_page_config(
    page_title="Dashboard de Exploración de Pórfido Cu-Au-Mo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Dashboard de Exploración de Pórfido Cu-Au-Mo")

# --- Menú de Navegación en la Parte Superior ---
st.markdown(
    """
    <style>
    .navbar {
        background-color: #2E3A59;
        padding: 10px;
        color: white;
        display: flex;
        justify-content: center;
        gap: 15px;
        border-radius: 10px;
    }
    .navbar a {
        color: white;
        text-decoration: none;
        font-size: 18px;
    }
    .navbar a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="navbar">
        <a href="#sondajes">Inicio</a>
        <a href="#sondajes">Sondajes</a>
        <a href="#dispersio">Dispersión</a>
        <a href="#lineas_telaraña">Líneas y Telaraña</a>
        <a href="#wavelet">Wavelet</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Filtros ---
st.sidebar.header("Filtros")
profundidad_min = st.sidebar.slider("Profundidad Mínima (m)", min_value=0, max_value=PROFUNDIDAD_SONDAJE, value=0)
profundidad_max = st.sidebar.slider("Profundidad Máxima (m)", min_value=0, max_value=PROFUNDIDAD_SONDAJE, value=PROFUNDIDAD_SONDAJE)

df_filtrado = df_sondajes_3d[(df_sondajes_3d["Profundidad"] >= profundidad_min) & (
        df_sondajes_3d["Profundidad"] <= profundidad_max)]

# --- Diseño en 2x2 ---
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("Visualización de Sondajes 3D")
    ley_a_visualizar = st.selectbox("Seleccionar Ley Mineral:", ["Cu (%)", "Au (g/t)", "Mo (%)"])
    mostrar_sondajes = st.checkbox("Mostrar Sondajes", value=True)
    mostrar_volumen = st.checkbox("Mostrar Volumen de Ley", value=True)
    mostrar_alteracion = st.checkbox("Mostrar Alteraciones", value=True)

    fig_3d = go.Figure()

    if mostrar_sondajes:
        for i, sondaje in df_filtrado.groupby("Sondaje"):
            x = [sondaje["X"].values[0]]
            y = [sondaje["Y"].values[0]]
            z = [sondaje["Z"].values[0]]
            fig_3d.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    name=f"Sondaje {sondaje['Sondaje'].values[0]}",
                    marker=dict(size=8, color="red"),
                )
            )

    if mostrar_volumen:
        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(df_filtrado["X"].min(), df_filtrado["X"].max(), NPTS),
            np.linspace(df_filtrado["Y"].min(), df_filtrado["Y"].max(), NPTS),
            np.linspace(df_filtrado["Z"].min(), df_filtrado["Z"].max(), NPTS),
            indexing='ij'
        )
        grid_values = griddata(
            (df_filtrado["X"], df_filtrado["Y"], df_filtrado["Z"]),
            df_filtrado[ley_a_visualizar],
            (grid_x, grid_y, grid_z),
            method="linear",
        )
        if grid_values is not None:
            fig_3d.add_trace(
                go.Isosurface(
                    x=grid_x.flatten(),
                    y=grid_y.flatten(),
                    z=grid_z.flatten(),
                    value=grid_values.flatten(),
                    isomin=df_filtrado[ley_a_visualizar].min(),
                    isomax=df_filtrado[ley_a_visualizar].max(),
                    surface_count=5,
                    colorscale="Viridis",
                    showscale=False,
                    opacity=0.3,
                )
            )

    if mostrar_alteracion:
        for alteracion_tipo in ["Sílice", "Potásica"]:
            df_alteracion = df_filtrado[df_filtrado["Alteración"] == alteracion_tipo]
            fig_3d.add_trace(
                go.Scatter3d(
                    x=df_alteracion["X"],
                    y=df_alteracion["Y"],
                    z=df_alteracion["Z"],
                    mode="markers",
                    name=alteracion_tipo,
                    marker=dict(
                        size=6,
                        symbol="diamond-open",
                        color="orange" if alteracion_tipo == "Potásica" else "blue",
                    ),
                )
            )

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Este (m)",
            yaxis_title="Norte (m)",
            zaxis_title="Elevación (m)",
            aspectmode='data'
        ),
        legend=dict(x=0.85, y=0.9, bgcolor="rgba(255,255,255,0.5)"),
        width=500,
        height=500,
        margin=dict(r=20, l=10, b=10, t=10),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    st.header("Gráfico de Dispersión")
    ley_dispersar = st.selectbox("Seleccionar Ley para Dispersión:", ["Cu (%)", "Au (g/t)"])
    fig_dispersion = go.Figure()
    fig_dispersion.add_trace(
        go.Scatter(
            x=df_sondajes_3d[ley_dispersar],
            y=df_sondajes_3d["Au (g/t)"],
            mode="markers",
            marker=dict(size=8, color=df_sondajes_3d["Cluster"], colorscale="Viridis", showscale=True),
            text=df_sondajes_3d["Sondaje"],
        )
    )
    fig_dispersion.update_layout(
        xaxis_title=f"Ley de {ley_dispersar}",
        yaxis_title="Ley de Au (g/t)",
        title="Dispersión de Leyes",
        width=500,
        height=500,
        margin=dict(r=10, l=10, b=10, t=30),
    )
    st.plotly_chart(fig_dispersion, use_container_width=True)

# --- Fila 2 ---
col3, col4 = st.columns(2, gap="large")

with col3:
    st.header("Gráfico de Líneas y Telaraña")
    visualizacion_seleccionada = st.selectbox(
        "Seleccionar Visualización",
        ["Gráfico de Líneas", "Gráfico de Telaraña"]
    )

    if visualizacion_seleccionada == "Gráfico de Líneas":
        ley_lineas = st.selectbox("Seleccionar Ley para Gráfico de Líneas:", ["Cu (%)", "Au (g/t)", "Mo (%)"])
        x = np.arange(1, len(df_sondajes_3d["Sondaje"].unique()) + 1)
        y = df_sondajes_3d.groupby("Sondaje")[ley_lineas].mean().values

        fig_lineas = go.Figure()
        fig_lineas.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'Ley de {ley_lineas}'))
        fig_lineas.update_layout(
            title=f"Gráfico de Líneas para Ley de {ley_lineas}",
            xaxis_title="Sondaje",
            yaxis_title=f"Ley de {ley_lineas}",
            width=500,
            height=500,
            margin=dict(r=10, l=10, b=10, t=30),
        )
        st.plotly_chart(fig_lineas, use_container_width=True)
    
    elif visualizacion_seleccionada == "Gráfico de Telaraña":
        categorias = ['Cu (%)', 'Au (g/t)', 'Mo (%)']
        valores = [df_sondajes_3d[categoria].mean() for categoria in categorias]
        fig_telaraña = go.Figure()
        fig_telaraña.add_trace(go.Scatterpolar(
            r=valores,
            theta=categorias,
            fill='toself',
            name='Medias de Leyes'
        ))
        fig_telaraña.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(valores) + 0.5])
            ),
            title="Gráfico de Telaraña",
            width=500,
            height=500,
            margin=dict(r=10, l=10, b=10, t=30),
        )
        st.plotly_chart(fig_telaraña, use_container_width=True)

with col4:
    st.header("Tesselación Wavelet")
    ley_wavelet = st.selectbox("Seleccionar Ley para Tesselación Wavelet:", ["Cu (%)", "Au (g/t)", "Mo (%)"])
    x, ley_original, ley_reconstruida = calcular_wavelet_transform(df_sondajes_3d[df_sondajes_3d["Cu (%)"] > 0])

    fig_wavelet = go.Figure()
    fig_wavelet.add_trace(
        go.Scatter(x=x, y=ley_original, mode='lines', name=f'Ley de {ley_wavelet} Original')
    )
    fig_wavelet.add_trace(
        go.Scatter(x=x, y=ley_reconstruida, mode='lines', name=f'Ley de {ley_wavelet} Reconstruida')
    )
    fig_wavelet.update_layout(
        title=f"Tesselación Wavelet de Ley de {ley_wavelet}",
        xaxis_title="Índice",
        yaxis_title=f"Ley de {ley_wavelet}",
        width=500,
        height=500,
        margin=dict(r=10, l=10, b=10, t=30),
    )
    st.plotly_chart(fig_wavelet, use_container_width=True)

# Footer opcional para más información
st.markdown(
    """
    <hr>
    <footer style="text-align: center; padding: 10px; background-color: #2E3A59; color: white;">
        <p>Desarrollado por tu nombre o tu empresa. Todos los derechos reservados.</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
