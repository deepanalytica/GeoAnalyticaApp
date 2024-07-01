import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import plotly.express as px

# --- Parámetros de la Simulación (Ajustables) ---
NUM_SONDAJES = 20
PROFUNDIDAD_SONDAJE = 200  # Metros
LEY_MEDIA = {"Cu": 0.7, "Au": 0.2, "Mo": 0.01}  # %Cu, g/t Au, % Mo
DESVIACION_ESTANDAR = {"Cu": 0.4, "Au": 0.1, "Mo": 0.005}

# --- Datos de los Sondajes (Ajustables) ---
datos_sondajes = {
    "Sondaje": [f"DH-{i+1}" for i in range(NUM_SONDAJES)],
    "Este (m)": [
        0, 20, 40, 60, 80, 10, 30, 50, 70, 90, 
        0, 20, 40, 60, 80, 10, 30, 50, 70, 90,
    ],
    "Norte (m)": [
        0, 10, 20, 30, 40, 10, 20, 30, 40, 50, 
        -10, -10, -10, -10, -10, -20, -20, -20, -20, -20,
    ],
    "Elevación (m)": [1000 + i - 5 for i in range(NUM_SONDAJES)],
    "Azimut (°)": [0, 45, 90, 135, 180, 225, 270, 315, 0, 45, 0, 45, 90, 135, 180, 225, 270, 315, 0, 45],
    "Inclinación (°)": [-60, -55, -50, -45, -60, -55, -50, -45, -60, -55] * 2,
    "Profundidad (m)": [PROFUNDIDAD_SONDAJE] * NUM_SONDAJES,
}
df_sondajes = pd.DataFrame(datos_sondajes)

# --- Funciones ---
def generar_leyes(profundidad, ley_media, desviacion_estandar, factor_zonificacion=1):
    """Genera leyes con tendencia, mayor variabilidad y zonificación."""
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
        x = sondaje_data["Este (m)"] - j * np.sin(np.deg2rad(sondaje_data["Inclinación (°)"])) * np.cos(np.deg2rad(sondaje_data["Azimut (°)"]))
        y = sondaje_data["Norte (m)"] - j * np.sin(np.deg2rad(sondaje_data["Inclinación (°)"])) * np.sin(np.deg2rad(sondaje_data["Azimut (°)"]))
        z = sondaje_data["Elevación (m)"] - j * np.cos(np.deg2rad(sondaje_data["Inclinación (°)"]))

        # Zonificación simple (distancia al centro)
        dist_centro = np.sqrt(x**2 + y**2)
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

# --- Generar datos (una sola vez) ---
@st.cache_data 
def cargar_datos():
    """Genera y almacena en caché los datos de los sondajes."""
    datos_sondajes_3d = []
    for i in range(len(df_sondajes)):
        datos_sondajes_3d.extend(generar_datos_sondaje(df_sondajes.iloc[i]))
    return pd.DataFrame(datos_sondajes_3d)

# --- Cargar datos usando la función cacheada ---
df_sondajes_3d = cargar_datos()

# --- Interfaz de usuario de Streamlit ---
st.sidebar.title("Visualización de Sondajes 3D")

# --- Opciones de visualización ---
st.sidebar.header("Opciones de Visualización")
ley_a_visualizar = st.sidebar.selectbox("Seleccionar Ley Mineral:", ["Cu (%)", "Au (g/t)", "Mo (%)"])
mostrar_volumen = st.sidebar.checkbox("Mostrar Volumen 3D", value=False)
mostrar_alteracion = st.sidebar.checkbox("Mostrar Alteración", value=True)

# --- Filtros ---
st.sidebar.header("Filtros")
profundidad_min = st.sidebar.slider("Profundidad Mínima (m)", min_value=0, max_value=PROFUNDIDAD_SONDAJE, value=0)
profundidad_max = st.sidebar.slider("Profundidad Máxima (m)", min_value=0, max_value=PROFUNDIDAD_SONDAJE, value=PROFUNDIDAD_SONDAJE)
df_filtrado = df_sondajes_3d[(df_sondajes_3d["Profundidad"] >= profundidad_min) & (df_sondajes_3d["Profundidad"] <= profundidad_max)]

# --- Visualización ---
st.title("Visualización 3D de Sondajes - Pórfido Cu-Au-Mo")

# --- Gráfico 3D ---
fig = go.Figure()

# Sondajes
for sondaje in df_filtrado["Sondaje"].unique():
    df_sondaje = df_filtrado[df_filtrado["Sondaje"] == sondaje]
    fig.add_trace(
        go.Scatter3d(
            x=df_sondaje["X"],
            y=df_sondaje["Y"],
            z=df_sondaje["Z"],
            mode="lines+markers",
            name=sondaje,
            marker=dict(
                size=4,
                color=df_sondaje[ley_a_visualizar],
                colorscale="Viridis",
                colorbar=dict(title=ley_a_visualizar, x=1.1, len=0.8),
                cmin=df_filtrado[ley_a_visualizar].min(),
                cmax=df_filtrado[ley_a_visualizar].max(),
            ),
            line=dict(width=2),
        )
    )

# Alteración
if mostrar_alteracion:
    for alteracion_tipo in ["Sílice", "Potásica"]:
        df_alteracion = df_filtrado[df_filtrado["Alteración"] == alteracion_tipo]
        fig.add_trace(
            go.Scatter3d(
                x=df_alteracion["X"],
                y=df_alteracion["Y"],
                z=df_alteracion["Z"],
                mode="markers",
                name=alteracion_tipo,
                marker=dict(
                    size=3,
                    symbol="diamond-open",
                    color="orange" if alteracion_tipo == "Potásica" else "blue",
                ),
            )
        )

# Teselación Wavelet
# Selección de puntos de interpolación
npts = 100
grid_x, grid_y = np.mgrid[
    df_filtrado["X"].min() : df_filtrado["X"].max() : complex(npts),
    df_filtrado["Y"].min() : df_filtrado["Y"].max() : complex(npts),
]
grid_z = griddata(
    (df_filtrado["X"], df_filtrado["Y"]),
    df_filtrado[ley_a_visualizar],
    (grid_x, grid_y),
    method="cubic",
)

fig.add_trace(
    go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale="Viridis",
        showscale=False,
        opacity=0.6,
    )
)

fig.update_layout(
    scene=dict(
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        zaxis_title="Elevación (m)",
    ),
    legend=dict(
        x=0.85, y=0.9, bgcolor="rgba(255,255,255,0.5)"
    ),
    width=800,
    height=800,
    margin=dict(r=20, l=10, b=10, t=10),
)

st.plotly_chart(fig)
