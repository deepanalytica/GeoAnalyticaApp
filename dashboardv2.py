import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# --- Parámetros de la Simulación (Ajustables) ---
NUM_SONDAJES = 20
PROFUNDIDAD_SONDAJE = 200  # Metros
LEY_MEDIA = {"Cu": 0.7, "Au": 0.2, "Mo": 0.01}  # %Cu, g/t Au, % Mo
DESVIACION_ESTANDAR = {"Cu": 0.4, "Au": 0.1, "Mo": 0.005}

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
mostrar_sondajes = st.sidebar.checkbox("Mostrar Sondajes", value=True)
mostrar_volumen = st.sidebar.checkbox("Mostrar Volumen 3D", value=True)
mostrar_alteracion = st.sidebar.checkbox("Mostrar Alteración", value=True)

# --- Filtros ---
st.sidebar.header("Filtros")
profundidad_min = st.sidebar.slider("Profundidad Mínima (m)", min_value=0, max_value=PROFUNDIDAD_SONDAJE, value=0)
profundidad_max = st.sidebar.slider("Profundidad Máxima (m)", min_value=0,
                                    max_value=PROFUNDIDAD_SONDAJE, value=PROFUNDIDAD_SONDAJE)
df_filtrado = df_sondajes_3d[(df_sondajes_3d["Profundidad"] >= profundidad_min) & (
        df_sondajes_3d["Profundidad"] <= profundidad_max)]

# --- Visualización ---
st.title("Visualización 3D de Sondajes - Pórfido Cu-Au-Mo")

# --- Gráfico 3D ---
fig = go.Figure()

# Sondajes (cilindros)
if mostrar_sondajes:
    for i in range(len(df_sondajes)):
        sondaje = df_sondajes.iloc[i]
        df_sondaje = df_filtrado[df_filtrado["Sondaje"] == sondaje["Sondaje"]]
        fig.add_trace(
            go.Scatter3d(
                x=df_sondaje["X"],
                y=df_sondaje["Y"],
                z=df_sondaje["Z"],
                mode="lines",
                line=dict(width=10, color="grey"),  # Cilindro con línea gruesa
                name=sondaje["Sondaje"],
                showlegend=False,
            )
        )

# Volumen 3D (Isosuperficie)
if mostrar_volumen:
    # Interpolación para la isosuperficie
    npts = 30  # Número de puntos de la grilla (reduce para menor resolución)
    grid_x, grid_y, grid_z = np.mgrid[
        df_filtrado["X"].min():df_filtrado["X"].max():npts * 1j,
        df_filtrado["Y"].min():df_filtrado["Y"].max():npts * 1j,
        df_filtrado["Z"].min():df_filtrado["Z"].max():npts * 1j,
    ]
    grid_values = griddata(
        (df_filtrado["X"], df_filtrado["Y"], df_filtrado["Z"]),
        df_filtrado[ley_a_visualizar],
        (grid_x, grid_y, grid_z),
        method="linear",
    )

    if grid_values is not None:  # Verificar si la interpolación es válida
        fig.add_trace(
            go.Isosurface(
                x=grid_x.flatten(),
                y=grid_y.flatten(),
                z=grid_z.flatten(),
                value=grid_values.flatten(),
                isomin=df_filtrado[ley_a_visualizar].min(),  # Valor mínimo de la isosuperficie
                isomax=df_filtrado[ley_a_visualizar].max(),  # Valor máximo de la isosuperficie
                surface_count=5,  # Controla la densidad de las superficies
                colorscale="Viridis",
                showscale=False,
                opacity=0.3,  # Transparencia de la isosuperficie
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

# Configuración del gráfico
fig.update_layout(
    scene=dict(
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        zaxis_title="Elevación (m)",
        aspectmode='data'
    ),
    legend=dict(x=0.85, y=0.9, bgcolor="rgba(255,255,255,0.5)"),
    width=800,
    height=800,
    margin=dict(r=20, l=10, b=10, t=10),
)

st.plotly_chart(fig)
