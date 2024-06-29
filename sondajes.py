import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# --- Parámetros de la Simulación (Ajustables) ---
NUM_SONDAJES = 10
PROFUNDIDAD_SONDAJE = 150  # Metros
LEY_MEDIA = {"Cu": 1.0, "Au": 0.1}  # %Cu, g/t Au
DESVIACION_ESTANDAR = {"Cu": 0.5, "Au": 0.05}

# --- Datos de los Sondajes (Ajustables) ---
# Los sondajes se distribuyen en un área más amplia
datos_sondajes = {
    "Sondaje": [f"DH-{i+1}" for i in range(NUM_SONDAJES)],
    "Este (m)": [0, 20, 40, 60, 80, 10, 30, 50, 70, 90],
    "Norte (m)": [0, 10, 20, 30, 40, 10, 20, 30, 40, 50],
    "Elevación (m)": [1000, 1002, 998, 1001, 999, 998, 1000, 1003, 1001, 999],
    "Azimut (°)": [0, 45, 90, 135, 180, 225, 270, 315, 0, 45],
    "Inclinación (°)": [-60, -60, -60, -45, -45, -45, -60, -60, -60, -45],
    "Profundidad (m)": [PROFUNDIDAD_SONDAJE] * NUM_SONDAJES,
}
df_sondajes = pd.DataFrame(datos_sondajes)

# --- Funciones ---
def generar_leyes(profundidad, ley_media, desviacion_estandar):
    """Genera leyes con una ligera tendencia y mayor variabilidad."""
    tendencia = np.random.normal(0, 0.005) * profundidad  # Mayor tendencia
    return np.maximum(
        0,
        np.random.normal(ley_media - tendencia, desviacion_estandar, profundidad),
    )


def generar_datos_sondaje(sondaje_data):
    """Genera datos de puntos de muestra a lo largo de un sondaje,
    incluyendo una alteración simulada.
    """
    datos = []
    for j in range(sondaje_data["Profundidad (m)"]):
        x = sondaje_data[
            "Este (m)"
        ] - j * np.sin(np.deg2rad(sondaje_data["Inclinación (°)"])) * np.cos(
            np.deg2rad(sondaje_data["Azimut (°)"])
        )
        y = sondaje_data[
            "Norte (m)"
        ] - j * np.sin(np.deg2rad(sondaje_data["Inclinación (°)"])) * np.sin(
            np.deg2rad(sondaje_data["Azimut (°)"])
        )
        z = sondaje_data["Elevación (m)"] - j * np.cos(
            np.deg2rad(sondaje_data["Inclinación (°)"])
        )
        # Simular alteración (1: Sílice, 0: Sin alteración)
        alteracion = 1 if np.random.rand() < 0.3 else 0  
        datos.append(
            {
                "Sondaje": sondaje_data["Sondaje"],
                "Profundidad": j + 1,
                "X": x,
                "Y": y,
                "Z": z,
                "Cu (%)": generar_leyes(
                    j + 1, LEY_MEDIA["Cu"], DESVIACION_ESTANDAR["Cu"]
                )[j],
                "Au (g/t)": generar_leyes(
                    j + 1, LEY_MEDIA["Au"], DESVIACION_ESTANDAR["Au"]
                )[j],
                "Alteración": alteracion,
            }
        )
    return datos

# --- Generar datos para todos los sondajes ---
datos_sondajes_3d = []
for i in range(len(df_sondajes)):
    datos_sondajes_3d.extend(generar_datos_sondaje(df_sondajes.iloc[i]))

df_sondajes_3d = pd.DataFrame(datos_sondajes_3d)

# --- Interfaz de usuario de Streamlit ---
st.sidebar.title("Parámetros de Visualización")
ley_a_visualizar = st.sidebar.selectbox(
    "Seleccionar Ley Mineral:", ["Cu (%)", "Au (g/t)"]
)

mostrar_volumen = st.sidebar.checkbox("Mostrar Volumen 3D", value=False)
mostrar_alteracion = st.sidebar.checkbox("Mostrar Alteración", value=False)

# --- Visualización 3D ---
st.title("Visualización 3D de Sondajes")

fig = go.Figure()

# --- Gráfico de dispersión 3D de los sondajes ---
for sondaje in df_sondajes_3d["Sondaje"].unique():
    df_sondaje = df_sondajes_3d[df_sondajes_3d["Sondaje"] == sondaje]
    fig.add_trace(
        go.Scatter3d(
            x=df_sondaje["X"],
            y=df_sondaje["Y"],
            z=df_sondaje["Z"],
            mode='lines+markers',
            name=sondaje,
            marker=dict(
                size=4,
                color=df_sondaje[ley_a_visualizar],
                colorscale="Viridis",
                colorbar=dict(title=ley_a_visualizar, x=1.1, len=0.8),
                cmin=df_sondajes_3d[ley_a_visualizar].min(),
                cmax=df_sondajes_3d[ley_a_visualizar].max(),
            ),
            line=dict(width=2),
        )
    )

# --- Generar volumen 3D ---
if mostrar_volumen:
    # Crear una malla 3D para la interpolación
    xi = np.linspace(df_sondajes_3d["X"].min(), df_sondajes_3d["X"].max(), 20)
    yi = np.linspace(df_sondajes_3d["Y"].min(), df_sondajes_3d["Y"].max(), 20)
    zi = np.linspace(df_sondajes_3d["Z"].min(), df_sondajes_3d["Z"].max(), 20)
    xi, yi, zi = np.meshgrid(xi, yi, zi)

    # Interpolar los valores de ley en la malla 3D
    valores_ley = griddata(
        (df_sondajes_3d["X"], df_sondajes_3d["Y"], df_sondajes_3d["Z"]),
        df_sondajes_3d[ley_a_visualizar],
        (xi, yi, zi),
        method='linear',
    )

    # Agregar el volumen 3D al gráfico
    fig.add_trace(
        go.Volume(
            x=xi.flatten(),
            y=yi.flatten(),
            z=zi.flatten(),
            value=valores_ley.flatten(),
            isomin=df_sondajes_3d[ley_a_visualizar].min(),
            isomax=df_sondajes_3d[ley_a_visualizar].max(),
            opacity=0.2,  # Ajustar la transparencia
            surface_count=15,  # Ajustar la resolución
            colorscale="Viridis",
            colorbar=dict(title=ley_a_visualizar, x=1.2, len=0.8),
        )
    )

# --- Mostrar alteración ---
if mostrar_alteracion:
    df_alteracion = df_sondajes_3d[df_sondajes_3d["Alteración"] == 1]
    fig.add_trace(
        go.Scatter3d(
            x=df_alteracion["X"],
            y=df_alteracion["Y"],
            z=df_alteracion["Z"],
            mode="markers",
            name="Alteración",
            marker=dict(
                size=3,
                color="yellow",  # Color para la alteración
                symbol="diamond-open",  
            ),
        )
    )

# --- Diseño del gráfico ---
fig.update_layout(
    scene=dict(
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        zaxis_title="Elevación (m)",
        aspectmode="data",
    ),
    width=800,
    height=600,
    margin=dict(l=65, r=100, b=65, t=90)
)

st.plotly_chart(fig)
