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
# Distribución más dispersa y realista de los sondajes
datos_sondajes = {
    "Sondaje": [f"DH-{i+1}" for i in range(NUM_SONDAJES)],
    "Este (m)": [
        0,
        20,
        40,
        60,
        80,
        10,
        30,
        50,
        70,
        90,
        0,
        20,
        40,
        60,
        80,
        10,
        30,
        50,
        70,
        90,
    ],
    "Norte (m)": [
        0,
        10,
        20,
        30,
        40,
        10,
        20,
        30,
        40,
        50,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -20,
        -20,
        -20,
        -20,
    ],
    "Elevación (m)": [1000 + i - 5 for i in range(NUM_SONDAJES)],
    "Azimut (°)": [
        0,
        45,
        90,
        135,
        180,
        225,
        270,
        315,
        0,
        45,
        0,
        45,
        90,
        135,
        180,
        225,
        270,
        315,
        0,
        45,
    ],
    "Inclinación (°)": [-60, -55, -50, -45, -60, -55, -50, -45, -60, -55] * 2,
    "Profundidad (m)": [PROFUNDIDAD_SONDAJE] * NUM_SONDAJES,
}
df_sondajes = pd.DataFrame(datos_sondajes)

# --- Funciones ---
def generar_leyes(
    profundidad, ley_media, desviacion_estandar, factor_zonificacion=1
):
    """Genera leyes con tendencia, mayor variabilidad y zonificación."""
    tendencia = np.random.normal(0, 0.005) * profundidad  # Mayor tendencia
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

        # Zonificación simple (distancia al centro)
        dist_centro = np.sqrt(x**2 + y**2)
        factor_cu = max(
            0.1, 1 - (dist_centro / 50)
        )  # Mayor ley de Cu hacia el centro
        factor_au = max(
            0.1, (dist_centro / 70)
        )  # Mayor ley de Au en la periferia

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
                    j + 1,
                    LEY_MEDIA["Au"],
                    DESVIACION_ESTANDAR["Au"],
                    factor_au,
                )[j],
                "Mo (%)": generar_leyes(
                    j + 1, LEY_MEDIA["Mo"], DESVIACION_ESTANDAR["Mo"]
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
    "Seleccionar Ley Mineral:", ["Cu (%)", "Au (g/t)", "Mo (%)"]
)

mostrar_volumen = st.sidebar.checkbox("Mostrar Volumen 3D", value=False)
mostrar_alteracion = st.sidebar.checkbox("Mostrar Alteración", value=True)

# --- Visualización 3D ---
st.title("Visualización 3D de Sondajes - Pórfido Cu-Au-Mo")

fig = go.Figure()

# --- Gráfico de dispersión 3D de los sondajes ---
for sondaje in df_sondajes_3d["Sondaje"].unique():
    df_sondaje = df_sondajes_3d[df_sondajes_3d["Sondaje"] == sondaje]
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
        method="linear",
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
    for alteracion_tipo in ["Sílice", "Potásica"]:
        df_alteracion = df_sondajes_3d[
            df_sondajes_3d["Alteración"] == alteracion_tipo
        ]
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
                    color="orange"
                    if alteracion_tipo == "Potásica"
                    else "blue",  
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
    margin=dict(l=65, r=100, b=65, t=90),
)

st.plotly_chart(fig)

st.plotly_chart(fig)
