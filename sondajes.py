import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Parámetros de la simulación
NUM_SONDAJES = 6
PROFUNDIDAD_SONDAJE = 120  # Metros
LEY_MEDIA = {"Cu": 1.0, "Au": 0.1}  # %Cu, g/t Au
DESVIACION_ESTANDAR = {"Cu": 0.3, "Au": 0.02}

# Parámetros de ubicación e inclinación de los sondajes
COORDENADAS_X = [0, 10, 20, 30, 40, 50]  # Coordenadas X de los collarines
COORDENADAS_Y = [0, 0, 0, 0, 0, 0]  # Coordenadas Y de los collarines
INCLINACIONES = [-60, -60, -60, -45, -45, -45]  # Inclinaciones en grados (negativo = hacia abajo)
AZIMUT = [0, 0, 0, 0, 0, 0]

# Función para generar leyes aleatorias con tendencia
def generar_leyes(profundidad, ley_media, desviacion_estandar):
    tendencia = np.random.normal(0, 0.001) * profundidad
    return np.maximum(0, np.random.normal(ley_media - tendencia, desviacion_estandar, profundidad))

# Generar datos de sondajes
datos_sondajes = []
for i in range(NUM_SONDAJES):
    for j in range(PROFUNDIDAD_SONDAJE):
        # Calcular coordenadas 3D del punto de la muestra
        x = COORDENADAS_X[i] - j * np.sin(np.deg2rad(INCLINACIONES[i])) * np.cos(np.deg2rad(AZIMUT[i]))
        y = COORDENADAS_Y[i] - j * np.sin(np.deg2rad(INCLINACIONES[i])) * np.sin(np.deg2rad(AZIMUT[i]))
        z = -j * np.cos(np.deg2rad(INCLINACIONES[i])) # z negativo hacia abajo

        datos_sondajes.append({
            "Sondaje": f"DH-{i+1}", # Nomenclatura de sondaje de diamantina (Diamond Drill Hole)
            "Profundidad": j + 1,
            "X": x,
            "Y": y,
            "Z": z,
            "Cu (%)": generar_leyes(j + 1, LEY_MEDIA["Cu"], DESVIACION_ESTANDAR["Cu"])[j],
            "Au (g/t)": generar_leyes(j + 1, LEY_MEDIA["Au"], DESVIACION_ESTANDAR["Au"])[j]
        })

df = pd.DataFrame(datos_sondajes)

# --- Visualización 3D ---
st.title("Visualización 3D de Sondajes")

# Seleccionar ley mineral para visualizar
ley_a_visualizar = st.selectbox("Seleccionar Ley Mineral:", ["Cu (%)", "Au (g/t)"])

# Crear figura 3D
fig = go.Figure()

# Agregar trazas de sondajes
for sondaje in df["Sondaje"].unique():
    df_sondaje = df[df["Sondaje"] == sondaje]
    fig.add_trace(go.Scatter3d(
        x=df_sondaje["X"],
        y=df_sondaje["Y"],
        z=df_sondaje["Z"],
        mode='lines+markers',
        name=sondaje,
        marker=dict(
            size=4,
            color=df_sondaje[ley_a_visualizar],
            colorscale="Viridis",
            colorbar=dict(title=ley_a_visualizar),
            cmin=df[ley_a_visualizar].min(),
            cmax=df[ley_a_visualizar].max(),
        ),
        line=dict(width=2)
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        zaxis_title="Profundidad (m)",
        aspectmode="data" 
    ),
    width=800,
    height=600,
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(fig)
