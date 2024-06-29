import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Datos de los Sondajes ---
datos_sondajes = {
    'Sondaje': ['DH-01', 'DH-02', 'DH-03', 'DH-04', 'DH-05', 'DH-06'],
    'Este (m)': [0, 10, 20, 30, 40, 50], 
    'Norte (m)': [0, 0, 0, 0, 0, 0],  
    'Elevación (m)': [1000, 1000, 1000, 1000, 1000, 1000], 
    'Azimut (°)': [0, 45, 90, 135, 180, 225],
    'Inclinación (°)': [-60, -60, -60, -45, -45, -45], 
    'Profundidad (m)': [120, 120, 120, 120, 120, 120]
}
df_sondajes = pd.DataFrame(datos_sondajes)

# --- Parámetros de la Simulación ---
LEY_MEDIA = {"Cu": 1.0, "Au": 0.1}  # %Cu, g/t Au
DESVIACION_ESTANDAR = {"Cu": 0.3, "Au": 0.02}

# --- Funciones ---
def generar_leyes(profundidad, ley_media, desviacion_estandar):
    """Genera leyes con una ligera tendencia a disminuir con la profundidad."""
    tendencia = np.random.normal(0, 0.001) * profundidad
    return np.maximum(0, np.random.normal(ley_media - tendencia, desviacion_estandar, profundidad))

def generar_datos_sondaje(sondaje_data):
    """Genera datos de puntos de muestra a lo largo de un sondaje."""
    datos = []
    for j in range(sondaje_data['Profundidad (m)']):
        x = sondaje_data['Este (m)'] - j * np.sin(np.deg2rad(sondaje_data['Inclinación (°)'])) * np.cos(np.deg2rad(sondaje_data['Azimut (°)']))
        y = sondaje_data['Norte (m)'] - j * np.sin(np.deg2rad(sondaje_data['Inclinación (°)'])) * np.sin(np.deg2rad(sondaje_data['Azimut (°)']))
        z = sondaje_data['Elevación (m)'] - j * np.cos(np.deg2rad(sondaje_data['Inclinación (°)'])) 

        datos.append({
            "Sondaje": sondaje_data['Sondaje'],
            "Profundidad": j + 1,
            "X": x,
            "Y": y,
            "Z": z,
            "Cu (%)": generar_leyes(j + 1, LEY_MEDIA["Cu"], DESVIACION_ESTANDAR["Cu"])[j],
            "Au (g/t)": generar_leyes(j + 1, LEY_MEDIA["Au"], DESVIACION_ESTANDAR["Au"])[j]
        })
    return datos

# --- Generar datos para todos los sondajes ---
datos_sondajes_3d = []
for i in range(len(df_sondajes)):
    datos_sondajes_3d.extend(generar_datos_sondaje(df_sondajes.iloc[i]))

df_sondajes_3d = pd.DataFrame(datos_sondajes_3d)

# --- Visualización 3D ---
st.title("Visualización 3D de Sondajes")

ley_a_visualizar = st.selectbox("Seleccionar Ley Mineral:", ["Cu (%)", "Au (g/t)"])

fig = go.Figure()

for sondaje in df_sondajes_3d["Sondaje"].unique():
    df_sondaje = df_sondajes_3d[df_sondajes_3d["Sondaje"] == sondaje]
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
            cmin=df_sondajes_3d[ley_a_visualizar].min(),
            cmax=df_sondajes_3d[ley_a_visualizar].max(),
        ),
        line=dict(width=2)
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        zaxis_title="Elevación (m)", 
        aspectmode="data" 
    ),
    width=800,
    height=600,
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(fig)
