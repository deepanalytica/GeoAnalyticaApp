import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Parámetros de la simulación
NUM_SONDAJES = 6
PROFUNDIDAD_SONDAJE = 120  # Metros
LEY_MEDIA = {"Cu": 1.0, "Au": 0.1} # %Cu, g/t Au
DESVIACION_ESTANDAR = {"Cu": 0.3, "Au": 0.02} 

# Función para generar leyes aleatorias con tendencia
def generar_leyes(profundidad, ley_media, desviacion_estandar):
    """Genera leyes con una ligera tendencia a disminuir con la profundidad."""
    tendencia = np.random.normal(0, 0.001) * profundidad
    return np.maximum(0, np.random.normal(ley_media - tendencia, desviacion_estandar, profundidad))

# Generar datos de sondajes
datos_sondajes = []
for i in range(NUM_SONDAJES):
    for j in range(PROFUNDIDAD_SONDAJE):
        datos_sondajes.append({
            "Sondaje": f"S{i+1}",
            "Profundidad": j + 1,
            "X": i * 10,  # Separación arbitraria entre sondajes
            "Y": 0,
            "Cu (%)": generar_leyes(j + 1, LEY_MEDIA["Cu"], DESVIACION_ESTANDAR["Cu"])[j],
            "Au (g/t)": generar_leyes(j + 1, LEY_MEDIA["Au"], DESVIACION_ESTANDAR["Au"])[j]
        })

df = pd.DataFrame(datos_sondajes)

#--- Visualización 3D ---
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
        z=df_sondaje["Profundidad"],
        mode='lines+markers',
        name=sondaje,
        marker=dict(
            size=6,
            color=df_sondaje[ley_a_visualizar],
            colorscale="Viridis",  # Escala de colores
            colorbar=dict(title=ley_a_visualizar),
            cmin=df[ley_a_visualizar].min(),
            cmax=df[ley_a_visualizar].max(),
        ),
        line=dict(width=3)
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        zaxis_title="Profundidad (m)",
        aspectmode="data"  # Ajusta la relación de aspecto para una vista real
    ),
    width=800,
    height=600,
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(fig)
