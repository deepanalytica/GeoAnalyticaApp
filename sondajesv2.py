import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import plotly.express as px  # Para mapas de calor y gráficos de dispersión

# --- Parámetros de la Simulación (Ajustables) ---
NUM_SONDAJES = 20
PROFUNDIDAD_SONDAJE = 200  # Metros
LEY_MEDIA = {"Cu": 0.7, "Au": 0.2, "Mo": 0.01}  # %Cu, g/t Au, % Mo
DESVIACION_ESTANDAR = {"Cu": 0.4, "Au": 0.1, "Mo": 0.005}

# --- Datos de los Sondajes (Ajustables) ---
# (Mismos datos que en el código anterior)

# --- Funciones ---
# (Las funciones generar_leyes y generar_datos_sondaje son iguales 
# al código anterior, no es necesario modificarlas)

# --- Generar datos (una sola vez) ---
@st.cache_data
def cargar_datos():
    """Genera y almacena en caché los datos de los sondajes."""
    datos_sondajes_3d = []
    for i in range(len(df_sondajes)):
        datos_sondajes_3d.extend(
            generar_datos_sondaje(df_sondajes.iloc[i])
        )
    return pd.DataFrame(datos_sondajes_3d)


df_sondajes_3d = cargar_datos()

# --- Interfaz de usuario de Streamlit ---
st.sidebar.title("Visualización de Sondajes 3D")

# --- Opciones de visualización ---
st.sidebar.header("Opciones de Visualización")
ley_a_visualizar = st.sidebar.selectbox(
    "Seleccionar Ley Mineral:", ["Cu (%)", "Au (g/t)", "Mo (%)"]
)
mostrar_volumen = st.sidebar.checkbox("Mostrar Volumen 3D", value=False)
mostrar_alteracion = st.sidebar.checkbox("Mostrar Alteración", value=True)

# --- Filtros ---
st.sidebar.header("Filtros")
profundidad_min = st.sidebar.slider(
    "Profundidad Mínima (m)",
    min_value=0,
    max_value=PROFUNDIDAD_SONDAJE,
    value=0,
)
profundidad_max = st.sidebar.slider(
    "Profundidad Máxima (m)",
    min_value=0,
    max_value=PROFUNDIDAD_SONDAJE,
    value=PROFUNDIDAD_SONDAJE,
)
df_filtrado = df_sondajes_3d[
    (df_sondajes_3d["Profundidad"] >= profundidad_min)
    & (df_sondajes_3d["Profundidad"] <= profundidad_max)
]

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

# Volumen 3D
if mostrar_volumen:
    # (El código para generar el volumen 3D es el mismo que en la versión anterior,
    #  pero usando df_filtrado en lugar de df_sondajes_3d)

# Alteración
if mostrar_alteracion:
    # (El código para mostrar la alteración es el mismo que en la versión anterior,
    #  pero usando df_filtrado en lugar de df_sondajes_3d)

# --- Diseño del gráfico 3D ---
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

# --- Sección Transversal ---
st.header("Sección Transversal")
direccion_seccion = st.selectbox("Dirección:", ["E-W", "N-S"])

if direccion_seccion == "E-W":
    coordenada_fija = df_filtrado["Y"].mean()  
    coordenada_variable = df_filtrado["X"]
    titulo_eje = "Este (m)"
else: 
    coordenada_fija = df_filtrado["X"].mean()  
    coordenada_variable = df_filtrado["Y"]
    titulo_eje = "Norte (m)"

fig_seccion = go.Figure(data=go.Scatter(
    x=coordenada_variable, y=df_filtrado["Z"], mode='markers',
    marker=dict(
        size=6,
        color=df_filtrado[ley_a_visualizar],
        colorscale='Viridis',
        colorbar=dict(title=ley_a_visualizar),
        cmin=df_filtrado[ley_a_visualizar].min(),
        cmax=df_filtrado[ley_a_visualizar].max(),
    )
))
fig_seccion.update_layout(
    title=f'Sección Transversal {direccion_seccion}',
    xaxis_title=titulo_eje,
    yaxis_title='Elevación (m)'
)
st.plotly_chart(fig_seccion)

# --- Mapa de calor 2D ---
st.header("Mapa de Calor 2D")
fig_heatmap = px.density_heatmap(
    df_filtrado, x="X", y="Y", z=ley_a_visualizar, 
    title="Mapa de Calor de Leyes",
    labels={"X": "Este (m)", "Y": "Norte (m)"}
)
st.plotly_chart(fig_heatmap)

# --- Gráfico de dispersión ---
st.header("Gráfico de Dispersión")
elemento_x = st.selectbox("Elemento X:", ["Cu (%)", "Au (g/t)", "Mo (%)"])
elemento_y = st.selectbox("Elemento Y:", ["Au (g/t)", "Cu (%)", "Mo (%)"])
fig_scatter = px.scatter(
    df_filtrado, x=elemento_x, y=elemento_y,
    title="Gráfico de Dispersión",
    trendline="ols"  # Agregar línea de tendencia
)
st.plotly_chart(fig_scatter)
