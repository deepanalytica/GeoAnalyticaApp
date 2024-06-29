import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Rbf
import plotly.express as px

# --- Parámetros de la Simulación (Ajustables) ---
NUM_SONDAJES = 20
PROFUNDIDAD_SONDAJE = 200  # Metros
LEY_MEDIA = {"Cu": 0.7, "Au": 0.2, "Mo": 0.01}  # %Cu, g/t Au, % Mo
DESVIACION_ESTANDAR = {"Cu": 0.4, "Au": 0.1, "Mo": 0.005}

# --- Datos de los Sondajes (Ajustables) ---
# ... (Mismos datos de sondajes que en el código anterior)

# --- Funciones ---
# ... (Funciones generar_leyes y generar_datos_sondaje, igual que antes)

# --- Generar datos (una sola vez) ---
@st.cache_data
def cargar_datos():
    # ... (Función igual que en el código anterior)

df_sondajes_3d = cargar_datos()

# --- Interfaz de usuario de Streamlit ---
st.sidebar.title("Visualización de Sondajes 3D")

# --- Opciones de visualización ---
st.sidebar.header("Opciones de Visualización")
elementos_a_visualizar = st.sidebar.multiselect(
    "Seleccionar Elementos:",
    ["Cu (%)", "Au (g/t)", "Mo (%)"],
    default=["Cu (%)"],
)
mostrar_volumen = st.sidebar.checkbox("Mostrar Volumen 3D", value=False)
mostrar_alteracion = st.sidebar.checkbox("Mostrar Alteración", value=True)

# --- Filtros ---
st.sidebar.header("Filtros")
profundidad_min = st.sidebar.slider(
    "Profundidad Mínima (m)", min_value=0, max_value=PROFUNDIDAD_SONDAJE, value=0
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

# --- Selección de Ley para Visualización ---
ley_a_visualizar = st.selectbox(
    "Seleccionar Ley Mineral:", ["Cu (%)", "Au (g/t)", "Mo (%)"]
)

# --- Visualización ---
st.title("Visualización 3D de Sondajes - Pórfido Cu-Au-Mo")

# --- Contenedor para el gráfico 3D y el scatterplot ---
col1, col2 = st.columns([3, 1])

# --- Scatterplot (Columna 2) ---
with col2:
    st.header("Gráfico de Dispersión")
    elemento_x = st.selectbox("Elemento X:", ["Cu (%)", "Au (g/t)", "Mo (%)"])
    elemento_y = st.selectbox("Elemento Y:", ["Au (g/t)", "Cu (%)", "Mo (%)"])
    fig_scatter = px.scatter(
        df_filtrado,
        x=elemento_x,
        y=elemento_y,
        title="Gráfico de Dispersión",
        trendline="ols",  # Agregar línea de tendencia
    )
    st.plotly_chart(fig_scatter)

# --- Gráfico 3D (Columna 1) ---
with col1:
    fig = go.Figure()

    # --- Sondajes en el gráfico 3D ---
    # ... (Código para los sondajes en el gráfico 3D, igual que antes)

    # --- Alteración en el gráfico 3D ---
    if mostrar_alteracion:
        # ... (Código para la alteración, igual que antes)

    # --- Volumen 3D ---
    if mostrar_volumen:
        # Crear una malla 3D para la interpolación
        num_puntos_malla = 25  # Ajustar para la resolución del volumen
        xi = np.linspace(
            df_filtrado["X"].min(), df_filtrado["X"].max(), num_puntos_malla
        )
        yi = np.linspace(
            df_filtrado["Y"].min(), df_filtrado["Y"].max(), num_puntos_malla
        )
        zi = np.linspace(
            df_filtrado["Z"].min(), df_filtrado["Z"].max(), num_puntos_malla
        )
        xi, yi, zi = np.meshgrid(xi, yi, zi)

        # Paletas de colores con mayor contraste
        escalas_colores = {
            "Cu (%)": "Inferno",
            "Au (g/t)": "Magma",
            "Mo (%)": "Viridis",
        }

        # Interpolación y visualización de volúmenes para cada elemento
        for elemento in elementos_a_visualizar:
            # Interpolación radial (RBF)
            rbf = Rbf(
                df_filtrado["X"],
                df_filtrado["Y"],
                df_filtrado["Z"],
                df_filtrado[elemento],
                function="multiquadric",  # Puedes probar con 'linear', 'gaussian', etc.
                smooth=0.8,  # Ajustar para suavizar la interpolación
            )
            valores_elemento = rbf(xi, yi, zi)

            # Agregar el volumen 3D al gráfico
            fig.add_trace(
                go.Volume(
                    x=xi.flatten(),
                    y=yi.flatten(),
                    z=zi.flatten(),
                    value=valores_elemento.flatten(),
                    isomin=df_filtrado[elemento].min(),
                    isomax=df_filtrado[elemento].max(),
                    opacity=0.2,  # Ajustar la transparencia
                    surface_count=20,  # Ajustar la resolución
                    colorscale=escalas_colores[
                        elemento
                    ],  # Asignar escala de color
                    colorbar=dict(
                        title=elemento,
                        x=1.1 + 0.15 * elementos_a_visualizar.index(elemento),
                        len=0.8,
                    ),
                    showscale=elementos_a_visualizar.index(elemento)
                    == 0,  # Mostrar solo una barra de color por defecto
                )
            )

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
        margin=dict(l=65, r=150, b=65, t=90),
        clickmode="event+select",
    )

    # --- Obtener puntos seleccionados en el scatterplot ---
    selected_points = (
        fig_scatter.data[0].selectedpoints
        if fig_scatter.data[0].selectedpoints is not None
        else []
    )

    # --- Resaltar puntos seleccionados en el gráfico 3D ---
    if selected_points:
        selected_indices = [p["pointIndex"] for p in selected_points]
        for i, trace in enumerate(fig.data):
            if "customdata" in trace:
                mask = np.isin(trace.customdata, selected_indices)
                fig.data[i].marker.color = np.where(
                    mask, "red", fig.data[i].marker.color
                )

    st.plotly_chart(fig)

# --- Sección Transversal ---
# ... (Código para la sección transversal, igual que antes)

# --- Mapa de calor 2D ---
st.header("Mapa de Calor 2D")
fig_heatmap_2d = px.density_heatmap(
    df_filtrado,
    x="X",
    y="Y",
    z=ley_a_visualizar,
    title="Mapa de Calor de Leyes 2D",
    labels={"X": "Este (m)", "Y": "Norte (m)"},
    color_continuous_scale="Inferno",  # Paleta de colores intensa
)
st.plotly_chart(fig_heatmap_2d)

# --- Mapa de calor 3D ---
st.header("Mapa de Calor 3D")
fig_heatmap_3d = go.Figure(
    data=go.Scatter3d(
        x=df_filtrado["X"],
        y=df_filtrado["Y"],
        z=df_filtrado["Z"],
        mode="markers",
        marker=dict(
            size=5,
            color=df_filtrado[ley_a_visualizar],
            colorscale="Inferno",  # Paleta de colores intensa
            colorbar=dict(title=ley_a_visualizar),
        ),
    )
)
fig_heatmap_3d.update_layout(
    scene=dict(
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        zaxis_title="Elevación (m)",
        aspectmode="cube",  # Mantener la relación de aspecto para una mejor visualización
    ),
    width=800,
    height=600,
)
st.plotly_chart(fig_heatmap_3d)
