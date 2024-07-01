import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import pywt
from sklearn.cluster import KMeans

# Configuración de la página
st.set_page_config(
    page_title="GeoAnalytica Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cargar datos
@st.cache_data
def load_data():
    # Cambia la ruta del archivo CSV a la ubicación correcta de tus datos
    df = pd.read_csv("data/sondajes.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.header("Configuración de Datos")
sondajes_seleccionados = st.sidebar.multiselect(
    "Seleccionar Sondajes:", df["Sondaje"].unique(), default=df["Sondaje"].unique()
)
df_filtrado = df[df["Sondaje"].isin(sondajes_seleccionados)]

# Creación de un DataFrame para 3D
df_sondajes_3d = df_filtrado[["Sondaje", "X", "Y", "Z", "Cu (%)", "Au (g/t)", "Mo (%)", "Alteración"]].copy()
df_sondajes_3d["Cluster"] = KMeans(n_clusters=3).fit_predict(df_sondajes_3d[["Cu (%)", "Au (g/t)", "Mo (%)"]])

# Título del Dashboard
st.title("GeoAnalytica Dashboard")

# --- Fila 1 ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("Sondajes en 3D")
    ley_a_visualizar = st.selectbox(
        "Seleccionar Ley para Volumen de Ley:",
        ["Cu (%)", "Au (g/t)", "Mo (%)"]
    )
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
        NPTS = 30  # Definición de la cantidad de puntos para la malla
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
        width=700,
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
        width=700,
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
            width=700,
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
            width=700,
            height=500,
            margin=dict(r=10, l=10, b=10, t=30),
        )
        st.plotly_chart(fig_telaraña, use_container_width=True)

with col4:
    st.header("Tesselation Wavelet")
    ley_wavelet = st.selectbox("Seleccionar Ley para Tesselation Wavelet:", ["Cu (%)", "Au (g/t)", "Mo (%)"])
    x = df_filtrado["X"]
    y = df_filtrado["Y"]
    z = df_filtrado[ley_wavelet]

    # Preparar datos para Wavelet
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)

    # Cálculo de dx/dz
    dx_dz = dx / dz

    fig_wavelet = go.Figure()
    fig_wavelet.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=dx_dz,
            colorscale="Viridis",
            colorbar=dict(title=f"dx/dz ({ley_wavelet})"),
        )
    )
    fig_wavelet.update_layout(
        title=f"Tesselation Wavelet de {ley_wavelet}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="dx/dz",
            aspectmode='data'
        ),
        width=700,
        height=500,
        margin=dict(r=10, l=10, b=10, t=30),
    )
    st.plotly_chart(fig_wavelet, use_container_width=True)
