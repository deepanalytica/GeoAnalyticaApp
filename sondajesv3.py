import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Rbf  # Interpolación radial
import plotly.express as px  # Para mapas de calor y gráficos de dispersión

# --- Parámetros de la Simulación (Ajustables) ---
NUM_SONDAJES = 20
PROFUNDIDAD_SONDAJE = 200  # Metros
LEY_MEDIA = {"Cu": 0.7, "Au": 0.2, "Mo": 0.01}  # %Cu, g/t Au, % Mo
DESVIACION_ESTANDAR = {"Cu": 0.4, "Au": 0.1, "Mo": 0.005}

# --- Datos de los Sondajes (Ajustables) ---
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
elementos_a_visualizar = st.sidebar.multiselect(
    "Seleccionar Elementos:", ["Cu (%)", "Au (g/t)", "Mo (%)"], default=["Cu (%)"]
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
                color="grey",  # Color base de los sondajes
                line=dict(width=2),
            ),
        )
    )

# Alteración
if mostrar_alteracion:
    for alteracion_tipo in ["Sílice", "Potásica"]:
        df_alteracion = df_filtrado[
            df_filtrado["Alteración"] == alteracion_tipo
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

    # Volumen 3D (dentro del bloque if mostrar_alteracion)
    if mostrar_volumen:
        # Crear una malla 3D para la interpolación
        num_puntos_malla = 30  # Ajustar para la resolución del volumen
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

        # Interpolación y visualización de volúmenes para cada elemento
        for elemento in elementos_a_visualizar:
            # Interpolación radial (RBF)
            rbf = Rbf(
                df_filtrado["X"],
                df_filtrado["Y"],
                df_filtrado["Z"],
                df_filtrado[elemento],
                function="linear",  # Puedes probar con 'multiquadric', 'gaussian', etc.
                smooth=0.5,  # Ajustar para suavizar la interpolación
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
                    opacity=0.15,  # Ajustar la transparencia
                    surface_count=15,  # Ajustar la resolución
                    colorscale="Hot"
                    if elemento == "Au (g/t)"
                    else "Viridis"
                    if elemento == "Cu (%)"
                    else "Cividis",  # Escalas de color diferentes
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
    margin=dict(l=65, r=150, b=65, t=90),  # Más espacio para las barras de color
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

fig_seccion = go.Figure(
    data=go.Scatter(
        x=coordenada_variable,
        y=df_filtrado["Z"],
        mode="markers",
        marker=dict(
            size=6,
            color=df_filtrado[ley_a_visualizar],
            colorscale="Viridis",
            colorbar=dict(title=ley_a_visualizar),
            cmin=df_filtrado[ley_a_visualizar].min(),
            cmax=df_filtrado[ley_a_visualizar].max(),
        ),
    )
)
fig_seccion.update_layout(
    title=f"Sección Transversal {direccion_seccion}",
    xaxis_title=titulo_eje,
    yaxis_title="Elevación (m)",
)
st.plotly_chart(fig_seccion)

# --- Mapa de calor 2D ---
st.header("Mapa de Calor 2D")
fig_heatmap = px.density_heatmap(
    df_filtrado,
    x="X",
    y="Y",
    z=ley_a_visualizar,
    title="Mapa de Calor de Leyes",
    labels={"X": "Este (m)", "Y": "Norte (m)"},
)
st.plotly_chart(fig_heatmap)

# --- Gráfico de dispersión ---
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
