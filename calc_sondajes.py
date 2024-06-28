import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Título de la aplicación
st.title("Optimización de Costos de Sondajes")

# Parámetros de perforación (metros por día)
PARAMETROS_PERFORACION = {
    "Diamantina": {"min": 30, "max": 60},
    "Aire Reverso": {"min": 60, "max": 120},
}

# Función para calcular los costos de sondaje
def calcular_costos(tipo_sondaje, metros):
    costo_metro = {"Diamantina": 100, "Aire Reverso": 80}
    return costo_metro.get(tipo_sondaje, "Tipo de sondaje inválido") * metros

# Función para crear el gráfico
def graficar_costos(costo_diamantina, costo_aire_reverso, tiempo_diamantina, tiempo_aire_reverso, metros_rango):
    fig = go.Figure()

    # Gráfico de costos
    fig.add_trace(go.Scatter(x=metros_rango, y=costo_diamantina,
                            mode='lines', name='Costo Diamantina',
                            line=dict(color='royalblue', dash='dot')))
    fig.add_trace(go.Scatter(x=metros_rango, y=costo_aire_reverso,
                            mode='lines', name='Costo Aire Reverso',
                            line=dict(color='firebrick', dash='dot')))

    # Gráfico de tiempo (como barras apiladas para rango)
    fig.add_trace(go.Bar(x=metros_rango, y=tiempo_diamantina["max"] - tiempo_diamantina["min"], 
                        base=tiempo_diamantina["min"], name="Tiempo Diamantina (rango)", 
                        marker_color='royalblue', opacity=0.5))
    fig.add_trace(go.Bar(x=metros_rango, y=tiempo_aire_reverso["max"] - tiempo_aire_reverso["min"], 
                        base=tiempo_aire_reverso["min"], name="Tiempo Aire Reverso (rango)", 
                        marker_color='firebrick', opacity=0.5)) 

    fig.update_layout(title="Comparación de Costos y Tiempo de Sondaje",
                      xaxis_title="Metros a Perforar",
                      yaxis_title="Costo total ($) / Tiempo (días)",
                      legend_title="Leyenda",
                      yaxis=dict(rangemode='tozero'),  # Ajusta el eje y para empezar en cero
                      hovermode="x unified")
    st.plotly_chart(fig)

# Barra lateral para ingresar los datos
st.sidebar.header("Parámetros de Sondaje")
tipo_sondaje = st.sidebar.selectbox("Tipo de Sondaje", ["Diamantina", "Aire Reverso"])
metros = st.sidebar.slider("Metros a Perforar", min_value=10, max_value=500, value=50, step=10)

# Cálculos para el gráfico
metros_rango = np.arange(10, metros + 10, 10)
costo_diamantina = [calcular_costos("Diamantina", m) for m in metros_rango]
costo_aire_reverso = [calcular_costos("Aire Reverso", m) for m in metros_rango]

# Calcular rango de tiempo en días
tiempo_diamantina = {"min": metros_rango / PARAMETROS_PERFORACION["Diamantina"]["max"],
                   "max": metros_rango / PARAMETROS_PERFORACION["Diamantina"]["min"]}
tiempo_aire_reverso = {"min": metros_rango / PARAMETROS_PERFORACION["Aire Reverso"]["max"],
                    "max": metros_rango / PARAMETROS_PERFORACION["Aire Reverso"]["min"]}

# Mostrar resultados
st.header("Resultados")
st.write(f"Costo total del sondaje: **${calcular_costos(tipo_sondaje, metros):.2f}**")

# Mostrar el gráfico
graficar_costos(costo_diamantina, costo_aire_reverso, tiempo_diamantina, tiempo_aire_reverso, metros_rango)

# Análisis
st.header("Análisis de Costos y Tiempo")

st.write("El gráfico muestra la relación entre costo, tiempo y metros perforados. Observa:")
st.write("- La línea punteada representa el costo total, mientras que la barra sólida representa el rango de tiempo (días) para completar la perforación.")
st.write("- Puedes analizar el trade-off entre costo y tiempo: Aire Reverso es generalmente más rápido pero puede ser más costoso en algunos casos.")

st.write("**Recomendaciones:**")
st.write("- Considera el costo total y el tiempo necesario para completar el proyecto.")
st.write("- Ajusta los parámetros de perforación en el código para reflejar con mayor precisión las capacidades de tu equipo y las condiciones del terreno.") 
