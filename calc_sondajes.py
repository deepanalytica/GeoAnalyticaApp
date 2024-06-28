import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Título de la aplicación
st.title("Optimización de Costos de Sondajes (3D)")

# Parámetros de perforación (metros por día)
PARAMETROS_PERFORACION = {
    "Diamantina": {"min": 30, "max": 60, "costo_metro": 100},
    "Aire Reverso": {"min": 60, "max": 120, "costo_metro": 80},
}

# Rango de metros y días a simular (global para consistencia)
metros_rango = np.arange(10, 510, 50)
dias_rango = np.arange(1, 21, 1)
metros_malla, dias_malla = np.meshgrid(metros_rango, dias_rango)

# Función para calcular datos de perforación
def calcular_datos_perforacion(tipo_sondaje):
    tasa = np.random.uniform(PARAMETROS_PERFORACION[tipo_sondaje]["min"],
                              PARAMETROS_PERFORACION[tipo_sondaje]["max"],
                              size=metros_malla.shape)
    metros_perforados = dias_malla * tasa
    metros_perforados = np.where(metros_perforados <= metros_rango[-1], metros_perforados, metros_rango[-1])
    costo = metros_perforados * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"]
    return costo, metros_perforados

# Función para crear el gráfico 3D
def graficar_costos_3d():
    costo_diamantina, metros_diamantina = calcular_datos_perforacion("Diamantina")
    costo_aire_reverso, metros_aire_reverso = calcular_datos_perforacion("Aire Reverso")

    fig = go.Figure()

    fig.add_trace(go.Surface(z=costo_diamantina, x=dias_malla, y=metros_diamantina,
                               colorscale='Blues',
                               colorbar=dict(title="Costo Diamantina ($)"),
                               name="Diamantina",
                               opacity=0.8))

    fig.add_trace(go.Surface(z=costo_aire_reverso, x=dias_malla, y=metros_aire_reverso,
                               colorscale='Reds',
                               colorbar=dict(title="Costo Aire Reverso ($)"),
                               name="Aire Reverso",
                               opacity=0.8))

    fig.update_layout(title="Comparación de Costos de Sondaje (3D)",
                      scene=dict(xaxis_title="Tiempo (días)",
                                 yaxis_title="Metros Perforados",
                                 zaxis_title="Costo Total ($)"),
                      autosize=False,
                      width=800, height=600)
    st.plotly_chart(fig)

# Función para crear gráfico de contorno
def graficar_contorno():
    costo_diamantina, _ = calcular_datos_perforacion("Diamantina")
    costo_aire_reverso, _ = calcular_datos_perforacion("Aire Reverso")

    fig = go.Figure()
    fig.add_trace(go.Contour(z=costo_diamantina, x=dias_rango, y=metros_rango, 
                             contours_coloring='lines',
                             line_width=2,
                             contours=dict(start=costo_diamantina.min(), 
                                           end=costo_diamantina.max(), 
                                           size=5000),
                             name='Diamantina'))
    fig.add_trace(go.Contour(z=costo_aire_reverso, x=dias_rango, y=metros_rango,
                             contours_coloring='lines',
                             line_width=2,
                             contours=dict(start=costo_aire_reverso.min(),
                                           end=costo_aire_reverso.max(),
                                           size=5000),
                             name='Aire Reverso'))

    fig.update_layout(title='Mapa de Contorno de Costos',
                      xaxis_title='Tiempo (días)',
                      yaxis_title='Metros Perforados',
                      legend_title='Método')
    st.plotly_chart(fig)


# Mostrar el gráfico 3D
graficar_costos_3d()

# Mostrar gráfico de contorno
graficar_contorno()

# Explicación del gráfico
st.write("""
**Interpretación:**

- **Gráfico 3D:** Muestra la relación entre costo, tiempo y metros perforados para ambos métodos.
- **Mapa de Contorno:**  Visualiza áreas de igual costo para cada método, lo que facilita la comparación en función del tiempo y la profundidad deseados.
""")
