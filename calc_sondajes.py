import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Título de la aplicación
st.title("Optimización de Costos de Sondajes")

# Parámetros de perforación (metros por día)
PARAMETROS_PERFORACION = {
    "Diamantina": {"min": 30, "max": 60, "costo_metro": 100},
    "Aire Reverso": {"min": 60, "max": 120, "costo_metro": 80},
}

# Rango de días a simular (calculado dinámicamente)
dias_rango = np.arange(1, 31, 1)  # Simular hasta 30 días

# Casilla para ingresar metros requeridos
metros_requeridos = st.number_input("Metros a Perforar:", min_value=10, value=100, step=10)

# Función para calcular datos de perforación
def calcular_datos_perforacion(tipo_sondaje):
    # Calcular el rango válido de metros perforados para cada día
    metros_perforados_min = dias_rango[:, np.newaxis] * PARAMETROS_PERFORACION[tipo_sondaje]["min"]
    metros_perforados_max = dias_rango[:, np.newaxis] * PARAMETROS_PERFORACION[tipo_sondaje]["max"]

    # Limitar el rango de metros perforados al valor requerido
    metros_perforados_min = np.clip(metros_perforados_min, metros_perforados_min.min(), metros_requeridos)
    metros_perforados_max = np.clip(metros_perforados_max, metros_perforados_max.min(), metros_requeridos)

    # Calcular el costo para los rangos mínimo y máximo de metros
    costo_min = metros_perforados_min * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"]
    costo_max = metros_perforados_max * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"]

    return costo_min, costo_max, metros_perforados_min

# Función para crear el gráfico 3D
def graficar_costos_3d():
    costo_diamantina_min, costo_diamantina_max, metros_diamantina = calcular_datos_perforacion("Diamantina")
    costo_aire_reverso_min, costo_aire_reverso_max, metros_aire_reverso = calcular_datos_perforacion("Aire Reverso")

    fig = go.Figure()

    for i, dias in enumerate(dias_rango):
        fig.add_trace(go.Scatter3d(x=np.array([dias, dias]), y=metros_diamantina[i], z=[costo_diamantina_min[i,0], costo_diamantina_max[i,0]],
                                     mode='lines', line=dict(color='blue'), showlegend=False))
        fig.add_trace(go.Scatter3d(x=np.array([dias, dias]), y=metros_aire_reverso[i], z=[costo_aire_reverso_min[i,0], costo_aire_reverso_max[i,0]],
                                     mode='lines', line=dict(color='red'), showlegend=False))

    fig.update_layout(title="Comparación de Costos de Sondaje (3D)",
                      scene=dict(xaxis_title="Tiempo (días)",
                                 yaxis_title="Metros Perforados",
                                 zaxis_title="Costo Total ($)"),
                      autosize=False,
                      width=800, height=600,
                      margin=dict(l=65, r=50, b=65, t=90),
                      showlegend=False)
    st.plotly_chart(fig)

# Función para crear gráfico de contorno
def graficar_contorno():
    costo_diamantina_min, costo_diamantina_max, metros_diamantina = calcular_datos_perforacion("Diamantina")
    costo_aire_reverso_min, costo_aire_reverso_max, metros_aire_reverso = calcular_datos_perforacion("Aire Reverso")

    fig = go.Figure()

    # Agregar áreas de costo para Diamantina
    fig.add_trace(go.Scatter(x=dias_rango, y=metros_diamantina[:, 0], mode='lines', line=dict(color='blue'), name='Diamantina (Min)'))
    fig.add_trace(go.Scatter(x=dias_rango, y=metros_diamantina[:, -1], mode='lines', line=dict(color='blue', dash='dash'), name='Diamantina (Max)'))
    fig.add_trace(go.Scatter(x=[dias_rango[-1], dias_rango[0]], y=[metros_diamantina[-1, 0], metros_diamantina[0, -1]], mode='lines', line=dict(color='blue'), showlegend=False))

    # Agregar áreas de costo para Aire Reverso
    fig.add_trace(go.Scatter(x=dias_rango, y=metros_aire_reverso[:, 0], mode='lines', line=dict(color='red'), name='Aire Reverso (Min)'))
    fig.add_trace(go.Scatter(x=dias_rango, y=metros_aire_reverso[:, -1], mode='lines', line=dict(color='red', dash='dash'), name='Aire Reverso (Max)'))
    fig.add_trace(go.Scatter(x=[dias_rango[-1], dias_rango[0]], y=[metros_aire_reverso[-1, 0], metros_aire_reverso[0, -1]], mode='lines', line=dict(color='red'), showlegend=False))

    fig.update_layout(title='Rango de Costos en Función del Tiempo',
                      xaxis_title='Tiempo (días)',
                      yaxis_title='Metros Perforados',
                      legend_title='Método',
                      yaxis_range=[0, metros_requeridos * 1.1])  # Ajustar el rango del eje Y para que se vea el valor completo de metros_requeridos
    st.plotly_chart(fig)


# Mostrar el gráfico 3D
graficar_costos_3d()

# Mostrar gráfico de contorno
graficar_contorno()

# Explicación del gráfico
st.write("""
**Interpretación:**

- **Gráfico 3D:** Muestra la relación entre costo, tiempo y metros perforados para ambos métodos.  Tenga en cuenta que los puntos en el gráfico están unidos por líneas para visualizar mejor el rango de costos posibles.
- **Gráfico de contorno:**  Visualiza el rango de costos para cada método en función del tiempo y los metros requeridos. Las líneas continua y discontinua representan el costo mínimo y máximo, respectivamente, para cada método.
""")
