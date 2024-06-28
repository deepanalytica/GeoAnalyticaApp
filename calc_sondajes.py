import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Título de la aplicación
st.title("Optimización de Costos de Sondajes (3D)")

# Parámetros de perforación (metros por día)
PARAMETROS_PERFORACION = {
    "Diamantina": {"min": 30, "max": 60},
    "Aire Reverso": {"min": 60, "max": 120},
}

# Función para calcular los costos de sondaje
def calcular_costos(tipo_sondaje, metros):
    costo_metro = {"Diamantina": 100, "Aire Reverso": 80}
    return costo_metro.get(tipo_sondaje, "Tipo de sondaje inválido") * metros

# Función para crear el gráfico 3D
def graficar_costos_3d(tipo_sondaje):
    # Rango de metros y días a simular
    metros_rango = np.arange(10, 510, 50) 
    dias_rango = np.arange(1, 21, 1)

    # Crear una malla con los rangos
    metros_malla, dias_malla = np.meshgrid(metros_rango, dias_rango)

    # Calcular la tasa de perforación (metros/día) para cada punto de la malla
    tasa_perforacion = np.random.uniform(PARAMETROS_PERFORACION[tipo_sondaje]["min"],
                                      PARAMETROS_PERFORACION[tipo_sondaje]["max"],
                                      size=metros_malla.shape)

    # Calcular los metros perforados en función del tiempo
    metros_perforados = dias_malla * tasa_perforacion

    # Asegurar que no se exceda el límite de metros 
    metros_perforados = np.where(metros_perforados <= metros_rango[-1], metros_perforados, metros_rango[-1])

    # Calcular el costo para cada punto
    costo_total = calcular_costos(tipo_sondaje, metros_perforados)

    # Crear el gráfico 3D
    fig = go.Figure(data=[go.Surface(z=costo_total, x=dias_malla, y=metros_perforados,
                                   colorscale='Viridis',
                                   colorbar=dict(title="Costo Total ($)"))])

    fig.update_layout(title=f"Relación Costo-Tiempo-Metros ({tipo_sondaje})",
                      scene=dict(xaxis_title="Tiempo (días)",
                                 yaxis_title="Metros Perforados",
                                 zaxis_title="Costo Total ($)"),
                      autosize=False,
                      width=800, height=600)
    st.plotly_chart(fig)

# Barra lateral para ingresar los datos
st.sidebar.header("Parámetros de Sondaje")
tipo_sondaje = st.sidebar.selectbox("Tipo de Sondaje", ["Diamantina", "Aire Reverso"])

# Mostrar el gráfico 3D
graficar_costos_3d(tipo_sondaje)

# Explicación del gráfico
st.write("""
**Interpretación del gráfico 3D:**

- **Eje X (Tiempo):** Representa el número de días de trabajo.
- **Eje Y (Metros Perforados):** Muestra la cantidad de metros perforados.
- **Eje Z (Costo Total):** Indica el costo total en función del tiempo y los metros perforados.

**Análisis:**

El gráfico te permite visualizar cómo interactúan el costo, el tiempo y los metros perforados. Puedes observar cómo el costo aumenta con el tiempo y la cantidad de metros. 
""")
