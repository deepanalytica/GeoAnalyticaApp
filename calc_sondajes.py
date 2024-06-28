import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Título de la aplicación
st.title("Optimización de Costos de Sondajes")

# Parámetros de perforación (metros por día)
PARAMETROS_PERFORACION = {
    "Diamantina": {"min": 30, "max": 60, "costo_metro": 100},
    "Aire Reverso": {"min": 60, "max": 120, "costo_metro": 80},
}

# Casilla para ingresar metros requeridos
metros_requeridos = st.number_input("Metros a Perforar:", min_value=10, value=100, step=10)

# Función para calcular datos de perforación
def calcular_datos_perforacion(tipo_sondaje, metros_requeridos):
    """
    Calcula los costos mínimos y máximos, así como los días necesarios 
    para alcanzar los metros requeridos.

    Args:
        tipo_sondaje (str): "Diamantina" o "Aire Reverso".
        metros_requeridos (int): Total de metros a perforar.

    Returns:
        tuple: Costo mínimo, costo máximo y días necesarios.
    """

    dias_min = int(np.ceil(metros_requeridos / PARAMETROS_PERFORACION[tipo_sondaje]["max"]))
    dias_max = int(np.ceil(metros_requeridos / PARAMETROS_PERFORACION[tipo_sondaje]["min"]))

    dias = np.arange(dias_min, dias_max + 1)
    costo_min = dias * PARAMETROS_PERFORACION[tipo_sondaje]["min"] * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"]
    costo_max = dias * PARAMETROS_PERFORACION[tipo_sondaje]["max"] * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"]

    # Ajustar costos para que no excedan el costo de los metros requeridos exactos
    costo_min = np.where(costo_min > metros_requeridos * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"],
                         metros_requeridos * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"], 
                         costo_min)
    costo_max = np.where(costo_max > metros_requeridos * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"],
                         metros_requeridos * PARAMETROS_PERFORACION[tipo_sondaje]["costo_metro"],
                         costo_max)
    
    return costo_min, costo_max, dias

# --- Gráfico 3D ---
def graficar_costos_3d():
    """Genera y muestra el gráfico 3D de comparación de costos."""
    fig = go.Figure()

    for tipo in ["Diamantina", "Aire Reverso"]:
        costo_min, costo_max, dias = calcular_datos_perforacion(tipo, metros_requeridos)
        for i in range(len(dias)):
            fig.add_trace(go.Scatter3d(
                x=np.array([dias[i], dias[i]]), 
                y=np.array([metros_requeridos, metros_requeridos]), 
                z=[costo_min[i], costo_max[i]],
                mode='lines',
                line=dict(color='blue' if tipo == "Diamantina" else 'red'),
                showlegend=False 
            ))

    fig.update_layout(
        title="Comparación de Costos de Sondaje (3D)",
        scene=dict(
            xaxis_title="Tiempo (días)",
            yaxis_title="Metros Perforados",
            zaxis_title="Costo Total ($)"
        ),
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    st.plotly_chart(fig)

# --- Gráfico de contorno ---
def graficar_contorno():
    """Genera y muestra el gráfico de contorno de comparación de costos."""
    fig = go.Figure()

    for tipo in ["Diamantina", "Aire Reverso"]:
        costo_min, costo_max, dias = calcular_datos_perforacion(tipo, metros_requeridos)
        fig.add_trace(go.Scatter(
            x=dias, 
            y=costo_min, 
            mode='lines', 
            line=dict(color='blue' if tipo == "Diamantina" else 'red'), 
            name=f'{tipo} (Min)'
        ))
        fig.add_trace(go.Scatter(
            x=dias, 
            y=costo_max, 
            mode='lines', 
            line=dict(color='blue' if tipo == "Diamantina" else 'red', dash='dash'), 
            name=f'{tipo} (Max)'
        ))

    fig.update_layout(
        title='Rango de Costos en Función del Tiempo',
        xaxis_title='Tiempo (días)',
        yaxis_title='Costo Total ($)',
        legend_title='Método'
    )
    st.plotly_chart(fig)

# --- Mostrar gráficos ---
graficar_costos_3d()
graficar_contorno()

# --- Explicación ---
st.write("""
**Interpretación:**

- **Gráfico 3D:** Muestra la relación entre el tiempo, los metros perforados (fijos en este caso) 
  y el rango de costos para ambos métodos.
- **Gráfico de contorno:**  Visualiza el rango de costos mínimo y máximo para cada método en 
  función del tiempo necesario para completar la perforación.
""")
