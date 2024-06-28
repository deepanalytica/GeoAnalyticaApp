import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Título de la aplicación
st.title("Optimización de Costos de Sondajes (3D)")

# Parámetros de perforación (metros por día)
PARAMETROS_PERFORACION = {
    "Diamantina": {"min": 30, "max": 60, "costo_metro": 100},
    "Aire Reverso": {"min": 60, "max": 120, "costo_metro": 80},
}

# Función para crear el gráfico 3D
def graficar_costos_3d():
    # Rango de metros y días a simular
    metros_rango = np.arange(10, 510, 50)
    dias_rango = np.arange(1, 21, 1)

    # Crear una malla con los rangos
    metros_malla, dias_malla = np.meshgrid(metros_rango, dias_rango)

    # Calcular datos para Diamantina
    tasa_diamantina = np.random.uniform(PARAMETROS_PERFORACION["Diamantina"]["min"],
                                        PARAMETROS_PERFORACION["Diamantina"]["max"],
                                        size=metros_malla.shape)
    metros_perforados_diamantina = dias_malla * tasa_diamantina
    metros_perforados_diamantina = np.where(metros_perforados_diamantina <= metros_rango[-1], 
                                          metros_perforados_diamantina, metros_rango[-1])
    costo_diamantina = metros_perforados_diamantina * PARAMETROS_PERFORACION["Diamantina"]["costo_metro"]

    # Calcular datos para Aire Reverso
    tasa_aire_reverso = np.random.uniform(PARAMETROS_PERFORACION["Aire Reverso"]["min"],
                                          PARAMETROS_PERFORACION["Aire Reverso"]["max"],
                                          size=metros_malla.shape)
    metros_perforados_aire_reverso = dias_malla * tasa_aire_reverso
    metros_perforados_aire_reverso = np.where(metros_perforados_aire_reverso <= metros_rango[-1],
                                              metros_perforados_aire_reverso, metros_rango[-1])
    costo_aire_reverso = metros_perforados_aire_reverso * PARAMETROS_PERFORACION["Aire Reverso"]["costo_metro"]

    # Crear el gráfico 3D
    fig = go.Figure()

    # Superficie para Diamantina
    fig.add_trace(go.Surface(z=costo_diamantina, x=dias_malla, y=metros_perforados_diamantina,
                               colorscale='Blues',
                               colorbar=dict(title="Costo Diamantina ($)"),
                               name="Diamantina",
                               opacity=0.8))

    # Superficie para Aire Reverso
    fig.add_trace(go.Surface(z=costo_aire_reverso, x=dias_malla, y=metros_perforados_aire_reverso,
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

# Mostrar el gráfico 3D
graficar_costos_3d()

# Explicación del gráfico
st.write("""
**Interpretación del gráfico 3D:**

- **Superficies:** La superficie azul representa los costos para Diamantina y la roja para Aire Reverso.
- **Transparencia:** Las superficies son semi-transparentes para visualizar la superposición y las áreas donde un método podría ser más económico que otro.
""")
