import streamlit as st

# Título de la aplicación
st.title("Optimización de Costos de Sondajes")

# Función para calcular los costos de sondaje
def calcular_costos(tipo_sondaje, metros):
  """
  Calcula los costos de sondaje según el tipo y los metros.

  Args:
      tipo_sondaje (str): "Diamantina" o "Aire Reverso"
      metros (float): Metros a perforar

  Returns:
      float: Costo total del sondaje
  """
  if tipo_sondaje == "Diamantina":
    costo_metro = 100
  elif tipo_sondaje == "Aire Reverso":
    costo_metro = 80
  else:
    return "Tipo de sondaje inválido"

  costo_total = costo_metro * metros
  return costo_total

# Barra lateral para ingresar los datos
st.sidebar.header("Parámetros de Sondaje")
tipo_sondaje = st.sidebar.selectbox("Tipo de Sondaje", ["Diamantina", "Aire Reverso"])
metros = st.sidebar.number_input("Metros a Perforar", min_value=0.0, step=1.0)

# Calcular los costos y el tiempo estimado
costo_total = calcular_costos(tipo_sondaje, metros)

# Mostrar los resultados
st.header("Resultados")
st.write(f"Costo total del sondaje: **${costo_total:.2f}**")

# Mostrar información adicional sobre el rango de avance diario
if tipo_sondaje == "Diamantina":
    st.write("El avance diario estimado para Diamantina es de 30 a 60 metros.")
elif tipo_sondaje == "Aire Reverso":
    st.write("El avance diario estimado para Aire Reverso es de 60 a 120 metros.")

# Sección para sugerencias de optimización
st.header("Sugerencias de Optimización")
st.markdown("""
* **Comparar precios de proveedores:** Busca diferentes proveedores de servicios de sondaje para obtener las mejores tarifas.
* **Planificar la ubicación de los sondajes:** Una buena planificación de la ubicación de los sondajes puede reducir la cantidad de metros necesarios.
* **Usar la técnica de sondaje adecuada:** Elegir la técnica de sondaje más eficiente para el tipo de suelo y la información que se busca puede reducir los costos.
* **Monitorear el avance del proyecto:** Un seguimiento constante del avance del proyecto permite identificar posibles retrasos y ajustar el presupuesto en consecuencia.
* **Considerar alternativas:** Dependiendo del objetivo, existen alternativas a los sondajes tradicionales, como la geofísica, que pueden resultar más económicas.
""")
