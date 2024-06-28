import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Generador de Tabla de Proyecto", layout="wide")

# Función para la página de inicio
def home():
    st.title("Generador de Tabla de Proyecto")
    st.write("Bienvenido a la herramienta de generación de tablas para proyectos. Utilice el menú de la izquierda para navegar a través de las diferentes secciones de la aplicación.")

# Función para la página de entrada de datos
def data_entry():
    st.title("Entrada de Datos del Proyecto")

    # Crear un formulario para ingresar datos
    with st.form("project_form"):
        st.subheader("Ingrese los datos del proyecto")

        # Sección de costos y gastos operativos
        transport_costs = st.number_input("Transport Costs Charges (US$mm)", value=0.0, help="Costo de transporte de los productos.")
        site_royalties = st.number_input("Site Royalties (US$mm)", value=0.0, help="Pagos de regalías por el sitio.")
        operating_expenses = st.number_input("Operating Expenses (US$mm)", value=0.0, help="Gastos operativos del proyecto.")
        ebitda = st.number_input("EBITDA (US$mm)", value=0.0, help="Ganancias antes de intereses, impuestos, depreciación y amortización.")
        sustaining_capital = st.number_input("Sustaining Capital (US$mm)", value=0.0, help="Capital necesario para mantener las operaciones.")
        change_working_capital = st.number_input("Change in Working Capital (US$mm)", value=0.0, help="Cambio en el capital de trabajo.")
        pre_tax_cash_flow = st.number_input("Pre-Tax Unlevered Free Cash Flow (US$mm)", value=0.0, help="Flujo de caja libre antes de impuestos.")
        tax_payable = st.number_input("Tax Payable (US$mm)", value=0.0, help="Impuestos a pagar.")
        post_tax_cash_flow = st.number_input("Post-Tax Unlevered Free Cash Flow (US$mm)", value=0.0, help="Flujo de caja libre después de impuestos.")

        # Sección de recursos minerales
        measured = st.number_input("Measured (Mt)", value=0.22, help="Recursos minerales medidos en millones de toneladas.")
        indicated = st.number_input("Indicated (Mt)", value=0.86, help="Recursos minerales indicados en millones de toneladas.")
        inferred = st.number_input("Inferred (Mt)", value=0.22, help="Recursos minerales inferidos en millones de toneladas.")
        grade_cut = st.number_input("Grade Cut-Off (%)", value=0.22, help="Ley de corte en porcentaje.")
        contained_metal = st.number_input("Contained Metal (kt)", value=419.837, help="Metal contenido en miles de toneladas.")

        # Sección de producción
        heap_leach_tonnes = st.number_input("Heap Leach Tonnes (kt)", value=0.0, help="Toneladas procesadas por lixiviación en montón.")
        rom_tonnes = st.number_input("ROM Tonnes (kt)", value=0.0, help="Toneladas de mineral run-of-mine.")
        waste_tonnes = st.number_input("Waste Tonnes (kt)", value=0.0, help="Toneladas de desecho.")
        heap_leach_grade = st.number_input("Heap Leach Cu Grade (%)", value=0.0, help="Ley de cobre en lixiviación en montón.")
        heap_leach_recovery = st.number_input("Heap Leach Recovery (%)", value=0.0, help="Recuperación de cobre en lixiviación en montón.")
        rom_grade = st.number_input("ROM Cu Grade (%)", value=0.0, help="Ley de cobre en mineral run-of-mine.")
        rom_recovery = st.number_input("ROM Recovery (%)", value=0.0, help="Recuperación de cobre en mineral run-of-mine.")
        total_processed = st.number_input("Total Tonnes Processed (kt)", value=0.0, help="Toneladas totales procesadas.")
        operation_life = st.number_input("Operation Life (yrs)", value=0, help="Duración de la operación en años.")

        # Sección de producción de cátodos de cobre
        heap_leach_recovered = st.number_input("Heap Leach Recovered Cu (kt)", value=0.0, help="Cobre recuperado por lixiviación en montón en miles de toneladas.")
        rom_recovered = st.number_input("ROM Recovered Cu (kt)", value=0.0, help="Cobre recuperado de mineral run-of-mine en miles de toneladas.")
        total_recovered = st.number_input("Total Recovered Cu (kt)", value=0.0, help="Total de cobre recuperado en miles de toneladas.")

        # Botón para enviar el formulario
        submit_button = st.form_submit_button(label="Generar Tabla")

    # Generar la tabla si se envió el formulario
    if submit_button:
        data = {
            "Transport Costs Charges (US$mm)": [transport_costs],
            "Site Royalties (US$mm)": [site_royalties],
            "Operating Expenses (US$mm)": [operating_expenses],
            "EBITDA (US$mm)": [ebitda],
            "Sustaining Capital (US$mm)": [sustaining_capital],
            "Change in Working Capital (US$mm)": [change_working_capital],
            "Pre-Tax Unlevered Free Cash Flow (US$mm)": [pre_tax_cash_flow],
            "Tax Payable (US$mm)": [tax_payable],
            "Post-Tax Unlevered Free Cash Flow (US$mm)": [post_tax_cash_flow],
            "Measured (Mt)": [measured],
            "Indicated (Mt)": [indicated],
            "Inferred (Mt)": [inferred],
            "Grade Cut-Off (%)": [grade_cut],
            "Contained Metal (kt)": [contained_metal],
            "Heap Leach Tonnes (kt)": [heap_leach_tonnes],
            "ROM Tonnes (kt)": [rom_tonnes],
            "Waste Tonnes (kt)": [waste_tonnes],
            "Heap Leach Cu Grade (%)": [heap_leach_grade],
            "Heap Leach Recovery (%)": [heap_leach_recovery],
            "ROM Cu Grade (%)": [rom_grade],
            "ROM Recovery (%)": [rom_recovery],
            "Total Tonnes Processed (kt)": [total_processed],
            "Operation Life (yrs)": [operation_life],
            "Heap Leach Recovered Cu (kt)": [heap_leach_recovered],
            "ROM Recovered Cu (kt)": [rom_recovered],
            "Total Recovered Cu (kt)": [total_recovered],
        }

        df = pd.DataFrame(data)
        st.subheader("Tabla Generada")
        st.dataframe(df)
        st.download_button("Descargar Datos en CSV", df.to_csv(index=False), "project_data.csv")

        # Generar gráficos
        st.subheader("Visualización de Datos")
        fig, ax = plt.subplots()
        sns.barplot(data=df, palette="viridis", ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

# Crear el menú de navegación
menu = ["Inicio", "Entrada de Datos", "Visualización"]
choice = st.sidebar.selectbox("Seleccione una Página", menu)

if choice == "Inicio":
    home()
elif choice == "Entrada de Datos":
    data_entry()
elif choice == "Visualización":
    st.title("Visualización de Datos")
    st.write("Seleccione 'Entrada de Datos' en el menú para ingresar los datos y generar la visualización.")
