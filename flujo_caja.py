import streamlit as st
import pandas as pd

# Título de la aplicación
st.title("Generador de Tabla de Flujo de Caja Proyecto)

# Crear un formulario para ingresar datos
with st.form("project_form"):
    st.subheader("Ingrese los datos del proyecto")

    # Sección de costos y gastos operativos
    transport_costs = st.number_input("Transport Costs Charges (US$mm)", value=0.0)
    site_royalties = st.number_input("Site Royalties (US$mm)", value=0.0)
    operating_expenses = st.number_input("Operating Expenses (US$mm)", value=0.0)
    ebitda = st.number_input("EBITDA (US$mm)", value=0.0)
    sustaining_capital = st.number_input("Sustaining Capital (US$mm)", value=0.0)
    change_working_capital = st.number_input("Change in Working Capital (US$mm)", value=0.0)
    pre_tax_cash_flow = st.number_input("Pre-Tax Unlevered Free Cash Flow (US$mm)", value=0.0)
    tax_payable = st.number_input("Tax Payable (US$mm)", value=0.0)
    post_tax_cash_flow = st.number_input("Post-Tax Unlevered Free Cash Flow (US$mm)", value=0.0)

    # Sección de recursos minerales
    measured = st.number_input("Measured (Mt)", value=0.22)
    indicated = st.number_input("Indicated (Mt)", value=0.86)
    inferred = st.number_input("Inferred (Mt)", value=0.22)
    grade_cut = st.number_input("Grade Cut-Off (%)", value=0.22)
    contained_metal = st.number_input("Contained Metal (kt)", value=419.837)

    # Sección de producción
    heap_leach_tonnes = st.number_input("Heap Leach Tonnes (kt)", value=0.0)
    rom_tonnes = st.number_input("ROM Tonnes (kt)", value=0.0)
    waste_tonnes = st.number_input("Waste Tonnes (kt)", value=0.0)
    heap_leach_grade = st.number_input("Heap Leach Cu Grade (%)", value=0.0)
    heap_leach_recovery = st.number_input("Heap Leach Recovery (%)", value=0.0)
    rom_grade = st.number_input("ROM Cu Grade (%)", value=0.0)
    rom_recovery = st.number_input("ROM Recovery (%)", value=0.0)
    total_processed = st.number_input("Total Tonnes Processed (kt)", value=0.0)
    operation_life = st.number_input("Operation Life (yrs)", value=0)

    # Sección de producción de cátodos de cobre
    heap_leach_recovered = st.number_input("Heap Leach Recovered Cu (kt)", value=0.0)
    rom_recovered = st.number_input("ROM Recovered Cu (kt)", value=0.0)
    total_recovered = st.number_input("Total Recovered Cu (kt)", value=0.0)

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
    st.dataframe(df)
