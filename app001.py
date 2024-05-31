import streamlit as st
import pandas as pd
import datetime

# Inyectar CSS personalizado para cambiar el color de la fuente en la barra lateral
st.markdown(
    """
    <style>
    .backgroundcolor {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Demostración de Widgets de Streamlit')

# Métricas superiores
num_varones = 10  # Ejemplo de número de varones, ajustar según datos reales
num_mujeres = 15  # Ejemplo de número de mujeres, ajustar según datos reales
edad_promedio_varones = 30  # Ejemplo de edad promedio, ajustar según datos reales
edad_promedio_mujeres = 28  # Ejemplo de edad promedio, ajustar según datos reales

col1, col2 = st.columns(2)
col1.metric(label="👦 Varones", value=num_varones, delta=f"Edad Promedio: {edad_promedio_varones} años")
col2.metric(label="👧 Mujeres", value=num_mujeres, delta=f"Edad Promedio: {edad_promedio_mujeres} años")



# Botón en la barra lateral
if st.sidebar.button('Saludar'):
    st.write('¡Hola!')

# Checkbox en la barra lateral
acuerdo = st.sidebar.checkbox('Estoy de acuerdo')
if acuerdo:
    st.write('¡Genial!')

# Radio Button en la barra lateral
genero = st.sidebar.radio(
    "¿Cuál es tu género de película favorito?",
    ('Comedia', 'Drama', 'Documental'))
if genero:
    st.write('Has seleccionado:', genero)

# Selectbox en la barra lateral
metodo_contacto = st.sidebar.selectbox(
    '¿Cómo te gustaría ser contactado?',
    ('Correo electrónico', 'Teléfono fijo', 'Teléfono móvil'))
st.write('Has seleccionado:', metodo_contacto)

# Multiselect en la barra lateral
colores = st.sidebar.multiselect(
    '¿Cuáles son tus colores favoritos?',
    ['Verde', 'Amarillo', 'Rojo', 'Azul'],
    ['Amarillo', 'Rojo'])
st.write('Has seleccionado:', colores)

# Slider en la barra lateral
edad = st.sidebar.slider('¿Cuántos años tienes?', 0, 130, 25)
st.write("Tengo ", edad, 'años')

# Text Input en la barra lateral
titulo = st.sidebar.text_input('Título de la película', 'Titanic')
st.write('El título de la película actual es', titulo)

# Date Input en la barra lateral
fecha = st.sidebar.date_input("¿Cuándo es tu cumpleaños?", datetime.date(2019, 7, 6))
st.write('Tu cumpleaños es:', fecha)

# File Uploader en la barra lateral
archivo_subido = st.sidebar.file_uploader("Elige un archivo")
if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.write(df)

# Color Picker en la barra lateral
color = st.sidebar.color_picker('Elige un color', '#00f900')
st.write('El color actual es', color)
