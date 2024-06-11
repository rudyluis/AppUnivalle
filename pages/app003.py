import altair as alt
import streamlit as st
from vega_datasets import data

# Configuración de la página
imagen = "logos/LogoCar.png"
st.set_page_config(page_title="CARS DATA",
                   page_icon=imagen,
                   layout="wide",
                   initial_sidebar_state="auto"
                   )

# Cargar datos de ejemplo sobre coches
df = data.cars()
st.title("Dashboard")  # Título del dashboard
st.write(df)  # Mostrar el dataframe en la aplicación

# Lista de columnas con datos numéricos
item_list = [
    col for col in df.columns if df[col].dtype in ['float64', 'int64']]
# Lista de orígenes únicos en el dataframe
origin_list = list(df['Origin'].unique())

# Crear una nueva columna 'YYYY' con el año extraído de la columna 'Year'
df['YYYY'] = df['Year'].apply(lambda x: x.year)
min_year = df['YYYY'].min().item()  # Año mínimo en los datos
max_year = df['YYYY'].max().item()  # Año máximo en los datos

# Diseño de la barra lateral
st.sidebar.image(imagen, width=200)
st.sidebar.markdown('###')
st.sidebar.markdown("### *Configuraciones*")
start_year, end_year = st.sidebar.slider(
    "Periodo",
    min_value=min_year, max_value=max_year,
    value=(min_year, max_year))

st.sidebar.markdown('###')
origins = st.sidebar.multiselect('Orígenes', origin_list,
                                 default=origin_list)
st.sidebar.markdown('###')
item1 = st.sidebar.selectbox('Elemento 1', item_list, index=0)
item2 = st.sidebar.selectbox('Elemento 2', item_list, index=3)

# Filtrar datos según el rango de años seleccionado y los orígenes seleccionados
df_rng = df[(df['YYYY'] >= start_year) & (df['YYYY'] <= end_year)]
source = df_rng[df_rng['Origin'].isin(origins)]

# Crear un objeto base para los gráficos Altair
base = alt.Chart(source).properties(height=300)

# Gráfico de barras para contar el número de registros por origen
bar = base.mark_bar().encode(
    x=alt.X('count(Origin):Q', title='Número de Registros'),
    y=alt.Y('Origin:N', title='Origen'),
    color=alt.Color('Origin:N', legend=None)
)

# Gráfico de dispersión para comparar dos elementos seleccionados
point = base.mark_circle(size=50).encode(
    x=alt.X(item1 + ':Q', title=item1),
    y=alt.Y(item2 + ':Q', title=item2),
    color=alt.Color('Origin:N', title='',
                    legend=alt.Legend(orient='bottom-left'))
)

# Gráfico de líneas para mostrar el promedio mensual del primer elemento seleccionado
line1 = base.mark_line(size=5).encode(
    x=alt.X('yearmonth(Year):T', title='Fecha'),
    y=alt.Y('mean(' + item1 + '):Q', title=item1),
    color=alt.Color('Origin:N', title='',
                    legend=alt.Legend(orient='bottom-left'))
)

# Gráfico de líneas para mostrar el promedio mensual del segundo elemento seleccionado
line2 = base.mark_line(size=5).encode(
    x=alt.X('yearmonth(Year):T', title='Fecha'),
    y=alt.Y('mean(' + item2 + '):Q', title=item2),
    color=alt.Color('Origin:N', title='',
                    legend=alt.Legend(orient='bottom-left'))
)

# Diseño del contenido principal dividido en dos columnas
left_column, right_column = st.columns(2)

# Mostrar los gráficos en las respectivas columnas
left_column.markdown(
    '**Número de Registros (' + str(start_year) + '-' + str(end_year) + ')**')
right_column.markdown(
    '**Diagrama de Dispersión de _' + item1 + '_ y _' + item2 + '_**')
left_column.altair_chart(bar, use_container_width=True)
right_column.altair_chart(point, use_container_width=True)

left_column.markdown('**_' + item1 + '_ (Promedio Mensual)**')
right_column.markdown('**_' + item2 + '_ (Promedio Mensual)**')
left_column.altair_chart(line1, use_container_width=True)
right_column.altair_chart(line2, use_container_width=True)

# Ocultar el estilo predeterminado de Streamlit
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Mensaje de crédito en el pie de página
st.markdown("""
      Realizado por Rudy Manzaneda - 2024
""")
