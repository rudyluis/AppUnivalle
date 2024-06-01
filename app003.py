import altair as alt
import streamlit as st
from vega_datasets import data

imagen = "logos/LogoCar.png"
st.set_page_config(page_title="CARS DATA",
					page_icon=imagen,
					layout="wide",
					initial_sidebar_state="auto"
					)
# Datos
df = data.cars()
st.title("Dashboard")
st.write(df)
item_list = [
    col for col in df.columns if df[col].dtype in ['float64', 'int64']]
origin_list = list(df['Origin'].unique())

df['YYYY'] = df['Year'].apply(lambda x: x.year)
min_year = df['YYYY'].min().item()
max_year = df['YYYY'].max().item()

# Diseño (Barra Lateral)
st.sidebar.image(imagen,  width=200)
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

df_rng = df[(df['YYYY'] >= start_year) & (df['YYYY'] <= end_year)]
source = df_rng[df_rng['Origin'].isin(origins)]

# Contenido
base = alt.Chart(source).properties(height=300)

bar = base.mark_bar().encode(
    x=alt.X('count(Origin):Q', title='Número de Registros'),
    y=alt.Y('Origin:N', title='Origen'),
    color=alt.Color('Origin:N', legend=None)
)

point = base.mark_circle(size=50).encode(
    x=alt.X(item1 + ':Q', title=item1),
    y=alt.Y(item2 + ':Q', title=item2),
    color=alt.Color('Origin:N', title='',
                    legend=alt.Legend(orient='bottom-left'))
)

line1 = base.mark_line(size=5).encode(
    x=alt.X('yearmonth(Year):T', title='Fecha'),
    y=alt.Y('mean(' + item1 + '):Q', title=item1),
    color=alt.Color('Origin:N', title='',
                    legend=alt.Legend(orient='bottom-left'))
)

line2 = base.mark_line(size=5).encode(
    x=alt.X('yearmonth(Year):T', title='Fecha'),
    y=alt.Y('mean(' + item2 + '):Q', title=item2),
    color=alt.Color('Origin:N', title='',
                    legend=alt.Legend(orient='bottom-left'))
)

# Diseño (Contenido)
left_column, right_column = st.columns(2)

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

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
				<style>
				#MainMenu {visibility: hidden;}
				footer {visibility: hidden;}
				header {visibility: hidden;}
				</style>
				"""
st.markdown(hide_st_style, unsafe_allow_html=True)		
st.markdown("""
	  Realizado por Rudy Manzaneda - 2024
""")