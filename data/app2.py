import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
df = pd.read_csv("LibroEncuestaLimpia.csv", encoding='latin-1', delimiter=';')

# Configuración de la página de Streamlit
st.set_page_config(page_title='Análisis de Productos Financieros', layout='wide')

# Título y descripción
st.title('Análisis de Uso de Productos Financieros')
st.write('Esta aplicación analiza el uso de productos financieros basado en datos demográficos.')

# Filtros en la barra lateral
st.sidebar.header('Filtros')
genero = st.sidebar.multiselect('Género', options=df['Genero'].unique(), default=df['Genero'].unique())
distrito = st.sidebar.multiselect('Distrito', options=df['distrito'].unique(), default=df['distrito'].unique())
rango_edad = st.sidebar.multiselect('Rango de Edad', options=df['rango_edad'].unique(), default=df['rango_edad'].unique())
nivel_educativo = st.sidebar.multiselect('Nivel Educativo', options=df['nivel_educativo'].unique(), default=df['nivel_educativo'].unique())
situacion_laboral = st.sidebar.multiselect('Situación Laboral', options=df['situacion_laboral'].unique(), default=df['situacion_laboral'].unique())

# Filtrar los datos
filtered_data = df[(df['Genero'].isin(genero)) & 
                   (df['distrito'].isin(distrito)) & 
                   (df['rango_edad'].isin(rango_edad)) & 
                   (df['nivel_educativo'].isin(nivel_educativo)) & 
                   (df['situacion_laboral'].isin(situacion_laboral))]

# Calcular métricas
total_registros = len(filtered_data)
edad_promedio = int(filtered_data['rango_edad'].str.extract('(\d+)').astype(int).mean())
edad_promedio_mujeres = int(filtered_data[filtered_data['Genero'] == 'Femenino']['rango_edad'].str.extract('(\d+)').astype(int).mean())

nivel_socioeconomico_mas_comun = filtered_data['nivel_socioeconomico'].mode()[0]
num_varones = filtered_data[filtered_data['Genero'] == 'Masculino'].shape[0]
num_mujeres = filtered_data[filtered_data['Genero'] == 'Femenino'].shape[0]

# Mostrar métricas
st.header('Métricas del Conjunto de Datos Filtrados')
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Registros", total_registros)
col2.metric("👦 Varones", num_varones, delta=f"Edad Promedio: {edad_promedio}")
col3.metric("👧 Mujeres", num_mujeres, delta=f"Edad Promedio: {edad_promedio_mujeres}")

col4.metric("Nivel Socioeconómico Medio", nivel_socioeconomico_mas_comun)

# Visualización de Datos
st.header('Visualización de Datos')

# Crear dos columnas para las gráficas
col1, col2 = st.columns(2)

# Gráfico de barras: Uso de Tarjeta de Débito
fig_bar_tarjeta_debito = px.histogram(filtered_data, x='Tarjeto de Debito', title='Uso de Tarjeta de Débito')
col1.plotly_chart(fig_bar_tarjeta_debito)

# Gráfico de pastel: Uso de Tarjeta de Débito
fig_pie_tarjeta_debito = px.pie(filtered_data, names='Tarjeto de Debito', title='Distribución del Uso de Tarjeta de Débito')
col2.plotly_chart(fig_pie_tarjeta_debito)

# Gráfico de barras: Plan de Presupuesto Familiar
fig_bar_presupuesto = px.histogram(filtered_data, x='plan_presupuesto_familiar', title='Plan de Presupuesto Familiar')
col1.plotly_chart(fig_bar_presupuesto)

# Gráfico de pastel: Plan de Presupuesto Familiar
fig_pie_presupuesto = px.pie(filtered_data, names='plan_presupuesto_familiar', title='Distribución del Plan de Presupuesto Familiar')
col2.plotly_chart(fig_pie_presupuesto)


# Histograma de Edades por Género
fig_histograma_edades = px.histogram(filtered_data, x='rango_edad', color='Genero', barmode='overlay', title='Histograma de Edades por Género')
col1.plotly_chart(fig_histograma_edades)

# Gráfico de Barras Apiladas por Género y Uso de Productos Financieros
fig_bar_productos_genero = px.bar(filtered_data, x='variedad_productos_financieros', color='Genero', title='Uso de Productos Financieros por Género', barmode='group')
col2.plotly_chart(fig_bar_productos_genero)

# Gráfico de Pastel de Distribución de Nivel Educativo por Género
fig_pie_nivel_educativo = px.pie(filtered_data, names='nivel_educativo', color='Genero', title='Distribución de Nivel Educativo por Género')
st.plotly_chart(fig_pie_nivel_educativo)




# Título y descripción
st.title('Visualización de Datos con Treemap')

st.subheader("🌳 Mapa de Arbol por Jerarquias")
			##df2 = px.data.gapminder().query("year == 2007")
			##st.write(df2)
			#fig = px.treemap(df2, path=[px.Constant('world'), 'continent', 'country'], values='pop',
			#			color='lifeExp', hover_data=['iso_alpha'])
			##st.write(dfi)
column_list = df.columns.tolist()
column_list.remove('ID')
		# Dropdown para seleccionar la columna
c1a, c1b = st.columns([2, 2])
with c1a:
	selected_columns_a = st.multiselect("Selecciona una columna (A):", column_list, placeholder="Seleccione las variables de Análisis",default=[column_list[0]])
		
	available_columns_b = [col for col in column_list if col not in selected_columns_a]
 
# En el segundo contenedor, permite seleccionar una columna (B) de las columnas disponibles
with c1b:
	selected_columns_b = st.multiselect("Selecciona una columna (B):",
                                    available_columns_b,
                                    default=[available_columns_b[0]],
                                    placeholder="Seleccione las variables de Análisis")
	encoder=LabelEncoder()
	data=df.copy()
	data = data.drop('ID', axis=1)
	for columna in selected_columns_a:
		data[columna]=encoder.fit_transform(data[columna])
			
	data = data.select_dtypes(include='number')
		
		
	correlation_matrix = data.corr()

			##df2 = px.data.gapminder().query("year == 2007")
			##st.write(df2)
			#fig = px.treemap(df2, path=[px.Constant('world'), 'continent', 'country'], values='pop',
			#			color='lifeExp', hover_data=['iso_alpha'])
if df[selected_columns_b].notnull().any().all():
	dfi=df.copy()
	for columna in selected_columns_b:
		dfi[columna] = encoder.fit_transform(dfi[columna]) + 1
		fig = px.treemap(dfi, path=selected_columns_a, values=columna,
							color=columna, hover_data=columna,color_continuous_scale='RdYlBu')
st.plotly_chart(fig,use_container_width=True)



# Análisis Detallado
st.header('Análisis Detallado')
# Gráfico de barras apiladas para comparar el uso de productos financieros por género
fig_bar_productos_genero = px.bar(filtered_data, x='Genero', y='variedad_productos_financieros', color='Genero', 
                                  title='Uso de Productos Financieros por Género', barmode='group')
st.plotly_chart(fig_bar_productos_genero)

# Gráfico de barras apiladas para comparar el uso de productos financieros por nivel educativo
fig_bar_productos_educativo = px.bar(filtered_data, x='nivel_educativo', y='variedad_productos_financieros', color='nivel_educativo', 
                                      title='Uso de Productos Financieros por Nivel Educativo', barmode='group')
st.plotly_chart(fig_bar_productos_educativo)

# Gráfico de barras apiladas para comparar el uso de productos financieros por situación laboral
fig_bar_productos_laboral = px.bar(filtered_data, x='situacion_laboral', y='variedad_productos_financieros', color='situacion_laboral', 
                                      title='Uso de Productos Financieros por Situación Laboral', barmode='group')
st.plotly_chart(fig_bar_productos_laboral)



# Ejecuta la aplicación con `streamlit run nombre_del_archivo.py`
