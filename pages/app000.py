import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
from sklearn.preprocessing import LabelEncoder
from pandas import set_option
from st_aggrid import AgGrid
import plotly.express as px
##import pygwalker as pyg 
import streamlit.components.v1 as stc 
import seaborn as sns
import matplotlib.pyplot as plt
import random
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
# DB Management
import sqlite3 
##conn = sqlite3.connect('data.db')
##c = conn.cursor()

color_discrete_sequence=['#FF9999', '#99CCFF', '#C2DFFF', '#FFD966', '#FFB6C1', '#C1FFC1', '#FFFF99', '#B0C4DE', '#FFD700']

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib

def agregar_borde_html(fig):
    return f"""
    <div style="border: 1px solid black; padding: 10px;">
        {fig.to_html(full_html=False)}
    </div>
    """
def pie(df,  names,titulografica):
	theme_plotly = None # None or streamlit

	fig = px.pie(df, names=names, title=titulografica,color_discrete_sequence=color_discrete_sequence)
	fig.update_layout(legend_title="Serie", legend_y=0.9)
	fig.update_traces(textinfo='percent+label', textposition='inside')
	st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

def barchart(df,val_x,val_y,titulografica):
	theme_plotly = None # None or streamlit
	fig = px.bar(df, y=val_y, x=val_x, text_auto='.2s',title=titulografica)
	fig.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
	st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def metrics(df):
	col1, col2, col3 = st.columns(3)

	col1.metric(label="🗂️ Total Registros", value=df.ID.count(), delta="Todos los registros")
	num_varones = df[df['Genero'] == 'Masculino'].shape[0]

	col2.metric(label="👦 Varones", value= num_varones,delta="Edad Promedio")

	num_mujeres = df[df['Genero'] == 'Femenino'].shape[0]

	col3.metric(label="👩 Mujeres", value= num_mujeres,delta="Edad Promedio")

	##col3.metric(label="Annual Salary", value= f"{ df_selection.AnnualSalary.max()-df.AnnualSalary.min():,.0f}",delta="Annual Salary Range")

	style_metric_cards(background_color="#fff2cc",border_left_color="#000000",box_shadow="3px")

def metric_m(df,selected_column):

	counts = df[selected_column].value_counts()
					# Convertir la Serie a un DataFrame
	df_counts = pd.DataFrame(counts).reset_index()

	df_counts.columns = ['Valor', 'Frecuencia']
	total = df_counts['Frecuencia'].sum()

			# Calcular los porcentajes y agregarlos como una nueva columna al DataFrame
	df_counts['Porcentaje'] = (df_counts['Frecuencia'] / total) * 100


	mean_value = round(df_counts['Frecuencia'].mean(),2)
	median_value = round(df_counts['Frecuencia'].median(),2)
	std_deviation = round(df_counts['Frecuencia'].std(),2)
	min_value = round(df_counts['Frecuencia'].min(),2)
	max_value = round(df_counts['Frecuencia'].max(),2)
	col1, col2, col3 = st.columns(3)
	col1.metric(label="✏️ Media", value=mean_value, delta="Min:"+str(min_value))
	

	col2.metric(label="🖇️ Mediana", value= median_value,delta="")

	
	col3.metric(label="📋 Desviacion Estandar", value= std_deviation,delta="Max:"+str(max_value))

	##col3.metric(label="Annual Salary", value= f"{ df_selection.AnnualSalary.max()-df.AnnualSalary.min():,.0f}",delta="Annual Salary Range")

	style_metric_cards(background_color="#fff2cc",border_left_color="#000000",box_shadow="3px")




def main():

	imagen = "logos/logoBien.png"
	st.set_page_config(page_title="Encuestas Bienestar",
					page_icon=imagen,
					layout="wide",
					initial_sidebar_state="auto"
					)
	
	# Mostrar la imagen en la barra lateral
	st.sidebar.image(imagen,  width=200)	
	st.subheader("📈 Encuesta de Bienestar Financiero a Hogares")
	with open('style.css')as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)	
	df = pd.read_csv("df/LibroEncuestaLimpia.csv",encoding='latin-1',delimiter=';')
	menu = ["Inicio","Analisis Exploratorio","Matriz de Correlacion"]

	iconos=["house","book","book","cast"],
	with st.sidebar:
			choice=option_menu(
			menu_title="Menu",
			#menu_title=None,
			options=menu,
			icons=iconos,
			menu_icon="cast", #option
			default_index=0, #option
			orientation="vertical", )
 


	if choice == "Analisis Dinamico":
		st.subheader("📈 Principal")
		# Mostrar el DataFrame
		##st.write("Contenido del archivo CSV:")
		##st.write(df)
		metrics(df)
		st.write("Contenido del archivo CSV:")
		##pyg_html = pyg.walk(df,return_html=True)
            # Render with components
		##stc.html(pyg_html,scrolling=True,height=1000)
		

	elif choice == "Inicio":
		st.subheader('📃 Resultados')
		
		column_list = df.columns.tolist()
		column_list.remove('ID')
		# Dropdown para seleccionar la columna
		selected_column = st.selectbox("Selecciona una columna:", column_list)
		metric_m(df,selected_column)
		colA,colB,colC=st.columns(3)
		with colA:
			st.subheader('🖼️ Grafica de Pie')
			pie(df, selected_column,f'Distribucion de {selected_column}')
		with colB:
			##barchart(df,selected_column,selected_column,'Grafica de Barras')
			st.subheader('📃 Grafica de Barras')
			if(selected_column=='menores_edad' or selected_column=='mayores_edad'):
				colores_personalizados = {
					'Ninguno': '#4B0082',
					'1': '#FF00FF',
					'2':'#99CCFF', 
					'3 o más':'#FFD700'
				}
				counts = df[selected_column].value_counts()
				# Convertir la Serie a un DataFrame
				df_counts = pd.DataFrame(counts).reset_index()

				# Renombrar las columnas
				df_counts.columns = ['Valor', 'Frecuencia']
				chart = alt.Chart(df_counts).mark_bar().encode(
					x=alt.X('Valor:N', title='Categoria', axis=alt.Axis(labelAngle=0, labelAlign='center',  titleFontSize=20, labelFontSize=12)),
					y=alt.Y('Frecuencia:Q', title='Cantidad', axis=alt.Axis(labelAngle=0, labelAlign='center', titleFontSize=20, labelFontSize=12)),  # Corregir aquí
					color=alt.Color('Valor:N', title='Categoria', scale=alt.Scale(domain=list(colores_personalizados.keys()), range=list(colores_personalizados.values()))),
					tooltip=['Valor:N', 'Frecuencia:Q']
				).properties(
					title=alt.Title(text=selected_column, fontSize=16)
				)
				st.altair_chart(chart, use_container_width=True)
			else:
				fig = px.histogram(df, x=selected_column, title=f'Distribución por {selected_column}',color_discrete_sequence=[color_discrete_sequence[random.randint(1, 8)]])
				st.plotly_chart(fig,use_container_width=True)
		with colC:
			st.subheader('💻 Tabla de Datos')
			counts = df[selected_column].value_counts()
					# Convertir la Serie a un DataFrame
			df_counts = pd.DataFrame(counts).reset_index()

			df_counts.columns = ['Valor', 'Frecuencia']
			total = df_counts['Frecuencia'].sum()

			# Calcular los porcentajes y agregarlos como una nueva columna al DataFrame
			df_counts['Porcentaje'] = (df_counts['Frecuencia'] / total) * 100

			# Mostrar el DataFrame actualizado en Streamlit
			
			st.write(df_counts)
		st.divider()
		### grafica de tipo de tarjeta
		cols_a_apilar = df.columns[9:15]
		df[cols_a_apilar] = df[cols_a_apilar].replace({'VERDADERO': 1, 'FALSO': 0})
		##st.write(cols_a_apilar)
		# Utilizar melt solo para esas columnas
		counts = df[cols_a_apilar].sum()

    # Crear el gráfico de barras con Plotly Express
		fig = px.bar(x=counts.index, y=counts.values, title='¿Ha utilizado productos financieros como tarjeta de crédito, tarjetas de débito, depósitos en cuentas de ahorro/corriente, préstamos personales, compras en cuotas con tarjeta de Crédito o depósito',color_discrete_sequence=[color_discrete_sequence[random.randint(1, 8)]])

    # Personalizar el gráfico
		fig.update_layout(xaxis_title='Uso de Productos Financieros', yaxis_title='Respuestas')

    # Mostrar el gráfico en Streamlit
		##st.plotly_chart(fig,use_container_width=True)
		
		
	elif choice == "Analisis Exploratorio":
		column_list = df.columns.tolist()
		column_list.remove('ID')

		# Interfaz de Streamlit
		st.subheader("💰 Análisis Exploratorio de Datos (EDA)")
		c1a, c1b = st.columns([2, 2])
		with c1a:
			selected_columns_a = st.selectbox("Selecciona una columna (A):", column_list, placeholder="Seleccione las variables de Análisis")

		available_columns_b = [col for col in column_list if col not in selected_columns_a]

# En el segundo contenedor, permite seleccionar una columna (B) de las columnas disponibles
		with c1b:
			selected_columns_b = st.selectbox("Selecciona una columna (B):",
                                  available_columns_b,
                                  placeholder="Seleccione las variables de Análisis")

		# Dropdown para seleccionar la columna
		##selected_column = st.selectbox("Selecciona una columna:", column_list)
	
		summary_a = df[selected_columns_a].describe()
		summary_b = df[selected_columns_b].describe()

			# Concatenar los resúmenes estadísticos
		summary_concatenated = pd.concat([summary_a, summary_b], axis=1)

		st.write("📊 Valores de " +selected_columns_a)
		col1, col2, col3, col4 = st.columns(4)
		col1.metric(label="✏️ Total Registros", value=summary_concatenated[selected_columns_a]['count'])
		col2.metric(label="🖇️ Unicos", value= summary_concatenated[selected_columns_a]['unique'])
		col3.metric(label="📋 Mas recurrente", value= summary_concatenated[selected_columns_a]['top'])
		col4.metric(label="📋 Frecuencia", value= summary_concatenated[selected_columns_a]['freq'])
		style_metric_cards(background_color="#fff2cc",border_left_color="#000000",box_shadow="3px")
		st.write("📊 Valores de " +selected_columns_b)
		col1, col2, col3, col4 = st.columns(4)
		col1.metric(label="✏️ Total Registros", value=summary_concatenated[selected_columns_b]['count'])
		col2.metric(label="🖇️ Unicos", value= summary_concatenated[selected_columns_b]['unique'])
		col3.metric(label="📋 Mas recurrente", value= summary_concatenated[selected_columns_b]['top'])
		col4.metric(label="📋 Frecuencia", value= summary_concatenated[selected_columns_b]['freq'])
		style_metric_cards(background_color="#fff2cc",border_left_color="#000000",box_shadow="3px")


		##st.write("📈 Visualización de la distribución de una variable categórica")
				
		# Convertir la Serie a un DataFrame
		df_filtered = df.loc[:, [selected_columns_a, selected_columns_b]]
		# Renombrar las columnas
		
		valores_unicos = df_filtered[selected_columns_a].unique().tolist()

		# Agregar "Todos" al principio de la lista de valores únicos
		valores_unicos.insert(0, "Todos")

		# Crear un selectbox con la opción "Todos"
		variable_c = st.selectbox("Selecciona un valor (C):", valores_unicos, placeholder="Seleccione ")



		c1, c2 = st.columns(2)
		encoder=LabelEncoder()

		
		df_filtered[selected_columns_b]=encoder.fit_transform(df_filtered[selected_columns_b])+1
		if variable_c !="Todos":
			df_filtered = df_filtered[df_filtered[selected_columns_a] == variable_c]

		with c1:
			st.write('📉 Grafica de Barras')
			fig9 = px.bar(df_filtered, x=selected_columns_a, y=selected_columns_b, color=selected_columns_a, barmode="group")
			st.plotly_chart(fig9,use_container_width=True)
			st.write('📊 Histograma')

			fig5 = px.histogram(df_filtered, x=selected_columns_a, y=selected_columns_b, color=selected_columns_b, marginal="rug", hover_data=df_filtered.columns)
			st.plotly_chart(fig5,use_container_width=True)
		with c2:
			st.write('📉 Grafico de Cajas')
			fig6 = px.box(df_filtered, x=selected_columns_a, y=selected_columns_b, color=selected_columns_a, notched=True)
			st.plotly_chart(fig6,use_container_width=True)
			st.write('📊 Grafico Distribucion Acumulativa (ECDF) ')
			fig7 = px.ecdf(df_filtered, x=selected_columns_a, color=selected_columns_b)
			st.plotly_chart(fig7,use_container_width=True)


	elif choice == "Matriz de Correlacion":
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
		
		st.subheader("🗞️ Matriz de Correlacion")
		##plt.figure(figsize=(6, 4))
		##sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
		##st.pyplot(plt)
		#3plt.figure(figsize=(4, 2))
		##st.divider()


		fig = px.imshow(correlation_matrix.values,
                labels=dict(color="Correlación"),
                x=correlation_matrix.index,
                y=correlation_matrix.columns,
                color_continuous_scale='RdBu',  # Utiliza el colormap RdBu de Plotly Express
                width=800,  # Ancho ajustado
                height=600)  # Altura ajustada
		st.plotly_chart(fig,use_container_width=True)
		st.divider()  # Línea divisoria


		st.subheader("🌳 Matriz Arbol ")
			##df2 = px.data.gapminder().query("year == 2007")
			##st.write(df2)
			#fig = px.treemap(df2, path=[px.Constant('world'), 'continent', 'country'], values='pop',
			#			color='lifeExp', hover_data=['iso_alpha'])
			##st.write(dfi)
		if df[selected_columns_b].notnull().any().all():
			dfi=df.copy()
			for columna in selected_columns_b:
				dfi[columna] = encoder.fit_transform(dfi[columna]) + 1
			fig = px.treemap(dfi, path=selected_columns_a, values=columna,
							color=columna, hover_data=columna,color_continuous_scale='RdYlBu')
			st.plotly_chart(fig,use_container_width=True)

	elif choice == "Modelo Prediccion":
		col1, col2, col3 = st.columns([0.3,0.3,0.4])
		with col1:
		
			genero=st.selectbox('Genero',df['Genero'].unique())
		with col2:
			distrito=st.selectbox('Distrito',df['distrito'].unique())
		with col3:
			nivel_educativo=st.selectbox('Nivel de Educacion',df['nivel_educativo'].unique())

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
if __name__ == '__main__':
    main()