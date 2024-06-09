import streamlit as st

imagen = "logos/logoStreamlit.png"
st.set_page_config(page_title="Encuestas Bienestar",
					page_icon=imagen,
					layout="wide",
					initial_sidebar_state="auto"
					)
	
	# Mostrar la imagen en la barra lateral

st.image("logos/logoStreamlit.png")
st.title('PyDay Cochabamba 2024')
st.subheader("📈 Aplicaciones Streamlit")

st.page_link("app.py", label="Home", icon="🏠")
st.page_link("pages/app000.py", label="App Financiero", icon="1️⃣")
st.page_link("pages/app001.py", label="Aplicando Widget", icon="2️⃣")
st.page_link("pages/app002.py", label="App Equipos computacion", icon="3️⃣")
st.page_link("pages/app003.py", label="App Car Data", icon="4️⃣")