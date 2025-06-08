import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Estilo general: fondo negro o azul marino, texto blanco, inputs y contenedores también oscuros
st.markdown(
    """
    <style>
    /* Asegura que el body completo y el HTML tengan el color de fondo */
    html, body {
        background-color: #001f3f !important; /* azul marino oscuro */
        color: white !important;
    }

    /* El contenedor principal de Streamlit */
    .stApp {
        background-color: #001f3f !important;
        color: white !important;
    }

    /* El contenedor de bloques principal de contenido */
    .block-container {
        background-color: #001f3f !important;
        color: white !important;
    }

    /* Cambiar los contenedores de los widgets (inputs, selectbox, botones) */
    .css-1d391kg, .css-1v3fvcr, .css-14xtw13 { /* Estos son selectores comunes, pueden variar ligeramente entre versiones de Streamlit */
        background-color: #001f3f !important;
        color: white !important;
    }
    
    /* Para los elementos de texto dentro de los contenedores que podrían no heredar */
    .stMarkdown, .stText, .stJson {
        color: white !important;
    }

    /* Inputs, selects, textareas y botones con fondo oscuro y texto blanco */
    input, select, textarea, button {
        background-color: #001f3f !important;
        color: white !important;
        border-color: #444 !important;
    }
    
    /* Estilo específico para los selectbox que a veces no toman el color de fondo */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #001f3f !important;
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] div[role="button"] {
        background-color: #001f3f !important;
        color: white !important;
        border-color: #444 !important;
    }


    /* Caja que muestra mensajes como st.success, st.error, etc. */
    .stAlert {
        background-color: #003366 !important; /* Un azul un poco más claro para los alerts */
        color: white !important;
    }
    
    /* Títulos y textos */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    /* Gráficos (contenedor canvas) */
    .stPlotly, .element-container, .stImage {
        background-color: #001f3f !important;
        color: white !important; /* Asegura que el texto dentro de estos también sea blanco si aplica */
    }

    /* Asegura que el contenedor superior de la barra lateral (si la hubiera) y el menú superior también sean oscuros */
    .css-1lcbmhc, .css-1lcbmhc.e1fqk00q3 { /* selector para la barra lateral principal */
        background-color: #001f3f !important;
        color: white !important;
    }

    /* Para el menú de hamburguesa superior derecho */
    .css-vk32pt.exg6vlp1 { /* Selector para el botón del menú de hamburguesa */
        background-color: #001f3f !important;
        color: white !important;
    }

    /* Target the main content wrapper that has padding */
    .css-18e3th9 { /* This class is often on the main content wrapper */
        background-color: #001f3f !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Columnas que usa el modelo
columnas = [
    "const", "year",
    "estado_Aguascalientes", "estado_Baja California",
    "estado_Baja California Sur", "estado_Campeche", "estado_Chiapas",
    "estado_Chihuahua", "estado_Ciudad de México", "estado_Coahuila de Zaragoza",
    "estado_Colima", "estado_Durango", "estado_Extranjero", "estado_Guanajuato",
    "estado_Guerrero", "estado_Hidalgo", "estado_Jalisco",
    "estado_Michoacán de Ocampo", "estado_Morelos", "estado_México",
    "estado_Nayarit", "estado_Nuevo León", "estado_Oaxaca", "estado_Puebla",
    "estado_Querétaro", "estado_Quintana Roo", "estado_San Luis Potosí",
    "estado_Sinaloa", "estado_Sonora", "estado_Tabasco", "estado_Tamaulipas",
    "estado_Tlaxcala", "estado_Veracruz de Ignacio de la Llave",
    "estado_Yucatán", "estado_Zacatecas"
]

# Carga el modelo entrenado
# Usar st.cache_resource para cargar el modelo una sola vez
@st.cache_resource
def load_model():
    return joblib.load("modelo1.joblib")

modelo = load_model()

st.title("Predicción de nacimientos por estado en los proximos años")

# Inputs
year_input = st.number_input("Año para la predicción", min_value=1900, max_value=2100, value=2023, step=1)
estado_options = [col.replace("estado_", "") for col in columnas if col.startswith("estado_")]
estado_seleccionado = st.selectbox("Estado", estado_options)

if st.button("Predecir y graficar"):
    estado_col = "estado_" + estado_seleccionado

    if estado_col not in columnas:
        st.error(f"Estado no válido: {estado_seleccionado}")
    else:
        # Predicción para el año seleccionado
        fila_pred = pd.DataFrame([[0] * len(columnas)], columns=columnas)
        fila_pred["const"] = 1
        fila_pred["year"] = year_input
        fila_pred[estado_col] = 1

        prediccion = modelo.predict(fila_pred).iloc[0] # Assuming .iloc[0] is correct based on your model's output
        st.success(f"Predicción para {estado_seleccionado} en {year_input}: {prediccion:.2f}")

        # Generar predicciones para rango de años para la gráfica
        año_inicio = 2000
        años = list(range(año_inicio, year_input + 1))
        predicciones = []

        for año in años:
            fila = pd.DataFrame([[0] * len(columnas)], columns=columnas)
            fila["const"] = 1
            fila["year"] = año
            fila[estado_col] = 1
            prediccion_año = modelo.predict(fila).iloc[0] # Assuming .iloc[0] is correct
            predicciones.append(prediccion_año)

        # Graficar con matplotlib
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#001f3f')  # fondo azul marino figura
        ax.set_facecolor('#001f3f')          # fondo azul marino gráfico
        ax.plot(años, predicciones, marker='o', color='white')
        ax.set_title(f"Predicción de nacimientos para {estado_seleccionado}", color='white')
        ax.set_xlabel("Año", color='white')
        ax.set_ylabel("Nacimientos", color='white')
        ax.grid(True, color='gray', linestyle='--', alpha=0.7) # Grid más suave
        ax.tick_params(axis='x', colors='white')  # color de los ticks del eje x
        ax.tick_params(axis='y', colors='white')  # color de los ticks del eje y
        ax.spines['bottom'].set_color('white') # color de las líneas del eje
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('#001f3f') # Esconde la línea superior
        ax.spines['right'].set_color('#001f3f') # Esconde la línea derecha

        st.pyplot(fig)
