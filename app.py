from flask import Flask, request, render_template
import pandas as pd
import joblib  # Cambiado de pickle a joblib

app = Flask(__name__)

# Columnas utilizadas para el modelo (debes haberlas usado igual al entrenar)
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

# Cargar modelo entrenado con joblib
modelo = joblib.load("modelo1.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None
    if request.method == "POST":
        try:
            year = int(request.form.get("year"))
            estado_input = request.form.get("estado")
            estado_col = "estado_" + estado_input

            if estado_col not in columnas:
                error = f"Estado no válido: {estado_input}"
            else:
                # Crear vector de entrada con ceros
                nueva_fila = pd.DataFrame([[0] * len(columnas)], columns=columnas)
                nueva_fila["const"] = 1
                nueva_fila["year"] = year
                nueva_fila[estado_col] = 1

                # Predicción
                prediccion = modelo.predict(nueva_fila).iloc[0]

        except Exception as e:
            error = f"Ocurrió un error: {str(e)}"

    return render_template("index.html", prediccion=prediccion, error=error)

if __name__ == "__main__":
    app.run(debug=True)
