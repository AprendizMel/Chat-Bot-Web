"""esta es la api que conecta el front con el backend"""
from flask import Flask, request, jsonify, render_template
from chat import get_response  # Importa la función del modelo

app = Flask(__name__)

@app.route('/')
def home():
    """Sirve el archivo HTML del frontend."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recibe un mensaje del frontend y devuelve la respuesta del chatbot."""
    data = request.get_json()  # Obtiene el mensaje enviado desde el frontend
    message = data.get("message")  # Extrae el texto del mensaje
    response = get_response(message)  # Llama a la función del chatbot
    return jsonify({"answer": response})  # Envía la respuesta al frontend

if __name__ == "__main__":
    app.run(debug=True)
