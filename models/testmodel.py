import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Carica il modello .tflite
interpreter = tf.lite.Interpreter(model_path="sine_model.tflite")
interpreter.allocate_tensors()

# Ottieni dettagli su input e output del modello
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Crea i dati di input (da 0 a 6.28 con passo di 0.1 radianti)
x_values = np.arange(0, 6.28, 0.1, dtype=np.float32)

# Array per salvare i valori di output del modello
y_predicted = []

# Itera attraverso i valori di input, esegui l'inferenza e salva i risultati
for x in x_values:
    # Prepara l'input per il modello
    input_data = np.array([[x]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Esegui l'inferenza
    interpreter.invoke()

    # Ottieni il risultato dall'output del modello
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Salva il risultato
    y_predicted.append(output_data[0][0])

# Calcola la funzione seno standard per confrontarla
y_true = np.sin(x_values)

# Salva i risultati in un file CSV
results = pd.DataFrame({
    'Input (radians)': x_values,
    'Predicted Sine': y_predicted,
    'True Sine': y_true
})
results.to_csv('sine_predictions_comparison.csv', index=False)

# Esegui il plot dei risultati
plt.plot(x_values, y_predicted, label='Predicted Sine', linestyle='--', color='blue')
plt.plot(x_values, y_true, label='True Sine', linestyle='-', color='red')
plt.xlabel('Input (radians)')
plt.ylabel('Output (sine)')
plt.title('Predicted vs True Sine Function')
plt.legend()
plt.grid(True)
plt.show()
