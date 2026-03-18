import jax
import jax.numpy as jnp
import jax.random as jrand
from data_processing import booksData
import models  
import mlflow
import mlflow.pyfunc

#Modelo identifica los libros que tienen potencial de ser virales".

def calcular_metricas(y_true, y_pred):
    # Aseguramos que sean enteros para las comparaciones lógicas
    y_true = y_true.astype(jnp.int32)
    y_pred = y_pred.astype(jnp.int32)

    # Cálculo de la matriz de confusión manual
    tp = jnp.sum((y_true == 1) & (y_pred == 1))
    fp = jnp.sum((y_true == 0) & (y_pred == 1))
    fn = jnp.sum((y_true == 1) & (y_pred == 0))
    tn = jnp.sum((y_true == 0) & (y_pred == 0))

    # Métricas derivadas
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-8)  # 1e-8 evita división por cero
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "acc": accuracy * 100,
        "pre": precision,
        "rec": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def run_pipeline():
    mlflow.set_experiment("ML_Final_Project_Elite_Jennifer")

    with mlflow.start_run():
        # Preparar Datos
        data = booksData('kindle_data-v2.csv')
        data.preprocess_data()
        data.extract_features_target()
        X_scaled = data.normalized_data()
        y = jnp.array(data.y)

        #Registrar Parámetros
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("epochs", 100)
        mlflow.log_param("n_features", X_scaled.shape[1])
        mlflow.log_param("umbral_elite", 365.0)

        # Inicializar Parámetros
        key = jrand.PRNGKey(42)
        n_features = X_scaled.shape[1]
        W = jrand.normal(key, (n_features,))
        b = 0.0
        params = (W, b)
        learning_rate = 0.05
        epochs = 100

        # --- ENTRENAMIENTO LINEAL ---
        print(f"\n--- Training with {n_features} features ---")
        for epoch in range(epochs):
            params = models.train_step(params, X_scaled, y, learning_rate)
        
        W_final_lin, b_final_lin = params
        y_pred_linear = models.linear_class(W_final_lin, b_final_lin, X_scaled)
        m_lin = calcular_metricas(y, y_pred_linear)
        print(f"Linear Model Accuracy: {m_lin['acc']:.2f}%")

        # --- ENTRENAMIENTO LOGÍSTICO ---
        # Reiniciamos parámetros para el logístico o usamos los del lineal como base
        for epoch in range(epochs):
            params = models.train_step_logistic(params, X_scaled, y, learning_rate)
            if epoch % 10 == 0:
                current_loss = models.loss_logistic(params, X_scaled, y)
                print(f"epoch {epoch}: Error (Loss) = {current_loss:.6f}")

        y_pred_logistic = models.classify(params, X_scaled)
        m_log = calcular_metricas(y, y_pred_logistic)
        print(f"--- Logistic Model Accuracy: {m_log['acc']:.2f}% ---")

        # ---GUARDAR ARTEFACTOS ---
        W_final, b_final = params
        # Guardamos W y b para poder usarlos en el oráculo/frontend
        jnp.savez("final_weights.npz", W=W_final, b=b_final)
        mlflow.log_artifact("final_weights.npz")

        print("\n--- FINAL RESULTS ---")
        print(f"{'METRICS':<20} | {'LINEAL':<15} | {'LOGISTIC':<15}")
        print("-" * 65)
        print(f"{'Accuracy (%)':<20} | {m_lin['acc']:>14.2f}% | {m_log['acc']:>14.2f}%")
        print(f"{'Precision':<20} | {m_lin['pre']:>15.4f} | {m_log['pre']:>15.4f}")
        print(f"{'Recall':<20} | {m_lin['rec']:>15.4f} | {m_log['rec']:>15.4f}")
        print(f"{'F1-Score':<20} | {m_lin['f1']:>15.4f} | {m_log['f1']:>15.4f}")

        #Confusion Matrix
        print(f"\nLINEAL CLASSIFIER")
        print(f"  (TN): {int(m_lin['tn'])}")
        print(f"  (TP): {int(m_lin['tp'])}")
        print(f"  (FP): {int(m_lin['fp'])}  <-- Libros normales confundidos con Élite")
        print(f"  (FN): {int(m_lin['fn'])}")

        print(f"\nLOGISTIC CLASSIFIER")
        print(f"  (TN): {int(m_log['tn'])}")
        print(f"  (TP): {int(m_log['tp'])}")
        print(f"  (FP): {int(m_log['fp'])}  <-- Libros normales confundidos con Élite")
        print(f"  (FN): {int(m_log['fn'])}")

        return params, data

def predict_interactive(params, data_obj):
    """ Función para que el usuario ingrese datos y el modelo categorice """
    W, b = params
    print("---Prediction oracle---")

    try:
        # Pedir inputs al usuario
        stars = float(input("Rating (1-5 estrellas): "))
        price = float(input("Precio del libro ($): "))
        print(f"Géneros disponibles: {data_obj.features[2:-3]}") # Muestra los géneros del One-Hot
        genre_choice = input("Escribe el género (ej. genre_Romance): ")
        ku = int(input("¿Está en Kindle Unlimited? (1: Sí, 0: No): "))
        pick = int(input("¿Es Editor's Pick? (1: Sí, 0: No): "))
        choice = int(input("¿Es Goodreads Choice? (1: Sí, 0: No): "))
        # Construir el vector X_input con el orden correcto
        # Inicializamos con ceros según el número de features (ej. 16)
        x_raw = [0.0] * len(data_obj.features)
       
        # Valores numéricos
        x_raw[0] = stars
        x_raw[1] = price
         
        # Buscamos el índice del género elegido y ponemos un 1
        if genre_choice in data_obj.features:
            idx = data_obj.features.index(genre_choice)
            x_raw[idx] = 1.0
            
        # Booleanos (las últimas 3 posiciones)
        x_raw[-3] = float(ku)
        x_raw[-2] = float(pick)
        x_raw[-1] = float(choice)

        # Normalizar usando los parámetros GUARDADOS en el entrenamiento
        x_jax = jnp.array(x_raw)
        x_scaled = (x_jax - data_obj.mu_train) / (data_obj.sigma_train + 1e-8)
        x_scaled = jnp.clip(x_scaled, -4.0, 4.0)
        # Inferencia con el modelo Logístico (Sigmoide)
        z = jnp.dot(x_scaled, W) + b
        probabilidad = 1 / (1 + jnp.exp(-z))

        clase = "ÉLITE" if probabilidad > 0.5 else "NORMAL"
        print(f"Predicción del Modelo: {clase}")
        print(f"Probabilidad de éxito: {probabilidad*100:.2f}%")

    except Exception as e:
        print(f"Error en la entrada de datos: {e}")

if __name__ == "__main__":
    final_params, trained_data = run_pipeline()
    
    # Iniciamos el modo interactivo
    while True:
        opcion = input("\n¿Deseas probar un libro nuevo? (s/n): ")
        if opcion.lower() == 's':
            predict_interactive(final_params, trained_data)
        else:
            print("Bye!")
            break