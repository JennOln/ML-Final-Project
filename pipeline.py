import jax
import jax.numpy as jnp
import jax.random as jrand
from data_processing import booksData
import models  

def run_pipeline():
    #Preparar Datos
    data = booksData('kindle_data-v2.csv')
    data.preprocess_data()
    data.extract_features_target()
    X_scaled = data.normalized_data()
    y = jnp.array(data.y) # Aseguramos que y sea un array de JAX

    # Inicializar Parámetros
    key = jrand.PRNGKey(42)
    n_features = X_scaled.shape[1]
    W = jrand.normal(key, (n_features,))
    b = 0.0
    params = (W, b) # Importante: params es una tupla

    learning_rate = 0.05 # Un lr un poco más bajo suele ser más estable
    epochs = 100

    print(f"\n--- Entrenando con {n_features} features ---")
    for epoch in range(epochs):
        params = models.train_step(params, X_scaled, y, learning_rate)

        if epoch % 10 == 0:
            loss = models.loss_linear(params, X_scaled, y)
            print(f"Época {epoch}: Error (Loss) = {loss:.6f}")


    # Usamos los pesos finales para calcular la predicción de todo el dataset
    W_final, b_final = params
    y_pred = models.linear_class(W_final, b_final, X_scaled)
    y_pred_logistic = models.logistic_class(W_final, b_final, X_scaled)

    # Comparamos posición por posición cuántos son iguales
    
    comparacion = (y_pred == y)
    comparacion_logistic = (y_pred_logistic == y)
          
    accuracy = jnp.mean(comparacion) * 100
    accuracy_logistic = jnp.mean(comparacion_logistic) * 100
    print(f"Precisión del modelo lineal: {accuracy:.2f}%")
    print(f"Precisión del modelo logístico: {accuracy_logistic:.2f}%")

    print("\n--- Diagnóstico de Predicciones ---")
    print(f"Total de libros en el dataset: {len(y)}")
    print(f"Cuántos predijo como Best Seller (1): {jnp.sum(y_pred == 1)}")
    print(f"Cuántos predijo como Normal (0): {jnp.sum(y_pred == 0)}")
    print(f"Cuántos SON Best Seller en realidad: {jnp.sum(y == 1)}")

    print(f"Valor MÁXIMO en X_scaled: {jnp.max(X_scaled)}")
    print(f"Valor MÍNIMO en X_scaled: {jnp.min(X_scaled)}")
    return params

    

if __name__ == "__main__":
    final_trained_params = run_pipeline()