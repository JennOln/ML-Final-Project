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
        # USAMOS TU FUNCIÓN DE MODELS.PY
        params = models.train_step(params, X_scaled, y, learning_rate)

        if epoch % 10 == 0:
            loss = models.loss_linear(params, X_scaled, y)
            print(f"Época {epoch}: Error (Loss) = {loss:.6f}")

    return params

if __name__ == "__main__":
    final_trained_params = run_pipeline()