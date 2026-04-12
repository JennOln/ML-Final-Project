import os
import jax
import jax.numpy as jnp
import jax.random as jrand
from data_processing import booksData
import models  
import mlflow
from fastapi import FastAPI, middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np 
import metaflow 

app= FastAPI()

# Configuración de CORS para que el navegador no bloquee las peticiones
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario global para almacenar las métricas y enviarlas al frontend
cache_metrics = {}

# Modelo identifica los libros que tienen potencial de ser virales.
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
    actual_dir = os.getcwd()
    db_path = f"sqlite:///{os.path.join(actual_dir, 'mlflow.db')}"
    mlflow.set_tracking_uri(db_path)
    mlflow.set_experiment("ML_Final_Project_Elite_Jennifer")

    with mlflow.start_run():
        # Prepare data
        data = booksData('kindle_data-v2.csv')
        data.preprocess_data()
        data.extract_features_target()
        X_scaled = data.normalized_data()
        y = jnp.array(data.y)

        # Initialize parameters for 
        key = jrand.PRNGKey(42)
        n_features = X_scaled.shape[1]
        
        W_init = jrand.normal(key, (n_features,))
        b_init = 0.0
        
        # Guardamos la "línea de salida" intacta para reiniciar el modelo
        params_iniciales = (W_init, b_init) 
        
        learning_rate = 0.1  # Subido para mejor convergencia
        epochs = 1000        # Subido para darle tiempo de aprender

        # Registrar Parámetros en MLflow
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("n_features", n_features)

        """LINEAL TRAINING"""
        print(f"\nTraining with {n_features} features")
        print(f"\nTraining Linear Classifier...")
        
        params = params_iniciales # El lineal arranca desde la línea de salida
        for epoch in range(epochs):
            params = models.LinearClassifier.train_step(params, X_scaled, y, learning_rate)
            
        W_final_lin, b_final_lin = params
        y_pred_linear = models.LinearClassifier.linear_class(W_final_lin, b_final_lin, X_scaled)
        m_lin = calcular_metricas(y, y_pred_linear)
        print(f"Linear Model Accuracy: {m_lin['acc']:.2f}%")
        mlflow.log_metric("Linear Model Accuracy", m_lin['acc'])


        """LOGISTIC TRAINING"""
        print(f"\nTraining Logistic Regression...")
        
        params = params_iniciales 
        
        for epoch in range(epochs):
            params = models.LogisticClassifier.train_step(params, X_scaled, y, learning_rate)
            if epoch % 100 == 0:
                current_loss = models.LogisticClassifier.loss_logistic(params, X_scaled, y)
                print(f"epoch {epoch}: Error (Loss) = {current_loss:.6f}")

        print("🔍 BUSCANDO EL UMBRAL ÓPTIMO (F1-SCORE)")
        probs_log = models.LogisticClassifier.predict_logistic(params, X_scaled)
        mejor_f1 = 0.0
        mejor_umbral = 0.25 # Por defecto
        
        for umbral_test in jnp.arange(0.20, 0.52, 0.02):
            y_pred_test = jnp.where(probs_log > umbral_test, 1, 0)
            metricas_test = calcular_metricas(y, y_pred_test)
            if metricas_test['f1'] > mejor_f1:
                mejor_f1 = metricas_test['f1']
                mejor_umbral = float(umbral_test)
        

        """MLP TRAINING"""
        print(f"\nTraining MLP Classifier...")
        
        # Inicializamos los parámetros de la red (Ej. 32 neuronas ocultas)
        key_mlp = jrand.PRNGKey(99)
        params_mlp = models.MLP.init_params(key_mlp, input_dim=n_features, hidden_dim1=64, hidden_dim2=32)
        
        for epoch in range(epochs):
            params_mlp = models.MLP.train_step(params_mlp, X_scaled, y, learning_rate)
            if epoch % 100 == 0:
                loss_mlp = models.MLP.loss_mlp(params_mlp, X_scaled, y)
                print(f"MLP epoch {epoch}: Error (Loss) = {loss_mlp:.6f}")
                
        # Búsqueda de umbral...
        probs_mlp = models.MLP.predict_proba(params_mlp, X_scaled)
        best_f1_mlp = 0.0
        best_threshold_mlp = 0.25

# Modelo identifica los libros que tienen potencial de ser virales.
def calcular_metricas(y_true, y_pred):
    # Convertimos los arreglos de JAX a Numpy estándar de CPU
    y_true_np = np.array(y_true).astype(int)
    y_pred_np = np.array(y_pred).astype(int)

    tp = int(np.sum((y_true_np == 1) & (y_pred_np == 1)))
    fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
    fn = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))
    tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))

    accuracy = (tp + tn) / len(y_true_np)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # IMPORTANTE: Todo debe ser float() o int() nativo de Python
    return {
        "acc": float(accuracy * 100),
        "pre": float(precision),
        "rec": float(recall),
        "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def run_pipeline():
    actual_dir = os.getcwd()
    db_path = f"sqlite:///{os.path.join(actual_dir, 'mlflow.db')}"
    mlflow.set_tracking_uri(db_path)
    mlflow.set_experiment("ML_Final_Project_Elite_Jennifer")

    with mlflow.start_run():
        """Prepare data and params"""
        data = booksData('kindle_data-v2.csv')
        data.preprocess_data()
        data.extract_features_target()
        X_scaled = data.normalized_data()
        y = jnp.array(data.y)

        # Initialize parameters for 
        key = jrand.PRNGKey(42)
        n_features = X_scaled.shape[1]
        W_init = jrand.normal(key, (n_features,))
        b_init = 0.0
        params_iniciales = (W_init, b_init) 
        
        learning_rate = 0.1  
        epochs = 1000

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("n_features", n_features)
        print(f"\nTraining with {n_features} features")

        """
            LINEAL TRAINING...
        """
        print(f"\nTraining Linear Classifier...")
        lineal_params = params_iniciales # El lineal arranca desde la línea de salida
        for epoch in range(epochs):
            lineal_params = models.LinearClassifier.train_step(lineal_params, X_scaled, y, learning_rate)
            
        W_final_lin, b_final_lin = lineal_params
        y_pred_linear = models.LinearClassifier.linear_class(W_final_lin, b_final_lin, X_scaled)
        m_lin = calcular_metricas(y, y_pred_linear)


        """
            LOGISTIC TRAINING
        """
        print(f"\nTraining Logistic Regression...")
        log_params = params_iniciales 
        
        for epoch in range(epochs):
            log_params = models.LogisticClassifier.train_step(log_params, X_scaled, y, learning_rate)
            if epoch % 200 == 0:
                current_loss = models.LogisticClassifier.loss_logistic(log_params, X_scaled, y)
                print(f"epoch {epoch}: Error (Loss) = {current_loss:.6f}")

    
        
        """
            MLP TRAINING
        """
        print(f"\nTraining MLP Classifier...")
        key_mlp = jrand.PRNGKey(99)
        params_mlp = models.MLP.init_params(key_mlp, input_dim=n_features, hidden_dim1=64, hidden_dim2=32)
        
        for epoch in range(epochs):
            params_mlp = models.MLP.train_step(params_mlp, X_scaled, y, learning_rate)
            if epoch % 100 == 0:
                loss_mlp = models.MLP.loss_mlp(params_mlp, X_scaled, y)
                print(f"MLP epoch {epoch}: Error (Loss) = {loss_mlp:.6f}")


        """
            Classificacion Treee
        """ 
        print(f"\n--- Entrenando Árbol de Clasificación (Decision Tree) ---")
        tree_model = models.DecisionTree(max_depth=5)
        tree_model.fit(X_scaled, y)
        
        # Hacemos la predicción directamente (El árbol devuelve 0 o 1, no probabilidades)
        y_pred_tree = tree_model.predict(X_scaled)
        m_tree = calcular_metricas(y, y_pred_tree)


        """
            Search best threshold (F1-SCORE)
        """
        print("\nSearching for optimal thresholds...")
        
        # Para el Logístico
        probs_log = models.LogisticClassifier.predict_logistic(log_params, X_scaled)
        mejor_f1_log = 0.0
        mejor_umbral_log = 0.25 
        for umbral_test in jnp.arange(0.20, 0.52, 0.02):
            y_pred_test = jnp.where(probs_log > umbral_test, 1, 0)
            m_test = calcular_metricas(y, y_pred_test)
            if m_test['f1'] > mejor_f1_log:
                mejor_f1_log = m_test['f1']
                mejor_umbral_log = float(umbral_test)
        
        # Para el MLP
        probs_mlp = models.MLP.predict_proba(params_mlp, X_scaled)
        mejor_f1_mlp = 0.0
        mejor_umbral_mlp = 0.25
        for umbral_test in jnp.arange(0.20, 0.52, 0.02):
            y_pred_test_mlp = jnp.where(probs_mlp > umbral_test, 1, 0)
            m_test_mlp = calcular_metricas(y, y_pred_test_mlp)
            if m_test_mlp['f1'] > mejor_f1_mlp:
                mejor_f1_mlp = m_test_mlp['f1']
                mejor_umbral_mlp = float(umbral_test)

        # Calculamos las métricas finales con los ganadores
        y_pred_log_final = jnp.where(probs_log > mejor_umbral_log, 1, 0)
        m_log = calcular_metricas(y, y_pred_log_final)

        y_pred_mlp_final = jnp.where(probs_mlp > mejor_umbral_mlp, 1, 0)
        m_mlp = calcular_metricas(y, y_pred_mlp_final)

        mlflow.log_param("best_threshold_logistic", mejor_umbral_log)
        mlflow.log_param("best_threshold_mlp", mejor_umbral_mlp)

        W_final, b_final = log_params # Guardamos los pesos logísticos para el oráculo
        jnp.savez("final_weights.npz", W=W_final, b=b_final)
        mlflow.log_artifact("final_weights.npz")
        
        mlflow.log_metrics({
            # Lineal
            "lineal_accuracy": float(m_lin['acc']), "lineal_precision": float(m_lin['pre']),
            "lineal_recall": float(m_lin['rec']), "lineal_f1": float(m_lin['f1']),
            "lineal_TP": float(m_lin['tp']), "lineal_TN": float(m_lin['tn']),
            "lineal_FP": float(m_lin['fp']), "lineal_FN": float(m_lin['fn']),
            # Logístico
            "logistic_accuracy": float(m_log['acc']), "logistic_precision": float(m_log['pre']),
            "logistic_recall": float(m_log['rec']), "logistic_f1": float(m_log['f1']),
            "logistic_TP": float(m_log['tp']), "logistic_TN": float(m_log['tn']),
            "logistic_FP": float(m_log['fp']), "logistic_FN": float(m_log['fn']),
            # MLP
            "mlp_accuracy": float(m_mlp['acc']), "mlp_precision": float(m_mlp['pre']),
            "mlp_recall": float(m_mlp['rec']), "mlp_f1": float(m_mlp['f1']),
            "mlp_TP": float(m_mlp['tp']), "mlp_TN": float(m_mlp['tn']),
            "mlp_FP": float(m_mlp['fp']), "mlp_FN": float(m_mlp['fn']),
            # Tree
            "tree_accuracy": float(m_tree['acc']), "tree_precision": float(m_tree['pre']),
            "tree_recall": float(m_tree['rec']), "tree_f1": float(m_tree['f1']),
            "tree_TP": float(m_tree['tp']), "tree_TN": float(m_tree['tn']),
            "tree_FP": float(m_tree['fp']), "tree_FN": float(m_tree['fn'])
            #Mixture Models
        })

        print("\n--- FINAL RESULTS ---")
        print(f"{'METRICS':<15} | {'LINEAL':<15} | {'LOGISTIC':<15} | {'MLP (RED NEUR.)':<18} | {'DECISION TREE':<15}")
        print("-" * 100)
        print(f"{'Accuracy (%)':<15} | {m_lin['acc']:>14.2f}% | {m_log['acc']:>14.2f}% | {m_mlp['acc']:>17.2f}% | {m_tree['acc']:>14.2f}%")
        print(f"{'Precision':<15} | {m_lin['pre']:>15.4f} | {m_log['pre']:>15.4f} | {m_mlp['pre']:>18.4f} | {m_tree['pre']:>15.4f}")
        print(f"{'Recall':<15} | {m_lin['rec']:>15.4f} | {m_log['rec']:>15.4f} | {m_mlp['rec']:>18.4f} | {m_tree['rec']:>15.4f}")
        print(f"{'F1-Score':<15} | {m_lin['f1']:>15.4f} | {m_log['f1']:>15.4f} | {m_mlp['f1']:>18.4f} | {m_tree['f1']:>15.4f}")


        print("\nConfusion Matrix:")
        print(f"\nLINEAL CLASSIFIER")
        print(f"  (TN): {int(m_lin['tn']):<8} |  (TP): {int(m_lin['tp']):<8}")
        print(f"  (FP): {int(m_lin['fp']):<8} |  (FN): {int(m_lin['fn']):<8}")

        print(f"\nLOGISTIC CLASSIFIER (Umbral: {mejor_umbral_log:.2f})")
        print(f"  (TN): {int(m_log['tn']):<8} |  (TP): {int(m_log['tp']):<8}")
        print(f"  (FP): {int(m_log['fp']):<8} |  (FN): {int(m_log['fn']):<8}")

        print(f"\nMLP CLASSIFIER (Umbral: {mejor_umbral_mlp:.2f})")
        print(f"  (TN): {int(m_mlp['tn']):<8} |  (TP): {int(m_mlp['tp']):<8}")
        print(f"  (FP): {int(m_mlp['fp']):<8} |  (FN): {int(m_mlp['fn']):<8}\n")

        print("\nDECISION TREE CLASSIFIER")
        print(f"  (TN): {int(m_tree['tn']):<8} |  (TP): {int(m_tree['tp']):<8}")
        print(f"  (FP): {int(m_tree['fp']):<8} |  (FN): {int(m_tree['fn']):<8}\n")

        #UVIRN
        metrics_package = {
        "lin": m_lin,
        "log": m_log,
        "mlp": m_mlp,
        "tree": m_tree
        }

        return params_mlp, data, mejor_umbral_mlp, metrics_package
    

  
def predict_interactive(params, data_obj, umbral):
    """ Función para que el usuario ingrese datos y el modelo (MLP) categorice """
    print("\n🔮 ORÁCULO DE PREDICCIÓN (Red Neuronal MLP) 🔮")

    try:
        # Pedir inputs
        stars = float(input("Rating (1-5 estrellas): "))
        price = float(input("Precio del libro ($): "))
        autor_famoso = float(input("¿Es un autor establecido/famoso? (1: Sí, 0: No): "))
        ku = float(input("¿Está en Kindle Unlimited? (1: Sí, 0: No): "))
        pick = float(input("¿Es Editor's Pick? (1: Sí, 0: No): "))
        choice = float(input("¿Es Goodreads Choice? (1: Sí, 0: No): "))
        
        generos_disp = [f for f in data_obj.features if f.startswith('genre_')]
        print(f"\nGéneros disponibles: {generos_disp}") 
        genre_choice = input("Escribe el género (ej. genre_Romance): ")
        
        # Construir el vector X_input 
        x_raw = [0.0] * len(data_obj.features)
        x_raw[0] = stars
        x_raw[1] = price
        
        def set_feature(feature_name, value):
            if feature_name in data_obj.features:
                x_raw[data_obj.features.index(feature_name)] = value

        set_feature('is_established_author', autor_famoso)
        set_feature('isKindleUnlimited', ku)
        set_feature('isEditorsPick', pick)
        set_feature('isGoodReadsChoice', choice)
        set_feature(genre_choice, 1.0)

        # Normalizar SOLO estrellas y precio
        x_num = jnp.array([x_raw[0], x_raw[1]])
        x_num_scaled = (x_num - data_obj.mu_train) / (data_obj.sigma_train + 1e-8)
        
        # Unir
        x_bin = jnp.array(x_raw[2:])
        x_scaled = jnp.concatenate([x_num_scaled, x_bin])
        
        #INFERENCIA CON RED NEURONAL (MLP)
        # Le pasamos todos los pesos (W1, b1, W2, b2) escondidos en 'params'
        prob_array = models.MLP.predict_proba(params, x_scaled)
        
        # JAX puede devolver un arreglo escalar, sacamos el número puro:
        probabilidad = float(prob_array[0]) if getattr(prob_array, 'size', 0) > 1 else float(prob_array)

        # Usamos el umbral que ganó en el entrenamiento
        clase = "ÉLITE" if probabilidad > umbral else "NORMAL"
        
        print("\n" + "─"*40)
        print(f"Predicción del Modelo: {clase}")
        print(f"Probabilidad de éxito: {probabilidad*100:.2f}%")
        print(f"(Umbral mínimo para ser Élite: {umbral*100:.2f}%)")
        print("─"*40)

    except Exception as e:
        print(f"Error en la entrada de datos: {e}")

# --- INTEGRACIÓN API ---
@app.on_event("startup")
async def startup_event():
    global cache_metrics, final_params, trained_data, best_threshold
    # Capturamos los 4 valores del return de run_pipeline
    final_params, trained_data, best_threshold, metrics = run_pipeline()
    cache_metrics = metrics
    
@app.get("/api/all-metrics")
async def get_metrics():
    return JSONResponse(content=cache_metrics)

@app.get("/")
async def serve_home():
    return FileResponse("index.html")

app.mount("/resources", StaticFiles(directory="resources"), name="resources")
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)