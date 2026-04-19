import os
import json
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import mlflow
from metaflow import FlowSpec, step
from data_processing import booksData
import models  

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

class BooksEliteFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """Prepare data and params"""
        data = booksData('kindle_data-v2.csv')
        data.preprocess_data()
        data.extract_features_target()
        
        self.X_scaled = data.normalized_data()
        self.y = jnp.array(data.y)
        self.n_features = self.X_scaled.shape[1]
        
        self.learning_rate = 0.1  
        self.epochs = 1000

        print(f"\nTraining with {self.n_features} features")
        self.next(self.train_models)

    @step
    def train_models(self):
        key = jrand.PRNGKey(42)
        W_init = jrand.normal(key, (self.n_features,))
        params_iniciales = (W_init, 0.0) 

        """
            LINEAL TRAINING...
        """
        print(f"\nTraining Linear Classifier...")
        self.lineal_params = params_iniciales # El lineal arranca desde la línea de salida
        for epoch in range(self.epochs):
            self.lineal_params = models.LinearClassifier.train_step(self.lineal_params, self.X_scaled, self.y, self.learning_rate)
            
        W_final_lin, b_final_lin = self.lineal_params
        y_pred_linear = models.LinearClassifier.linear_class(W_final_lin, b_final_lin, self.X_scaled)
        self.m_lin = calcular_metricas(self.y, y_pred_linear)
        print(f"Linear Model Accuracy: {self.m_lin['acc']:.2f}%")

        """
            LOGISTIC TRAINING
        """
        print(f"\nTraining Logistic Regression...")
        self.log_params = params_iniciales 
        
        for epoch in range(self.epochs):
            self.log_params = models.LogisticClassifier.train_step(self.log_params, self.X_scaled, self.y, self.learning_rate)
            if epoch % 100 == 0:
                current_loss = models.LogisticClassifier.loss_logistic(self.log_params, self.X_scaled, self.y)
                print(f"epoch {epoch}: Error (Loss) = {current_loss:.6f}")
        
        """
            MLP TRAINING
        """
        print(f"\nTraining MLP Classifier...")
        key_mlp = jrand.PRNGKey(99)
        self.params_mlp = models.MLP.init_params(key_mlp, input_dim=self.n_features, hidden_dim1=64, hidden_dim2=32)
        
        for epoch in range(self.epochs):
            self.params_mlp = models.MLP.train_step(self.params_mlp, self.X_scaled, self.y, self.learning_rate)
            if epoch % 100 == 0:
                loss_mlp = models.MLP.loss_mlp(self.params_mlp, self.X_scaled, self.y)
                print(f"MLP epoch {epoch}: Error (Loss) = {loss_mlp:.6f}")

        """
            Classificacion Treee
        """ 
        print(f"\n--- Entrenando Árbol de Clasificación (Decision Tree) ---")
        self.tree_model = models.DecisionTree(max_depth=5)
        self.tree_model.fit(self.X_scaled, self.y)
        
        # Hacemos la predicción directamente
        y_pred_tree = self.tree_model.predict(self.X_scaled)
        self.m_tree = calcular_metricas(self.y, y_pred_tree)
        # Gaussian Mixture Model (GMM)
        print("Entrenando Mixture Models (GMM)...")
        # El K=2 significa que usaremos 2 gaussianas para los Normales y 2 para los Élite
        self.params_gmm = models.GMMClassifier.fit(self.X_scaled, self.y, K=2, iters=50)

        """
            AdaBoost
        """
        #AdaBoost
        print("\nTraining AdaBoost (Ensemble of 50 experts)...")
        self.adaboost_model = models.AdaBoostClassifier(n_clf=50)
        self.adaboost_model.fit(self.X_scaled, self.y)

        self.next(self.evaluate_and_log)

        

    @step
    def evaluate_and_log(self):
        """
            Search best threshold (F1-SCORE)
        """
        print("\nSearching for optimal thresholds...")
        
        # Para el Logístico
        probs_log = models.LogisticClassifier.predict_logistic(self.log_params, self.X_scaled)
        self.mejor_f1_log = 0.0
        self.mejor_umbral_log = 0.25 
        for umbral_test in jnp.arange(0.20, 0.52, 0.02):
            y_pred_test = jnp.where(probs_log > umbral_test, 1, 0)
            m_test = calcular_metricas(self.y, y_pred_test)
            if m_test['f1'] > self.mejor_f1_log:
                self.mejor_f1_log = m_test['f1']
                self.mejor_umbral_log = float(umbral_test)
        
        # Para el MLP
        probs_mlp = models.MLP.predict_proba(self.params_mlp, self.X_scaled)
        self.mejor_f1_mlp = 0.0
        self.mejor_umbral_mlp = 0.25
        for umbral_test in jnp.arange(0.20, 0.52, 0.02):
            y_pred_test_mlp = jnp.where(probs_mlp > umbral_test, 1, 0)
            m_test_mlp = calcular_metricas(self.y, y_pred_test_mlp)
            if m_test_mlp['f1'] > self.mejor_f1_mlp:
                self.mejor_f1_mlp = m_test_mlp['f1']
                self.mejor_umbral_mlp = float(umbral_test)

        # GMM 
        probs_gmm = models.GMMClassifier.predict_proba(self.params_gmm, self.X_scaled)
        self.mejor_f1_gmm = 0.0
        self.mejor_umbral_gmm = 0.25
        for umbral_test in jnp.arange(0.20, 0.52, 0.02):
            y_pred_test_gmm = jnp.where(probs_gmm > umbral_test, 1, 0)
            m_test_gmm = calcular_metricas(self.y, y_pred_test_gmm)
            if m_test_gmm['f1'] > self.mejor_f1_gmm:
                self.mejor_f1_gmm = m_test_gmm['f1']
                self.mejor_umbral_gmm = float(umbral_test)
    
        # Calculamos las métricas finales con los ganadores
        y_pred_log_final = jnp.where(probs_log > self.mejor_umbral_log, 1, 0)
        self.m_log = calcular_metricas(self.y, y_pred_log_final)

        y_pred_mlp_final = jnp.where(probs_mlp > self.mejor_umbral_mlp, 1, 0)
        self.m_mlp = calcular_metricas(self.y, y_pred_mlp_final)

        y_pred_gmm_final = jnp.where(probs_gmm > self.mejor_umbral_gmm, 1, 0)
        self.m_gmm = calcular_metricas(self.y, y_pred_gmm_final)

        # AdaBoost
        probs_ada = self.adaboost_model.predict_proba(self.X_scaled)
        self.mejor_f1_ada = 0.0
        self.mejor_umbral_ada = 0.50 
        for umbral_test in jnp.arange(0.30, 0.70, 0.02):
            y_pred_test_ada = jnp.where(probs_ada > umbral_test, 1, 0)
            m_test_ada = calcular_metricas(self.y, y_pred_test_ada)
            if m_test_ada['f1'] > self.mejor_f1_ada:
                self.mejor_f1_ada = m_test_ada['f1']
                self.mejor_umbral_ada = float(umbral_test)

        y_pred_ada_final = jnp.where(probs_ada > self.mejor_umbral_ada, 1, 0)
        self.m_ada = calcular_metricas(self.y, y_pred_ada_final)

        

        # Imprimir Resultados Exactamente como los tenías
        print("\n--- FINAL RESULTS ---")
        print(f"{'METRICS':<15} | {'LINEAL':<12} | {'LOGISTIC':<12} | {'MLP':<12} | {'TREE':<12} | {'GMM':<12}")
        print("-" * 85)
        print(f"{'Accuracy (%)':<15} | {self.m_lin['acc']:>11.2f}% | {self.m_log['acc']:>11.2f}% | {self.m_mlp['acc']:>11.2f}% | {self.m_tree['acc']:>11.2f}% | {self.m_gmm['acc']:>11.2f}%")
        print(f"{'Precision':<15} | {self.m_lin['pre']:>12.4f} | {self.m_log['pre']:>12.4f} | {self.m_mlp['pre']:>12.4f} | {self.m_tree['pre']:>12.4f} | {self.m_gmm['pre']:>12.4f}")
        print(f"{'Recall':<15} | {self.m_lin['rec']:>12.4f} | {self.m_log['rec']:>12.4f} | {self.m_mlp['rec']:>12.4f} | {self.m_tree['rec']:>12.4f} | {self.m_gmm['rec']:>12.4f}")
        print(f"{'F1-Score':<15} | {self.m_lin['f1']:>12.4f} | {self.m_log['f1']:>12.4f} | {self.m_mlp['f1']:>12.4f} | {self.m_tree['f1']:>12.4f} | {self.m_gmm['f1']:>12.4f}")

        print("\nConfusion Matrix:")
        print(f"\nLINEAL CLASSIFIER")
        print(f"  (TN): {int(self.m_lin['tn']):<8} |  (TP): {int(self.m_lin['tp']):<8}")
        print(f"  (FP): {int(self.m_lin['fp']):<8} |  (FN): {int(self.m_lin['fn']):<8}")

        print(f"\nLOGISTIC CLASSIFIER (Umbral: {self.mejor_umbral_log:.2f})")
        print(f"  (TN): {int(self.m_log['tn']):<8} |  (TP): {int(self.m_log['tp']):<8}")
        print(f"  (FP): {int(self.m_log['fp']):<8} |  (FN): {int(self.m_log['fn']):<8}")

        print(f"\nMLP CLASSIFIER (Umbral: {self.mejor_umbral_mlp:.2f})")
        print(f"  (TN): {int(self.m_mlp['tn']):<8} |  (TP): {int(self.m_mlp['tp']):<8}")
        print(f"  (FP): {int(self.m_mlp['fp']):<8} |  (FN): {int(self.m_mlp['fn']):<8}\n")

        print("\nDECISION TREE CLASSIFIER")
        print(f"  (TN): {int(self.m_tree['tn']):<8} |  (TP): {int(self.m_tree['tp']):<8}")
        print(f"  (FP): {int(self.m_tree['fp']):<8} |  (FN): {int(self.m_tree['fn']):<8}\n")

        print(f"\nGMM CLASSIFIER (Umbral: {self.mejor_umbral_gmm:.2f})")
        print(f"  (TN): {int(self.m_gmm['tn']):<8} |  (TP): {int(self.m_gmm['tp']):<8}")
        print(f"  (FP): {int(self.m_gmm['fp']):<8} |  (FN): {int(self.m_gmm['fn']):<8}\n")

        # Preparar y guardar JSON para la API (Uvicorn)
        metrics_package = {
            "lin": self.m_lin,
            "log": self.m_log,
            "mlp": self.m_mlp,
            "tree": self.m_tree,
            "gmm": self.m_gmm,
            "ada": self.m_ada
        }
        with open("metrics_cache.json", "w") as f:
            json.dump(metrics_package, f)

        # MLflow (Oculto al final para no chocar con Metaflow)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("ML_Final_Project_Elite_Jennifer")
        with mlflow.start_run():
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("n_features", self.n_features)
            mlflow.log_param("best_threshold_logistic", self.mejor_umbral_log)
            mlflow.log_param("best_threshold_mlp", self.mejor_umbral_mlp)
            mlflow.log_param("best_threshold_gmm", self.mejor_umbral_gmm)

            jnp.savez("final_weights.npz", *self.params_mlp)
            mlflow.log_artifact("final_weights.npz")
            
            mlflow.log_metrics({
                "lineal_accuracy": float(self.m_lin['acc']), "lineal_precision": float(self.m_lin['pre']),
                "lineal_recall": float(self.m_lin['rec']), "lineal_f1": float(self.m_lin['f1']),
                "lineal_TP": float(self.m_lin['tp']), "lineal_TN": float(self.m_lin['tn']),
                "lineal_FP": float(self.m_lin['fp']), "lineal_FN": float(self.m_lin['fn']),
                
                "logistic_accuracy": float(self.m_log['acc']), "logistic_precision": float(self.m_log['pre']),
                "logistic_recall": float(self.m_log['rec']), "logistic_f1": float(self.m_log['f1']),
                "logistic_TP": float(self.m_log['tp']), "logistic_TN": float(self.m_log['tn']),
                "logistic_FP": float(self.m_log['fp']), "logistic_FN": float(self.m_log['fn']),

                "mlp_accuracy": float(self.m_mlp['acc']), "mlp_precision": float(self.m_mlp['pre']),
                "mlp_recall": float(self.m_mlp['rec']), "mlp_f1": float(self.m_mlp['f1']),
                "mlp_TP": float(self.m_mlp['tp']), "mlp_TN": float(self.m_mlp['tn']),
                "mlp_FP": float(self.m_mlp['fp']), "mlp_FN": float(self.m_mlp['fn']),
                
                "tree_accuracy": float(self.m_tree['acc']), "tree_precision": float(self.m_tree['pre']),
                "tree_recall": float(self.m_tree['rec']), "tree_f1": float(self.m_tree['f1']),
                "tree_TP": float(self.m_tree['tp']), "tree_TN": float(self.m_tree['tn']),
                "tree_FP": float(self.m_tree['fp']), "tree_FN": float(self.m_tree['fn']),

                "gmm_accuracy": float(self.m_gmm['acc']), "gmm_precision": float(self.m_gmm['pre']),
                "gmm_recall": float(self.m_gmm['rec']), "gmm_f1": float(self.m_gmm['f1']),
                "gmm_TP": float(self.m_gmm['tp']), "gmm_TN": float(self.m_gmm['tn']),
                "gmm_FP": float(self.m_gmm['fp']), "gmm_FN": float(self.m_gmm['fn']),

                "ada_accuracy": float(self.m_ada['acc']), "ada_precision": float(self.m_ada['pre']),
                "ada_recall": float(self.m_ada['rec']), "ada_f1": float(self.m_ada['f1']),
                "ada_TP": float(self.m_ada['tp']), "ada_TN": float(self.m_ada['tn']),
                "ada_FP": float(self.m_ada['fp']), "ada_FN": float(self.m_ada['fn'])
            })

        self.next(self.end)

    @step
    def end(self):
        print("\nPipeline Finalizado. El archivo JSON y los pesos están listos.")

if __name__ == '__main__':
    BooksEliteFlow()