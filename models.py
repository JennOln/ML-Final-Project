import numpy as np
import jax.numpy as jnp
from jax import grad
import jax


"""funciones compartidas"""
def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def relu(x):
    """Función de activación para capas ocultas (evita que el gradiente muera)"""
    return jnp.maximum(0, x)


"""Linear Classifier"""
class LinearClassifier:

    @staticmethod
    def linear_class(W, b, X):
        """     
        y_hat = XW + b    
        """
        y_hat = jnp.dot(X, W) + b
        return jnp.where(y_hat > 0.5, 1, 0)

    @staticmethod
    def loss_linear(params, X, y):
        """
        Loss Function (MSE) para el clasificador lineal.
        """
        W, b = params
        y_hat = jnp.dot(X, W) + b
        return jnp.mean((y_hat - y)**2)

    @staticmethod
    def train_step(params, X, y, learning_rate=0.01):
        grads = jax.grad(LinearClassifier.loss_linear)(params, X, y)
        
        # Actualizar los parametros W y b en contra del gradiente
        W, b = params
        grad_W, grad_b = grads
        
        new_W = W - learning_rate * grad_W
        new_b = b - learning_rate * grad_b
        
        return new_W, new_b

"""Logistic Regression"""
# Función Sigmoide convierte en una probabilidad (0 a 1)
class LogisticClassifier:
    def __init__(self, input_dim):
        self.W = jnp.random.normal(size=(input_dim, 1)) * 0.01
        self.b = jnp.zeros(1)

    @staticmethod
    def predict_logistic(params, X):
        W, b = params
        z = jnp.dot(X, W) + b
        return sigmoid(z)

    @staticmethod
    def loss_logistic(params, X, y):
        probs = LogisticClassifier.predict_logistic(params, X)
        loss = -jnp.mean(y * jnp.log(probs + 1e-7) + (1 - y) * jnp.log(1 - probs + 1e-7))
        return loss

    @staticmethod
    def classify(params, X, threshold=0.35): # Ajustado a la proporción real de Élite
        probs = LogisticClassifier.predict_logistic(params, X)
        return jnp.where(probs > threshold, 1, 0)

    @staticmethod
    def train_step(params, X, y, lr):
        grads = grad(LogisticClassifier.loss_logistic)(params, X, y)
        W, b = params
        grad_W, grad_b = grads
        new_W = W - lr * grad_W
        new_b = b - lr * grad_b
        return (new_W, new_b)

"""MLP Classifier"""
class MLP:    
    @staticmethod
    def init_params(key, input_dim, hidden_dim1=64, hidden_dim2=32):
        """
        Para clasificación binaria, output_dim SIEMPRE es 1.
        hidden_dim son las 'neuronas' en el medio.
        """

        key1, key2, key3 = jax.random.split(key, 3)
        
        #Hidden Layer 1 (Input to 64 neurons)
        W1 = jax.random.normal(key1, (input_dim, hidden_dim1)) * 0.1
        b1 = jnp.zeros(hidden_dim1)
        
        #hidden layer 2 (64 neurons to 32)
        W2 = jax.random.normal(key2, (hidden_dim1, hidden_dim2)) * 0.1
        b2 = jnp.zeros(hidden_dim2)

        # Capa de salida: Toma las 'hidden_dim' neuronas y las reduce a 1 predicción
        W3 = jax.random.normal(key3, (hidden_dim2, 1)) * 0.1
        b3 = jnp.zeros(1)
        
        return (W1, b1, W2, b2, W3, b3)

    @staticmethod
    def predict_proba(params, X):
        W1, b1, W2, b2, W3, b3= params
        # hidden layer 1
        a1 = relu(jnp.dot(X, W1) + b1)
        # hidden layer 2
        a2 = relu(jnp.dot(a1, W2) + b2)
        # output layer
        probs = sigmoid(jnp.dot(a2, W3) + b3)
        # flatten() aplana la matriz de (N, 1) a (N,) para que coincida con tu vector 'y'
        return probs.flatten() 
    
    @staticmethod
    def loss_mlp(params, X, y):
        probs = MLP.predict_proba(params, X)
        # Log Loss (Binary Cross-Entropy)
        return -jnp.mean(y * jnp.log(probs + 1e-7) + (1 - y) * jnp.log(1 - probs + 1e-7))
    
    @staticmethod
    def train_step(params, X, y, lr):
        # JAX calcula el gradiente para las 6 variables a la vez (W1, b1, W2, b2, W3, b3)
        grads = jax.grad(MLP.loss_mlp)(params, X, y)
        W1, b1, W2, b2, W3, b3 = params
        grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = grads
        
        new_W1 = W1 - lr * grad_W1
        new_b1 = b1 - lr * grad_b1

        new_W2 = W2 - lr * grad_W2
        new_b2 = b2 - lr * grad_b2

        new_W3 = W3 - lr * grad_W3
        new_b3 = b3 - lr * grad_b3

        return (new_W1, new_b1, new_W2, new_b2, new_W3, new_b3)

"""Decision Tree Classifier"""
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # Para nodos de decisión
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # Para nodos hoja (terminales)
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=7):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X, y):
        # Convertimos a numpy normal porque JAX no maneja bien arreglos de tamaño variable en recursión
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y)
        
    def _build_tree(self, X, y, current_depth=0):
        num_samples, num_features = np.shape(X)
        
        # Reglas de Parada (Stop-Splitting Rules)
        if num_samples >= self.min_samples_split and current_depth < self.max_depth:
            # Encontrar el mejor corte (Algoritmo página 63 de tu PDF)
            best_split = self._get_best_split(X, y, num_samples, num_features)
            
            if best_split["info_gain"] > 0: # Si hay ganancia, seguimos dividiendo
                left_subtree = self._build_tree(best_split["X_left"], best_split["y_left"], current_depth + 1)
                right_subtree = self._build_tree(best_split["X_right"], best_split["y_right"], current_depth + 1)
                
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # Si se detiene, calcular el valor de la hoja (Clase mayoritaria)
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)
    
    def _get_best_split(self, X, y, num_samples, num_features):
        best_split = {"info_gain": -1}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                # Generar XtY (izquierda) y XtN (derecha)
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                
                if len(left_indices) > 0 and len(right_indices) > 0:
                    y_left, y_right = y[left_indices], y[right_indices]
                    
                    # Calcular disminución de impureza (Ganancia de Información)
                    current_info_gain = self._information_gain(y, y_left, y_right)
                    
                    if current_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "X_left": X[left_indices, :],
                            "y_left": y_left,
                            "X_right": X[right_indices, :],
                            "y_right": y_right,
                            "info_gain": current_info_gain
                        }
                        max_info_gain = current_info_gain
                        
        return best_split
    
    def _information_gain(self, parent, l_child, r_child):
        # Delta I(t) = I(t) - (NtY/Nt)*I(tY) - (NtN/Nt)*I(tN)
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self._entropy(parent) - (weight_l * self._entropy(l_child) + weight_r * self._entropy(r_child))
        return gain
    
    def _entropy(self, y):
        # Shannon Entropy I(t) = - SUM P(w|t) log2 P(w|t)
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += - p_cls * np.log2(p_cls)
        return entropy
        
    def _calculate_leaf_value(self, y):
        # arg max P(w|t) (THe most common class in the leaf)
        y = list(y)
        return max(y, key=y.count)
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self._make_prediction(x, self.root) for x in X])
    
    def _make_prediction(self, x, tree):
        if tree.value is not None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)
        

"""Gaussian Mixture Model (GMM) Classifier"""
class GMMClassifier:
    
    @staticmethod
    def _log_gaussian_pdf(X, mu, var):
        """
        Calcula la probabilidad logarítmica de que X pertenezca a una Gaussiana (mu, var).
        Usamos varianza diagonal para estabilidad numérica extrema en JAX.
        """
        D = X.shape[1]
        log_det = jnp.sum(jnp.log(var))
        log_norm_const = -0.5 * D * jnp.log(2 * jnp.pi) - 0.5 * log_det
        mahalanobis = -0.5 * jnp.sum(((X - mu)**2) / var, axis=1)
        return log_norm_const + mahalanobis

    @staticmethod
    def _e_step(X, weights, mus, vars):
        """ Expectation: Calcula la 'responsabilidad' de cada Gaussiana sobre cada punto """
        K = weights.shape[0]
        
        # Calculamos el log(PDF) + log(peso) para cada grupo (K)
        log_pdfs = jnp.stack([
            GMMClassifier._log_gaussian_pdf(X, mus[k], vars[k]) + jnp.log(weights[k])
            for k in range(K)
        ], axis=1)

        # Usamos logsumexp para evitar overflow/underflow (Matemáticas seguras)
        log_marginal = jax.scipy.special.logsumexp(log_pdfs, axis=1, keepdims=True)
        
        # Responsabilidades (gamma)
        resp = jnp.exp(log_pdfs - log_marginal)
        return resp

    @staticmethod
    def _m_step(X, resp):
        """ Maximization: Actualiza pesos, medias y varianzas """
        # Agregamos 1e-10 para evitar divisiones por cero
        N_k = jnp.sum(resp, axis=0) + 1e-10 
        
        # 1. Actualizar Pesos
        new_weights = N_k / X.shape[0]

        # 2. Actualizar Medias
        new_mus = jnp.dot(resp.T, X) / N_k[:, None]

        # 3. Actualizar Varianzas (Diagonales)
        diff_sq = (X[:, None, :] - new_mus[None, :, :]) ** 2
        new_vars = jnp.sum(resp[:, :, None] * diff_sq, axis=0) / N_k[:, None]
        
        # Suavizado estadístico (evita que la varianza sea 0 exacto)
        new_vars = new_vars + 1e-6 

        return new_weights, new_mus, new_vars

    @staticmethod
    def _train_single_gmm(X, K, key, iters=50):
        """ Entrena una sola mezcla Gaussiana usando el Algoritmo EM """
        N, D = X.shape
        
        # Inicialización aleatoria usando puntos reales de los datos
        idx = jax.random.randint(key, (K,), 0, N)
        mus = X[idx]
        vars = jnp.ones((K, D))
        weights = jnp.ones(K) / K

        for _ in range(iters):
            resp = GMMClassifier._e_step(X, weights, mus, vars)
            weights, mus, vars = GMMClassifier._m_step(X, resp)

        return weights, mus, vars

    @staticmethod
    def fit(X, y, K=2, iters=50):
        """
        Clasificador Generativo Completo:
        Ajusta un GMM para la clase Normal (y=0) y otro para Élite (y=1).
        """
        # Convertir a numpy normal temporalmente para el filtrado seguro de máscaras
        X_np, y_np = np.array(X), np.array(y)
        
        X_0 = jnp.array(X_np[y_np == 0])
        X_1 = jnp.array(X_np[y_np == 1])

        # Probabilidades Base (Priors)
        prior_0 = len(X_0) / len(X)
        prior_1 = len(X_1) / len(X)

        key = jax.random.PRNGKey(42)
        key0, key1 = jax.random.split(key)

        # Entrenar K campanas de Gauss para cada clase (K=2 por defecto)
        w0, m0, v0 = GMMClassifier._train_single_gmm(X_0, K, key0, iters)
        w1, m1, v1 = GMMClassifier._train_single_gmm(X_1, K, key1, iters)

        # Empaquetamos todo en "params" como el resto de tus modelos
        return (w0, m0, v0, w1, m1, v1, prior_0, prior_1)

    @staticmethod
    def predict_proba(params, X):
        """ Inferencia usando el Teorema de Bayes """
        w0, m0, v0, w1, m1, v1, prior_0, prior_1 = params

        # Log-Probabilidad de haber sido generado por la Clase 0 (Normal)
        log_pdfs_0 = jnp.stack([
            GMMClassifier._log_gaussian_pdf(X, m0[k], v0[k]) + jnp.log(w0[k])
            for k in range(w0.shape[0])
        ], axis=1)
        log_prob_0 = jax.scipy.special.logsumexp(log_pdfs_0, axis=1) + jnp.log(prior_0)

        # Log-Probabilidad de haber sido generado por la Clase 1 (Élite)
        log_pdfs_1 = jnp.stack([
            GMMClassifier._log_gaussian_pdf(X, m1[k], v1[k]) + jnp.log(w1[k])
            for k in range(w1.shape[0])
        ], axis=1)
        log_prob_1 = jax.scipy.special.logsumexp(log_pdfs_1, axis=1) + jnp.log(prior_1)

        # P(y=1 | X) usando probabilidad de Bayes (con truco logsumexp para evitar NaN)
        log_total = jax.scipy.special.logsumexp(jnp.stack([log_prob_0, log_prob_1], axis=1), axis=1)
        return jnp.exp(log_prob_1 - log_total)
    

"""AdaBoost Classifier (con Decision Stumps)"""
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None 

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoostClassifier:
    def __init__(self, n_clf=50):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # Convertimos [0, 1] a [-1, 1] para la matemática de AdaBoost
        y_ = np.where(y == 0, -1, 1) 
        n_samples, n_features = X.shape
        
        # Inicializar pesos
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    p = 1
                    predictions = np.ones(np.shape(y_))
                    predictions[X[:, feature_i] < threshold] = -1

                    # Error ponderado
                    error = sum(w[y_ != predictions])

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y_ * predictions)
            w /= np.sum(w) 

            self.clfs.append(clf)

    def predict_proba(self, X):
        X = np.array(X)
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred_continuous = np.sum(clf_preds, axis=0)
        # Regresión Logística Aditiva (pasarlo a probabilidad 0-1)
        return 1 / (1 + jnp.exp(-y_pred_continuous))