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