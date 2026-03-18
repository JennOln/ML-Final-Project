import jax.numpy as jnp
from jax import grad
import jax

def linear_class(W, b, X):
    """
    y_hat = XW + b
    """
    y_hat = jnp.dot(X, W) + b
    # para clasificar, 1 si y_hat > 0, sino 0
    return jnp.where(y_hat > jnp.mean(y_hat), 1, 0)

def loss_linear(params, X, y):
    """
    Loss Function (MSE) para el clasificador lineal.
    """
    W, b = params
    y_hat = jnp.dot(X, W) + b
    return jnp.mean((y_hat - y)**2)

def predict_book(new_book, params, data_obj):
    """
    nuevo_libro: diccionario con {'stars': 4.5, 'reviews': 1000, 'price': 9.99, 'category_name': 'Fiction', 'isKindleUnlimited': True, 'isEditorsPick': False, 'isGoodReadsChoice': False}
    params: (W, b) ya entrenados
    data_obj: Objeto de booksData (para usar la media y std originales)
    """

    W, b = params
    # 1. Aplicar Logaritmo a las reseñas (Como hicimos en la limpieza)
    rev_log = jnp.log1p(new_book['reviews'])
    
    # 2. Normalizar usando la MEDIA y STD que calculó el modelo antes
    X_num = jnp.array([new_book['stars'], rev_log, new_book['price']])
    X_num_scaled = (X_num - data_obj.mu_train) / (data_obj.sigma_train + 1e-8)
    
    #(One-Hot)
    X_genero = jnp.zeros(len(data_obj.features) - 3)
    X_input = jnp.concatenate([X_num_scaled, X_genero])
    z = jnp.dot(X_input, W) + b
    
    es_bestseller = jnp.where(z > 0, "SÍ es Best Seller", "NO es Best Seller")
    return es_bestseller, z


def train_step(params, X, y, learning_rate=0.01):
    grads = jax.grad(loss_linear)(params, X, y)
    
    # Actualizar los parametros W y b en contra del gradiente
    W, b = params
    grad_W, grad_b = grads
    
    new_W = W - learning_rate * grad_W
    new_b = b - learning_rate * grad_b
    
    return new_W, new_b

# Función Sigmoide convierte en una probabilidad (0 a 1)
def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def predict_logistic(params, X):
    W, b = params
    z = jnp.dot(X, W) + b
    return sigmoid(z)

# Loss fuction Binary Cross-Entropy (BCE)
def loss_logistic(params, X, y):
    probs = predict_logistic(params, X)
    loss = -jnp.mean(y * jnp.log(probs + 1e-7) + (1 - y) * jnp.log(1 - probs + 1e-7))
    return loss

#Clasificación Final (Umbral de 0.5)
def classify(params, X):
    probs = predict_logistic(params, X)
    return jnp.where(probs > 0.5, 1, 0)

def train_step_logistic(params, X, y, lr):
    grads = grad(loss_logistic)(params, X, y)
    W, b = params
    grad_W, grad_b = grads
    new_W = W - lr * grad_W
    new_b = b - lr * grad_b
    return (new_W, new_b)
