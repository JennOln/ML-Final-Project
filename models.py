import jax.numpy as jnp
import jax

def linear_class(W, b, X):
    """
    y_hat = XW + b
    """
    y_hat = jnp.dot(X, W) + b
    # Para clasificar, devolvemos 1 si y_hat > 0, de lo contrario 0
    return jnp.where(y_hat > 0, 1, 0)

def loss_linear(params, X, y):
    """
    Loss Function (MSE) para el clasificador lineal.
    Incluso en clasificación, el modelo lineal trata de acercar 'y_hat' a 'y'.
    """
    W, b = params
    y_hat = jnp.dot(X, W) + b
    return jnp.mean((y_hat - y)**2)

def train_step(params, X, y, learning_rate=0.01):
    # Calculamos el gradiente de la función de pérdida
    grads = jax.grad(loss_linear)(params, X, y)
    
    # Actualizamos los parámetros (W y b) moviéndonos en contra del gradiente
    W, b = params
    grad_W, grad_b = grads
    
    new_W = W - learning_rate * grad_W
    new_b = b - learning_rate * grad_b
    
    return new_W, new_b