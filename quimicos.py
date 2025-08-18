import pandas as pd
import numpy as np
def calcular_sse_y_varest(X, y, beta):
    """
    Calculamos la suma de cuadrados del error (SSE) y la varianza estimada del error (varest)
    a partir de una regresión lineal.

    Parámetros:
        X (np.ndarray): Matriz de diseño (n x p)
        y (np.ndarray): Vector de respuestas (n x 1) o (n,)
        beta (np.ndarray): Coeficientes estimados (p x 1) o (p,)

    Retorna:
        sse (float): Suma de cuadrados del error
        varest (float): Varianza estimada del error
    """

    # Nos aseguramos de que las formas sean correctas
    y = np.array(y).reshape(-1, 1)
    beta = np.array(beta).reshape(-1, 1)

    # Calculamos de SSE
    sse = float((y.T @ y - beta.T @ X.T @ y)[0, 0])

    # Grados de libertad: n - p
    n = X.shape[0]
    p = X.shape[1]

    varest = sse / (n - p)

    return sse, varest
def calcular_beta(X,y):
    """
    Calculamos los coeficientes beta de una regresión lineal
    usando la fórmula de mínimos cuadrados ordinarios:
        beta = (X^T X)^(-1) X^T y

    Parámetros:
        X (np.ndarray): Matriz de diseño (n x p)
        y (np.ndarray): Vector de respuestas (n x 1) o (n,)

    Retorna:
        beta (np.ndarray): Vector de coeficientes (p x 1)
    """
    # Nos aseguramos de que y sea un vector columna
    y = np.array(y).reshape(-1, 1)
    XtX = X.T @ X  # Calculamos X^T X
    XtY = X.T @ y # Calculamos X^T y
    beta = np.linalg.inv(XtX) @ XtY # Calculamos beta = (Xᵀ X)^(-1) * Xᵀ y
    return beta
df = pd.DataFrame({
    'Experimento': range(1, 18),
    'x1': [41.9, 43.4, 43.9, 44.5, 47.3, 47.5, 47.9, 50.2, 52.8, 53.2, 56.7, 57.0, 63.5, 64.3, 71.1, 77.0, 77.8],
    'x2': [29.1, 29.3, 29.5, 29.7, 29.9, 30.3, 30.5, 30.7, 30.8, 30.9, 31.5, 31.7, 31.9, 32.0, 32.1, 32.5, 32.9],
    'y':  [251.3, 251.3, 248.3, 267.5, 273.0, 276.5, 270.3, 274.9, 285.0, 290.0, 297.0, 302.5, 304.5, 309.3, 321.7, 330.7, 349.0]
})
idv = np.ones(len(df))
# Creamos X
X = np.column_stack((idv, df['x1'], df['x2']))
# Creamos Y
y = df['y']
## En esta parte se usa la función calcular_beta() para calcular los coeficientes beta de manera explicita
beta = calcular_beta(X,y)
print("Beta: ", beta)
## En esta parte se calcula beta usando la función np.linalg.lstsq() integrada en la libreria numpy para calcular \hat{\beta}
beta1 = np.linalg.lstsq(X, y, rcond=None)[0]
print("Beta (usando np.linalg.lstsq()):", beta1)
## En esta parte usanmos calcular_sse_y_varest() para calcular suma de cuadrados del error (SSE) y la varianza estimada del error (varest)
sse, varest = calcular_sse_y_varest(X,y,beta)
print("SSE:", sse)
print("Varianza estimada del error:", varest)
