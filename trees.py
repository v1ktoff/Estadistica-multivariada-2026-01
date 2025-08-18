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
#Importamos trees.csv
df = pd.read_csv('trees.csv')
# Se muestran las primeras filas
print(df.head())
# Creamos el intercepto
idv = np.ones(len(df))
# Creamos X
X = np.column_stack((idv, df['Girth'], df['Height']))
# Creamos Y
y= df['Volume']
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
