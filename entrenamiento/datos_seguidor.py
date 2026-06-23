import numpy as np

# =========================
# 1) DATASET
# =========================
X = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1],
], dtype=float)

Y = np.array([
    [0.50, 0.50],  # 000
    [0.95, 0.20],  # 001
    [0.85, 0.85],  # 010
    [0.92, 0.45],  # 011
    [0.20, 0.95],  # 100
    [0.75, 0.75],  # 101
    [0.45, 0.92],  # 110
    [0.70, 0.70],  # 111
], dtype=float)

# =========================
# 2) HIPERPARÁMETROS
# =========================
np.random.seed(0)   # importante para reproducibilidad
n_in = 3
n_h = 4
n_out = 2
lr = 0.1
epochs = 20000

# =========================
# 3) INICIALIZACIÓN
# =========================
W1 = np.random.randn(n_h, n_in) * 0.5
b1 = np.zeros(n_h)
W2 = np.random.randn(n_out, n_h) * 0.5
b2 = np.zeros(n_out)

def tanh(x):
    return np.tanh(x)

def dtanh(a):
    return 1.0 - a*a

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(a):
    return a * (1.0 - a)

# =========================
# 4) ENTRENAMIENTO
# =========================
for epoch in range(epochs):
    # Forward
    z1 = X @ W1.T + b1
    a1 = tanh(z1)

    z2 = a1 @ W2.T + b2
    a2 = sigmoid(z2)

    # Error
    err = a2 - Y
    loss = np.mean(err**2)

    # Backprop
    dz2 = (2 / len(X)) * err * dsigmoid(a2)
    dW2 = dz2.T @ a1
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2
    dz1 = da1 * dtanh(a1)
    dW1 = dz1.T @ X
    db1 = dz1.sum(axis=0)

    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# =========================
# 5) RESULTADOS
# =========================
np.set_printoptions(precision=8, suppress=False)

print("W1 =")
print(W1)
print("\nb1 =")
print(b1)
print("\nW2 =")
print(W2)
print("\nb2 =")
print(b2)
print("\nMSE final =", loss)

# =========================
# 6) VERIFICACIÓN
# =========================
z1 = X @ W1.T + b1
a1 = tanh(z1)
z2 = a1 @ W2.T + b2
Y_pred = sigmoid(z2)

print("\nSalidas predichas:")
for x, y_t, y_p in zip(X, Y, Y_pred):
    print(f"X={x.astype(int)}  Y_obj={y_t}  Y_pred={np.round(y_p,4)}")