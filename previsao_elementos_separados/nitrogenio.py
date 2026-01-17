import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def kennard_stone(X, n_train):
    X = np.asarray(X)
    n_samples = X.shape[0]

    # Distância Euclidiana
    dist = np.linalg.norm(X[:, None] - X[None, :], axis=2)

    # Primeiro par mais distante
    i1, i2 = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [i1, i2]

    while len(selected) < n_train:
        remaining = list(set(range(n_samples)) - set(selected))
        min_dist = []

        for r in remaining:
            d = np.min([dist[r, s] for s in selected])
            min_dist.append(d)

        selected.append(remaining[np.argmax(min_dist)])

    train_idx = np.array(selected)
    test_idx = np.array(list(set(range(n_samples)) - set(train_idx)))

    return train_idx, test_idx

script_dir = os.path.dirname(os.path.abspath(__file__))
caminho_excel = os.path.join(script_dir, '..', 'Dados', 'Dados.xlsx')

df = pd.read_excel(caminho_excel, sheet_name='Bancada')

X = df.iloc[:, 2:553].values   # espectros
y = df['% N'].values          # variável alvo

# Savitzky-Golay: 1ª derivada
X_sg = savgol_filter(
    X,
    window_length=11,
    polyorder=2,
    deriv=0
)


n_train = int(0.7 * X_sg.shape[0])

train_idx, test_idx = kennard_stone(X_sg, n_train)

X_train, X_test = X_sg[train_idx], X_sg[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

max_components = 20
rmse_cv = []

for n in range(1, max_components + 1):
    pls = PLSRegression(n_components=n)
    y_cv = cross_val_predict(pls, X_train, y_train, cv=5)
    rmse = np.sqrt(mean_squared_error(y_train, y_cv))
    rmse_cv.append(rmse)

best_n = np.argmin(rmse_cv) + 1

print(f"Melhor número de componentes: {best_n}")

pls = PLSRegression(n_components=best_n)
pls.fit(X_train, y_train)

y_pred = pls.predict(X_test).ravel()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
