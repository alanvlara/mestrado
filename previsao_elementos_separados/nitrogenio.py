import os
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Leitura dos dados
script_dir = os.path.dirname(os.path.abspath(__file__))
caminho_excel = os.path.join(script_dir, '..', 'Dados', 'Dados.xlsx')

df = pd.read_excel(caminho_excel, sheet_name='Bancada')

# 2. Separação do espectro (X) e variável alvo (y)
X = df.iloc[:, 2:553].values
y = df['% N'].values

# 3. Divisão treino/teste (30% teste, 70% treino)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Treinamento do modelo PLS com número fixo de componentes (ex: 10)
n_components = 10
pls = PLSRegression(n_components=n_components)
pls.fit(X_train, y_train)

# 5. Predição
y_pred = pls.predict(X_test).ravel()

# 6. Cálculo das métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

# 7. Gráfico valores reais vs preditos
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, c='crimson', edgecolors='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'g--', lw=2)
plt.xlabel('Valor Real (% N)')
plt.ylabel('Valor Predito (% N)')
plt.title('Predição de Nitrogênio com PLS')
plt.grid(True)
plt.tight_layout()
plt.show()
