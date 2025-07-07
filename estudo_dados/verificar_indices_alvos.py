import pandas as pd

# Carregando o arquivo Excel
df = pd.read_excel('Dados/Dados.xlsx', sheet_name='Bancada')  # ajuste caminho se necessário

# Lista com os nomes das colunas que você quer localizar
alvos = ['% Proteina', '% N', 'P (ppm)', '% P', 'K (ppm)']

# Dicionário para armazenar o índice de cada coluna
colunas_indices = {}

for alvo in alvos:
    if alvo in df.columns:
        colunas_indices[alvo] = df.columns.get_loc(alvo)
    else:
        print(f'❌ Coluna não encontrada: {alvo}')

# Mostrando os índices encontrados
print("✅ Índices das colunas alvo:")
for coluna, indice in colunas_indices.items():
    print(f"{coluna}: {indice}")
