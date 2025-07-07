# 📁 Dataset NIR - Bancada

Este projeto utiliza dados espectrais obtidos via espectroscopia no infravermelho próximo (NIR) para prever características químicas de amostras.

---

## 🧾 Estrutura dos Dados

O dataset está armazenado em um arquivo `.xlsx` com a seguinte organização:

| Coluna | Tipo de dado                         | Descrição                                                |
|--------|--------------------------------------|----------------------------------------------------------|
| 0–1    | Identificação                        | Contém `ID`, nome ou descrição da amostra (opcional)     |
| 2–552  | **Espectro NIR**                     | 551 colunas com valores de refletância para cada comprimento de onda (ex: de 1100 nm a 2500 nm) |
| 553–557| **Variáveis alvo (targets)**         | Propriedades químicas a serem previstas:                 |
|        | `% Proteina`                         | Teor de proteína na amostra                              |
|        | `% N`                                | Percentual de nitrogênio                                 |
|        | `P (ppm)`                            | Fósforo em partes por milhão                             |
|        | `% P`                                | Percentual de fósforo                                    |
|        | `K (ppm)`                            | Potássio em partes por milhão                            |

---

## 📊 Dimensões

- **Número de amostras**: 188
- **Número de colunas espectrais (NIR)**: 551
- **Número de variáveis-alvo (químicas)**: 5

---