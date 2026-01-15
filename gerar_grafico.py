import pandas as pd
import matplotlib.pyplot as plt

# Nome do arquivo CSV
arquivo = "resultados_cartpole_ppo_bc.csv"

# Carregar o CSV
df = pd.read_csv(arquivo)

# Estatísticas
media = df["recompensa_total"].mean()
minimo = df["recompensa_total"].min()
maximo = df["recompensa_total"].max()

print(f"Média: {media}")
print(f"Mínimo: {minimo}")
print(f"Máximo: {maximo}")

# Criar o gráfico
plt.figure(figsize=(12, 6))
plt.plot(df["episodio"], df["recompensa_total"], marker="o", linestyle="-", color="blue", label="Recompensa")

# Linhas horizontais de estatísticas
plt.axhline(media, color="orange", linestyle="--", label=f"Média ({media:.2f})")
plt.axhline(minimo, color="red", linestyle="--", label=f"Mínimo ({minimo:.2f})")
plt.axhline(maximo, color="green", linestyle="--", label=f"Máximo ({maximo:.2f})")

# Títulos e rótulos
plt.title("Recompensa Total por Episódio")
plt.xlabel("Episódio")
plt.ylabel("Recompensa Total")

# Legenda e grade
plt.legend()
plt.grid(True)

# Mostrar o gráfico
plt.show()

