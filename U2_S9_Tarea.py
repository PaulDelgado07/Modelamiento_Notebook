import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Simulación de los datos reales de la ventanilla (No modificar esta semilla)
np.random.seed(24)
tiempos_reales = np.random.exponential(scale=3.8, size=120)

# Estimar el parámetro 'scale' usando Máxima Verosimilitud (MLE)
# Utilizar stats.expon.fit()
parametros = stats.expon.fit(tiempos_reales)
print(f"Parámetro Estimado (Media Teórica): {parametros} minutos")

# Ejecutar la Prueba de Bondad de Ajuste Kolmogorov-Smirnov (K-S)
# Utilizar stats.kstest() comparando con 'expon'
d_stat, p_valor = stats.kstest(tiempos_reales, 'expon', args=parametros)
print(f"Estadístico D de K-S: {d_stat:.4f}")
print(f"P-Valor obtenido: {p_valor:.4f}")

# Graficar el Histograma de los datos junto a la Curva Teórica Ajustada
plt.figure(figsize=(7, 4.5))
plt.hist(tiempos_reales, bins=10, density=True, alpha=0.6, color='seagreen', label='Datos de Ventanilla')

x = np.linspace(0, max(tiempos_reales), 200)
pdf_teorica = stats.expon.pdf(x, *parametros)
plt.plot(x, pdf_teorica, color='darkred', lw=2, label='Curva Exponencial Ajustada')

plt.title('Validación de Tiempos de Atención en Ventanilla')
plt.xlabel('Minutos')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()