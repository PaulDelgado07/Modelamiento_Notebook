import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats

print("-----------------------------------------------------------")
#Calcular los datos crudos de la cafateria existentes
print("Datos crudos de la cafetería (tiempo entre llegadas en minutos):")
datos_crudos= [0.8, 1.2, 0.5, 2.1, 1.5, 0.9, 3.0, 1.1, 0.6, 1.8, 2.5, 0.4, 1.3, 2.2, 0.7]
print(datos_crudos)

#Calcular la media y la desviacion estandar de los datos
media = np.mean(datos_crudos)
desviacion = np.std(datos_crudos)
print("-----------------------------------------------------------")
print("-- Análisis de datos crudos --")
print(f"Cantidad de datos dispuestos: {len(datos_crudos)}")
print(f"Media calculada tiempo medio entre llegadas a la cafetería: {media:.2f} minutos")
print("-----------------------------------------------------------")
#ajustar los datos a una distribucion exponencial utilizando MLE
print(f"--- Resultados de la Prueba Kolmogorov-Smirnov ---")

parametros = stats.expon.fit(datos_crudos, floc=0)
d_stat, p_valor = stats.kstest(datos_crudos, 'expon', args=parametros)

print(f"Estadístico D de K-S: {d_stat:.4f}")
print(f"P-Valor obtenido: {p_valor:.4f}")
print("-----------------------------------------------------------")
#Indiciar si se acepta o rechaza la hipotesis nula
alpha = 0.05
if p_valor > alpha:
    print("Resultado: SE APRUEBA la Hipótesis Nula (H0).")
    print("Se acepta la hipótesis nula: Los datos siguen una distribución exponencial.")
else:    
    print("Resultado: SE RECHAZA la Hipótesis Nula (H0).")
    print("Se rechaza la hipótesis nula: Los datos no siguen una distribución exponencial.")
print("-----------------------------------------------------------")
