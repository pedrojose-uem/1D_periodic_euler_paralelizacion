import matplotlib.pyplot as plt

# ==========================
# Datos de strong scaling
# ==========================

procesos_strong = [1, 2, 4, 8]
tiempos_strong = [101.01, 54.807, 48.3158, 38.5879]

# Cálculo de speedup y eficiencia
t1 = tiempos_strong[0]
speedup = [t1 / t for t in tiempos_strong]
eficiencia = [s / p * 100 for s, p in zip(speedup, procesos_strong)]

print("Strong scaling")
print("Procesos | Tiempo (s) | Speedup | Eficiencia (%)")
for p, t, s, e in zip(procesos_strong, tiempos_strong, speedup, eficiencia):
    print(f"{p:8d} | {t:10.4f} | {s:7.3f} | {e:13.2f}")

# ==========================
# Gráfica 1: Speedup
# ==========================

plt.figure(figsize=(7, 5))
plt.plot(procesos_strong, speedup, marker="o", label="Speedup medido")
plt.plot(procesos_strong, procesos_strong, linestyle="--", label="Speedup ideal")

plt.xlabel("Número de procesos")
plt.ylabel("Speedup")
plt.title("Strong scaling: speedup vs número de procesos")
plt.xticks(procesos_strong)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("strong_scaling_speedup.png", dpi=300)
plt.savefig("strong_scaling_speedup.pdf")
plt.close()

# ==========================
# Gráfica 2: Eficiencia
# ==========================

plt.figure(figsize=(7, 5))
plt.plot(procesos_strong, eficiencia, marker="o")

plt.xlabel("Número de procesos")
plt.ylabel("Eficiencia (%)")
plt.title("Strong scaling: eficiencia paralela")
plt.xticks(procesos_strong)
plt.grid(True)
plt.tight_layout()
plt.savefig("strong_scaling_eficiencia.png", dpi=300)
plt.savefig("strong_scaling_eficiencia.pdf")
plt.close()

# ==========================
# Datos de weak scaling
# ==========================

procesos_weak = [1, 2, 4, 8]
puntos_totales_weak = [25000, 50000, 100000, 200000]
tiempos_weak = [5.57268, 14.48, 46.9543, 186.021]

# ==========================
# Gráfica 3: Weak scaling
# ==========================

plt.figure(figsize=(7, 5))
plt.plot(procesos_weak, tiempos_weak, marker="o")

plt.xlabel("Número de procesos")
plt.ylabel("Tiempo de computación (s)")
plt.title("Weak scaling: tiempo vs número de procesos")
plt.xticks(procesos_weak)
plt.grid(True)
plt.tight_layout()
plt.savefig("weak_scaling_tiempo.png", dpi=300)
plt.savefig("weak_scaling_tiempo.pdf")
plt.close()

# ==========================
# Tabla weak scaling
# ==========================

print("\nWeak scaling")
print("Procesos | Puntos totales | Puntos/proceso | Tiempo (s)")
for p, n, t in zip(procesos_weak, puntos_totales_weak, tiempos_weak):
    print(f"{p:8d} | {n:14d} | {n//p:14d} | {t:10.4f}")

print("\nGráficas generadas:")
print("- strong_scaling_speedup.png")
print("- strong_scaling_speedup.pdf")
print("- strong_scaling_eficiencia.png")
print("- strong_scaling_eficiencia.pdf")
print("- weak_scaling_tiempo.png")
print("- weak_scaling_tiempo.pdf")
