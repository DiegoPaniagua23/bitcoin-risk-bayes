import numpy as np
import pandas as pd
import time
import os
import sys

# Agregar el directorio raíz al path para poder importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.mcmc.sampler import run_gibbs_sampler

def test_mcmc():
    # 1. Cargar datos
    data_path = 'data/btc_log_returns.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encuentra el archivo {data_path}")
        return

    print("📂 Cargando datos...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 2. Preprocesamiento: Escalar por 100 para estabilidad numérica
    # Retornos en porcentaje (ej. 1.5 en lugar de 0.015)
    y = df['Log_Return'].values * 100
    T = len(y)
    print(f"✅ Datos cargados: {T} observaciones.")
    print(f"   Varianza global (scaled): {np.var(y):.4f}")

    # 3. Configuración del Sampler
    N_ITER = 10000
    BURN_IN = 2000

    print(f"\n🚀 Iniciando Gibbs Sampler (JIT-Compiled)...")
    print(f"   Iteraciones: {N_ITER}")

    # Primera ejecución (incluye tiempo de compilación)
    start_compile = time.time()
    # Corremos pocas iteraciones para disparar la compilación
    run_gibbs_sampler(y, 10, min_obs=50)
    end_compile = time.time()
    print(f"   ⏱️  Tiempo de compilación JIT: {end_compile - start_compile:.4f} s")

    # Ejecución real
    start_run = time.time()
    sigma1_sq, sigma2_sq, k_samples = run_gibbs_sampler(y, N_ITER, min_obs=50)
    end_run = time.time()

    total_time = end_run - start_run
    iter_per_sec = N_ITER / total_time

    print(f"   ⏱️  Tiempo de ejecución: {total_time:.4f} s")
    print(f"   ⚡ Velocidad: {iter_per_sec:.0f} iter/s")

    # 4. Resultados
    # Descartar burn-in
    s1_mean = np.mean(sigma1_sq[BURN_IN:])
    s2_mean = np.mean(sigma2_sq[BURN_IN:])
    k_mean = int(np.mean(k_samples[BURN_IN:]))

    # Fecha del cambio estimado
    date_change = df.index[k_mean]

    print("\n📊 Resultados Preliminares (Posterior Means):")
    print(f"   Sigma1^2 (Antes): {s1_mean:.4f}")
    print(f"   Sigma2^2 (Después): {s2_mean:.4f}")
    print(f"   Punto de Cambio (k): {k_mean} (aprox. {date_change.date()})")

    if s1_mean > s2_mean:
        print("   📉 La volatilidad disminuyó después del cambio.")
    else:
        print("   📈 La volatilidad aumentó después del cambio.")

    # 5. Guardar Cadenas para Análisis Posterior
    results_dir = 'results/models'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Guardamos en formato comprimido .npz
    output_file = os.path.join(results_dir, 'mcmc_chains.npz')
    np.savez(output_file,
             sigma1_sq=sigma1_sq,
             sigma2_sq=sigma2_sq,
             k_samples=k_samples,
             burn_in=BURN_IN,
             dates=df.index.astype(str).values) # Guardamos fechas como strings para referencia

    print(f"\n💾 Cadenas MCMC guardadas en: {output_file}")

if __name__ == "__main__":
    test_mcmc()
