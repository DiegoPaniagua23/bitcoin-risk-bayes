import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import norm

# Agregar root al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.validation.backtest import kupiec_pof_test
from src.validation.bootstrap import circular_block_bootstrap, calculate_historical_var

# Configuración
plt.style.use('ggplot')
ALPHA = 0.05 # VaR 95%

def compare_models():
    print("⚔️  Iniciando Comparación de Modelos: MCMC vs GARCH...")

    # Rutas
    mcmc_path = 'results/models/mcmc_chains.npz'
    garch_path = 'results/models/garch_results.csv'
    figures_dir = 'results/figures'

    if not os.path.exists(mcmc_path) or not os.path.exists(garch_path):
        print("❌ Faltan archivos de resultados. Asegúrate de haber corrido ambos modelos.")
        return

    # 1. Cargar Resultados GARCH
    print("   🔹 Cargando GARCH...")
    df_garch = pd.read_csv(garch_path, index_col=0, parse_dates=True)
    returns = df_garch['Log_Return'].values # Ya están escalados x100
    dates = df_garch.index
    garch_var = df_garch['VaR_95'].values
    garch_vol = df_garch['Volatility'].values

    # 2. Cargar y Procesar Resultados MCMC
    print("   🔹 Cargando MCMC...")
    mcmc_data = np.load(mcmc_path, allow_pickle=True)
    sigma1_sq_chain = mcmc_data['sigma1_sq']
    sigma2_sq_chain = mcmc_data['sigma2_sq']
    k_chain = mcmc_data['k_samples']
    burn_in = int(mcmc_data['burn_in'])

    # Calcular medias posteriores
    s1_hat = np.mean(sigma1_sq_chain[burn_in:])
    s2_hat = np.mean(sigma2_sq_chain[burn_in:])
    k_hat = int(np.mean(k_chain[burn_in:]))

    # Construir serie de volatilidad MCMC
    # Sigma_t = sqrt(s1) si t < k else sqrt(s2)
    T = len(returns)
    mcmc_vol = np.zeros(T)
    mcmc_vol[:k_hat] = np.sqrt(s1_hat)
    mcmc_vol[k_hat:] = np.sqrt(s2_hat)

    # Calcular VaR MCMC (Asumiendo Normalidad según propuesta 5.2)
    # VaR = mu + sigma * Z_alpha (mu=0)
    z_score = norm.ppf(ALPHA)
    mcmc_var = mcmc_vol * z_score

    # 3. Validación: Test de Kupiec
    print("\n🧪 Ejecutando Test de Kupiec (Backtesting)...")

    kupiec_garch = kupiec_pof_test(returns, garch_var, alpha=ALPHA)
    kupiec_mcmc = kupiec_pof_test(returns, mcmc_var, alpha=ALPHA)

    print(f"\n   🏆 Resultados Kupiec (Alpha={ALPHA}):")
    print(f"   {'Modelo':<10} | {'Fallos':<8} | {'Tasa Obs.':<10} | {'p-value':<10} | {'Decisión'}")
    print("-" * 65)
    print(f"   {'GARCH':<10} | {kupiec_garch['failures']:<8} | {kupiec_garch['observed_rate']:.4f}     | {kupiec_garch['p_value']:.4f}     | {'✅ Acepta' if kupiec_garch['decision'] == 0 else '❌ Rechaza'}")
    print(f"   {'MCMC':<10}  | {kupiec_mcmc['failures']:<8} | {kupiec_mcmc['observed_rate']:.4f}     | {kupiec_mcmc['p_value']:.4f}     | {'✅ Acepta' if kupiec_mcmc['decision'] == 0 else '❌ Rechaza'}")

    # 4. Bootstrap de Residuos (MCMC)
    # Validamos si la asunción de normalidad del MCMC fue correcta analizando sus residuos
    print("\nbootstrapping MCMC residuals...")
    mcmc_resid = returns / mcmc_vol

    # Función auxiliar para el bootstrap (quantile 5%)
    def calc_q05(x): return np.percentile(x, 5)

    lb, obs, ub = circular_block_bootstrap(mcmc_resid, calc_q05, block_size=50, n_bootstrap=1000)

    print(f"   Intervalo de Confianza (95%) para el cuantil 5% de residuos MCMC:")
    print(f"   [{lb:.4f}, {ub:.4f}] (Teórico Normal: {z_score:.4f})")
    if lb <= z_score <= ub:
        print("   ✅ La asunción de Normalidad es razonable (el valor teórico cae en el IC).")
    else:
        print("   ⚠️ La asunción de Normalidad podría ser incorrecta (colas más pesadas detectadas).")

    # 5. Gráfica Comparativa Final
    print("\n📈 Generando gráfica comparativa...")
    plt.figure(figsize=(15, 7))

    # Retornos
    plt.plot(dates, returns, color='gray', alpha=0.3, label='Retornos BTC', lw=0.5)

    # VaR GARCH
    plt.plot(dates, garch_var, color='tab:blue', label='VaR GARCH-Student', lw=1.5)

    # VaR MCMC
    plt.plot(dates, mcmc_var, color='tab:purple', label='VaR MCMC-Normal', lw=1.5, linestyle='--')

    # Puntos de fallo (Excepciones)
    failures_garch = returns < garch_var
    plt.scatter(dates[failures_garch], returns[failures_garch], color='red', s=10, label='Fallos GARCH', zorder=5)

    plt.title('Comparación de Modelos de Riesgo: GARCH vs MCMC (Bitcoin 2019-2024)')
    plt.ylabel('Retornos / VaR (%)')
    plt.legend()

    save_path = os.path.join(figures_dir, 'model_comparison.png')
    plt.savefig(save_path)
    print(f"✅ Gráfica guardada en: {save_path}")

if __name__ == "__main__":
    compare_models()
