import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Configuración de estilo
plt.style.use('ggplot')

def visualize_mcmc_results():
    # Rutas
    results_path = 'results/models/mcmc_chains.npz'
    data_path = 'data/btc_log_returns.csv'
    figures_dir = 'results/figures'

    if not os.path.exists(results_path):
        print(f"❌ No se encontró el archivo de resultados: {results_path}")
        return

    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Cargar resultados
    print("📂 Cargando resultados MCMC...")
    data = np.load(results_path, allow_pickle=True)
    sigma1_sq = data['sigma1_sq']
    sigma2_sq = data['sigma2_sq']
    k_samples = data['k_samples']
    burn_in = int(data['burn_in'])
    dates_str = data['dates']

    # Convertir fechas de string a datetime
    dates = pd.to_datetime(dates_str)

    # Cargar datos originales para contexto
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    y = df['Log_Return'].values * 100 # Recordar escalar

    # Filtrar burn-in
    s1_clean = sigma1_sq[burn_in:]
    s2_clean = sigma2_sq[burn_in:]
    k_clean = k_samples[burn_in:]

    # --- Gráfica 1: Trace Plots (Diagnóstico de Convergencia) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(sigma1_sq, color='tab:blue', alpha=0.6, lw=1)
    axes[0].axvline(burn_in, color='red', linestyle='--', label='Burn-in')
    axes[0].set_ylabel(r'$\sigma_1^2$ (Varianza Pre)')
    axes[0].legend()
    axes[0].set_title('Trace Plots: Convergencia de las Cadenas MCMC')

    axes[1].plot(sigma2_sq, color='tab:orange', alpha=0.6, lw=1)
    axes[1].axvline(burn_in, color='red', linestyle='--')
    axes[1].set_ylabel(r'$\sigma_2^2$ (Varianza Post)')

    axes[2].plot(k_samples, color='tab:green', alpha=0.6, lw=1)
    axes[2].axvline(burn_in, color='red', linestyle='--')
    axes[2].set_ylabel(r'$k$ (Día de Cambio)')
    axes[2].set_xlabel('Iteración')

    plt.tight_layout()
    save_path1 = os.path.join(figures_dir, 'mcmc_traceplots.png')
    plt.savefig(save_path1)
    print(f"✅ Trace plots guardados en: {save_path1}")
    plt.close()

    # --- Gráfica 2: Distribuciones Posteriores ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(s1_clean, bins=50, color='tab:blue', density=True, alpha=0.7)
    axes[0].set_title(r'Posterior $\sigma_1^2$')
    axes[0].set_xlabel('Varianza')

    axes[1].hist(s2_clean, bins=50, color='tab:orange', density=True, alpha=0.7)
    axes[1].set_title(r'Posterior $\sigma_2^2$')
    axes[1].set_xlabel('Varianza')

    # Histograma de fechas de cambio
    # Convertimos los índices k a fechas para el histograma
    k_dates = dates[k_clean]
    axes[2].hist(k_dates, bins=30, color='tab:green', density=True, alpha=0.7)
    axes[2].set_title('Posterior del Punto de Cambio')
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    save_path2 = os.path.join(figures_dir, 'mcmc_posteriors.png')
    plt.savefig(save_path2)
    print(f"✅ Posteriores guardados en: {save_path2}")
    plt.close()

    # --- Gráfica 3: Serie de Tiempo con Cambio Detectado ---
    k_mean = int(np.mean(k_clean))
    date_change = dates[k_mean]

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, y, color='gray', alpha=0.5, label='Retornos Log (%)', lw=0.8)

    # Línea del cambio
    plt.axvline(date_change, color='red', linestyle='-', linewidth=2, label=f'Cambio Detectado ({date_change.date()})')

    # Intervalo de credibilidad del cambio (95%)
    k_lower = int(np.percentile(k_clean, 2.5))
    k_upper = int(np.percentile(k_clean, 97.5))
    plt.axvspan(dates[k_lower], dates[k_upper], color='red', alpha=0.1, label='IC 95% Cambio')

    # Regímenes de volatilidad (bandas visuales)
    s1_mean = np.mean(s1_clean)
    s2_mean = np.mean(s2_clean)

    # Dibujar bandas de +/- 2 desviaciones estándar estimadas
    std1 = np.sqrt(s1_mean)
    std2 = np.sqrt(s2_mean)

    plt.hlines(2*std1, dates[0], date_change, colors='blue', linestyles='--', lw=1.5)
    plt.hlines(-2*std1, dates[0], date_change, colors='blue', linestyles='--', lw=1.5, label=r'Régimen 1 ($\pm 2\sigma$)')

    plt.hlines(2*std2, date_change, dates[-1], colors='orange', linestyles='--', lw=1.5)
    plt.hlines(-2*std2, date_change, dates[-1], colors='orange', linestyles='--', lw=1.5, label=r'Régimen 2 ($\pm 2\sigma$)')

    plt.title('Detección de Cambio de Régimen en Volatilidad de Bitcoin')
    plt.ylabel('Retorno Diario (%)')
    plt.legend(loc='upper left')

    plt.tight_layout()
    save_path3 = os.path.join(figures_dir, 'btc_volatility_change.png')
    plt.savefig(save_path3)
    print(f"✅ Serie de tiempo guardada en: {save_path3}")
    plt.close()

if __name__ == "__main__":
    visualize_mcmc_results()
