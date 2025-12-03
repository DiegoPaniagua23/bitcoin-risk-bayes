import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import pickle

# Agregar root al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.garch.model import GarchModel

# Configuración de estilo
plt.style.use('ggplot')

def train_and_analyze():
    # Rutas
    data_path = 'data/btc_log_returns.csv'
    models_dir = 'results/models'
    figures_dir = 'results/figures'

    if not os.path.exists(models_dir): os.makedirs(models_dir)
    if not os.path.exists(figures_dir): os.makedirs(figures_dir)

    # 1. Cargar Datos
    print("📂 Cargando datos...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # ESCALADO IMPORTANTE: Multiplicar por 100 para estabilidad numérica
    # El modelo trabajará con porcentajes (ej. -5.2%)
    returns_scaled = df['Log_Return'] * 100

    # 2. Instanciar y Ajustar Modelo
    print("⚙️  Ajustando GJR-GARCH(1,1) con errores t-Student...")
    # p=1 (GARCH), o=1 (GJR/Asimetría), q=1 (ARCH)
    garch = GarchModel(p=1, o=1, q=1, dist='t')
    res = garch.fit(returns_scaled)

    # 3. Mostrar Resultados
    print("\n📊 Resumen del Modelo:")
    print(res.summary())

    metrics = garch.get_aic_bic()
    print(f"\n📏 Métricas de Ajuste: AIC={metrics['AIC']:.2f}, BIC={metrics['BIC']:.2f}")

    # 4. Obtener Volatilidad y VaR
    volatility = garch.get_conditional_volatility()
    var_95 = garch.calculate_var(alpha=0.05)

    # 5. Guardar Resultados Numéricos
    # Guardamos un DataFrame con todo alineado por fecha
    results_df = pd.DataFrame({
        'Log_Return': returns_scaled,
        'Volatility': volatility,
        'VaR_95': var_95,
        'Std_Resid': garch.get_standardized_residuals()
    }, index=df.index)

    results_file = os.path.join(models_dir, 'garch_results.csv')
    results_df.to_csv(results_file)
    print(f"\n💾 Resultados numéricos guardados en: {results_file}")

    # Guardar objeto del modelo (opcional, por si queremos reusarlo sin reentrenar)
    # Nota: 'arch' objects a veces son tricky con pickle, pero probemos
    try:
        with open(os.path.join(models_dir, 'garch_model.pkl'), 'wb') as f:
            pickle.dump(res, f)
    except Exception as e:
        print(f"⚠️ No se pudo serializar el modelo completo: {e}")

    # 6. Visualización
    print("📈 Generando gráficas...")

    # Gráfica A: Retornos vs VaR
    plt.figure(figsize=(14, 6))
    plt.plot(results_df.index, results_df['Log_Return'], color='gray', alpha=0.5, label='Retornos (%)', lw=0.8)
    plt.plot(results_df.index, results_df['VaR_95'], color='red', label='VaR 95% (GJR-GARCH)', lw=1.5)
    plt.title('Bitcoin: Retornos Diarios vs Value at Risk (GJR-GARCH)')
    plt.ylabel('Retorno (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'garch_var_95.png'))

    # Gráfica B: Volatilidad Condicional
    plt.figure(figsize=(14, 6))
    plt.plot(results_df.index, results_df['Volatility'], color='blue', label='Volatilidad Condicional ($\sigma_t$)', lw=1)
    plt.title('Volatilidad Estimada por GJR-GARCH')
    plt.ylabel('Volatilidad (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'garch_volatility.png'))

    print("✅ Gráficas guardadas en results/figures/")

if __name__ == "__main__":
    train_and_analyze()
