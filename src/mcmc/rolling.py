import numpy as np
import pandas as pd
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.mcmc.sampler import run_gibbs_sampler

def run_rolling_mcmc(returns, window_size=252, n_iter=1000, burn_in=200):
    """
    Realiza una estimación de ventana móvil del modelo Bayesiano de Cambio de Régimen.
    
    Para cada día t, usamos los datos [t-window_size : t] para estimar sigma.
    Usamos la sigma del 'último régimen' detectado en la ventana como la predicción para t+1.
    """
    T = len(returns)
    rolling_vol = np.zeros(T)
    rolling_vol[:] = np.nan # Inicializar con NaN para los primeros window_size días
    
    print(f"Iniciando Rolling MCMC (Ventana: {window_size}, Iter: {n_iter})...")
    
    # Iteramos desde el primer día donde tenemos una ventana completa
    for t in range(window_size, T):
        if t % 100 == 0:
            print(f"Procesando día {t}/{T}...")
        # Datos de la ventana actual
        window_data = returns[t-window_size : t]
        
        # Correr Gibbs Sampler
        # alpha=2.0, beta=1.0 son priors poco informativos pero razonables para retornos escalados
        s1_chain, s2_chain, k_chain = run_gibbs_sampler(window_data, n_iter=n_iter)
        
        # Eliminar burn-in
        s1_clean = s1_chain[burn_in:]
        s2_clean = s2_chain[burn_in:]
        k_clean = k_chain[burn_in:]
        
        # Estrategia de Predicción:
        # El modelo estima un cambio de régimen en 'k' dentro de la ventana (0 a window_size).
        # Queremos la volatilidad vigente al FINAL de la ventana (para predecir t).
        # Si k < window_size, el final de la ventana está en el régimen 2.
        # Si k se estima cerca del final, podría ser ambiguo, pero generalmente:
        # Vol_t = Promedio(sigma2) dado que k suele estar en el medio o pasado.
        # Sin embargo, para ser robustos, calculamos la volatilidad implícita en el último punto
        # para cada muestra de la cadena y promediamos.
        
        # Vectorizado: Para cada iteración i, si k_i <= window_size (que siempre es cierto por definición),
        # el régimen al final (índice window_size-1) es el 2 si k_i < window_size.
        # Pero ojo, k es el índice donde EMPIEZA el régimen 2.
        # Si k=window_size, entonces todo es régimen 1.
        
        # Vamos a calcular la volatilidad esperada en el último punto de la ventana
        # Para cada muestra de la posterior:
        # vol_last = s2 if k <= (window_size - 1) else s1
        
        # Nota: k_samples va de min_obs a T-min_obs.
        
        # Calculamos la media posterior de la varianza en el último punto
        # Mascara: True si estamos en regimen 2 al final de la ventana
        is_regime2 = k_clean <= (window_size - 1)
        
        # Varianza para cada muestra
        var_samples = np.where(is_regime2, s2_clean, s1_clean)
        
        # Volatilidad estimada (media de desviaciones estándar o raíz de media de varianza)
        # Usualmente E[sigma]
        vol_est = np.mean(np.sqrt(var_samples))
        
        rolling_vol[t] = vol_est
        
    return rolling_vol

if __name__ == "__main__":
    # Cargar datos
    data_path = 'data/btc_log_returns.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        returns = df['Log_Return'].values * 100 # Escalar x100 para consistencia con GARCH
        
        # Ejecutar rolling
        # Aumentamos ventana a 365 días para mayor estabilidad
        vol = run_rolling_mcmc(returns, window_size=365, n_iter=1000)
        
        # Guardar
        np.savez('results/models/mcmc_rolling_results.npz', rolling_vol=vol, dates=df.index)
        print("Resultados guardados en results/models/mcmc_rolling_results.npz")
    else:
        print("No se encontró data/btc_log_returns.csv")
