import numpy as np
from typing import Tuple, Callable

def calculate_historical_var(data: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calcula el Valor en Riesgo (VaR) Histórico.
    
    Args:
        data: Arreglo de retornos o residuales.
        alpha: Nivel de significancia (ej. 0.05 para 95% de confianza).
        
    Returns:
        El valor del VaR (retorno negativo).
    """
    if len(data) == 0:
        return np.nan
    return np.percentile(data, alpha * 100)

def circular_block_bootstrap(
    data: np.ndarray, 
    statistic: Callable[[np.ndarray], float], 
    block_size: int, 
    n_bootstrap: int = 1000, 
    alpha_ci: float = 0.95,
    random_state: int = None
) -> Tuple[float, float, float]:
    """
    Realiza un Bootstrap de Bloques Circulares para estimar el intervalo de confianza de un estadístico.
    
    Args:
        data: Datos de serie de tiempo (arreglo 1D).
        statistic: Función para calcular el estadístico de interés (ej. VaR).
        block_size: Tamaño de los bloques para el remuestreo.
        n_bootstrap: Número de réplicas del bootstrap.
        alpha_ci: Nivel de confianza para el intervalo (ej. 0.95).
        random_state: Semilla para reproducibilidad.
        
    Returns:
        Tupla conteniendo (limite_inferior, estadistico_observado, limite_superior).
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    
    # Extender datos para circularidad
    data_extended = np.concatenate([data, data[:block_size]])
    
    bootstrap_stats = []
    
    # Calcular estadístico observado
    observed_stat = statistic(data)
    
    # Número de bloques necesarios
    num_blocks = int(np.ceil(n / block_size))
    
    for _ in range(n_bootstrap):
        # Índices de inicio aleatorios para los bloques
        indices = rng.integers(0, n, size=num_blocks)
        
        # Construir muestra bootstrap
        sample = []
        for idx in indices:
            sample.extend(data_extended[idx : idx + block_size])
        
        # Truncar a la longitud original
        sample = np.array(sample[:n])
        
        # Calcular estadístico en la muestra bootstrap
        bootstrap_stats.append(statistic(sample))
        
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calcular Intervalo de Confianza (Método de Percentiles)
    lower_percentile = (1 - alpha_ci) / 2
    upper_percentile = 1 - lower_percentile
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile * 100)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile * 100)
    
    return lower_bound, observed_stat, upper_bound
