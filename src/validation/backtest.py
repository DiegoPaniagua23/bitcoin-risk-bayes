import numpy as np
from scipy.stats import chi2
from typing import Dict

def kupiec_pof_test(
    returns: np.ndarray, 
    var_estimates: np.ndarray, 
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Realiza la prueba de Proporción de Fallos (POF) de Kupiec.
    
    Args:
        returns: Arreglo de retornos reales.
        var_estimates: Arreglo de estimaciones de VaR (deben ser valores negativos, coincidiendo con la escala de retornos).
                       Si el VaR es positivo (pérdida), asegurar que la lógica de comparación lo maneje.
                       Aquí asumimos que los retornos tienen signo (+/-) y el VaR es un umbral negativo.
                       Ocurre una excepción si retorno < VaR.
        alpha: Tasa de fallos esperada (ej. 0.05 para VaR al 95%).
        
    Returns:
        Diccionario conteniendo:
            - 'LR_POF': Estadístico de Razón de Verosimilitud.
            - 'p_value': valor p de la prueba.
            - 'failures': Número de excepciones.
            - 'expected_failures': Número esperado de excepciones (N * alpha).
            - 'N': Total de observaciones.
            - 'decision': 1 si se rechaza H0 (modelo inexacto), 0 en caso contrario (a 0.05 de significancia para la prueba misma).
    """
    returns = np.array(returns)
    var_estimates = np.array(var_estimates)
    
    if len(returns) != len(var_estimates):
        raise ValueError("Los retornos y las estimaciones de VaR deben tener la misma longitud.")
    
    N = len(returns)
    
    # Contar fallos: casos donde el retorno real es peor (menor) que el VaR
    # Asumiendo VaR negativo (ej. -0.02). Si retorno es -0.03, es un fallo.
    failures = np.sum(returns < var_estimates)
    
    x = failures
    p = alpha
    p_hat = x / N
    
    # Manejar casos borde para cálculo de logaritmo
    if x == 0:
        # Si hay 0 fallos, el cálculo de LR requiere cuidado. 
        # Límite de p_hat^x cuando x->0 es 1.
        # Denominador del Término 2 se vuelve (1-0)^N * 1 = 1.
        # Numerador del Término 1 es (1-p)^N * p^0 = (1-p)^N.
        # LR = -2 ln( (1-p)^N / 1 ) = -2 * N * ln(1-p)
        lr_pof = -2 * N * np.log(1 - p)
    elif x == N:
         # Si todos son fallos
         lr_pof = -2 * N * np.log(p)
    else:
        numerator = ((1 - p) ** (N - x)) * (p ** x)
        denominator = ((1 - p_hat) ** (N - x)) * (p_hat ** x)
        lr_pof = -2 * np.log(numerator / denominator)
    
    # valor p de distribución Chi-cuadrada con 1 grado de libertad
    p_value = 1 - chi2.cdf(lr_pof, df=1)
    
    # Decisión: Rechazar H0 si valor p < 0.05 (significancia estándar para la prueba)
    decision = 1 if p_value < 0.05 else 0
    
    return {
        "LR_POF": lr_pof,
        "p_value": p_value,
        "failures": int(x),
        "expected_failures": N * p,
        "observed_rate": p_hat,
        "target_rate": p,
        "N": N,
        "decision": decision
    }
