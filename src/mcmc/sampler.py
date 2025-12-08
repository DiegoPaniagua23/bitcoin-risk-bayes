import numpy as np
from numba import jit

@jit(nopython=True)
def run_gibbs_sampler(y, n_iter, alpha=2.0, beta=1.0, min_obs=20):
    """
    Ejecuta el muestreo de Gibbs para un modelo de punto de cambio de varianza.

    Modelo:
    y_t ~ N(0, sigma_1^2) para t < k
    y_t ~ N(0, sigma_2^2) para t >= k

    Priors:
    sigma_i^2 ~ InvGamma(alpha, beta)
    k ~ Uniform(min_obs, T - min_obs)

    Args:
        y (np.array): Array de retornos logarítmicos.
        n_iter (int): Número de iteraciones del sampler.
        alpha (float): Parámetro de forma del prior Inverse-Gamma.
        beta (float): Parámetro de escala del prior Inverse-Gamma.
        min_obs (int): Mínimo de observaciones por régimen.

    Returns:
        tuple: (sigma1_sq_samples, sigma2_sq_samples, k_samples)
    """
    T = len(y)

    # Pre-cálculo de sumas acumuladas de cuadrados para eficiencia O(1) en el cálculo de verosimilitud
    y_sq = y**2
    cumsum_y_sq = np.cumsum(y_sq)
    total_sum_sq = cumsum_y_sq[-1]

    # Inicialización de arrays para guardar muestras
    sigma1_sq_samples = np.zeros(n_iter)
    sigma2_sq_samples = np.zeros(n_iter)
    k_samples = np.zeros(n_iter, dtype=np.int64)

    # Valores iniciales
    k = T // 2
    # Estimación inicial simple
    sigma1_sq = np.var(y[:k])
    sigma2_sq = np.var(y[k:])

    # Evitar varianza cero o nula inicial
    if sigma1_sq < 1e-6: sigma1_sq = 1.0
    if sigma2_sq < 1e-6: sigma2_sq = 1.0

    for i in range(n_iter):
        # -------------------------------------------------------
        # 1. Muestrear sigma1_sq | y, k
        # -------------------------------------------------------
        # Datos en régimen 1: y[0] ... y[k-1] (total k observaciones)
        n1 = k
        # Suma de cuadrados: cumsum_y_sq[k-1]
        sum_sq1 = cumsum_y_sq[k-1]

        # Parámetros del posterior Inverse-Gamma
        # IG(a, b) -> a_post = a + n/2, b_post = b + sum_sq/2
        alpha_post1 = alpha + n1 / 2.0
        beta_post1 = beta + sum_sq1 / 2.0

        # Muestrear de Inverse-Gamma: 1 / Gamma(alpha, 1/beta)
        # Nota: np.random.gamma toma (shape, scale), donde scale = 1/rate
        sigma1_sq = 1.0 / np.random.gamma(alpha_post1, 1.0 / beta_post1)

        # -------------------------------------------------------
        # 2. Muestrear sigma2_sq | y, k
        # -------------------------------------------------------
        # Datos en régimen 2: y[k] ... y[T-1] (total T-k observaciones)
        n2 = T - k
        sum_sq2 = total_sum_sq - sum_sq1

        alpha_post2 = alpha + n2 / 2.0
        beta_post2 = beta + sum_sq2 / 2.0

        sigma2_sq = 1.0 / np.random.gamma(alpha_post2, 1.0 / beta_post2)

        # -------------------------------------------------------
        # 3. Muestrear k | y, sigma1_sq, sigma2_sq
        # -------------------------------------------------------
        # Calculamos la log-probabilidad (proporcional) para cada posible k
        # Rango posible de k: [min_obs, T - min_obs]

        log_probs = np.zeros(T)
        # Llenamos con -inf fuera del rango permitido
        log_probs[:] = -np.inf

        # Constantes para el cálculo
        log_sigma1 = np.log(sigma1_sq)
        log_sigma2 = np.log(sigma2_sq)
        inv_2_sigma1 = 1.0 / (2.0 * sigma1_sq)
        inv_2_sigma2 = 1.0 / (2.0 * sigma2_sq)

        # Iteramos sobre los posibles puntos de corte j
        # j representa el inicio del segundo régimen
        for j in range(min_obs, T - min_obs + 1):
            # Log-Likelihood del régimen 1 (0 a j-1)
            # LL1 = - (n1/2)*ln(2*pi) - (n1/2)*ln(sigma1^2) - (1/(2*sigma1^2)) * sum_sq1
            # Ignoramos constantes como ln(2*pi) que se cancelan al normalizar

            n1_j = j
            n2_j = T - j

            sum_sq1_j = cumsum_y_sq[j-1]
            sum_sq2_j = total_sum_sq - sum_sq1_j

            ll_1 = - (n1_j / 2.0) * log_sigma1 - sum_sq1_j * inv_2_sigma1
            ll_2 = - (n2_j / 2.0) * log_sigma2 - sum_sq2_j * inv_2_sigma2

            log_probs[j] = ll_1 + ll_2

        # "Truco log-sum-exp" para estabilidad numérica al convertir log-probs a probs
        # Restamos el máximo para evitar overflow de exp
        max_log_prob = -np.inf
        for j in range(min_obs, T - min_obs + 1):
            if log_probs[j] > max_log_prob:
                max_log_prob = log_probs[j]

        # Calcular probabilidades no normalizadas
        probs = np.zeros(T)
        sum_probs = 0.0
        for j in range(min_obs, T - min_obs + 1):
            p = np.exp(log_probs[j] - max_log_prob)
            probs[j] = p
            sum_probs += p

        # Muestrear k de la distribución discreta
        # Generamos un número aleatorio uniforme escalado por la suma total
        r = np.random.random() * sum_probs
        cumulative = 0.0
        new_k = k # Fallback
        for j in range(min_obs, T - min_obs + 1):
            cumulative += probs[j]
            if r <= cumulative:
                new_k = j
                break
        k = new_k

        # Guardar muestras
        sigma1_sq_samples[i] = sigma1_sq
        sigma2_sq_samples[i] = sigma2_sq
        k_samples[i] = k

    return sigma1_sq_samples, sigma2_sq_samples, k_samples
