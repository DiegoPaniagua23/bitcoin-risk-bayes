import numpy as np
import pandas as pd
from arch import arch_model

class GarchModel:
    """
    Wrapper para el modelo GJR-GARCH(1,1) con distribución t-Student usando la librería 'arch'.
    """
    def __init__(self, p=1, o=1, q=1, dist='t'):
        """
        Inicializa el modelo GJR-GARCH.

        Args:
            p (int): Lag de varianza (GARCH).
            o (int): Lag de asimetría (GJR). Si o>0, es GJR-GARCH.
            q (int): Lag de error (ARCH).
            dist (str): Distribución de los errores ('t' para Student, 'norm' para Normal).
        """
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.model = None
        self.res = None

    def fit(self, returns):
        """
        Ajusta el modelo a la serie de retornos.

        Args:
            returns (pd.Series or np.array): Serie de retornos (preferiblemente escalados x100).
        """
        # rescale=False asume que el usuario ya escaló los datos si es necesario (recomendado x100)
        self.model = arch_model(returns, vol='Garch', p=self.p, o=self.o, q=self.q, dist=self.dist, rescale=False)
        self.res = self.model.fit(disp='off')
        return self.res

    def get_conditional_volatility(self):
        """Retorna la volatilidad condicional estimada (sigma_t)."""
        if self.res is None:
            raise ValueError("El modelo no ha sido ajustado. Ejecuta .fit() primero.")
        return self.res.conditional_volatility

    def get_standardized_residuals(self):
        """Retorna los residuos estandarizados (epsilon_t / sigma_t)."""
        if self.res is None:
            raise ValueError("El modelo no ha sido ajustado. Ejecuta .fit() primero.")
        return self.res.std_resid

    def get_aic_bic(self):
        """Retorna AIC y BIC del modelo."""
        if self.res is None:
            raise ValueError("El modelo no ha sido ajustado.")
        return {'AIC': self.res.aic, 'BIC': self.res.bic}

    def summary(self):
        """Retorna el resumen estadístico del ajuste."""
        if self.res is None:
            raise ValueError("El modelo no ha sido ajustado.")
        return self.res.summary()

    def calculate_var(self, alpha=0.05):
        """
        Calcula el Value at Risk (VaR) histórico condicional.

        Args:
            alpha (float): Nivel de significancia (ej. 0.05 para 95% confianza).

        Returns:
            pd.Series: VaR estimado para cada punto en el tiempo.
        """
        if self.res is None:
            raise ValueError("El modelo no ha sido ajustado.")

        # Parámetros estimados
        params = self.res.params
        mu = params['mu'] # Media constante estimada
        sigma = self.res.conditional_volatility

        # Obtener el cuantil de la distribución ajustada
        if self.dist == 't':
            nu = params['nu'] # Grados de libertad
            # ppf: Percent point function (inverse of cdf)
            q = self.model.distribution.ppf(alpha, nu)
        elif self.dist == 'norm':
            q = self.model.distribution.ppf(alpha)
        else:
            # Fallback genérico si se usa skewt u otros
            q = self.model.distribution.ppf(alpha, params)

        # VaR_t = mu + sigma_t * q_alpha
        # Nota: q_alpha será negativo para alpha < 0.5 (pérdidas)
        var_series = mu + sigma * q
        return var_series
