# 📄 Propuesta de Proyecto Final
**Centro de Investigación en Matemáticas (CIMAT) - Unidad Monterrey**
**Materia:** Cómputo Estadístico

---

## 1. Título del Proyecto 🏷️
**Inferencia Bayesiana de Puntos de Cambio y Riesgo en Bitcoin: JIT-MCMC vs. GJR-GARCH**

### Resumen General
Este proyecto investiga la dinámica de la volatilidad extrema en el mercado de Bitcoin (BTC-USD) durante el periodo 2019-2024. Se propone contrastar dos paradigmas de modelado avanzados:
* Un enfoque **econométrico robusto** (GJR-GARCH con errores t-Student).
* Frente a una metodología **computacional intensiva** basada en Inferencia Bayesiana (Change-Point Detection vía Gibbs Sampling).

El diferenciador técnico reside en la optimización computacional mediante compilación **Just-In-Time (JIT)** y el uso de técnicas de **Bootstrap** para construir intervalos de confianza sobre las métricas de riesgo (VaR), evaluando si el costo computacional adicional justifica la ganancia en precisión predictiva.

---

## 2. Objetivo General 🎯
Implementar y comparar un algoritmo de **Muestreo de Gibbs (Gibbs Sampler)** diseñado *ad-hoc* y optimizado con **Numba**, frente a modelos de volatilidad asimétrica (**GJR-GARCH**), evaluando su capacidad para capturar "colas pesadas" y predecir riesgos extremos mediante intervalos de confianza generados por Bootstrap en la serie de tiempo de retornos logarítmicos del Bitcoin.

---

## 3. Descripción del Problema 📉
Las series de tiempo de criptoactivos violan los supuestos clásicos de normalidad. Presentan dos fenómenos críticos que los modelos básicos ignoran:

* **Colas Pesadas (Fat Tails):** La probabilidad de eventos extremos (pérdidas > 10%) es significativamente mayor a lo que predice una distribución Normal.
* **Clusters de Volatilidad:** La varianza no es constante, sino que sufre cambios de régimen abruptos debido a *shocks* exógenos.

El reto consiste en superar a los modelos econométricos estándar mediante técnicas de cómputo intensivo que se adapten más rápido a estos cambios estructurales.

---

## 4. Dataset 💾
Se utilizarán datos históricos diarios de precios de cierre ajustados.

* **Fuente:** Yahoo Finance API (`yfinance`).
* **Activo:** Bitcoin vs Dólar Estadounidense (BTC-USD).
* **Periodo:** Enero 2019 – Enero 2024 (Incluye crisis COVID-19 y colapso FTX).
* **Transformación:** Retornos Logarítmicos Diarios:
  $$r_{t}=ln(P_{t})-ln(P_{t-1})$$

---

## 5. Metodología 🛠️

### 5.1. Baseline Competitivo (Estándar Robusto)
Implementación de un modelo **GJR-GARCH (1,1)** que captura la asimetría de los shocks (efecto apalancamiento).

* **Ecuación de Varianza:**
  $$\sigma_{t}^{2}=\omega+(\alpha+\gamma I_{t-1})\epsilon_{t-1}^{2}+\beta\sigma_{t-1}^{2}$$
* **Innovación:** Se asumirá que los errores siguen una distribución **t-Student estandarizada** (no Normal) para capturar explícitamente las colas pesadas.

### 5.2. Propuesta Bayesiana (Cómputo Intensivo)
Desarrollo de un modelo jerárquico de Punto de Cambio estimado mediante **Gibbs Sampling**.

* **Modelo:**
  $$y_{t}\sim\mathcal{N}(0,\sigma_{t}^{2})$$
  Donde la varianza cambia en $t=k$.
* **Optimización JIT:** Se utilizará la librería **Numba** para compilar las funciones de muestreo a código máquina, buscando reducir el tiempo de inferencia de horas a minutos.

### 5.3. Estimación de Incertidumbre (Bootstrap)
Para robustecer la validación, no solo se calculará el VaR puntual. Se aplicará **Bootstrap de Bloques** sobre los residuos estandarizados para generar Intervalos de Confianza al 95% para el VaR.

---

## 6. Métricas de Evaluación 📏
1.  **Ajuste:** Criterios de información **AIC** y **BIC** penalizando la complejidad.
2.  **Riesgo de Cola:** Value at Risk (VaR) al $5\%$ y $1\%$.
3.  **Backtesting:** Prueba de Kupiec (POF) para validar si la frecuencia de excepciones coincide con el nivel de confianza teórico ($H_{0}$: Tasa de fallos $=c$).

---

## 7. Diseño Experimental 🧪
El experimento se dividirá en tres fases secuenciales:

1.  **Calibración:** Ajuste de modelos sobre el 80% del dataset (Entrenamiento).
2.  **Diagnóstico:** Verificación de convergencia MCMC (Trace Plots, Gelman-Rubin) y análisis de residuales GARCH (Ljung-Box).
3.  **Inferencia y Validación:** Generación de la distribución predictiva y cálculo de intervalos de confianza vía Bootstrap.

---

## 8. Resultados Esperados 📊
* **Eficiencia Computacional:** Demostrar que la implementación con Numba acelera el proceso MCMC al menos **50x** comparado con Python puro.
* **Sensibilidad:** Se espera que el modelo Bayesiano detecte cambios de régimen estructurales antes que el GARCH.
* **Robustez:** Obtención de intervalos de confianza para el VaR que permitan una comparación estadística rigurosa más allá de la estimación puntual.

---

## 9. División de Roles y Arquitectura 👥
La carga de trabajo se distribuye para aprovechar las 3 unidades clave del curso.

| Integrante | Rol | Responsabilidad Técnica Principal |
| :--- | :--- | :--- |
| **Miembro 1** | Ingeniero MCMC | Implementación del Gibbs Sampler y optimización de bajo nivel con decoradores `@jit` (Numba). |
| **Miembro 2** | Analista de Validación | Implementación de técnicas de Bootstrap para intervalos de confianza y ejecución del Test de Kupiec. |
| **Miembro 3** | Data Scientist | Implementación del GJR-GARCH con distribución t-Student y benchmarking de modelos (AIC/BIC). |

---

## 10. Stack Tecnológico 💻
* **Lenguaje:** Python 3.10+.
* **HPC / Optimización:** Numba (Compilador JIT).
* **Librerías Científicas:** `scipy.stats`, `numpy`, `arch` (econometría).
* **Entorno:** Linux (Ubuntu) + VS Code / Neovim.

---

## 11. Flujo de Trabajo GitHub 🐙
Se utilizará una estrategia de **Feature Branching** para colaboración paralela.

* **Ramas (Branches):**
    * `main`: Producción (Código final del reporte).
    * `develop`: Integración (Donde se unen las partes).
    * `feat/mcmc-core`: Desarrollo del motor Bayesiano (Miembro 1).
    * `feat/risk-boot`: Implementación de Bootstrap y validación (Miembro 2).
    * `feat/garch-model`: Modelado econométrico (Miembro 3).

* **Reglas de Merge:** Todo código debe pasar por **Pull Request (PR)** con revisión cruzada.

---

## 12. Conclusión 🏁
Este proyecto eleva la complejidad del análisis estándar al integrar técnicas de optimización de código y modelos estadísticos robustos frente a la no-normalidad. Se busca proveer evidencia empírica sobre si el costo computacional de los métodos Bayesianos e intensivos (MCMC, Bootstrap) se justifica en entornos de alta frecuencia y riesgo extremo.
