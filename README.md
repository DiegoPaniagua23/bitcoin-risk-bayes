# 🪙 Bitcoin Risk Bayes: JIT-MCMC vs. GJR-GARCH

Este proyecto investiga la dinámica de la volatilidad extrema en el mercado de Bitcoin (BTC-USD) durante el periodo 2019-2024. Se propone contrastar dos paradigmas de modelado avanzados: un enfoque **econométrico robusto** (GJR-GARCH con errores t-Student) frente a una metodología **computacional intensiva** basada en Inferencia Bayesiana (Change-Point Detection vía Gibbs Sampling).

## 📂 Estructura del Proyecto

```text
bitcoin-risk-bayes/
├── data/               # Datos crudos y procesados (BTC-USD)
├── notebooks/          # Notebooks para exploración y prototipado
├── results/            # Resultados generados
│   ├── figures/        # Gráficas y visualizaciones
│   └── models/         # Modelos serializados (.pkl, .json)
├── src/                # Código fuente del proyecto
│   ├── garch/          # Implementación del modelo GJR-GARCH
│   ├── mcmc/           # Implementación del Gibbs Sampler (Numba)
│   ├── utils/          # Funciones de utilidad y helpers
│   │   └── download_data.py # Script para descarga de datos
│   └── main.py         # Punto de entrada principal
├── prompts/            # Documentación de prompts y propuestas
├── pyproject.toml      # Configuración de dependencias (uv)
└── README.md           # Documentación general
```

## 🚀 Quick Start

Este proyecto utiliza `uv` para la gestión de dependencias y entornos virtuales.

### 1. Clonar el repositorio
```bash
git clone https://github.com/DiegoPaniagua23/bitcoin-risk-bayes.git
cd bitcoin-risk-bayes
```

### 2. Configurar el entorno
```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear entorno virtual e instalar dependencias
uv sync
```

### 3. Descargar datos
```bash
uv run python src/utils/download_data.py
```

## 🛠️ Stack Tecnológico

*   **Lenguaje:** Python 3.11
*   **Gestor de Paquetes:** `uv`
*   **Optimización:** `numba` (JIT Compilation)
*   **Econometría:** `arch`
*   **Análisis de Datos:** `pandas`, `numpy`, `scipy`
*   **Visualización:** `matplotlib`

## 👥 Colaboración

El proyecto sigue una estrategia de **Feature Branching**:

*   `main`: Rama de producción.
*   `develop`: Rama de integración principal.
*   `feat/mcmc-core`: Desarrollo del motor Bayesiano.
*   `feat/risk-boot`: Implementación de Bootstrap y validación.
*   `feat/garch-model`: Modelado econométrico.

---
**CIMAT - Cómputo Estadístico**
