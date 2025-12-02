import yfinance as yf
import numpy as np
import pandas as pd
import os

# Configuración
TICKER = "BTC-USD"
START_DATE = "2019-01-01"
END_DATE = "2024-02-01" # Para incluir todo Enero 2024
DATA_DIR = "data"

def download_and_process():
    # Crear carpeta data si no existe
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Directorio '{DATA_DIR}' creado.")

    print(f"⬇️  Descargando datos para {TICKER} ({START_DATE} a {END_DATE})...")

    # Descarga de datos
    try:
        df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    except Exception as e:
        print(f"❌ Error al descargar: {e}")
        return

    if df.empty:
        print("❌ No se obtuvieron datos. Verifica tu conexión.")
        return

    print(f"✅ Datos descargados: {len(df)} registros.")

    # Seleccionar precio de cierre ajustado
    # yfinance a veces devuelve columnas complejas, simplificamos si es necesario
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(TICKER, axis=1, level=1) if TICKER in df.columns.levels[1] else df

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    print(f"ℹ️  Usando columna: {price_col}")

    # Cálculo de Retornos Logarítmicos: r_t = ln(P_t) - ln(P_{t-1})
    df['Log_Return'] = np.log(df[price_col]) - np.log(df[price_col].shift(1))

    # Limpieza (eliminar el primer NaN generado por el shift)
    df_clean = df.dropna(subset=['Log_Return'])

    # Guardar archivos
    raw_file = os.path.join(DATA_DIR, "btc_raw.csv")
    processed_file = os.path.join(DATA_DIR, "btc_log_returns.csv")

    df.to_csv(raw_file)
    df_clean[['Log_Return']].to_csv(processed_file)

    print("\n💾 Archivos guardados:")
    print(f"   1. Crudos:      {raw_file}")
    print(f"   2. Procesados:  {processed_file}")

    print("\n📊 Vista previa de los retornos:")
    print(df_clean[['Log_Return']].head())
    print(f"\nEstadísticas básicas:")
    print(df_clean['Log_Return'].describe())

if __name__ == "__main__":
    download_and_process()
