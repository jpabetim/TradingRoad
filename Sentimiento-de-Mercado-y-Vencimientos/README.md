# Dashboard de Análisis de Opciones y Derivados

Un dashboard de análisis técnico que visualiza datos de mercado en tiempo real, combinando información de las APIs de **Deribit** y **Binance** para ofrecer una visión completa del mercado de derivados de criptomonedas.

![Screenshot del Dashboard](https://i.imgur.com/r6nB9t8.png)
*(Recomendación: Reemplaza esta imagen de ejemplo con una captura de tu propio dashboard.)*

## Características Principales

### Métricas Generales y de Mercado
- **Panel Superior:** Métricas clave como el Interés Abierto (OI) Total Promedio, Funding Rate, tiempo para la próxima financiación, precio actual y máximos/mínimos de la semana.
- **Punto de Máximo Dolor (Max Pain):** Cálculo del Max Pain que se actualiza dinámicamente según la fecha de vencimiento seleccionada.

### Análisis de Opciones (Deribit)
- **Interés Abierto por Strike:** Gráfico de barras que muestra la distribución de OI para Puts y Calls en cada precio de ejercicio.
- **Métricas Detalladas por Vencimiento:** Un panel de datos integrado que se actualiza con la fecha seleccionada y muestra:
  - Call y Put Open Interest.
  - Total Open Interest.
  - **Put/Call Ratio** (por Interés Abierto y por Volumen).
  - Valor Nocional total de las posiciones.
- **Volumen por Strike (24h):** Gráfico de barras apiladas que muestra el volumen de negociación de Puts y Calls por strike, ideal para detectar actividad reciente.
- **Sonrisa de Volatilidad (IV Smile):** Gráfico de líneas que se activa al seleccionar un vencimiento para visualizar la Volatilidad Implícita (IV) de Puts y Calls, mostrando el "skew" o sesgo del mercado.

### Indicadores de Sentimiento y Mercado
- **Historial de Volatilidad (DVOL):** Gráfico de área que muestra la evolución del índice de volatilidad de Deribit (con una media móvil de 7 días para suavizar la tendencia).
- **Historial de Interés Abierto (Binance):** Gráfico de la evolución del OI agregado en Binance.
- **Ratio Long/Short (Binance):** Historial del ratio de posiciones largas vs. cortas de las cuentas de Binance.
- **Historial de Funding Rate:** Evolución de la tasa de financiación para el contrato perpetuo.

### Herramientas Adicionales
- **Libro de Órdenes (Deribit):** Visualización en tiempo real del libro de órdenes del contrato perpetuo, con refresco cada 10 segundos y niveles de agregación personalizables (Detalle, 100, 1K, 5K, 10K).

## Tecnologías Utilizadas

- **Backend:** Python, Flask, Pandas, Requests, Numpy.
- **Frontend:** HTML5, CSS3, JavaScript.
- **Visualización de Datos:** ApexCharts.js.

## Instalación y Uso

Para ejecutar este proyecto en tu máquina local, necesitarás tener dos terminales abiertas.

### Terminal 1: Activar el Backend
```bash
# 1. Navega a la carpeta del backend
cd ruta/a/tu/proyecto/backend

# 2. Crea y activa un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instala las dependencias
pip install -r requirements.txt

# 4. Ejecuta el servidor de Flask
python app.py