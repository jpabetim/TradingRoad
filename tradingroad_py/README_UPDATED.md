# TradeRoad - Plataforma de Análisis de Trading con IA

## 📊 Descripción

TradeRoad es una plataforma avanzada de análisis de trading que combina análisis técnico tradicional con inteligencia artificial. Especializada en metodologías Wyckoff y Smart Money Concepts (SMC), la plataforma ofrece:

- **Análisis técnico con IA**: Asistente de trading inteligente que puede "ver" y analizar gráficos
- **Datos en tiempo real**: Integración con múltiples APIs (Binance, Deribit, Finnhub, FMP)
- **Dashboard interactivo**: Resumen del mercado con datos reales
- **Análisis de sentimiento**: Procesamiento de noticias financieras
- **Gestión de volatilidad**: Análisis de opciones y derivados

## 🚀 Características Principales

### ✅ **Problemas Resueltos:**

1. **Asistente IA que VE el gráfico**: El asistente ahora puede recibir tanto datos OHLCV como imágenes del gráfico para análisis completo
2. **Dashboard con datos reales**: El resumen del mercado ahora carga datos en vivo de APIs reales

### 🎯 **Funcionalidades:**

- **Análisis IA Avanzado**: Integración con Google Gemini para análisis de gráficos
- **Datos de Mercado Reales**: APIs de Binance, Deribit, FMP y Finnhub
- **Análisis de Opciones**: Cálculos de Max Pain, ratios Put/Call, volatilidad implícita
- **Sentimiento de Mercado**: Análisis de Open Interest, ratios Long/Short, funding rates
- **Noticias Financieras**: Agregación y análisis de sentimiento automático
- **Calendario Económico**: Eventos importantes del mercado

## 🛠️ Instalación

### Requisitos Previos
- Python 3.8+
- pip

### Instalación Rápida
```bash
# Clonar el repositorio
cd tradingroad_py

# Ejecutar script de instalación
chmod +x install.sh
./install.sh

# O instalación manual:
pip install -r requirements.txt
pip install -r requirements_ai.txt
```

### Configuración
1. Copia el archivo de configuración:
```bash
cp .env.example .env
```

2. Edita `.env` con tus API keys:
```env
VITE_GEMINI_API_KEY=tu_api_key_de_gemini_aqui
FINNHUB_API_KEY=tu_finnhub_key
FMP_API_KEY=tu_fmp_key
```

3. Ejecutar la aplicación:
```bash
python app.py
```

## 🌐 URLs de Acceso

- **Dashboard Principal**: http://localhost:5000
- **Análisis con IA**: http://localhost:5000/analysis  
- **Análisis de Volatilidad**: http://localhost:5000/volatility
- **Noticias**: http://localhost:5000/news
- **Calendario Económico**: http://localhost:5000/calendar

## 🔧 APIs Integradas

### Datos de Criptomonedas
- **Binance**: Precios spot, futuros, funding rates, sentimiento
- **Deribit**: Opciones, volatilidad implícita, order book

### Datos Tradicionales  
- **Financial Modeling Prep**: Acciones, ETFs, indicadores económicos
- **Finnhub**: Noticias financieras, datos fundamentales

### Inteligencia Artificial
- **Google Gemini**: Análisis de gráficos e interpretación de datos

## 📡 Endpoints API Principales

### Mercado
- `GET /api/market/summary` - Resumen del mercado en tiempo real
- `GET /api/market/data` - Datos OHLCV
- `GET /api/market/symbols` - Símbolos disponibles

### IA y Análisis
- `POST /api/ai/analyze-chart` - Análisis de gráfico con IA
- `GET /api/data/<currency>` - Datos de opciones filtrados
- `GET /api/consolidated-metrics/<symbol>` - Métricas consolidadas

### Noticias y Sentimiento
- `GET /api/news` - Noticias financieras agregadas
- `GET /api/sentiment/<symbol>` - Análisis de sentimiento

## 🎨 Interfaz de Usuario

La aplicación utiliza un enfoque **Backend-first** con Flask:
- Templates Jinja2 para renderizado server-side
- JavaScript vanilla para interactividad
- Tailwind CSS para styling moderno
- WebSockets para actualizaciones en tiempo real

## 🤖 Funcionalidades de IA

### Análisis de Gráficos
El asistente IA puede analizar:
- **Datos OHLCV**: Precios, volúmenes, patrones
- **Indicadores técnicos**: RSI, MACD, medias móviles
- **Imágenes de gráficos**: Análisis visual de patrones
- **Contexto de mercado**: Noticias, sentimiento, eventos

### Metodologías Soportadas
- **Análisis Wyckoff**: Fases de mercado, volumen, price action
- **Smart Money Concepts**: Order blocks, liquidity zones, market structure
- **Análisis técnico clásico**: Soportes, resistencias, tendencias

## 📊 Análisis de Datos Avanzado

### Opciones y Derivados
- Cálculo de Max Pain en tiempo real
- Ratios Put/Call por strike y expiración
- Análisis de volatilidad implícita
- Order book de Deribit

### Sentimiento de Mercado
- Open Interest histórico
- Ratios Long/Short de traders
- Funding rates de perpetuos
- Análisis de noticias automatizado

## 🔄 Arquitectura

```
tradingroad_py/
├── app.py                 # Aplicación principal Flask
├── modules/               # Módulos de negocio
│   ├── market/           # Servicios de mercado
│   ├── news/             # Agregación de noticias
│   ├── calendar/         # Calendario económico
│   └── analysis/         # Análisis técnico
├── templates/            # Templates HTML
├── static/               # Assets estáticos
└── config/               # Configuraciones
```

## 🚦 Estado del Proyecto

### ✅ Completado
- [x] Backend Flask funcional
- [x] Integración con APIs de mercado
- [x] Asistente IA que puede "ver" gráficos
- [x] Dashboard con datos reales
- [x] Análisis de opciones y derivados
- [x] Sistema de noticias agregadas
- [x] Calendario económico

### 🔄 En Desarrollo
- [ ] Interfaz React (opcional)
- [ ] Alertas automatizadas
- [ ] Backtesting automatizado
- [ ] Más indicadores técnicos

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE.md](LICENSE.md) para detalles.

## 🙏 Agradecimientos

- APIs utilizadas: Binance, Deribit, Finnhub, Financial Modeling Prep
- Google Gemini para capacidades de IA
- Comunidad open source por las librerías utilizadas

---

**Desarrollado con ❤️ para la comunidad de trading**
