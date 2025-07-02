// TradingRoad Professional - Advanced Trading Platform

class TradingRoadChart {
    constructor() {
        this.chart = null;
        this.activeIndicators = new Map();
        this.currentSymbol = 'BTCUSDT';
        this.currentTimeframe = '4h';
        this.currentExchange = 'binance';
        this.websocket = null;
        this.candleData = [];
        this.indicators = new Map();
        this.isRealTime = true;
        this.priceScale = 'right';

        // APIs de exchanges configuradas
        this.exchangeAPIs = {
            binance: {
                rest: 'https://api.binance.com/api/v3',
                websocket: 'wss://stream.binance.com:9443/ws',
                klineEndpoint: '/klines',
                tickerEndpoint: '/ticker/24hr'
            },
            bybit: {
                rest: 'https://api.bybit.com/v2/public',
                websocket: 'wss://stream.bybit.com/realtime',
                klineEndpoint: '/kline/list',
                tickerEndpoint: '/tickers'
            },
            coinbase: {
                rest: 'https://api.exchange.coinbase.com',
                websocket: 'wss://ws-feed.exchange.coinbase.com',
                klineEndpoint: '/products',
                tickerEndpoint: '/products'
            }
        };

        // Configuración de timeframes para cada exchange
        this.timeframeMapping = {
            binance: {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
            },
            bybit: {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '4h': '240', '1d': 'D', '1w': 'W'
            },
            coinbase: {
                '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800
            }
        };

        // Colores profesionales TradingRoad
        this.colors = {
            bull: '#089981',
            bear: '#F23645',
            background: '#0C111C',
            grid: '#2A2E39',
            text: '#D1D4DC',
            accent: '#2962FF',
            volume: '#434651'
        };

        this.init();
    }

    init() {
        this.initChart();
        this.bindEvents();
        this.loadInitialData();
        this.setupIndicators();
    }

    // Configuración inicial del gráfico ECharts para TradingRoad
    initChart() {
        const chartContainer = document.getElementById('tradingChart');
        if (!chartContainer) return;

        this.chart = echarts.init(chartContainer, null, {
            backgroundColor: this.colors.background
        });

        // Configuración base del gráfico
        const option = {
            backgroundColor: this.colors.background,
            animation: false,
            grid: [{
                id: 'mainGrid',
                left: '10%',
                right: '8%',
                top: '8%',
                height: '65%'
            }, {
                id: 'volumeGrid',
                left: '10%',
                right: '8%',
                top: '75%',
                height: '20%'
            }],
            xAxis: [{
                type: 'category',
                data: [],
                axisLine: { lineStyle: { color: this.colors.grid } },
                axisTick: { show: false },
                axisLabel: { 
                    color: this.colors.text,
                    fontSize: 11
                },
                splitLine: { show: false },
                gridIndex: 0
            }, {
                type: 'category',
                data: [],
                axisLine: { lineStyle: { color: this.colors.grid } },
                axisTick: { show: false },
                axisLabel: { show: false },
                splitLine: { show: false },
                gridIndex: 1
            }],
            yAxis: [{
                type: 'value',
                scale: true,
                position: this.priceScale,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { 
                    color: this.colors.text,
                    fontSize: 11,
                    formatter: (value) => this.formatPrice(value)
                },
                splitLine: { 
                    lineStyle: { 
                        color: this.colors.grid,
                        opacity: 0.3
                    }
                },
                gridIndex: 0
            }, {
                type: 'value',
                scale: true,
                position: this.priceScale,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { 
                    color: this.colors.text,
                    fontSize: 11,
                    formatter: (value) => this.formatVolume(value)
                },
                splitLine: { 
                    lineStyle: { 
                        color: this.colors.grid,
                        opacity: 0.2
                    }
                },
                gridIndex: 1
            }],
            series: [{
                name: 'Candlestick',
                type: 'candlestick',
                data: [],
                itemStyle: {
                    color: this.colors.bull,
                    color0: this.colors.bear,
                    borderColor: this.colors.bull,
                    borderColor0: this.colors.bear
                },
                xAxisIndex: 0,
                yAxisIndex: 0
            }, {
                name: 'Volume',
                type: 'bar',
                data: [],
                itemStyle: {
                    color: (params) => {
                        const candle = this.candleData[params.dataIndex];
                        if (candle && candle[1] <= candle[2]) {
                            return this.colors.bear + '80';
                        }
                        return this.colors.bull + '80';
                    }
                },
                xAxisIndex: 1,
                yAxisIndex: 1
            }],
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    lineStyle: {
                        color: this.colors.accent,
                        opacity: 0.8
                    }
                },
                backgroundColor: '#1E222D',
                borderColor: this.colors.grid,
                textStyle: {
                    color: this.colors.text,
                    fontSize: 12
                },
                formatter: (params) => this.formatTooltip(params)
            },
            dataZoom: [{
                type: 'inside',
                xAxisIndex: [0, 1],
                start: 80,
                end: 100
            }, {
                show: true,
                xAxisIndex: [0, 1],
                type: 'slider',
                bottom: 10,
                height: 20,
                backgroundColor: this.colors.background,
                fillerColor: this.colors.accent + '40',
                borderColor: this.colors.grid,
                handleStyle: {
                    color: this.colors.accent
                },
                textStyle: {
                    color: this.colors.text
                }
            }],
            brush: {
                xAxisIndex: 'all',
                brushLink: 'all',
                outOfBrush: {
                    colorAlpha: 0.1
                }
            }
        };

        this.chart.setOption(option);
        this.setupChartEvents();
    }

    // Eventos del gráfico
    setupChartEvents() {
        this.chart.on('datazoom', (params) => {
            this.onDataZoom(params);
        });

        this.chart.on('click', (params) => {
            this.onChartClick(params);
        });

        // Redimensionar cuando cambie el tamaño de la ventana
        window.addEventListener('resize', () => {
            this.chart.resize();
        });
    }

    // Cargar datos iniciales
    async loadInitialData() {
        try {
            this.showLoading(true);
            await this.loadCandleData();
            this.updateChart();
            this.showLoading(false);
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showLoading(false);
            this.showError('Error al cargar datos iniciales');
        }
    }

    // Cargar datos de velas desde la API
    async loadCandleData() {
        const api = this.exchangeAPIs[this.currentExchange];
        const timeframe = this.timeframeMapping[this.currentExchange][this.currentTimeframe];
        
        let url;
        switch (this.currentExchange) {
            case 'binance':
                url = `${api.rest}${api.klineEndpoint}?symbol=${this.currentSymbol}&interval=${timeframe}&limit=500`;
                break;
            case 'bybit':
                url = `${api.rest}${api.klineEndpoint}?symbol=${this.currentSymbol}&interval=${timeframe}&limit=500`;
                break;
            case 'coinbase':
                const end = Math.floor(Date.now() / 1000);
                const start = end - (500 * timeframe);
                url = `${api.rest}${api.klineEndpoint}/${this.currentSymbol}/candles?start=${start}&end=${end}&granularity=${timeframe}`;
                break;
        }

        try {
            const response = await fetch(url);
            const data = await response.json();
            this.candleData = this.parseKlineData(data);
        } catch (error) {
            console.error('Error fetching candle data:', error);
            // Fallback a datos simulados
            this.candleData = this.generateMockData();
        }
    }

    // Parsear datos de velas según el exchange
    parseKlineData(data) {
        switch (this.currentExchange) {
            case 'binance':
                return data.map(kline => [
                    parseFloat(kline[1]), // open
                    parseFloat(kline[4]), // close
                    parseFloat(kline[3]), // low
                    parseFloat(kline[2]), // high
                    parseFloat(kline[5]), // volume
                    new Date(kline[0]).toISOString()
                ]);
            case 'bybit':
                return data.result.map(kline => [
                    parseFloat(kline.open),
                    parseFloat(kline.close),
                    parseFloat(kline.low),
                    parseFloat(kline.high),
                    parseFloat(kline.volume),
                    new Date(kline.open_time * 1000).toISOString()
                ]);
            case 'coinbase':
                return data.map(kline => [
                    parseFloat(kline[3]), // open
                    parseFloat(kline[4]), // close
                    parseFloat(kline[1]), // low
                    parseFloat(kline[2]), // high
                    parseFloat(kline[5]), // volume
                    new Date(kline[0] * 1000).toISOString()
                ]);
            default:
                return this.generateMockData();
        }
    }

    // Generar datos simulados para fallback
    generateMockData() {
        const data = [];
        let basePrice = 45000;
        const now = new Date();
        
        for (let i = 499; i >= 0; i--) {
            const timestamp = new Date(now.getTime() - i * this.getTimeframeMilliseconds()).toISOString();
            const volatility = 0.02;
            const change = (Math.random() - 0.5) * volatility;
            
            const open = basePrice;
            const close = basePrice * (1 + change);
            const high = Math.max(open, close) * (1 + Math.random() * 0.01);
            const low = Math.min(open, close) * (1 - Math.random() * 0.01);
            const volume = Math.random() * 1000000;
            
            data.push([open, close, low, high, volume, timestamp]);
            basePrice = close;
        }
        
        return data;
    }

    // Obtener millisegundos por timeframe
    getTimeframeMilliseconds() {
        const timeframes = {
            '1m': 60000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '4h': 14400000,
            '1d': 86400000,
            '1w': 604800000
        };
        return timeframes[this.currentTimeframe] || 3600000;
    }

    // Actualizar el gráfico con nuevos datos
    updateChart() {
        if (!this.chart || !this.candleData.length) return;

        const times = this.candleData.map(d => this.formatTime(d[5]));
        const candlestickData = this.candleData.map(d => [d[0], d[1], d[2], d[3]]);
        const volumeData = this.candleData.map(d => d[4]);

        this.chart.setOption({
            xAxis: [{
                data: times
            }, {
                data: times
            }],
            series: [{
                data: candlestickData
            }, {
                data: volumeData
            }]
        });

        this.updateIndicators();
    }

    // Configurar indicadores técnicos
    setupIndicators() {
        this.indicators = new Map([
            ['SMA_20', { name: 'SMA 20', color: '#FF6B6B', visible: false }],
            ['SMA_50', { name: 'SMA 50', color: '#4ECDC4', visible: false }],
            ['EMA_20', { name: 'EMA 20', color: '#45B7D1', visible: false }],
            ['RSI', { name: 'RSI', color: '#FFA726', visible: false }],
            ['MACD', { name: 'MACD', color: '#AB47BC', visible: false }],
            ['Bollinger', { name: 'Bollinger Bands', color: '#66BB6A', visible: false }]
        ]);

        this.renderIndicatorControls();
    }

    // Renderizar controles de indicadores
    renderIndicatorControls() {
        const container = document.getElementById('indicatorsList');
        if (!container) return;

        let html = '';
        this.indicators.forEach((indicator, key) => {
            html += `
                <div class="indicator-item">
                    <input type="checkbox" id="indicator_${key}" class="indicator-checkbox" 
                           ${indicator.visible ? 'checked' : ''}>
                    <label for="indicator_${key}" class="indicator-label">${indicator.name}</label>
                    <div class="indicator-color" style="background-color: ${indicator.color}"></div>
                </div>
            `;
        });

        container.innerHTML = html;
        this.bindIndicatorEvents();
    }

    // Eventos de los indicadores
    bindIndicatorEvents() {
        document.querySelectorAll('.indicator-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const indicatorKey = e.target.id.replace('indicator_', '');
                const indicator = this.indicators.get(indicatorKey);
                if (indicator) {
                    indicator.visible = e.target.checked;
                    this.updateIndicators();
                }
            });
        });
    }

    // Actualizar indicadores en el gráfico
    updateIndicators() {
        const series = [];
        
        this.indicators.forEach((indicator, key) => {
            if (indicator.visible) {
                const data = this.calculateIndicator(key);
                if (data) {
                    series.push({
                        name: indicator.name,
                        type: 'line',
                        data: data,
                        lineStyle: {
                            color: indicator.color,
                            width: 1
                        },
                        symbol: 'none',
                        xAxisIndex: 0,
                        yAxisIndex: 0
                    });
                }
            }
        });

        // Agregar series de indicadores manteniendo las series base
        const option = this.chart.getOption();
        option.series = option.series.slice(0, 2).concat(series);
        this.chart.setOption(option);
    }

    // Calcular indicadores técnicos
    calculateIndicator(type) {
        const prices = this.candleData.map(d => d[1]); // precios de cierre
        
        switch (type) {
            case 'SMA_20':
                return this.calculateSMA(prices, 20);
            case 'SMA_50':
                return this.calculateSMA(prices, 50);
            case 'EMA_20':
                return this.calculateEMA(prices, 20);
            case 'RSI':
                return this.calculateRSI(prices, 14);
            case 'MACD':
                return this.calculateMACD(prices);
            case 'Bollinger':
                return this.calculateBollingerBands(prices, 20, 2);
            default:
                return null;
        }
    }

    // Calcular SMA (Simple Moving Average)
    calculateSMA(prices, period) {
        const sma = [];
        for (let i = 0; i < prices.length; i++) {
            if (i < period - 1) {
                sma.push(null);
            } else {
                const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
                sma.push(sum / period);
            }
        }
        return sma;
    }

    // Calcular EMA (Exponential Moving Average)
    calculateEMA(prices, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);
        
        for (let i = 0; i < prices.length; i++) {
            if (i === 0) {
                ema.push(prices[i]);
            } else {
                ema.push((prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier)));
            }
        }
        return ema;
    }

    // Calcular RSI (Relative Strength Index)
    calculateRSI(prices, period) {
        const gains = [];
        const losses = [];
        
        for (let i = 1; i < prices.length; i++) {
            const change = prices[i] - prices[i - 1];
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? Math.abs(change) : 0);
        }
        
        const rsi = [null];
        for (let i = period - 1; i < gains.length; i++) {
            const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
            const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
            
            if (avgLoss === 0) {
                rsi.push(100);
            } else {
                const rs = avgGain / avgLoss;
                rsi.push(100 - (100 / (1 + rs)));
            }
        }
        
        return rsi;
    }

    // Calcular MACD
    calculateMACD(prices) {
        const ema12 = this.calculateEMA(prices, 12);
        const ema26 = this.calculateEMA(prices, 26);
        const macdLine = ema12.map((val, i) => val && ema26[i] ? val - ema26[i] : null);
        return macdLine;
    }

    // Calcular Bollinger Bands
    calculateBollingerBands(prices, period, stdDev) {
        const sma = this.calculateSMA(prices, period);
        const upperBand = [];
        const lowerBand = [];
        
        for (let i = 0; i < prices.length; i++) {
            if (i < period - 1) {
                upperBand.push(null);
                lowerBand.push(null);
            } else {
                const slice = prices.slice(i - period + 1, i + 1);
                const mean = sma[i];
                const variance = slice.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / period;
                const standardDeviation = Math.sqrt(variance);
                
                upperBand.push(mean + (standardDeviation * stdDev));
                lowerBand.push(mean - (standardDeviation * stdDev));
            }
        }
        
        return { upper: upperBand, middle: sma, lower: lowerBand };
    }

    // Formatear precio para display
    formatPrice(value) {
        if (value >= 1000) {
            return value.toFixed(0);
        } else if (value >= 1) {
            return value.toFixed(2);
        } else {
            return value.toFixed(4);
        }
    }

    // Formatear volumen para display
    formatVolume(value) {
        if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        } else if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        }
        return value.toFixed(0);
    }

    // Formatear tiempo para el eje X
    formatTime(timestamp) {
        const date = new Date(timestamp);
        switch (this.currentTimeframe) {
            case '1m':
            case '5m':
            case '15m':
            case '30m':
                return date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
            case '1h':
            case '4h':
                return date.toLocaleDateString('es-ES', { month: 'short', day: 'numeric', hour: '2-digit' });
            case '1d':
            case '1w':
                return date.toLocaleDateString('es-ES', { month: 'short', day: 'numeric' });
            default:
                return date.toLocaleDateString('es-ES', { month: 'short', day: 'numeric' });
        }
    }

    // Formatear tooltip
    formatTooltip(params) {
        if (!params || !params.length) return '';
        
        const candleParam = params.find(p => p.seriesName === 'Candlestick');
        if (!candleParam) return '';
        
        const data = candleParam.data;
        const volumeParam = params.find(p => p.seriesName === 'Volume');
        
        let html = `
            <div style="font-weight: bold; margin-bottom: 5px;">${this.currentSymbol}</div>
            <div>O: ${this.formatPrice(data[1])}</div>
            <div>H: ${this.formatPrice(data[4])}</div>
            <div>L: ${this.formatPrice(data[3])}</div>
            <div>C: ${this.formatPrice(data[2])}</div>
        `;
        
        if (volumeParam) {
            html += `<div>Vol: ${this.formatVolume(volumeParam.data)}</div>`;
        }
        
        return html;
    }

    // Eventos de controles
    bindEvents() {
        // Cambio de símbolo
        const symbolInput = document.getElementById('symbolInput');
        const searchBtn = document.getElementById('searchSymbol');
        
        if (symbolInput && searchBtn) {
            const handleSymbolChange = () => {
                const newSymbol = symbolInput.value.toUpperCase().replace('/', '');
                if (newSymbol !== this.currentSymbol) {
                    this.currentSymbol = newSymbol;
                    this.loadInitialData();
                }
            };
            
            searchBtn.addEventListener('click', handleSymbolChange);
            symbolInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') handleSymbolChange();
            });
        }

        // Cambio de timeframe
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.currentTimeframe = e.target.dataset.tf;
                this.loadInitialData();
            });
        });

        // Cambio de exchange
        const exchangeSelect = document.getElementById('exchangeSelect');
        if (exchangeSelect) {
            exchangeSelect.addEventListener('change', (e) => {
                this.currentExchange = e.target.value;
                this.loadInitialData();
            });
        }

        // Botón de indicadores
        const addIndicatorBtn = document.getElementById('addIndicator');
        if (addIndicatorBtn) {
            addIndicatorBtn.addEventListener('click', () => {
                this.toggleIndicatorPanel();
            });
        }
    }

    // Toggle panel de indicadores
    toggleIndicatorPanel() {
        const panel = document.getElementById('indicatorPanel');
        if (panel) {
            panel.classList.toggle('hidden');
        }
    }

    // Mostrar/ocultar loading
    showLoading(show) {
        const loader = document.getElementById('chartLoader');
        if (loader) {
            loader.style.display = show ? 'block' : 'none';
        }
    }

    // Mostrar error
    showError(message) {
        console.error(message);
        // Aquí podrías agregar una notificación visual del error
    }

    // Eventos del gráfico
    onDataZoom(params) {
        // Manejar zoom del gráfico
        console.log('DataZoom event:', params);
    }

    onChartClick(params) {
        // Manejar clicks en el gráfico
        console.log('Chart click:', params);
    }

    // Destruir instancia
    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }
        if (this.chart) {
            this.chart.dispose();
        }
    }
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    window.tradingRoadChart = new TradingRoadChart();
});
