// TealStreet Pro - Implementación profesional con APIs reales y indicadores técnicos

class TealStreetPro {
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

        // Colores profesionales estilo TealStreet
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

    // Configuración inicial del gráfico ECharts estilo TealStreet
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
                gridIndex: 0,
                data: [],
                scale: true,
                boundaryGap: false,
                axisLine: { onZero: false, lineStyle: { color: this.colors.grid } },
                axisTick: { show: false },
                axisLabel: {
                    color: this.colors.text,
                    fontSize: 10,
                    formatter: this.formatTime.bind(this)
                },
                splitLine: { show: false }
            }, {
                type: 'category',
                gridIndex: 1,
                data: [],
                scale: true,
                boundaryGap: false,
                axisLine: { onZero: false, lineStyle: { color: this.colors.grid } },
                axisTick: { show: false },
                axisLabel: { show: false },
                splitLine: { show: false }
            }],
            yAxis: [{
                scale: true,
                gridIndex: 0,
                position: 'right',
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: {
                    color: this.colors.text,
                    fontSize: 10,
                    formatter: value => this.formatPrice(value)
                },
                splitLine: {
                    lineStyle: {
                        color: this.colors.grid,
                        type: 'dashed',
                        opacity: 0.3
                    }
                }
            }, {
                scale: true,
                gridIndex: 1,
                position: 'right',
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: {
                    color: this.colors.text,
                    fontSize: 9,
                    formatter: value => this.formatVolume(value)
                },
                splitLine: { show: false }
            }],
            dataZoom: [{
                type: 'inside',
                xAxisIndex: [0, 1],
                filterMode: 'filter'
            }, {
                type: 'slider',
                xAxisIndex: [0, 1],
                bottom: '2%',
                height: '3%',
                backgroundColor: this.colors.background,
                fillerColor: this.colors.accent + '20',
                borderColor: this.colors.grid,
                handleStyle: { color: this.colors.accent },
                textStyle: { color: this.colors.text }
            }],
            series: [{
                name: 'Candlestick',
                type: 'candlestick',
                xAxisIndex: 0,
                yAxisIndex: 0,
                data: [],
                itemStyle: {
                    color: this.colors.bull,
                    color0: this.colors.bear,
                    borderColor: this.colors.bull,
                    borderColor0: this.colors.bear,
                    borderWidth: 1
                }
            }, {
                name: 'Volume',
                type: 'bar',
                xAxisIndex: 1,
                yAxisIndex: 1,
                data: [],
                itemStyle: {
                    color: function (params) {
                        return params.data[1] > params.data[0] ?
                            this.colors.bull + '40' : this.colors.bear + '40';
                    }.bind(this)
                }
            }],
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    lineStyle: { color: this.colors.accent, opacity: 0.5 }
                },
                backgroundColor: '#131722',
                borderColor: '#2A2E39',
                borderWidth: 1,
                textStyle: { color: this.colors.text, fontSize: 11 },
                formatter: this.formatTooltip.bind(this)
            }
        };

        this.chart.setOption(option, true);
        this.setupChartEvents();
    }

    // Eventos del gráfico
    setupChartEvents() {
        this.chart.on('dataZoom', (params) => {
            this.onChartZoom(params);
        });

        this.chart.on('brush', (params) => {
            this.onChartBrush(params);
        });

        // Redimensionar cuando cambie el tamaño de la ventana
        window.addEventListener('resize', () => {
            this.chart.resize();
        });
    }

    // Carga de datos inicial
    async loadInitialData() {
        try {
            await this.loadHistoricalData();
            if (this.isRealTime) {
                this.setupWebSocket();
            }
            this.updateSymbolInfo();
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showNotification('Error loading market data', 'error');
        }
    }

    // Carga de datos históricos desde la API del backend
    async loadHistoricalData() {
        try {
            // Usar nuestro backend como proxy para las APIs de exchanges
            const url = `/api/exchange/klines/${this.currentExchange}`;
            const params = new URLSearchParams({
                symbol: this.currentSymbol,
                interval: this.currentTimeframe,
                limit: 500
            });

            const response = await fetch(`${url}?${params}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            this.candleData = result.data || [];
            console.log('Raw data received:', result.data);
            console.log('Candle data length:', this.candleData.length);
            console.log('First candle:', this.candleData[0]);

            this.updateChart();

            // Mostrar notificación de éxito
            this.showNotification(`Loaded ${this.candleData.length} candles from ${this.currentExchange}`, 'success');

        } catch (error) {
            console.error('Error fetching historical data:', error);
            this.showNotification('Failed to load real data, using fallback', 'warning');
            // Fallback a datos mock si la API falla
            this.loadMockData();
        }
    }

    // Procesar datos históricos según el exchange
    processHistoricalData(data) {
        this.candleData = [];

        switch (this.currentExchange) {
            case 'binance':
                data.forEach(candle => {
                    this.candleData.push({
                        timestamp: candle[0],
                        open: parseFloat(candle[1]),
                        high: parseFloat(candle[2]),
                        low: parseFloat(candle[3]),
                        close: parseFloat(candle[4]),
                        volume: parseFloat(candle[5])
                    });
                });
                break;

            case 'bybit':
                data.result.forEach(candle => {
                    this.candleData.push({
                        timestamp: candle.open_time * 1000,
                        open: parseFloat(candle.open),
                        high: parseFloat(candle.high),
                        low: parseFloat(candle.low),
                        close: parseFloat(candle.close),
                        volume: parseFloat(candle.volume)
                    });
                });
                break;

            case 'coinbase':
                data.forEach(candle => {
                    this.candleData.push({
                        timestamp: candle[0] * 1000,
                        open: candle[3],
                        high: candle[2],
                        low: candle[1],
                        close: candle[4],
                        volume: candle[5]
                    });
                });
                break;
        }

        // Ordenar por timestamp
        this.candleData.sort((a, b) => a.timestamp - b.timestamp);
    }

    // Configurar WebSocket para datos en tiempo real
    setupWebSocket() {
        if (this.websocket) {
            this.websocket.disconnect();
        }

        // Usar SocketIO para conexión en tiempo real
        this.websocket = io();

        // Eventos de conexión
        this.websocket.on('connect', () => {
            console.log('WebSocket conectado');
            this.showNotification('Conectado a datos en tiempo real', 'success');

            // Suscribirse a ticker y klines
            this.websocket.emit('subscribe_ticker', {
                symbol: this.currentSymbol,
                exchange: this.currentExchange
            });

            this.websocket.emit('subscribe_klines', {
                symbol: this.currentSymbol,
                exchange: this.currentExchange,
                interval: this.currentTimeframe
            });
        });

        this.websocket.on('disconnect', () => {
            console.log('WebSocket desconectado');
            this.showNotification('Desconectado de datos en tiempo real', 'warning');
        });

        // Actualización de ticker
        this.websocket.on('ticker_update', (data) => {
            this.updateTickerData(data);
        });

        // Actualización de klines
        this.websocket.on('kline_update', (data) => {
            this.updateKlineData(data);
        });

        this.websocket.on('status', (data) => {
            console.log('WebSocket status:', data.msg);
        });
    }

    // Suscribirse al símbolo actual en WebSocket
    subscribeToSymbol() {
        switch (this.currentExchange) {
            case 'binance':
                // Binance se conecta directamente al stream
                break;

            case 'bybit':
                this.websocket.send(JSON.stringify({
                    op: 'subscribe',
                    args: [`klineV2.${this.timeframeMapping.bybit[this.currentTimeframe]}.${this.currentSymbol}`]
                }));
                break;

            case 'coinbase':
                this.websocket.send(JSON.stringify({
                    type: 'subscribe',
                    product_ids: [this.currentSymbol.replace('USDT', '-USD')],
                    channels: ['ticker']
                }));
                break;
        }
    }

    // Manejar mensajes del WebSocket
    handleWebSocketMessage(data) {
        switch (this.currentExchange) {
            case 'binance':
                if (data.k && data.k.x) { // Vela cerrada
                    this.updateCandleData(data.k);
                }
                break;

            case 'bybit':
                if (data.topic && data.topic.includes('klineV2')) {
                    this.updateCandleData(data.data[0]);
                }
                break;

            case 'coinbase':
                if (data.type === 'ticker') {
                    this.updateTicker(data);
                }
                break;
        }
    }

    // Actualizar datos de vela en tiempo real
    updateCandleData(klineData) {
        let newCandle;

        switch (this.currentExchange) {
            case 'binance':
                newCandle = {
                    timestamp: klineData.t,
                    open: parseFloat(klineData.o),
                    high: parseFloat(klineData.h),
                    low: parseFloat(klineData.l),
                    close: parseFloat(klineData.c),
                    volume: parseFloat(klineData.v)
                };
                break;

            case 'bybit':
                newCandle = {
                    timestamp: klineData.open_time * 1000,
                    open: parseFloat(klineData.open),
                    high: parseFloat(klineData.high),
                    low: parseFloat(klineData.low),
                    close: parseFloat(klineData.close),
                    volume: parseFloat(klineData.volume)
                };
                break;
        }

        // Actualizar o añadir la vela
        const lastIndex = this.candleData.length - 1;
        if (lastIndex >= 0 && this.candleData[lastIndex].timestamp === newCandle.timestamp) {
            this.candleData[lastIndex] = newCandle;
        } else {
            this.candleData.push(newCandle);
        }

        this.updateChart();
        this.updateSymbolInfo();
    }

    // Actualizar datos de ticker en tiempo real
    updateTickerData(data) {
        // Actualizar información del símbolo en la UI
        const symbolInfo = document.querySelector('.symbol-info .symbol-name');
        const priceInfo = document.querySelector('.current-price');
        const changeInfo = document.querySelector('.price-change');

        if (symbolInfo) symbolInfo.textContent = data.symbol;
        if (priceInfo) priceInfo.textContent = `$${data.price.toLocaleString()}`;
        if (changeInfo) {
            changeInfo.textContent = `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)}%`;
            changeInfo.className = `price-change ${data.change >= 0 ? 'positive' : 'negative'}`;
        }
    }

    // Actualizar datos de klines en tiempo real
    updateKlineData(data) {
        if (!data.is_closed) return; // Solo procesar velas cerradas

        const newCandle = {
            timestamp: data.timestamp,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            volume: data.volume
        };

        // Añadir o actualizar la última vela
        if (this.candleData.length > 0) {
            const lastCandle = this.candleData[this.candleData.length - 1];
            if (lastCandle.timestamp === newCandle.timestamp) {
                // Actualizar vela existente
                this.candleData[this.candleData.length - 1] = newCandle;
            } else {
                // Añadir nueva vela
                this.candleData.push(newCandle);
                // Mantener solo las últimas 500 velas
                if (this.candleData.length > 500) {
                    this.candleData.shift();
                }
            }
        } else {
            this.candleData.push(newCandle);
        }

        // Actualizar el gráfico
        this.updateChart();
    }

    // Actualizar el gráfico con nuevos datos
    updateChart() {
        console.log('updateChart called');
        console.log('this.chart:', this.chart);
        console.log('this.candleData.length:', this.candleData.length);

        if (!this.chart) {
            console.error('Chart not initialized!');
            return;
        }

        if (this.candleData.length === 0) {
            console.error('No candle data available!');
            return;
        }

        console.log('Processing candle data...');

        const candleSeriesData = this.candleData.map(candle => [
            candle.open, candle.close, candle.low, candle.high
        ]);

        const volumeSeriesData = this.candleData.map(candle => [
            candle.open, candle.close, candle.volume
        ]);

        const timeLabels = this.candleData.map(candle =>
            this.formatTime(new Date(candle.timestamp))
        );

        console.log('candleSeriesData sample:', candleSeriesData.slice(0, 3));
        console.log('timeLabels sample:', timeLabels.slice(0, 3));

        this.chart.setOption({
            xAxis: [{
                data: timeLabels
            }, {
                data: timeLabels
            }],
            series: [{
                data: candleSeriesData
            }, {
                data: volumeSeriesData
            }]
        });

        console.log('Chart updated successfully');

        // Actualizar indicadores activos
        this.updateActiveIndicators();
    }

    // Hacer petición a la API con manejo de errores
    async makeAPIRequest(url, params) {
        const queryString = new URLSearchParams(params).toString();
        const fullUrl = `${url}?${queryString}`;

        const response = await fetch(fullUrl, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    // Cargar datos mock como fallback
    loadMockData() {
        console.log('Loading mock data as fallback...');
        // Generar datos mock realistas
        const now = Date.now();
        const interval = this.getIntervalMs(this.currentTimeframe);

        this.candleData = [];
        let price = 45000 + Math.random() * 10000;

        for (let i = 500; i >= 0; i--) {
            const timestamp = now - (i * interval);
            const change = (Math.random() - 0.5) * price * 0.02;
            const open = price;
            const close = price + change;
            const high = Math.max(open, close) + Math.random() * Math.abs(change);
            const low = Math.min(open, close) - Math.random() * Math.abs(change);
            const volume = 1000 + Math.random() * 5000;

            this.candleData.push({
                timestamp,
                open,
                high,
                low,
                close,
                volume
            });

            price = close;
        }

        this.updateChart();
    }

    // Obtener intervalo en millisegundos
    getIntervalMs(timeframe) {
        const intervals = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        };
        return intervals[timeframe] || intervals['1h'];
    }

    // Formatear tiempo para labels
    formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();

        if (this.currentTimeframe.includes('m') || this.currentTimeframe.includes('h')) {
            return date.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
            });
        } else {
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric'
            });
        }
    }

    // Formatear precio
    formatPrice(value) {
        if (value >= 1000) {
            return value.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
        }
        return value.toFixed(4);
    }

    // Formatear volumen
    formatVolume(value) {
        if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B';
        if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M';
        if (value >= 1e3) return (value / 1e3).toFixed(1) + 'K';
        return value.toFixed(0);
    }

    // Formatear tooltip
    formatTooltip(params) {
        if (!params || params.length === 0) return '';

        const candleData = params.find(p => p.seriesName === 'Candlestick');
        const volumeData = params.find(p => p.seriesName === 'Volume');

        if (!candleData) return '';

        const data = candleData.data;
        const timestamp = this.candleData[candleData.dataIndex]?.timestamp;
        const volume = this.candleData[candleData.dataIndex]?.volume;

        return `
            <div style="padding: 8px; font-size: 11px;">
                <div style="color: ${this.colors.accent}; font-weight: bold; margin-bottom: 4px;">
                    ${new Date(timestamp).toLocaleString()}
                </div>
                <div>Open: <span style="color: ${this.colors.text};">${this.formatPrice(data[0])}</span></div>
                <div>High: <span style="color: ${this.colors.bull};">${this.formatPrice(data[3])}</span></div>
                <div>Low: <span style="color: ${this.colors.bear};">${this.formatPrice(data[2])}</span></div>
                <div>Close: <span style="color: ${this.colors.text};">${this.formatPrice(data[1])}</span></div>
                <div>Volume: <span style="color: ${this.colors.text};">${this.formatVolume(volume)}</span></div>
            </div>
        `;
    }

    // Configurar indicadores técnicos
    setupIndicators() {
        this.indicators = new Map([
            ['SMA', { name: 'Simple Moving Average', calculate: this.calculateSMA.bind(this) }],
            ['EMA', { name: 'Exponential Moving Average', calculate: this.calculateEMA.bind(this) }],
            ['RSI', { name: 'Relative Strength Index', calculate: this.calculateRSI.bind(this) }],
            ['MACD', { name: 'MACD', calculate: this.calculateMACD.bind(this) }],
            ['BB', { name: 'Bollinger Bands', calculate: this.calculateBollingerBands.bind(this) }],
            ['VOLUME', { name: 'Volume', calculate: this.calculateVolume.bind(this) }]
        ]);
    }

    // Calcular SMA (Simple Moving Average)
    calculateSMA(data, period = 20) {
        const result = [];
        for (let i = period - 1; i < data.length; i++) {
            const sum = data.slice(i - period + 1, i + 1).reduce((acc, val) => acc + val.close, 0);
            result.push(sum / period);
        }
        return result;
    }

    // Calcular EMA (Exponential Moving Average)
    calculateEMA(data, period = 20) {
        const result = [];
        const multiplier = 2 / (period + 1);
        let ema = data[0].close;

        result.push(ema);

        for (let i = 1; i < data.length; i++) {
            ema = (data[i].close * multiplier) + (ema * (1 - multiplier));
            result.push(ema);
        }
        return result;
    }

    // Calcular RSI (Relative Strength Index)
    calculateRSI(data, period = 14) {
        const gains = [];
        const losses = [];

        for (let i = 1; i < data.length; i++) {
            const change = data[i].close - data[i - 1].close;
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? Math.abs(change) : 0);
        }

        const result = [];

        for (let i = period - 1; i < gains.length; i++) {
            const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;
            const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;

            if (avgLoss === 0) {
                result.push(100);
            } else {
                const rs = avgGain / avgLoss;
                const rsi = 100 - (100 / (1 + rs));
                result.push(rsi);
            }
        }

        return result;
    }

    // Calcular MACD
    calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
        const fastEMA = this.calculateEMA(data, fastPeriod);
        const slowEMA = this.calculateEMA(data, slowPeriod);

        const macdLine = [];
        const startIndex = slowPeriod - fastPeriod;

        for (let i = startIndex; i < fastEMA.length; i++) {
            macdLine.push(fastEMA[i] - slowEMA[i - startIndex]);
        }

        const signalLine = this.calculateEMA(
            macdLine.map((value, index) => ({ close: value })),
            signalPeriod
        );

        const histogram = [];
        for (let i = 0; i < signalLine.length; i++) {
            histogram.push(macdLine[i + signalPeriod - 1] - signalLine[i]);
        }

        return { macd: macdLine, signal: signalLine, histogram };
    }

    // Calcular Bollinger Bands
    calculateBollingerBands(data, period = 20, stdDev = 2) {
        const sma = this.calculateSMA(data, period);
        const upperBand = [];
        const lowerBand = [];

        for (let i = period - 1; i < data.length; i++) {
            const slice = data.slice(i - period + 1, i + 1);
            const mean = sma[i - period + 1];
            const variance = slice.reduce((acc, val) => acc + Math.pow(val.close - mean, 2), 0) / period;
            const standardDeviation = Math.sqrt(variance);

            upperBand.push(mean + (standardDeviation * stdDev));
            lowerBand.push(mean - (standardDeviation * stdDev));
        }

        return {
            middle: sma,
            upper: upperBand,
            lower: lowerBand
        };
    }

    // Calcular Volume
    calculateVolume(data) {
        return data.map(candle => ({
            time: candle.time,
            volume: candle.volume || 0
        }));
    }

    // Actualizar indicadores activos
    updateActiveIndicators() {
        this.activeIndicators.forEach((config, name) => {
            this.addIndicatorToChart(name, config);
        });
    }

    // Añadir indicador al gráfico usando cálculos del backend
    async addIndicatorToChart(name, config) {
        if (!this.indicators.has(name) || this.candleData.length === 0) return;

        try {
            // Usar el backend para calcular indicadores
            const response = await fetch('/api/indicators/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.candleData)
            });

            const params = new URLSearchParams({
                type: name.toLowerCase(),
                ...config.params.reduce((acc, param, index) => {
                    const paramNames = {
                        'SMA': ['period'],
                        'EMA': ['period'],
                        'RSI': ['period'],
                        'MACD': ['fast', 'slow', 'signal'],
                        'BB': ['period', 'stddev']
                    };
                    if (paramNames[name] && paramNames[name][index]) {
                        acc[paramNames[name][index]] = param;
                    }
                    return acc;
                }, {})
            });

            const urlWithParams = `/api/indicators/calculate?${params}`;
            const indicatorResponse = await fetch(urlWithParams, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.candleData)
            });

            if (indicatorResponse.ok) {
                const result = await indicatorResponse.json();
                const calculatedData = result.data;

                // Configurar serie según el tipo de indicador
                this.addIndicatorSeriesToChart(name, calculatedData, config);

                // Añadir chip de indicador activo
                this.addActiveIndicatorChip(name, config);

                this.showNotification(`${name} indicator added`, 'success');
                return;
            }
        } catch (error) {
            console.error('Error calculating indicator on backend:', error);
        }

        // Fallback a cálculo local
        const indicator = this.indicators.get(name);
        const calculatedData = indicator.calculate(this.candleData, ...config.params);
        this.addIndicatorSeriesToChart(name, calculatedData, config);
        this.addActiveIndicatorChip(name, config);
    }

    // Añadir series de indicador al gráfico
    addIndicatorSeriesToChart(name, calculatedData, config) {
        let seriesConfig;

        switch (name) {
            case 'SMA':
            case 'EMA':
                seriesConfig = {
                    name: `${name}(${config.params[0]})`,
                    type: 'line',
                    data: calculatedData,
                    smooth: true,
                    lineStyle: { color: config.color, width: 1.5 },
                    showSymbol: false,
                    yAxisIndex: 0
                };
                break;

            case 'RSI':
                // RSI se muestra en un panel separado
                this.addIndicatorPanel('RSI', calculatedData, config.color);
                return;

            case 'MACD':
                this.addIndicatorPanel('MACD', calculatedData, config.color);
                return;

            case 'BB':
                this.addBollingerBands(calculatedData, config.color);
                return;
        }

        if (seriesConfig) {
            this.chart.setOption({
                series: [seriesConfig]
            }, false);
        }
    }

    // Añadir chip de indicador activo
    addActiveIndicatorChip(name, config) {
        const container = document.querySelector('.active-indicators');
        if (!container) return;

        const chip = document.createElement('div');
        chip.className = 'indicator-chip';
        chip.dataset.indicator = name;
        chip.innerHTML = `
            ${name}(${config.params.join(',')})
            <button class="remove-indicator" onclick="window.tealStreetChart.removeIndicator('${name}')">
                <i class="fas fa-times"></i>
            </button>
        `;

        container.appendChild(chip);
    }

    // Remover indicador
    removeIndicator(name) {
        this.activeIndicators.delete(name);

        // Remover del gráfico
        const option = this.chart.getOption();
        const newSeries = option.series.filter(s => !s.name.startsWith(name));

        this.chart.setOption({
            series: newSeries
        }, true);

        // Remover chip
        const chip = document.querySelector(`[data-indicator="${name}"]`);
        if (chip) chip.remove();

        this.showNotification(`${name} indicator removed`, 'info');
    }

    // Eventos de la interfaz
    bindEvents() {
        // Cambio de símbolo
        const symbolInput = document.querySelector('.symbol-input');
        if (symbolInput) {
            symbolInput.addEventListener('change', (e) => {
                this.changeSymbol(e.target.value.toUpperCase());
            });
        }

        // Cambio de exchange
        const exchangeSelect = document.querySelector('.exchange-select');
        if (exchangeSelect) {
            exchangeSelect.addEventListener('change', (e) => {
                this.changeExchange(e.target.value);
            });
        }

        // Cambio de timeframe
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.changeTimeframe(e.target.dataset.tf);
            });
        });

        // Botón de indicadores
        const indicatorsBtn = document.querySelector('[data-action="indicators"]');
        if (indicatorsBtn) {
            indicatorsBtn.addEventListener('click', () => {
                this.showIndicatorsModal();
            });
        }

        // Herramientas de dibujo
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectDrawingTool(e.target.dataset.tool);
            });
        });
    }

    // Cambiar símbolo
    async changeSymbol(symbol) {
        if (symbol === this.currentSymbol) return;

        this.currentSymbol = symbol;
        document.querySelector('.symbol-input').value = symbol;

        // Cerrar WebSocket actual
        if (this.websocket) {
            this.websocket.close();
        }

        // Cargar nuevos datos
        await this.loadHistoricalData();
        if (this.isRealTime) {
            this.setupWebSocket();
        }
    }

    // Cambiar exchange
    async changeExchange(exchange) {
        if (exchange === this.currentExchange) return;

        this.currentExchange = exchange;

        // Cerrar WebSocket actual
        if (this.websocket) {
            this.websocket.close();
        }

        // Cargar nuevos datos
        await this.loadHistoricalData();
        if (this.isRealTime) {
            this.setupWebSocket();
        }
    }

    // Cambiar timeframe
    async changeTimeframe(timeframe) {
        if (timeframe === this.currentTimeframe) return;

        this.currentTimeframe = timeframe;

        // Cerrar WebSocket actual
        if (this.websocket) {
            this.websocket.close();
        }

        // Cargar nuevos datos
        await this.loadHistoricalData();
        if (this.isRealTime) {
            this.setupWebSocket();
        }
    }

    // Actualizar información del símbolo usando datos reales
    async updateSymbolInfo() {
        try {
            // Obtener ticker data del backend
            const url = `/api/exchange/ticker/${this.currentExchange}`;
            const params = new URLSearchParams({
                symbol: this.currentSymbol
            });

            const response = await fetch(`${url}?${params}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const tickerData = await response.json();

                // Actualizar elementos de la UI con datos reales
                const symbolName = document.querySelector('.symbol-name');
                const symbolPrice = document.querySelector('.symbol-price');
                const symbolChange = document.querySelector('.symbol-change');
                const exchangeTag = document.querySelector('.exchange-tag');

                if (symbolName) symbolName.textContent = tickerData.symbol;
                if (symbolPrice) symbolPrice.textContent = this.formatPrice(tickerData.price);
                if (exchangeTag) exchangeTag.textContent = this.currentExchange.toUpperCase();

                if (symbolChange) {
                    const changeText = `${tickerData.change >= 0 ? '+' : ''}${this.formatPrice(tickerData.change)} (${tickerData.changePercent.toFixed(2)}%)`;
                    symbolChange.textContent = changeText;
                    symbolChange.className = `symbol-change ${tickerData.change >= 0 ? 'positive' : 'negative'}`;
                }

                // Actualizar estadísticas de 24h
                const stats = document.querySelectorAll('.stat-value');
                if (stats.length >= 3) {
                    stats[0].textContent = this.formatPrice(tickerData.high);
                    stats[1].textContent = this.formatPrice(tickerData.low);
                    stats[2].textContent = this.formatVolume(tickerData.volume);
                }

                return;
            }
        } catch (error) {
            console.error('Error updating symbol info:', error);
        }

        // Fallback a cálculo local si la API falla
        if (this.candleData.length === 0) return;

        const lastCandle = this.candleData[this.candleData.length - 1];
        const prevCandle = this.candleData[this.candleData.length - 2];

        if (!lastCandle || !prevCandle) return;

        const price = lastCandle.close;
        const change = price - prevCandle.close;
        const changePercent = (change / prevCandle.close) * 100;

        // Actualizar elementos de la UI con datos locales
        const symbolName = document.querySelector('.symbol-name');
        const symbolPrice = document.querySelector('.symbol-price');
        const symbolChange = document.querySelector('.symbol-change');

        if (symbolName) symbolName.textContent = this.currentSymbol;
        if (symbolPrice) symbolPrice.textContent = this.formatPrice(price);

        if (symbolChange) {
            symbolChange.textContent = `${change >= 0 ? '+' : ''}${this.formatPrice(change)} (${changePercent.toFixed(2)}%)`;
            symbolChange.className = `symbol-change ${change >= 0 ? 'positive' : 'negative'}`;
        }
    }

    // Mostrar modal de indicadores
    showIndicatorsModal() {
        const modal = document.querySelector('.tealstreet-modal');
        const overlay = document.querySelector('.modal-overlay');

        if (modal && overlay) {
            modal.classList.add('show');
            overlay.classList.add('show');
        }
    }

    // Cerrar modal de indicadores
    hideIndicatorsModal() {
        const modal = document.querySelector('.tealstreet-modal');
        const overlay = document.querySelector('.modal-overlay');

        if (modal && overlay) {
            modal.classList.remove('show');
            overlay.classList.remove('show');
        }
    }

    // Mostrar notificación
    showNotification(message, type = 'info') {
        console.log(`[${type.toUpperCase()}] ${message}`);

        // Crear elemento de notificación
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

        // Añadir estilos CSS si no existen
        if (!document.getElementById('notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 12px 16px;
                    border-radius: 6px;
                    color: white;
                    font-size: 14px;
                    z-index: 10000;
                    max-width: 300px;
                    animation: slideIn 0.3s ease-out;
                }
                .notification-success { background: #10B981; }
                .notification-error { background: #EF4444; }
                .notification-warning { background: #F59E0B; }
                .notification-info { background: #3B82F6; }
                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(styles);
        }

        // Añadir al DOM
        document.body.appendChild(notification);

        // Remover después de 3 segundos
        setTimeout(() => {
            notification.style.animation = 'slideIn 0.3s ease-out reverse';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }

    // Cleanup al destruir la instancia
    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }

        if (this.chart) {
            this.chart.dispose();
        }

        window.removeEventListener('resize', this.chart?.resize);
    }
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    window.tealStreetChart = new TealStreetPro();
});

// Indicadores técnicos adicionales y utilidades
class TechnicalIndicators {
    static stochastic(data, kPeriod = 14, dPeriod = 3) {
        const k = [];
        const d = [];

        for (let i = kPeriod - 1; i < data.length; i++) {
            const slice = data.slice(i - kPeriod + 1, i + 1);
            const high = Math.max(...slice.map(candle => candle.high));
            const low = Math.min(...slice.map(candle => candle.low));
            const close = data[i].close;

            const kValue = ((close - low) / (high - low)) * 100;
            k.push(kValue);
        }

        for (let i = dPeriod - 1; i < k.length; i++) {
            const sum = k.slice(i - dPeriod + 1, i + 1).reduce((a, b) => a + b);
            d.push(sum / dPeriod);
        }

        return { k, d };
    }

    static williamsR(data, period = 14) {
        const result = [];

        for (let i = period - 1; i < data.length; i++) {
            const slice = data.slice(i - period + 1, i + 1);
            const high = Math.max(...slice.map(candle => candle.high));
            const low = Math.min(...slice.map(candle => candle.low));
            const close = data[i].close;

            const wr = ((high - close) / (high - low)) * -100;
            result.push(wr);
        }

        return result;
    }

    static atr(data, period = 14) {
        const trueRanges = [];

        for (let i = 1; i < data.length; i++) {
            const current = data[i];
            const previous = data[i - 1];

            const tr1 = current.high - current.low;
            const tr2 = Math.abs(current.high - previous.close);
            const tr3 = Math.abs(current.low - previous.close);

            trueRanges.push(Math.max(tr1, tr2, tr3));
        }

        const result = [];
        for (let i = period - 1; i < trueRanges.length; i++) {
            const sum = trueRanges.slice(i - period + 1, i + 1).reduce((a, b) => a + b);
            result.push(sum / period);
        }

        return result;
    }
}
