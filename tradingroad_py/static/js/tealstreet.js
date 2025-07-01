// TealStreet JavaScript - Implementación profesional con APIs reales

class TealStreetChart {
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

        // APIs de exchanges configuradas
        this.exchangeAPIs = {
            binance: {
                rest: 'https://api.binance.com/api/v3',
                websocket: 'wss://stream.binance.com:9443/ws',
                klineEndpoint: '/klines'
            },
            bybit: {
                rest: 'https://api.bybit.com/v2/public',
                websocket: 'wss://stream.bybit.com/realtime',
                klineEndpoint: '/kline/list'
            },
            coinbase: {
                rest: 'https://api.pro.coinbase.com',
                websocket: 'wss://ws-feed.pro.coinbase.com',
                klineEndpoint: '/products'
            }
        };

        this.init();
    }

    init() {
        this.initChart();
        this.bindEvents();
        this.loadMarketData();
    }

    initChart() {
        const chartContainer = document.getElementById('tradingChart');
        if (!chartContainer) return;

        this.chart = echarts.init(chartContainer, {
            backgroundColor: '#0E1421',
            theme: 'dark'
        });

        // Configuración inicial del gráfico
        const option = {
            backgroundColor: '#0E1421',
            grid: {
                left: '60px',
                right: '60px',
                top: '60px',
                bottom: '60px'
            },
            xAxis: {
                type: 'category',
                data: [],
                axisLine: { lineStyle: { color: '#374151' } },
                axisTick: { lineStyle: { color: '#374151' } },
                axisLabel: { color: '#9CA3AF' }
            },
            yAxis: {
                type: 'value',
                scale: true,
                position: 'right',
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: {
                    color: '#9CA3AF',
                    formatter: value => value.toFixed(2)
                },
                splitLine: {
                    lineStyle: {
                        color: '#374151',
                        type: 'dashed'
                    }
                }
            },
            series: [{
                type: 'candlestick',
                data: [],
                itemStyle: {
                    color: '#10B981',
                    color0: '#EF4444',
                    borderColor: '#10B981',
                    borderColor0: '#EF4444'
                }
            }],
            tooltip: {
                trigger: 'axis',
                backgroundColor: '#161D2E',
                borderColor: '#374151',
                textStyle: { color: '#F9FAFB' },
                formatter: function (params) {
                    if (params && params[0]) {
                        const data = params[0].data;
                        return `
                            <div style="padding: 8px;">
                                <div style="color: #0EA5E9; font-weight: bold; margin-bottom: 4px;">
                                    ${params[0].name}
                                </div>
                                <div>Open: <span style="color: #F9FAFB;">${data[1]}</span></div>
                                <div>High: <span style="color: #10B981;">${data[4]}</span></div>
                                <div>Low: <span style="color: #EF4444;">${data[3]}</span></div>
                                <div>Close: <span style="color: #F9FAFB;">${data[2]}</span></div>
                            </div>
                        `;
                    }
                    return '';
                }
            },
            dataZoom: [{
                type: 'inside',
                xAxisIndex: 0,
                filterMode: 'none'
            }]
        };

        this.chart.setOption(option);

        // Resize handler
        window.addEventListener('resize', () => {
            this.chart?.resize();
        });
    }

    bindEvents() {
        // Timeframe buttons
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.currentTimeframe = e.target.dataset.tf;
                this.loadMarketData();
            });
        });

        // Exchange selector
        const exchangeSelect = document.querySelector('.exchange-select');
        if (exchangeSelect) {
            exchangeSelect.addEventListener('change', (e) => {
                this.currentExchange = e.target.value;
                this.loadMarketData();
            });
        }

        // Symbol input
        const symbolInput = document.querySelector('.symbol-input');
        if (symbolInput) {
            symbolInput.addEventListener('change', (e) => {
                this.currentSymbol = e.target.value.toUpperCase();
                this.updateSymbolDisplay();
                this.loadMarketData();
            });
        }

        // Indicators modal
        const indicatorsBtn = document.getElementById('indicatorsBtn');
        const indicatorsModal = document.getElementById('indicatorsModal');
        const modalOverlay = document.getElementById('modalOverlay');
        const closeIndicators = document.getElementById('closeIndicators');

        if (indicatorsBtn && indicatorsModal) {
            indicatorsBtn.addEventListener('click', () => {
                this.showIndicatorsModal();
            });
        }

        if (closeIndicators) {
            closeIndicators.addEventListener('click', () => {
                this.hideIndicatorsModal();
            });
        }

        if (modalOverlay) {
            modalOverlay.addEventListener('click', () => {
                this.hideIndicatorsModal();
            });
        }

        // Tool buttons
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
            });
        });

        // Zoom controls
        document.querySelectorAll('.zoom-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.title;
                this.handleZoom(action);
            });
        });

        // Indicator toggles
        document.querySelectorAll('.switch input').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const indicator = e.target.id.replace('-toggle', '');
                if (e.target.checked) {
                    this.addIndicator(indicator);
                } else {
                    this.removeIndicator(indicator);
                }
            });
        });

        // Category buttons
        document.querySelectorAll('.category-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.filterIndicators(e.target.dataset.category);
            });
        });

        // Search functionality
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchIndicators(e.target.value);
            });
        }

        // Apply indicators
        const applyBtn = document.getElementById('applyIndicators');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                this.applyIndicators();
                this.hideIndicatorsModal();
            });
        }

        // Reset indicators
        const resetBtn = document.getElementById('resetIndicators');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetIndicators();
            });
        }
    }

    async loadMarketData() {
        try {
            const response = await fetch(`/api/market/data?symbol=${this.currentSymbol}&timeframe=${this.currentTimeframe}&source=${this.currentExchange}`);
            const data = await response.json();

            if (data && data.length > 0) {
                this.updateChart(data);
                this.updatePriceDisplay(data[data.length - 1]);
            }
        } catch (error) {
            console.error('Error loading market data:', error);
            this.showNotification('Error loading market data', 'error');
        }
    }

    updateChart(data) {
        if (!this.chart) return;

        const chartData = data.map(item => [
            item.time,
            parseFloat(item.open),
            parseFloat(item.close),
            parseFloat(item.low),
            parseFloat(item.high)
        ]);

        const times = data.map(item => {
            const date = new Date(item.time);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        });

        this.chart.setOption({
            xAxis: {
                data: times
            },
            series: [{
                data: chartData
            }]
        });
    }

    updatePriceDisplay(latestData) {
        if (!latestData) return;

        const symbolName = document.querySelector('.symbol-name');
        const currentPrice = document.querySelector('.current-price');
        const priceChange = document.querySelector('.price-change');

        if (symbolName) symbolName.textContent = this.currentSymbol;
        if (currentPrice) currentPrice.textContent = `$${parseFloat(latestData.close).toLocaleString()}`;

        if (priceChange) {
            const change = parseFloat(latestData.close) - parseFloat(latestData.open);
            const changePercent = (change / parseFloat(latestData.open)) * 100;

            priceChange.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent.toFixed(2)}%)`;
            priceChange.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
        }
    }

    updateSymbolDisplay() {
        const symbolName = document.querySelector('.symbol-name');
        if (symbolName) symbolName.textContent = this.currentSymbol;
    }

    showIndicatorsModal() {
        const modal = document.getElementById('indicatorsModal');
        const overlay = document.getElementById('modalOverlay');

        if (modal && overlay) {
            modal.classList.add('show');
            overlay.classList.add('show');
            document.body.style.overflow = 'hidden';
        }
    }

    hideIndicatorsModal() {
        const modal = document.getElementById('indicatorsModal');
        const overlay = document.getElementById('modalOverlay');

        if (modal && overlay) {
            modal.classList.remove('show');
            overlay.classList.remove('show');
            document.body.style.overflow = '';
        }
    }

    handleZoom(action) {
        if (!this.chart) return;

        switch (action) {
            case 'Zoom In':
                this.chart.dispatchAction({ type: 'dataZoom', start: 10, end: 90 });
                break;
            case 'Zoom Out':
                this.chart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 });
                break;
            case 'Reset Zoom':
                this.chart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 });
                break;
        }
    }

    addIndicator(indicator) {
        this.activeIndicators.add(indicator);
        this.updateActiveIndicatorsDisplay();
        console.log(`Added indicator: ${indicator}`);
    }

    removeIndicator(indicator) {
        this.activeIndicators.delete(indicator);
        this.updateActiveIndicatorsDisplay();
        console.log(`Removed indicator: ${indicator}`);
    }

    updateActiveIndicatorsDisplay() {
        const container = document.querySelector('.active-indicators');
        if (!container) return;

        container.innerHTML = '';

        this.activeIndicators.forEach(indicator => {
            const chip = document.createElement('span');
            chip.className = 'indicator-chip';
            chip.textContent = this.getIndicatorDisplayName(indicator);
            container.appendChild(chip);
        });
    }

    getIndicatorDisplayName(indicator) {
        const names = {
            'rsi': 'RSI(14)',
            'macd': 'MACD(12,26,9)',
            'bb': 'BB(20,2)',
            'ema': 'EMA(20)'
        };
        return names[indicator] || indicator.toUpperCase();
    }

    filterIndicators(category) {
        const cards = document.querySelectorAll('.indicator-card');

        cards.forEach(card => {
            if (category === 'all') {
                card.style.display = 'block';
            } else {
                const indicatorType = card.dataset.indicator;
                const shouldShow = this.getIndicatorCategory(indicatorType) === category;
                card.style.display = shouldShow ? 'block' : 'none';
            }
        });
    }

    getIndicatorCategory(indicator) {
        const categories = {
            'rsi': 'momentum',
            'macd': 'momentum',
            'bb': 'volatility',
            'ema': 'trend'
        };
        return categories[indicator] || 'other';
    }

    searchIndicators(query) {
        const cards = document.querySelectorAll('.indicator-card');
        const searchTerm = query.toLowerCase();

        cards.forEach(card => {
            const name = card.querySelector('.indicator-name').textContent.toLowerCase();
            const desc = card.querySelector('.indicator-desc').textContent.toLowerCase();

            const matches = name.includes(searchTerm) || desc.includes(searchTerm);
            card.style.display = matches ? 'block' : 'none';
        });
    }

    applyIndicators() {
        console.log('Applying indicators:', Array.from(this.activeIndicators));
        this.showNotification('Indicators applied successfully', 'success');
    }

    resetIndicators() {
        document.querySelectorAll('.switch input').forEach(toggle => {
            toggle.checked = false;
        });

        this.activeIndicators.clear();
        this.updateActiveIndicatorsDisplay();
        this.showNotification('All indicators reset', 'info');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);

        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Estilos para notificaciones
const notificationStyles = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 12px 16px;
    border-radius: 6px;
    color: white;
    font-size: 14px;
    font-weight: 500;
    z-index: 2000;
    animation: slideInRight 0.3s ease-out;
    max-width: 300px;
}

.notification.success {
    background: #10B981;
}

.notification.error {
    background: #EF4444;
}

.notification.info {
    background: #0EA5E9;
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TealStreetChart();
});

// === Funciones de inicialización ===

function initTopNavigation() {
    // Evento para los botones de timeframe
    document.querySelectorAll('.tf-button').forEach(button => {
        button.addEventListener('click', function () {
            // Quitar clase activa de todos los botones
            document.querySelectorAll('.tf-button').forEach(btn => {
                btn.classList.remove('active');
            });

            // Agregar clase activa a este botón
            this.classList.add('active');

            // Cambiar el timeframe y actualizar datos
            currentTimeframe = this.getAttribute('data-timeframe');
            loadMarketData();
        });
    });

    // Evento para el selector de fuente
    const sourceSelect = document.getElementById('dataSource');
    if (sourceSelect) {
        sourceSelect.addEventListener('change', function () {
            currentSource = this.value;
            updateExchangeDisplay();
            loadMarketData();
        });
    }

    // Búsqueda de símbolos
    const symbolSearch = document.getElementById('symbolSearch');
    if (symbolSearch) {
        symbolSearch.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                const searchValue = this.value.trim().toUpperCase();
                if (searchValue) {
                    // Convertir según el formato que use cada exchange
                    currentSymbol = formatSymbolForExchange(searchValue, currentSource);
                    loadMarketData();
                }
            }
        });

        // Establecer el valor inicial
        symbolSearch.value = 'BTCUSDT';
    }
}

function initDrawingTools() {
    // Eventos para herramientas de dibujo
    document.querySelectorAll('.tool-button').forEach(button => {
        button.addEventListener('click', function () {
            // Si ya está activo, desactivar
            if (this.classList.contains('active')) {
                this.classList.remove('active');
                // Desactivar herramienta (código para desactivar aquí)
                return;
            }

            // Desactivar todas las herramientas
            document.querySelectorAll('.tool-button').forEach(btn => {
                btn.classList.remove('active');
            });

            // Activar esta herramienta
            this.classList.add('active');

            // Código para activar la herramienta según data-tool
            const toolType = this.getAttribute('data-tool');
            activateDrawingTool(toolType);
        });
    });
}

function initZoomControls() {
    // Eventos para los botones de zoom
    document.querySelectorAll('.zoom-button').forEach(button => {
        button.addEventListener('click', function () {
            const zoomType = this.getAttribute('data-zoom');

            if (chart) {
                // Implementar función de zoom según el tipo
                switch (zoomType) {
                    case 'in':
                        chart.timeScale().zoom(0.5);
                        break;
                    case 'out':
                        chart.timeScale().zoom(-0.5);
                        break;
                    case 'reset':
                        chart.timeScale().fitContent();
                        break;
                }
            }
        });
    });
}

function initIndicatorsButton() {
    const indicatorsBtn = document.getElementById('indicatorsBtn');
    if (indicatorsBtn) {
        indicatorsBtn.addEventListener('click', function () {
            // Mostrar modal de indicadores
            showIndicatorsModal();
        });
    }
}

function showIndicatorsModal() {
    // Crear modal si no existe
    let modal = document.getElementById('indicators-modal');

    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'indicators-modal';
        modal.className = 'indicator-modal';

        // Estructura del modal
        modal.innerHTML = `
                <div class="modal-header">
                    <h3 class="modal-title">Indicadores</h3>
                    <button class="close-modal">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="indicator-group">
                        <div class="indicator-group-header">
                            <span class="indicator-name">Medias Móviles</span>
                            <button class="indicator-toggle active">Activo</button>
                        </div>
                        <div class="indicator-params">
                            <div class="param-group">
                                <span class="param-label">EMA 12</span>
                                <input type="color" class="param-input" value="#00FF00">
                            </div>
                            <div class="param-group">
                                <span class="param-label">EMA 20</span>
                                <input type="color" class="param-input" value="#FFFFFF">
                            </div>
                            <div class="param-group">
                                <span class="param-label">SMA 50</span>
                                <input type="color" class="param-input" value="#FF69B4">
                            </div>
                            <div class="param-group">
                                <span class="param-label">SMA 200</span>
                                <input type="color" class="param-input" value="#FF0000">
                            </div>
                        </div>
                    </div>
                    <div class="indicator-group">
                        <div class="indicator-group-header">
                            <span class="indicator-name">RSI</span>
                            <button class="indicator-toggle">Inactivo</button>
                        </div>
                        <div class="indicator-params" style="display:none">
                            <div class="param-group">
                                <span class="param-label">Período</span>
                                <input type="number" class="param-input" value="14">
                            </div>
                            <div class="param-group">
                                <span class="param-label">Sobrecompra</span>
                                <input type="number" class="param-input" value="70">
                            </div>
                            <div class="param-group">
                                <span class="param-label">Sobreventa</span>
                                <input type="number" class="param-input" value="30">
                            </div>
                        </div>
                    </div>
                    <div class="indicator-group">
                        <div class="indicator-group-header">
                            <span class="indicator-name">MACD</span>
                            <button class="indicator-toggle">Inactivo</button>
                        </div>
                    </div>
                    <div class="indicator-group">
                        <div class="indicator-group-header">
                            <span class="indicator-name">Volumen</span>
                            <button class="indicator-toggle active">Activo</button>
                        </div>
                    </div>
                </div>
            `;

        document.body.appendChild(modal);

        // Eventos del modal
        modal.querySelector('.close-modal').addEventListener('click', function () {
            modal.classList.remove('visible');
        });

        // Eventos de los toggles de indicadores
        modal.querySelectorAll('.indicator-toggle').forEach(toggle => {
            toggle.addEventListener('click', function () {
                this.classList.toggle('active');
                const isActive = this.classList.contains('active');
                this.textContent = isActive ? 'Activo' : 'Inactivo';

                // Mostrar u ocultar parámetros
                const params = this.closest('.indicator-group').querySelector('.indicator-params');
                if (params) {
                    params.style.display = isActive ? 'grid' : 'none';
                }

                // Actualizar indicadores en el gráfico según corresponda
                const indicatorName = this.closest('.indicator-group-header').querySelector('.indicator-name').textContent;
                updateIndicator(indicatorName, isActive);
            });
        });
    }

    // Mostrar el modal
    modal.classList.add('visible');
}

// === Funciones para el gráfico y datos ===

function initChart() {
    const chartContainer = document.getElementById('tradingViewContainer');

    // Configuración del gráfico con estilo similar a TealStreet
    chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight - 40, // Restar altura del symbol-info-bar
        layout: {
            background: { type: 'solid', color: '#131722' },
            textColor: '#D9D9D9',
            fontSize: 12,
            fontFamily: "'Trebuchet MS', Roboto, Ubuntu, sans-serif",
        },
        grid: {
            vertLines: { color: '#1E222D' },
            horzLines: { color: '#1E222D' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: '#2962FF', width: 1, style: 1, labelBackgroundColor: '#2962FF' },
            horzLine: { color: '#2962FF', width: 1, style: 1, labelBackgroundColor: '#2962FF' },
        },
        rightPriceScale: {
            borderColor: '#1E222D',
            scaleMargins: { top: 0.1, bottom: 0.2 },
        },
        timeScale: {
            borderColor: '#1E222D',
            timeVisible: true,
            secondsVisible: false,
        },
        watermark: {
            visible: true,
            fontSize: 48,
            horzAlign: 'center',
            vertAlign: 'center',
            color: 'rgba(100, 100, 140, 0.1)',
            text: 'TradingRoad',
        },
    });

    // Crear serie de velas
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26A69A',
        downColor: '#EF5350',
        borderVisible: false,
        wickUpColor: '#26A69A',
        wickDownColor: '#EF5350',
    });

    // Ajustar tamaño en cambios de ventana
    window.addEventListener('resize', function () {
        if (chart) {
            chart.applyOptions({
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight - 40,
            });
        }
    });

    // Cargar datos iniciales
    loadMarketData();
}

function loadMarketData() {
    // Mostrar indicador de carga
    document.querySelector('.chart-loading').style.display = 'flex';

    // Construir la URL de la API
    const url = `/api/market/data?source=${currentSource}&symbol=${currentSymbol}&timeframe=${currentTimeframe}&limit=500`;

    // Actualizar UI mientras se carga
    updateChartUI();

    // Fetch data
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                marketData = data.map(candle => ({
                    time: candle.time / 1000, // Convertir milisegundos a segundos para Lightweight Charts
                    open: candle.open,
                    high: candle.high,
                    low: candle.low,
                    close: candle.close,
                    volume: candle.volume
                }));

                // Actualizar el gráfico
                candleSeries.setData(marketData);

                // Añadir indicadores predeterminados
                addDefaultIndicators();

                // Actualizar información en la UI
                updateChartInfoDisplay();

                // Iniciar contador de vela
                startCandleCountdown();

                console.log(`Datos cargados: ${marketData.length} velas para ${currentSymbol}`);
            } else {
                console.error('No se recibieron datos del mercado');
            }

            // Ocultar indicador de carga
            document.querySelector('.chart-loading').style.display = 'none';
        })
        .catch(error => {
            console.error('Error al cargar datos:', error);
            document.querySelector('.chart-loading').style.display = 'none';
        });
}

function addDefaultIndicators() {
    // Configuración predeterminada para medias móviles
    const defaultMAs = [
        { type: 'ema', period: 12, color: '#00FF00', width: 2, opacity: 0.8 }, // EMA 12 en verde
        { type: 'ema', period: 20, color: '#FFFFFF', width: 2, opacity: 0.8 }, // EMA 20 en blanco
        { type: 'sma', period: 50, color: '#FF69B4', width: 2, opacity: 0.8 }, // SMA 50 en fucsia claro
        { type: 'sma', period: 200, color: '#FF0000', width: 2, opacity: 0.8 } // SMA 200 en rojo
    ];

    // Aplicar medias móviles predeterminadas
    defaultMAs.forEach(ma => {
        addMovingAverage(ma);
    });

    // Agregar volumen
    addVolumeIndicator();

    // Actualizar indicadores activos en la UI
    updateActiveIndicatorsList();
}

function addMovingAverage(params) {
    if (!chart || !candleSeries || !marketData || marketData.length === 0) return;

    const { type, period, color, width, opacity } = params;
    const maColor = color + Math.round(opacity * 255).toString(16).padStart(2, '0');

    // Calcular el indicador
    const ma = calculateMA(marketData, type, period);

    // Crear la serie
    const maSeries = chart.addLineSeries({
        color: maColor,
        lineWidth: width,
        priceScaleId: 'right',
        title: `${type.toUpperCase()} ${period}`,
    });

    // Asignar datos
    maSeries.setData(ma);

    // Mantener un registro de las medias activas
    if (!window.movingAverages) window.movingAverages = [];

    window.movingAverages.push({
        id: `${type}-${period}`,
        series: maSeries,
        params: params
    });
}

function calculateMA(data, type, period) {
    const result = [];
    const length = data.length;

    if (length === 0 || period <= 0) return result;

    if (type === 'sma') {
        // Cálculo de la Media Móvil Simple (SMA)
        let sum = 0;
        for (let i = 0; i < length; i++) {
            sum += data[i].close;
            if (i >= period) {
                sum -= data[i - period].close;
                result.push({ time: data[i].time, value: sum / period });
            } else if (i === period - 1) {
                result.push({ time: data[i].time, value: sum / period });
            }
        }
    } else if (type === 'ema') {
        // Cálculo de la Media Móvil Exponencial (EMA)
        const multiplier = 2 / (period + 1);
        let ema = data[0].close;

        for (let i = 0; i < length; i++) {
            const close = data[i].close;
            ema = (close - ema) * multiplier + ema;
            if (i >= period - 1) {
                result.push({ time: data[i].time, value: ema });
            }
        }
    }

    return result;
}

function addVolumeIndicator() {
    if (!chart || !marketData || marketData.length === 0) return;

    // Crear el panel del volumen
    const volumePane = chart.addHistogramSeries({
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: 'volume',
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });

    // Calcular volúmenes y colores
    const volumeData = marketData.map(d => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
    }));

    // Asignar datos
    volumePane.setData(volumeData);

    // Guardar referencia
    window.volumeSeries = volumePane;
}

function updateChartUI() {
    // Actualizar elementos visuales mientras se espera la carga de datos
    document.getElementById('chart-symbol-pair').textContent = formatSymbolDisplay(currentSymbol) + ' · ' + currentTimeframe.toUpperCase();
    document.getElementById('chart-timeframe').textContent = currentTimeframe.toUpperCase();
    document.getElementById('chart-exchange').textContent = currentSource.toUpperCase();

    // Actualizar botones de timeframe
    document.querySelectorAll('.tf-button').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-timeframe') === currentTimeframe);
    });
}

function updateChartInfoDisplay() {
    if (!marketData || marketData.length === 0) return;

    const lastCandle = marketData[marketData.length - 1];
    const prevCandle = marketData.length > 1 ? marketData[marketData.length - 2] : null;

    // Actualizar info de OHLC
    document.querySelector('#chart-price-open span').textContent = formatPrice(lastCandle.open);
    document.querySelector('#chart-price-high span').textContent = formatPrice(lastCandle.high);
    document.querySelector('#chart-price-low span').textContent = formatPrice(lastCandle.low);
    document.querySelector('#chart-price span').textContent = formatPrice(lastCandle.close);

    // Cambio y porcentaje
    if (prevCandle) {
        const change = lastCandle.close - prevCandle.close;
        const changePercent = (change / prevCandle.close) * 100;
        const isPositive = change >= 0;

        const changeElement = document.getElementById('chart-change');
        changeElement.textContent = `${isPositive ? '+' : ''}${formatPrice(change)} (${isPositive ? '+' : ''}${changePercent.toFixed(2)}%)`;
        changeElement.className = `price-change ${isPositive ? 'positive' : 'negative'}`;
    }

    // Volumen
    document.getElementById('chart-volume-value').textContent = formatVolume(lastCandle.volume);

    // Máximos y mínimos
    const maxPrice = Math.max(...marketData.map(c => c.high));
    const minPrice = Math.min(...marketData.map(c => c.low));

    document.getElementById('chart-max-price-marker').textContent = formatPrice(maxPrice);
    document.getElementById('chart-min-price-marker').textContent = formatPrice(minPrice);
}

function startCandleCountdown() {
    // Detener cualquier contador existente
    if (window.candleCountdownInterval) {
        clearInterval(window.candleCountdownInterval);
    }

    const countdownElement = document.getElementById('chart-countdown');

    // Obtener el intervalo en milisegundos
    const timeframeMap = {
        '30s': 30 * 1000,
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
        '1M': 30 * 24 * 60 * 60 * 1000
    };

    const interval = timeframeMap[currentTimeframe] || 60 * 1000;

    // Calcular el tiempo restante para la próxima vela
    window.candleCountdownInterval = setInterval(() => {
        const now = new Date();
        const currentTimestamp = now.getTime();
        const remainder = currentTimestamp % interval;
        const timeToNext = interval - remainder;

        const minutes = Math.floor(timeToNext / (60 * 1000));
        const seconds = Math.floor((timeToNext % (60 * 1000)) / 1000);

        countdownElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }, 1000);
}

function updateActiveIndicatorsList() {
    // Actualizar la lista de indicadores activos
    const activeIndicatorsList = document.getElementById('active-indicators-list');

    if (activeIndicatorsList && window.movingAverages) {
        const indicatorNames = window.movingAverages.map(ma =>
            `${ma.params.type.toUpperCase()}${ma.params.period}`
        );

        const volumeActive = window.volumeSeries ? 'VOL' : '';
        const rsiActive = window.rsiSeries ? 'RSI' : '';
        const macdActive = window.macdSeries ? 'MACD' : '';

        // Filtrar indicadores activos
        const activeIndicators = [
            ...indicatorNames,
            volumeActive,
            rsiActive,
            macdActive
        ].filter(Boolean);

        // Mostrar en la UI
        activeIndicatorsList.textContent = activeIndicators.join(', ') || 'Ninguno';
    }
}

// === Funciones auxiliares ===

function formatSymbolForExchange(symbol, exchange) {
    // Eliminar caracteres no alfanuméricos excepto /
    let formatted = symbol.replace(/[^A-Z0-9/]/g, '');

    // Si no tiene / pero probablemente es un par con USDT, añadir USDT
    if (!formatted.includes('/') && !formatted.includes('USDT') && !formatted.includes('USD')) {
        formatted += 'USDT';
    }

    // Convertir según el formato que use cada exchange
    switch (exchange.toLowerCase()) {
        case 'binance':
        case 'bybit':
        case 'kucoin':
            // Estos exchanges usan BTCUSDT (sin /)
            return formatted.replace('/', '');
        case 'kraken':
            // Kraken usa XBT en lugar de BTC
            return formatted.replace('BTC', 'XBT');
        default:
            return formatted;
    }
}

function formatSymbolDisplay(symbol) {
    // Para mostrar el símbolo de forma más amigable
    return symbol
        .replace('USDT', '') // Quitar USDT para mostrarlo por separado
        .replace('USD', '')  // Quitar USD para mostrarlo por separado
        .replace('/', '');   // Quitar la barra si existe
}

function formatPrice(price) {
    // Formato para precios según su tamaño
    if (price >= 1000) {
        return price.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    } else if (price >= 1) {
        return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } else {
        // Para precios muy pequeños (menos de 1), usar notación según tamaño
        return price.toLocaleString(undefined, { minimumFractionDigits: 6, maximumFractionDigits: 6 });
    }
}

function formatVolume(volume) {
    // Formato para volúmenes grandes
    if (volume >= 1000000000) {
        return (volume / 1000000000).toFixed(2) + 'B';
    } else if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    } else {
        return volume.toFixed(2);
    }
}

function updateExchangeDisplay() {
    // Actualizar el nombre del exchange mostrado
    document.getElementById('chart-exchange').textContent = currentSource.toUpperCase();
}

// === Funciones de dibujo ===

function activateDrawingTool(toolType) {
    // Por implementar la funcionalidad de herramientas de dibujo
    console.log(`Activada herramienta: ${toolType}`);

    // Aquí implementaríamos el comportamiento específico de cada herramienta
}

// === Funciones para indicadores ===

function updateIndicator(name, active) {
    // Activar o desactivar un indicador según su nombre
    console.log(`${name} está ahora ${active ? 'activo' : 'inactivo'}`);

    switch (name.toLowerCase()) {
        case 'medias móviles':
            toggleMovingAverages(active);
            break;
        case 'rsi':
            toggleRSI(active);
            break;
        case 'macd':
            toggleMACD(active);
            break;
        case 'volumen':
            toggleVolume(active);
            break;
        default:
            console.log(`Indicador no reconocido: ${name}`);
    }

    // Actualizar lista de indicadores activos
    updateActiveIndicatorsList();
}

function toggleMovingAverages(active) {
    if (window.movingAverages) {
        window.movingAverages.forEach(ma => {
            ma.series.applyOptions({ visible: active });
        });
    }
}

function toggleRSI(active) {
    // Implementar activación/desactivación del RSI
}

function toggleMACD(active) {
    // Implementar activación/desactivación del MACD
}

function toggleVolume(active) {
    if (window.volumeSeries) {
        window.volumeSeries.applyOptions({ visible: active });
    }
}
