/**
 * ECharts Trading Chart for TradingRoad
 * Advanced financial chart with multiple indicators and interactive tools
 */

class EChartsTradingChart {
    constructor(config) {
        this.container = config.container || 'tradingViewContainer';
        this.symbol = config.symbol || 'BTCUSDT';
        this.timeframe = config.timeframe || '1h';
        this.dataSource = config.dataSource || 'bybit';
        this.theme = config.theme || 'dark';

        // Chart instances
        this.mainChart = null;
        this.volumeChart = null;
        this.indicatorCharts = {};

        // Data
        this.candleData = [];
        this.volumeData = [];
        this.indicators = {};
        this.activeIndicators = [];

        // UI State
        this.isLoading = false;
        this.drawingMode = false;
        this.currentDrawingTool = null;

        // Initialize
        this.init();
    }

    init() {
        // Create DOM structure
        this.createChartDOM();

        // Initialize ECharts instances
        this.initMainChart();
        this.initVolumeChart();

        // Set up event listeners
        this.setupEventListeners();

        // Load data
        this.loadData();

        // Handle window resize
        window.addEventListener('resize', () => {
            if (this.mainChart) this.mainChart.resize();
            if (this.volumeChart) this.volumeChart.resize();
            Object.values(this.indicatorCharts).forEach(chart => {
                if (chart) chart.resize();
            });
        });
    }

    createChartDOM() {
        const container = document.getElementById(this.container);
        if (!container) return;

        // Clear existing content
        container.innerHTML = '';

        // Create structure
        container.innerHTML = `
            <div class="echarts-loading">
                <div class="spinner"></div>
                <span>Cargando datos...</span>
            </div>
            <div class="echarts-main-chart" id="mainChart"></div>
            <div class="echarts-volume-chart" id="volumeChart"></div>
            <div class="echarts-indicators-container" id="indicatorsContainer"></div>
            <div class="echarts-drawing-tools">
                <button class="tool-button" data-tool="line"><i class="fas fa-chart-line"></i></button>
                <button class="tool-button" data-tool="horizontal"><i class="fas fa-grip-lines"></i></button>
                <button class="tool-button" data-tool="vertical"><i class="fas fa-grip-lines-vertical"></i></button>
                <button class="tool-button" data-tool="rectangle"><i class="far fa-square"></i></button>
                <button class="tool-button" data-tool="fibonacci"><i class="fas fa-ruler-combined"></i></button>
                <button class="tool-button" data-tool="text"><i class="fas fa-font"></i></button>
            </div>
            <div class="echarts-zoom-controls">
                <button class="zoom-button" data-zoom="in"><i class="fas fa-search-plus"></i></button>
                <button class="zoom-button" data-zoom="out"><i class="fas fa-search-minus"></i></button>
                <button class="zoom-button" data-zoom="reset"><i class="fas fa-expand"></i></button>
            </div>
        `;

        // Add custom styles
        const style = document.createElement('style');
        style.textContent = `
            .echarts-loading {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(19, 23, 34, 0.7);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                color: #D9D9D9;
            }
            .echarts-main-chart {
                height: 70%;
                width: 100%;
            }
            .echarts-volume-chart {
                height: 15%;
                width: 100%;
            }
            .echarts-indicators-container {
                height: 15%;
                width: 100%;
                display: flex;
                flex-direction: column;
            }
        `;
        document.head.appendChild(style);
    }

    initMainChart() {
        const chartDom = document.getElementById('mainChart');
        if (!chartDom) return;

        this.mainChart = echarts.init(chartDom, this.theme);

        const option = {
            animation: false,
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    crossStyle: {
                        color: '#999'
                    }
                }
            },
            grid: {
                left: '10%',
                right: '10%',
                bottom: '15%'
            },
            xAxis: {
                type: 'category',
                data: [],
                scale: true,
                boundaryGap: false,
                axisLine: { lineStyle: { color: '#8392A5' } },
                splitLine: { show: false },
                min: 'dataMin',
                max: 'dataMax'
            },
            yAxis: {
                scale: true,
                splitArea: {
                    show: false
                }
            },
            dataZoom: [
                {
                    type: 'inside',
                    start: 0,
                    end: 100
                }
            ],
            series: [
                {
                    name: this.symbol,
                    type: 'candlestick',
                    data: [],
                    itemStyle: {
                        color: '#26a69a',
                        color0: '#ef5350',
                        borderColor: '#26a69a',
                        borderColor0: '#ef5350'
                    }
                }
            ]
        };

        this.mainChart.setOption(option);
    }

    initVolumeChart() {
        const chartDom = document.getElementById('volumeChart');
        if (!chartDom) return;

        this.volumeChart = echarts.init(chartDom, this.theme);

        const option = {
            animation: false,
            xAxis: {
                type: 'category',
                data: [],
                show: false,
                min: 'dataMin',
                max: 'dataMax'
            },
            yAxis: {
                type: 'value',
                show: true,
                position: 'right',
                axisLabel: {
                    formatter: function (value) {
                        return value >= 1000 ? (value / 1000).toFixed(1) + 'K' : value;
                    }
                }
            },
            grid: {
                left: '10%',
                right: '10%',
                top: '10%',
                bottom: '10%'
            },
            dataZoom: [
                {
                    type: 'inside',
                    start: 0,
                    end: 100,
                    xAxisIndex: [0]
                }
            ],
            series: [
                {
                    name: 'Volumen',
                    type: 'bar',
                    data: [],
                    itemStyle: {
                        color: function (params) {
                            // If close price is higher than open, green volume bar
                            return params.value > 0 ? '#26a69a' : '#ef5350';
                        }
                    }
                }
            ]
        };

        this.volumeChart.setOption(option);
    }

    setupEventListeners() {
        // Sync zooming between charts
        if (this.mainChart && this.volumeChart) {
            this.mainChart.on('dataZoom', params => {
                if (params.batch) {
                    this.volumeChart.dispatchAction({
                        type: 'dataZoom',
                        start: params.batch[0].start,
                        end: params.batch[0].end
                    });

                    // Update indicator charts as well
                    Object.values(this.indicatorCharts).forEach(chart => {
                        chart.dispatchAction({
                            type: 'dataZoom',
                            start: params.batch[0].start,
                            end: params.batch[0].end
                        });
                    });
                }
            });
        }

        // Drawing tools events
        document.querySelectorAll('.tool-button').forEach(button => {
            button.addEventListener('click', () => {
                const tool = button.getAttribute('data-tool');
                this.activateDrawingTool(tool, button);
            });
        });

        // Zoom controls
        document.querySelectorAll('.zoom-button').forEach(button => {
            button.addEventListener('click', () => {
                const zoom = button.getAttribute('data-zoom');
                this.handleZoom(zoom);
            });
        });
    }

    activateDrawingTool(tool, buttonEl) {
        // Deactivate all tools
        document.querySelectorAll('.tool-button').forEach(btn => {
            btn.classList.remove('active');
        });

        if (this.currentDrawingTool === tool) {
            // Toggle off
            this.drawingMode = false;
            this.currentDrawingTool = null;
        } else {
            // Toggle on
            this.drawingMode = true;
            this.currentDrawingTool = tool;
            buttonEl.classList.add('active');
        }

        // Implement drawing functionality based on tool type
        // This would need canvas-based drawing over the ECharts instances
        console.log(`Drawing tool: ${tool} ${this.drawingMode ? 'activated' : 'deactivated'}`);
    }

    handleZoom(type) {
        if (!this.mainChart) return;

        const option = this.mainChart.getOption();
        let dataZoom = option.dataZoom[0];
        let start = dataZoom.start;
        let end = dataZoom.end;

        switch (type) {
            case 'in':
                // Zoom in - decrease visible range by 20% from both sides
                const zoomInAmount = (end - start) * 0.2;
                start = Math.min(start + zoomInAmount, end - 10);
                end = Math.max(end - zoomInAmount, start + 10);
                break;

            case 'out':
                // Zoom out - increase visible range by 20% on both sides
                const zoomOutAmount = (end - start) * 0.2;
                start = Math.max(start - zoomOutAmount, 0);
                end = Math.min(end + zoomOutAmount, 100);
                break;

            case 'reset':
                start = 0;
                end = 100;
                break;
        }

        this.mainChart.dispatchAction({
            type: 'dataZoom',
            start: start,
            end: end
        });
    }

    setLoading(isLoading) {
        this.isLoading = isLoading;
        const loadingEl = document.querySelector('.echarts-loading');
        if (loadingEl) {
            loadingEl.style.display = isLoading ? 'flex' : 'none';
        }
    }

    async loadData() {
        this.setLoading(true);

        try {
            // Format the API endpoint based on your backend
            const endpoint = `/api/market/data?symbol=${this.symbol}&timeframe=${this.timeframe}&source=${this.dataSource}`;

            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error('API error: ' + response.statusText);
            }

            const data = await response.json();

            if (!data || !Array.isArray(data)) {
                throw new Error('Invalid data format');
            }

            this.processData(data);
            this.updateCharts();
            this.calculateIndicators();

        } catch (error) {
            console.error('Error loading market data:', error);
            // Display error in UI
            const container = document.getElementById(this.container);
            if (container) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'chart-error';
                errorDiv.textContent = `Error cargando datos: ${error.message}`;
                container.appendChild(errorDiv);
            }
        } finally {
            this.setLoading(false);
        }
    }

    processData(data) {
        // Assuming data format: [time, open, high, low, close, volume]
        this.candleData = data.map(item => {
            const timestamp = new Date(item[0]);
            return {
                time: timestamp,
                value: [
                    item[1], // Open
                    item[2], // High
                    item[3], // Low
                    item[4]  // Close
                ]
            };
        });

        // Extract volume data
        this.volumeData = data.map(item => {
            const timestamp = new Date(item[0]);
            return {
                time: timestamp,
                value: item[5],
                color: item[4] >= item[1] ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
            };
        });
    }

    updateCharts() {
        if (!this.mainChart || !this.volumeChart) return;

        // Process data for ECharts format
        const categoryData = this.candleData.map(item => item.time);
        const values = this.candleData.map(item => item.value);
        const volumes = this.volumeData.map(item => item.value);

        // Update main chart
        this.mainChart.setOption({
            xAxis: {
                data: categoryData
            },
            series: [
                {
                    data: values
                }
            ]
        });

        // Update volume chart
        this.volumeChart.setOption({
            xAxis: {
                data: categoryData
            },
            series: [
                {
                    data: volumes
                }
            ]
        });
    }

    calculateIndicators() {
        // Example: Calculate SMA indicator
        this.addSMA(20);
        this.addSMA(50);
        this.addSMA(200);
    }

    addSMA(period) {
        if (!this.candleData || this.candleData.length === 0) return;

        const closePrices = this.candleData.map(item => item.value[3]);
        const sma = this.calculateSMA(closePrices, period);

        // Add to main chart
        this.mainChart.setOption({
            series: [
                ...this.mainChart.getOption().series,
                {
                    name: `SMA ${period}`,
                    type: 'line',
                    data: sma,
                    smooth: true,
                    lineStyle: {
                        opacity: 0.8,
                        width: 1.5,
                        color: this.getSMAColor(period)
                    },
                    symbol: 'none',
                    z: 1
                }
            ]
        });
    }

    calculateSMA(data, period) {
        const result = [];

        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                // Not enough data yet, push null
                result.push(null);
                continue;
            }

            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += data[i - j];
            }
            result.push(sum / period);
        }

        return result;
    }

    getSMAColor(period) {
        // Common color scheme for SMAs
        switch (period) {
            case 20: return '#22a7f0'; // Blue
            case 50: return '#f62459'; // Pink
            case 100: return '#f39c12'; // Orange
            case 200: return '#8e44ad'; // Purple
            default: return '#2ecc71'; // Green
        }
    }

    // Add indicator to a separate panel below the main chart
    addIndicatorChart(name, type) {
        // Create DOM element for indicator
        const indicatorsContainer = document.getElementById('indicatorsContainer');
        if (!indicatorsContainer) return;

        const indicatorId = `indicator-${name.toLowerCase().replace(/\s+/g, '-')}`;

        // Check if indicator already exists
        if (document.getElementById(indicatorId)) return;

        // Create indicator container
        const indicatorDiv = document.createElement('div');
        indicatorDiv.id = indicatorId;
        indicatorDiv.className = 'echarts-indicator-chart';
        indicatorDiv.style.height = '100%';
        indicatorsContainer.appendChild(indicatorDiv);

        // Initialize ECharts for this indicator
        const indicatorChart = echarts.init(indicatorDiv, this.theme);

        // Configure based on indicator type
        switch (type) {
            case 'rsi':
                this.configureRSIChart(indicatorChart);
                break;
            case 'macd':
                this.configureMACDChart(indicatorChart);
                break;
            // Add more indicator types as needed
        }

        this.indicatorCharts[name] = indicatorChart;
    }

    configureRSIChart(chart) {
        // Calculate RSI values (basic implementation)
        if (!this.candleData || this.candleData.length === 0) return;

        const closePrices = this.candleData.map(item => item.value[3]);
        const rsiPeriod = 14;
        const rsiValues = this.calculateRSI(closePrices, rsiPeriod);

        chart.setOption({
            title: {
                text: 'RSI (14)',
                left: 'center',
                textStyle: {
                    color: '#D9D9D9',
                    fontSize: 12
                }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross'
                }
            },
            grid: {
                left: '10%',
                right: '10%',
                top: '15%',
                bottom: '10%'
            },
            xAxis: {
                type: 'category',
                data: this.candleData.map(item => item.time),
                show: true,
                axisLabel: {
                    show: false
                },
                min: 'dataMin',
                max: 'dataMax'
            },
            yAxis: {
                type: 'value',
                min: 0,
                max: 100,
                splitNumber: 4,
                splitLine: {
                    show: true,
                    lineStyle: {
                        color: '#2A2E39'
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#8392A5'
                    }
                }
            },
            dataZoom: [
                {
                    type: 'inside',
                    start: 0,
                    end: 100,
                    xAxisIndex: [0]
                }
            ],
            series: [
                {
                    name: 'RSI',
                    type: 'line',
                    data: rsiValues,
                    lineStyle: {
                        color: '#9B7DFF',
                        width: 1.5
                    },
                    symbol: 'none',
                    markLine: {
                        data: [
                            { yAxis: 70, lineStyle: { color: '#ef5350' } },
                            { yAxis: 30, lineStyle: { color: '#26a69a' } }
                        ],
                        symbol: ['none', 'none'],
                        label: {
                            formatter: '{c}'
                        }
                    }
                }
            ]
        });
    }

    configureMACDChart(chart) {
        // Calculate MACD values (basic implementation)
        if (!this.candleData || this.candleData.length === 0) return;

        const closePrices = this.candleData.map(item => item.value[3]);
        const fastPeriod = 12;
        const slowPeriod = 26;
        const signalPeriod = 9;
        const macdData = this.calculateMACD(closePrices, fastPeriod, slowPeriod, signalPeriod);

        chart.setOption({
            title: {
                text: 'MACD (12, 26, 9)',
                left: 'center',
                textStyle: {
                    color: '#D9D9D9',
                    fontSize: 12
                }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross'
                }
            },
            grid: {
                left: '10%',
                right: '10%',
                top: '15%',
                bottom: '10%'
            },
            xAxis: {
                type: 'category',
                data: this.candleData.map(item => item.time),
                show: true,
                axisLabel: {
                    show: false
                },
                min: 'dataMin',
                max: 'dataMax'
            },
            yAxis: {
                type: 'value',
                splitLine: {
                    show: true,
                    lineStyle: {
                        color: '#2A2E39'
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#8392A5'
                    }
                }
            },
            dataZoom: [
                {
                    type: 'inside',
                    start: 0,
                    end: 100,
                    xAxisIndex: [0]
                }
            ],
            series: [
                {
                    name: 'MACD',
                    type: 'line',
                    data: macdData.macd,
                    lineStyle: {
                        color: '#9B7DFF',
                        width: 1.5
                    },
                    symbol: 'none'
                },
                {
                    name: 'Signal',
                    type: 'line',
                    data: macdData.signal,
                    lineStyle: {
                        color: '#FFA500',
                        width: 1.5
                    },
                    symbol: 'none'
                },
                {
                    name: 'Histogram',
                    type: 'bar',
                    data: macdData.histogram,
                    itemStyle: {
                        color: function (params) {
                            return params.value >= 0 ? '#26a69a' : '#ef5350';
                        }
                    }
                }
            ]
        });
    }

    calculateRSI(data, period) {
        const rsi = [];
        let gains = 0;
        let losses = 0;

        // Fill initial null values
        for (let i = 0; i < data.length; i++) {
            if (i < period) {
                rsi.push(null);
                continue;
            }

            if (i === period) {
                // First RSI calculation
                for (let j = 1; j <= period; j++) {
                    const difference = data[j] - data[j - 1];
                    if (difference >= 0) {
                        gains += difference;
                    } else {
                        losses -= difference;
                    }
                }

                const avgGain = gains / period;
                const avgLoss = losses / period;
                const rs = avgLoss > 0 ? avgGain / avgLoss : 100;
                rsi.push(100 - (100 / (1 + rs)));
            } else {
                // Use smoothed method
                const difference = data[i] - data[i - 1];
                let currentGain = 0;
                let currentLoss = 0;

                if (difference >= 0) {
                    currentGain = difference;
                } else {
                    currentLoss = -difference;
                }

                const smoothedGain = ((gains * (period - 1)) + currentGain) / period;
                const smoothedLoss = ((losses * (period - 1)) + currentLoss) / period;

                gains = smoothedGain;
                losses = smoothedLoss;

                const rs = losses > 0 ? gains / losses : 100;
                rsi.push(100 - (100 / (1 + rs)));
            }
        }

        return rsi;
    }

    calculateMACD(data, fastPeriod, slowPeriod, signalPeriod) {
        const emaFast = this.calculateEMA(data, fastPeriod);
        const emaSlow = this.calculateEMA(data, slowPeriod);

        const macdLine = [];

        // Calculate MACD line
        for (let i = 0; i < data.length; i++) {
            if (i < slowPeriod - 1) {
                macdLine.push(null);
            } else {
                macdLine.push(emaFast[i] - emaSlow[i]);
            }
        }

        // Calculate signal line (9-period EMA of MACD line)
        const signalLine = this.calculateEMA(macdLine, signalPeriod);

        // Calculate histogram
        const histogram = [];
        for (let i = 0; i < data.length; i++) {
            if (i < slowPeriod + signalPeriod - 2) {
                histogram.push(null);
            } else {
                histogram.push(macdLine[i] - signalLine[i]);
            }
        }

        return {
            macd: macdLine,
            signal: signalLine,
            histogram: histogram
        };
    }

    calculateEMA(data, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);

        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                // Not enough data yet
                ema.push(null);
                continue;
            }

            if (i === period - 1) {
                // First EMA is SMA
                let sum = 0;
                for (let j = 0; j < period; j++) {
                    sum += data[i - j];
                }
                ema.push(sum / period);
            } else {
                // EMA = Price(t) * k + EMA(y) * (1 – k)
                ema.push(data[i] * multiplier + ema[i - 1] * (1 - multiplier));
            }
        }

        return ema;
    }

    // Additional indicator methods would be added here
    addBollingerBands(period = 20, deviation = 2) {
        if (!this.candleData || this.candleData.length === 0) return;

        const closePrices = this.candleData.map(item => item.value[3]);
        const middle = this.calculateSMA(closePrices, period);

        // Calculate standard deviation
        const upperBand = [];
        const lowerBand = [];

        for (let i = 0; i < closePrices.length; i++) {
            if (i < period - 1) {
                upperBand.push(null);
                lowerBand.push(null);
                continue;
            }

            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += Math.pow(closePrices[i - j] - middle[i], 2);
            }

            const stdDev = Math.sqrt(sum / period);
            upperBand.push(middle[i] + (stdDev * deviation));
            lowerBand.push(middle[i] - (stdDev * deviation));
        }

        // Add to chart
        this.mainChart.setOption({
            series: [
                ...this.mainChart.getOption().series,
                {
                    name: 'BB Middle',
                    type: 'line',
                    data: middle,
                    smooth: true,
                    lineStyle: { opacity: 0.8, width: 1.5, color: '#7e57c2' },
                    symbol: 'none'
                },
                {
                    name: 'BB Upper',
                    type: 'line',
                    data: upperBand,
                    smooth: true,
                    lineStyle: { opacity: 0.5, width: 1.5, color: '#7e57c2', type: 'dashed' },
                    symbol: 'none'
                },
                {
                    name: 'BB Lower',
                    type: 'line',
                    data: lowerBand,
                    smooth: true,
                    lineStyle: { opacity: 0.5, width: 1.5, color: '#7e57c2', type: 'dashed' },
                    symbol: 'none'
                }
            ]
        });
    }
}

// Global initialization function
function initEChartsTradingChart(options = {}) {
    // Make sure the DOM is fully loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            const chart = new EChartsTradingChart(options);
            window.tradingChart = chart;  // Make it globally accessible
        });
    } else {
        const chart = new EChartsTradingChart(options);
        window.tradingChart = chart;  // Make it globally accessible
    }
}
