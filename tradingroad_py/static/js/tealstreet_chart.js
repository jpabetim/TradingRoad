/**
 * TealStreet Chart - Professional Trading Chart Implementation
 * Based on ECharts with advanced indicators and real-time data
 */

class TealStreetChart {
    constructor() {
        // Core chart properties
        this.chart = null;
        this.candleData = [];
        this.currentExchange = 'binance';
        this.currentSymbol = 'BTCUSDT';
        this.currentTimeframe = '1h';
        this.activeIndicators = new Map();
        
        // UI elements
        this.modal = document.getElementById('indicatorModal');
        this.indicatorsList = document.querySelector('.indicators-grid');
        this.activeIndicatorsContainer = document.querySelector('.active-indicators');

        // Colors for the chart
        this.colors = {
            bull: '#089981',  // Green for up candles
            bear: '#F23645',  // Red for down candles
            background: '#0C111C',
            grid: '#2A2E39',
            text: '#D1D4DC',
            volume: {
                up: 'rgba(8, 153, 129, 0.3)',
                down: 'rgba(242, 54, 69, 0.3)'
            },
            indicator: {
                sma: '#2962FF',
                ema: '#B71CFF',
                rsi: '#FF9800',
                macd: {
                    line: '#2962FF',
                    signal: '#FF9800',
                    histogram: {
                        positive: 'rgba(8, 153, 129, 0.6)',
                        negative: 'rgba(242, 54, 69, 0.6)'
                    }
                },
                bollinger: {
                    upper: '#FF9800',
                    middle: '#2962FF',
                    lower: '#FF9800'
                }
            }
        };
        
        // Initialize chart
        this.init();
    }
    
    /**
     * Initialize the chart and event listeners
     */
    init() {
        console.log("Initializing TealStreet Chart...");
        
        // Initialize the ECharts instance
        this.initChart();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load initial data
        this.loadChartData();
        
        // Update market info
        this.updateMarketInfo();
    }
    
    /**
     * Initialize the ECharts chart
     */
    initChart() {
        const chartContainer = document.getElementById('mainChart');
        
        if (!chartContainer) {
            console.error("Chart container not found!");
            return;
        }
        
        console.log("Creating chart instance...");
        
        this.chart = echarts.init(chartContainer);
        
        // Set chart options
        const option = {
            backgroundColor: this.colors.background,
            animation: false,
            axisPointer: {
                link: [{
                    xAxisIndex: [0, 1]
                }]
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    lineStyle: {
                        color: this.colors.grid,
                        width: 1
                    }
                },
                backgroundColor: '#131722',
                borderColor: '#2A2E39',
                textStyle: {
                    color: this.colors.text
                },
                borderWidth: 1,
                position: function (pos, params, el, elRect, size) {
                    const obj = { top: 10 };
                    obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
                    return obj;
                }
            },
            grid: [
                {
                    left: '10%',
                    right: '8%',
                    top: '60',
                    height: '60%'
                },
                {
                    left: '10%',
                    right: '8%',
                    top: '75%',
                    height: '15%'
                }
            ],
            xAxis: [
                {
                    type: 'category',
                    data: [],
                    scale: true,
                    boundaryGap: false,
                    axisLine: {
                        onZero: false,
                        lineStyle: {
                            color: this.colors.grid
                        }
                    },
                    splitLine: {
                        show: false
                    },
                    axisLabel: {
                        show: true,
                        color: this.colors.text,
                        fontSize: 11
                    },
                    min: 'dataMin',
                    max: 'dataMax'
                },
                {
                    type: 'category',
                    gridIndex: 1,
                    data: [],
                    axisLabel: { show: false },
                    axisLine: {
                        lineStyle: {
                            color: this.colors.grid
                        }
                    },
                    axisTick: { show: false },
                    splitLine: { show: false }
                }
            ],
            yAxis: [
                {
                    scale: true,
                    position: 'right',
                    axisLabel: {
                        color: this.colors.text,
                        fontSize: 11,
                        formatter: (value) => {
                            return value >= 1000 ? 
                                value >= 10000 ? 
                                    value >= 1000000 ? 
                                        `${(value / 1000000).toFixed(1)}M` : 
                                        `${(value / 1000).toFixed(1)}K` : 
                                    value.toFixed(1) : 
                                value.toFixed(2);
                        }
                    },
                    splitLine: {
                        lineStyle: {
                            color: this.colors.grid,
                            opacity: 0.2
                        }
                    },
                    axisLine: {
                        show: false
                    },
                    axisTick: {
                        show: false
                    }
                },
                {
                    scale: true,
                    gridIndex: 1,
                    position: 'right',
                    axisLabel: {
                        color: this.colors.text,
                        fontSize: 11,
                        formatter: (value) => {
                            if (value >= 10000) {
                                return `${(value / 1000).toFixed(0)}K`;
                            } else if (value >= 1000) {
                                return `${(value / 1000).toFixed(1)}K`;
                            }
                            return value;
                        }
                    },
                    splitLine: {
                        show: false
                    },
                    axisLine: {
                        show: false
                    },
                    axisTick: {
                        show: false
                    }
                }
            ],
            dataZoom: [
                {
                    type: 'inside',
                    xAxisIndex: [0, 1],
                    start: 50,
                    end: 100
                },
                {
                    show: true,
                    xAxisIndex: [0, 1],
                    type: 'slider',
                    bottom: '5%',
                    height: 20,
                    borderColor: this.colors.grid,
                    fillerColor: 'rgba(41, 98, 255, 0.2)',
                    textStyle: {
                        color: this.colors.text
                    },
                    handleIcon: 'path://M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                    handleSize: '80%',
                    handleStyle: {
                        color: '#2962FF',
                        shadowBlur: 3,
                        shadowColor: 'rgba(0, 0, 0, 0.5)',
                        shadowOffsetX: 2,
                        shadowOffsetY: 2
                    },
                    start: 50,
                    end: 100
                }
            ],
            series: [
                {
                    name: 'Candlestick',
                    type: 'candlestick',
                    data: [],
                    itemStyle: {
                        color: this.colors.bull,
                        color0: this.colors.bear,
                        borderColor: this.colors.bull,
                        borderColor0: this.colors.bear,
                    },
                    markPoint: {
                        label: {
                            formatter: function(param) {
                                return param.name;
                            }
                        },
                        data: []
                    }
                },
                {
                    name: 'Volume',
                    type: 'bar',
                    xAxisIndex: 1,
                    yAxisIndex: 1,
                    data: [],
                    itemStyle: {
                        color: params => {
                            return params.value[2] > params.value[1] 
                                ? this.colors.volume.up 
                                : this.colors.volume.down;
                        }
                    }
                }
            ]
        };
        
        this.chart.setOption(option);
        
        // Handle resizing
        window.addEventListener('resize', () => this.chart.resize());
        
        console.log("Chart instance created successfully");
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        console.log("Setting up event listeners...");
        
        // Timeframe buttons
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', (event) => {
                // Remove active class from all timeframe buttons
                document.querySelectorAll('.tf-btn').forEach(el => {
                    el.classList.remove('active');
                });
                
                // Add active class to clicked button
                event.target.classList.add('active');
                
                // Update timeframe and reload data
                this.currentTimeframe = event.target.dataset.tf;
                this.loadChartData();
            });
        });
        
        // Exchange selector
        document.getElementById('exchangeSelect').addEventListener('change', (event) => {
            this.currentExchange = event.target.value;
            this.loadChartData();
        });
        
        // Symbol search
        document.getElementById('searchSymbol').addEventListener('click', () => {
            const symbolInput = document.getElementById('symbolInput');
            if (symbolInput.value.trim() !== '') {
                this.currentSymbol = symbolInput.value.trim().toUpperCase();
                this.loadChartData();
            }
        });
        
        // Symbol input enter key
        document.getElementById('symbolInput').addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                const symbolInput = document.getElementById('symbolInput');
                if (symbolInput.value.trim() !== '') {
                    this.currentSymbol = symbolInput.value.trim().toUpperCase();
                    this.loadChartData();
                }
            }
        });
        
        // Indicator modal
        document.getElementById('addIndicator').addEventListener('click', () => {
            this.showIndicatorModal();
        });
        
        // Close modal
        document.querySelector('.close-modal').addEventListener('click', () => {
            this.hideIndicatorModal();
        });
        
        // Indicator category buttons
        document.querySelectorAll('.category-btn').forEach(btn => {
            btn.addEventListener('click', (event) => {
                // Remove active class from all category buttons
                document.querySelectorAll('.category-btn').forEach(el => {
                    el.classList.remove('active');
                });
                
                // Add active class to clicked button
                event.target.classList.add('active');
                
                const category = event.target.dataset.category;
                this.filterIndicators(category);
            });
        });
        
        // Add indicator buttons
        document.querySelectorAll('.add-btn').forEach(btn => {
            btn.addEventListener('click', (event) => {
                const indicator = event.target.dataset.indicator;
                this.addIndicator(indicator);
                this.hideIndicatorModal();
            });
        });
        
        // Indicator search
        document.getElementById('indicatorSearch').addEventListener('input', (event) => {
            this.searchIndicators(event.target.value);
        });
        
        console.log("Event listeners setup complete");
    }
    
    /**
     * Load chart data from API
     */
    async loadChartData() {
        console.log(`Loading chart data for ${this.currentSymbol} on ${this.currentExchange} with timeframe ${this.currentTimeframe}...`);
        
        try {
            // Show loading animation
            this.chart.showLoading({
                text: 'Cargando datos...',
                color: '#2962FF',
                textColor: '#D1D4DC',
                maskColor: 'rgba(12, 17, 28, 0.8)'
            });
            
            // En este momento hay problemas con los endpoints del backend
            // Por lo que generamos datos localmente
            console.log("Skipping API call due to endpoint issues, using generated data");
            
            // Generate data directly
            this.generateLocalMockData();
            
            // Hide loading animation
            this.chart.hideLoading();
            
            // Update market info with mock data
            this.updateMarketInfoMock();
            
            // Update indicators
            this.updateActiveIndicators();
            
        } catch (error) {
            console.error("Error loading chart data:", error);
            
            // Hide loading animation
            this.chart.hideLoading();
            
            // Load mock data as fallback
            this.generateLocalMockData();
            
            // Show error notification
            this.showNotification(`Error cargando datos: ${error.message}`, 'error');
        }
    }
    
    /**
     * Update chart with new data
     * @param {Array} data - Array of candle data
     */
    updateChartData(data) {
        if (!data || !Array.isArray(data) || data.length === 0) {
            console.error("Invalid data format received", data);
            return;
        }
        
        console.log(`Updating chart with ${data.length} candles`);
        console.log("First candle sample:", data[0]);
        
        // Store the data
        this.candleData = data;
        
        // Format data for candlestick chart
        const categoryData = [];
        const values = [];
        const volumes = [];
        
        data.forEach(item => {
            // Format date
            const date = new Date(item.timestamp);
            const formattedDate = this.formatDate(date);
            
            categoryData.push(formattedDate);
            
            // Format candle data [open, close, lowest, highest]
            values.push([item.open, item.close, item.low, item.high]);
            
            // Format volume data [0, 0, volume] (0 and 1 are for visualizing colors)
            volumes.push([item.open, item.close, item.volume]);
        });
        
        // Update chart options
        this.chart.setOption({
            xAxis: [{
                data: categoryData
            }, {
                data: categoryData
            }],
            series: [{
                data: values
            }, {
                data: volumes
            }]
        });
        
        console.log("Chart data updated successfully");
    }
    
    /**
     * Update market information
     */
    async updateMarketInfo() {
        console.log("Updating market info...");
        
        try {
            // Fetch current ticker data
            const url = `/api/exchange/ticker/${this.currentExchange}?symbol=${this.currentSymbol}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Update symbol info
            document.querySelector('.symbol-name').textContent = this.currentSymbol;
            document.querySelector('.exchange-tag').textContent = this.currentExchange.charAt(0).toUpperCase() + this.currentExchange.slice(1);
            
            // Update price info if we have ticker data
            if (data && data.price) {
                const price = parseFloat(data.price);
                const change = parseFloat(data.change) || 0;
                const volume = parseFloat(data.volume) || 0;
                
                // Format price
                document.querySelector('.current-price').textContent = `$${this.formatPrice(price)}`;
                
                // Format price change
                const priceChangeElement = document.querySelector('.price-change');
                const changeAmount = data.changeAmount || 0;
                
                priceChangeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}% (${changeAmount >= 0 ? '+' : ''}${this.formatPrice(changeAmount)})`;
                priceChangeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
                
                // Update market stats
                const statElements = document.querySelectorAll('.stat-value');
                if (statElements.length >= 3) {
                    // Volume
                    statElements[0].textContent = this.formatVolume(volume);
                    
                    // High & Low from latest candles if available
                    if (this.candleData && this.candleData.length > 0) {
                        // Take the most recent 24 candles (or less if not available)
                        const recentCandles = this.candleData.slice(-24);
                        const high = Math.max(...recentCandles.map(c => c.high));
                        const low = Math.min(...recentCandles.map(c => c.low));
                        
                        statElements[1].textContent = this.formatPrice(high);
                        statElements[2].textContent = this.formatPrice(low);
                    }
                }
            }
            
            console.log("Market info updated successfully");
            
        } catch (error) {
            console.error("Error updating market info:", error);
        }
    }
    
    /**
     * Show indicator modal
     */
    showIndicatorModal() {
        this.modal.style.display = 'block';
    }
    
    /**
     * Hide indicator modal
     */
    hideIndicatorModal() {
        this.modal.style.display = 'none';
    }
    
    /**
     * Filter indicators by category
     * @param {string} category - Category to filter by
     */
    filterIndicators(category) {
        const cards = document.querySelectorAll('.indicator-card');
        
        cards.forEach(card => {
            if (category === 'all' || card.dataset.category === category) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });
    }
    
    /**
     * Search indicators
     * @param {string} query - Search query
     */
    searchIndicators(query) {
        const cards = document.querySelectorAll('.indicator-card');
        const searchTerm = query.toLowerCase();
        
        cards.forEach(card => {
            const title = card.querySelector('h4').textContent.toLowerCase();
            const description = card.querySelector('p').textContent.toLowerCase();
            
            if (title.includes(searchTerm) || description.includes(searchTerm)) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });
    }
    
    /**
     * Add indicator to chart
     * @param {string} indicator - Indicator to add
     */
    addIndicator(indicator) {
        console.log(`Adding indicator: ${indicator}`);
        
        // Check if we have candleData
        if (!this.candleData || this.candleData.length === 0) {
            console.error("No chart data available for indicators");
            this.showNotification("No hay datos para calcular indicadores", "error");
            return;
        }
        
        // Default config for each indicator type
        const defaultConfigs = {
            sma: { period: 20, color: this.colors.indicator.sma },
            ema: { period: 14, color: this.colors.indicator.ema },
            rsi: { period: 14, color: this.colors.indicator.rsi },
            macd: { 
                fastPeriod: 12, 
                slowPeriod: 26, 
                signalPeriod: 9,
                colors: this.colors.indicator.macd
            },
            bollinger: { 
                period: 20, 
                stdDev: 2,
                colors: this.colors.indicator.bollinger
            }
        };
        
        // Get the default config for this indicator
        const config = defaultConfigs[indicator] || {};
        
        // If the indicator is already active, remove it first
        if (this.activeIndicators.has(indicator)) {
            this.removeIndicator(indicator);
            return;
        }
        
        // Calculate indicator data based on type
        let indicatorData;
        let name;
        let subtitle;
        
        switch (indicator) {
            case 'sma':
                name = "SMA";
                subtitle = `Simple (${config.period})`;
                indicatorData = this.calculateSMA(this.candleData, config.period);
                break;
                
            case 'ema':
                name = "EMA";
                subtitle = `Exponencial (${config.period})`;
                indicatorData = this.calculateEMA(this.candleData, config.period);
                break;
                
            case 'rsi':
                name = "RSI";
                subtitle = `(${config.period})`;
                indicatorData = this.calculateRSI(this.candleData, config.period);
                
                // Create a new pane for RSI
                this.addIndicatorPane("RSI", indicatorData, {
                    min: 0,
                    max: 100,
                    splitLine: { 
                        show: true,
                        lineStyle: { color: this.colors.grid, opacity: 0.2 }
                    },
                    axisLabel: { color: this.colors.text },
                    position: "right"
                });
                break;
                
            case 'macd':
                name = "MACD";
                subtitle = `(${config.fastPeriod},${config.slowPeriod},${config.signalPeriod})`;
                indicatorData = this.calculateMACD(
                    this.candleData, 
                    config.fastPeriod, 
                    config.slowPeriod, 
                    config.signalPeriod
                );
                
                // Create a new pane for MACD
                this.addMACDPane(indicatorData, config.colors);
                break;
                
            case 'bollinger':
                name = "Bollinger";
                subtitle = `(${config.period}, ${config.stdDev})`;
                indicatorData = this.calculateBollingerBands(this.candleData, config.period, config.stdDev);
                
                // Add three lines: upper, middle, lower
                this.addLine(
                    `Bollinger Upper (${config.period}, ${config.stdDev})`, 
                    indicatorData.upper, 
                    config.colors.upper
                );
                this.addLine(
                    `Bollinger Middle (${config.period})`, 
                    indicatorData.middle, 
                    config.colors.middle
                );
                this.addLine(
                    `Bollinger Lower (${config.period}, ${config.stdDev})`, 
                    indicatorData.lower, 
                    config.colors.lower
                );
                break;
                
            case 'custom':
                // Show dialog for custom indicator (placeholder)
                this.showNotification("Funcionalidad de indicadores personalizados próximamente", "info");
                return;
                
            default:
                console.error(`Unknown indicator type: ${indicator}`);
                return;
        }
        
        // Add to active indicators map
        this.activeIndicators.set(indicator, {
            name,
            subtitle,
            config
        });
        
        // Add indicator chip to UI
        this.addIndicatorChip(indicator, name, subtitle);
        
        console.log(`Indicator ${indicator} added successfully`);
    }
    
    /**
     * Remove indicator from chart
     * @param {string} indicator - Indicator to remove
     */
    removeIndicator(indicator) {
        console.log(`Removing indicator: ${indicator}`);
        
        // Get all series
        const option = this.chart.getOption();
        
        // Filter out the indicator series
        switch (indicator) {
            case 'sma':
                option.series = option.series.filter(s => !s.name || !s.name.includes('SMA'));
                break;
                
            case 'ema':
                option.series = option.series.filter(s => !s.name || !s.name.includes('EMA'));
                break;
                
            case 'rsi':
                // Remove RSI series and its grid/yAxis
                option.series = option.series.filter(s => !s.name || !s.name.includes('RSI'));
                
                // Find the grid index and remove it
                const rsiGridIndex = option.yAxis.findIndex(axis => axis.name === "RSI");
                if (rsiGridIndex > 1) {
                    option.yAxis.splice(rsiGridIndex, 1);
                    option.grid.splice(rsiGridIndex, 1);
                }
                break;
                
            case 'macd':
                // Remove MACD series and its grid/yAxis
                option.series = option.series.filter(s => !s.name || !s.name.includes('MACD'));
                
                // Find the grid index and remove it
                const macdGridIndex = option.yAxis.findIndex(axis => axis.name === "MACD");
                if (macdGridIndex > 1) {
                    option.yAxis.splice(macdGridIndex, 1);
                    option.grid.splice(macdGridIndex, 1);
                }
                break;
                
            case 'bollinger':
                option.series = option.series.filter(s => !s.name || !s.name.includes('Bollinger'));
                break;
        }
        
        // Update the chart
        this.chart.setOption(option, true);
        
        // Remove from active indicators map
        this.activeIndicators.delete(indicator);
        
        // Remove indicator chip from UI
        const chip = document.querySelector(`[data-indicator="${indicator}"]`);
        if (chip) {
            chip.remove();
        }
        
        console.log(`Indicator ${indicator} removed successfully`);
    }
    
    /**
     * Add indicator chip to UI
     * @param {string} indicator - Indicator type
     * @param {string} name - Indicator display name
     * @param {string} subtitle - Indicator parameters
     */
    addIndicatorChip(indicator, name, subtitle) {
        const chip = document.createElement('div');
        chip.className = 'indicator-chip';
        chip.dataset.indicator = indicator;
        
        chip.innerHTML = `
            <span>${name}</span>
            <span class="text-secondary">${subtitle}</span>
            <button class="close-btn">×</button>
        `;
        
        // Add click handler to close button
        chip.querySelector('.close-btn').addEventListener('click', () => {
            this.removeIndicator(indicator);
        });
        
        // Add to active indicators container
        this.activeIndicatorsContainer.appendChild(chip);
    }
    
    /**
     * Update all active indicators after data changes
     */
    updateActiveIndicators() {
        console.log("Updating active indicators...");
        
        // Clear existing indicators
        const option = this.chart.getOption();
        
        // Keep only the first two series (candles and volume)
        const baseSeries = option.series.slice(0, 2);
        option.series = baseSeries;
        
        // Keep only the first two grids and yAxis (main chart and volume)
        const baseGrids = option.grid.slice(0, 2);
        const baseYAxis = option.yAxis.slice(0, 2);
        option.grid = baseGrids;
        option.yAxis = baseYAxis;
        
        // Update chart
        this.chart.setOption(option, true);
        
        // Re-add all active indicators
        for (const [indicator, details] of this.activeIndicators.entries()) {
            this.addIndicator(indicator);
        }
        
        console.log("Active indicators updated successfully");
    }
    
    /**
     * Add a line to the main chart
     * @param {string} name - Line name
     * @param {Array} data - Line data
     * @param {string} color - Line color
     */
    addLine(name, data, color) {
        // Get current options
        const option = this.chart.getOption();
        
        // Add the line series
        option.series.push({
            name: name,
            type: 'line',
            data: data,
            showSymbol: false,
            lineStyle: {
                color: color,
                width: 1
            },
            z: 100
        });
        
        // Update chart
        this.chart.setOption(option);
    }
    
    /**
     * Add a new indicator pane to the chart
     * @param {string} name - Indicator name
     * @param {Array} data - Indicator data
     * @param {Object} yAxisConfig - Y-axis configuration
     */
    addIndicatorPane(name, data, yAxisConfig) {
        // Get current options
        const option = this.chart.getOption();
        
        // Calculate new grid index
        const gridIndex = option.grid.length;
        
        // Calculate position for the new grid
        let top = 0;
        const lastGrid = option.grid[option.grid.length - 1];
        top = parseFloat(lastGrid.top) + parseFloat(lastGrid.height) + 10;
        
        // Add new grid
        option.grid.push({
            left: '10%',
            right: '8%',
            top: `${top}px`,
            height: '150px'
        });
        
        // Add new Y axis
        option.yAxis.push({
            name: name,
            gridIndex: gridIndex,
            position: yAxisConfig.position || 'right',
            min: yAxisConfig.min,
            max: yAxisConfig.max,
            splitLine: yAxisConfig.splitLine || {
                show: false
            },
            axisLabel: yAxisConfig.axisLabel || {
                color: this.colors.text
            },
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            }
        });
        
        // Add new series
        option.series.push({
            name: name,
            type: 'line',
            xAxisIndex: 0,
            yAxisIndex: gridIndex,
            data: data,
            showSymbol: false,
            lineStyle: {
                color: yAxisConfig.color || this.colors.indicator.rsi,
                width: 1
            }
        });
        
        // Update chart
        this.chart.setOption(option);
    }
    
    /**
     * Add MACD indicator pane to the chart
     * @param {Object} macdData - MACD data
     * @param {Object} colors - MACD colors
     */
    addMACDPane(macdData, colors) {
        // Get current options
        const option = this.chart.getOption();
        
        // Calculate new grid index
        const gridIndex = option.grid.length;
        
        // Calculate position for the new grid
        let top = 0;
        const lastGrid = option.grid[option.grid.length - 1];
        top = parseFloat(lastGrid.top) + parseFloat(lastGrid.height) + 10;
        
        // Add new grid
        option.grid.push({
            left: '10%',
            right: '8%',
            top: `${top}px`,
            height: '150px'
        });
        
        // Add new Y axis
        option.yAxis.push({
            name: "MACD",
            gridIndex: gridIndex,
            position: 'right',
            splitLine: {
                show: true,
                lineStyle: { color: this.colors.grid, opacity: 0.2 }
            },
            axisLabel: {
                color: this.colors.text
            },
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            }
        });
        
        // Add MACD line
        option.series.push({
            name: "MACD Line",
            type: 'line',
            xAxisIndex: 0,
            yAxisIndex: gridIndex,
            data: macdData.macdLine,
            showSymbol: false,
            lineStyle: {
                color: colors.line,
                width: 1
            }
        });
        
        // Add signal line
        option.series.push({
            name: "MACD Signal",
            type: 'line',
            xAxisIndex: 0,
            yAxisIndex: gridIndex,
            data: macdData.signalLine,
            showSymbol: false,
            lineStyle: {
                color: colors.signal,
                width: 1
            }
        });
        
        // Add histogram
        option.series.push({
            name: "MACD Histogram",
            type: 'bar',
            xAxisIndex: 0,
            yAxisIndex: gridIndex,
            data: macdData.histogram,
            barWidth: '70%',
            itemStyle: {
                color: params => {
                    return params.value >= 0 ? colors.histogram.positive : colors.histogram.negative;
                }
            }
        });
        
        // Update chart
        this.chart.setOption(option);
    }
    
    /**
     * Calculate Simple Moving Average
     * @param {Array} data - Candle data
     * @param {number} period - SMA period
     * @returns {Array} - SMA data
     */
    calculateSMA(data, period = 20) {
        const result = [];
        
        // We need at least 'period' data points
        if (data.length < period) {
            console.warn(`Not enough data points for SMA calculation. Need ${period}, have ${data.length}`);
            return result;
        }
        
        // For the first 'period-1' points, we can't calculate SMA
        for (let i = 0; i < period - 1; i++) {
            result.push(null);
        }
        
        // Calculate SMA for the rest of the points
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += data[i - j].close;
            }
            result.push(sum / period);
        }
        
        return result;
    }
    
    /**
     * Calculate Exponential Moving Average
     * @param {Array} data - Candle data
     * @param {number} period - EMA period
     * @returns {Array} - EMA data
     */
    calculateEMA(data, period = 14) {
        const result = [];
        
        // We need at least 'period' data points
        if (data.length < period) {
            console.warn(`Not enough data points for EMA calculation. Need ${period}, have ${data.length}`);
            return result;
        }
        
        // For the first 'period-1' points, we can't calculate EMA
        for (let i = 0; i < period - 1; i++) {
            result.push(null);
        }
        
        // First EMA is SMA
        let sum = 0;
        for (let i = 0; i < period; i++) {
            sum += data[i].close;
        }
        let ema = sum / period;
        result.push(ema);
        
        // Calculate EMA for the rest of the points
        const multiplier = 2 / (period + 1);
        for (let i = period; i < data.length; i++) {
            ema = (data[i].close - ema) * multiplier + ema;
            result.push(ema);
        }
        
        return result;
    }
    
    /**
     * Calculate Relative Strength Index
     * @param {Array} data - Candle data
     * @param {number} period - RSI period
     * @returns {Array} - RSI data
     */
    calculateRSI(data, period = 14) {
        const result = [];
        
        // We need at least 'period+1' data points
        if (data.length <= period) {
            console.warn(`Not enough data points for RSI calculation. Need ${period+1}, have ${data.length}`);
            return result;
        }
        
        // For the first 'period' points, we can't calculate RSI
        for (let i = 0; i < period; i++) {
            result.push(null);
        }
        
        // Calculate first RS
        let sumGain = 0;
        let sumLoss = 0;
        
        for (let i = 1; i <= period; i++) {
            const change = data[i].close - data[i-1].close;
            if (change >= 0) {
                sumGain += change;
            } else {
                sumLoss -= change;
            }
        }
        
        // Avoid division by zero
        sumLoss = sumLoss || 0.0001;
        
        let rs = sumGain / sumLoss;
        let rsi = 100 - (100 / (1 + rs));
        result.push(rsi);
        
        // Calculate RS and RSI for the rest of the points
        for (let i = period + 1; i < data.length; i++) {
            const change = data[i].close - data[i-1].close;
            
            let gain = 0;
            let loss = 0;
            
            if (change >= 0) {
                gain = change;
            } else {
                loss = -change;
            }
            
            // Use EMA-based approach for gains and losses
            sumGain = ((sumGain * (period - 1)) + gain) / period;
            sumLoss = ((sumLoss * (period - 1)) + loss) / period;
            
            // Avoid division by zero
            if (sumLoss === 0) sumLoss = 0.0001;
            
            rs = sumGain / sumLoss;
            rsi = 100 - (100 / (1 + rs));
            result.push(rsi);
        }
        
        return result;
    }
    
    /**
     * Calculate Moving Average Convergence Divergence
     * @param {Array} data - Candle data
     * @param {number} fastPeriod - Fast EMA period
     * @param {number} slowPeriod - Slow EMA period
     * @param {number} signalPeriod - Signal line period
     * @returns {Object} - MACD data
     */
    calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
        // Calculate fast and slow EMAs
        const fastEMA = this.calculateEMA(data, fastPeriod);
        const slowEMA = this.calculateEMA(data, slowPeriod);
        
        // Calculate MACD line (fast EMA - slow EMA)
        const macdLine = [];
        const maxLength = Math.max(fastEMA.length, slowEMA.length);
        
        for (let i = 0; i < maxLength; i++) {
            if (fastEMA[i] !== null && slowEMA[i] !== null) {
                macdLine.push(fastEMA[i] - slowEMA[i]);
            } else {
                macdLine.push(null);
            }
        }
        
        // Calculate signal line (EMA of MACD line)
        const signalLine = [];
        let validMacdCount = 0;
        let macdSum = 0;
        
        // Fill with null until we have enough data
        for (let i = 0; i < macdLine.length; i++) {
            if (macdLine[i] === null) {
                signalLine.push(null);
                continue;
            }
            
            validMacdCount++;
            macdSum += macdLine[i];
            
            if (validMacdCount < signalPeriod) {
                signalLine.push(null);
                continue;
            }
            
            if (validMacdCount === signalPeriod) {
                // First signal value is SMA
                signalLine.push(macdSum / signalPeriod);
                continue;
            }
            
            // EMA calculation for signal line
            const multiplier = 2 / (signalPeriod + 1);
            const lastSignal = signalLine[signalLine.length - 1];
            const signal = (macdLine[i] - lastSignal) * multiplier + lastSignal;
            signalLine.push(signal);
        }
        
        // Calculate histogram (MACD line - signal line)
        const histogram = [];
        
        for (let i = 0; i < macdLine.length; i++) {
            if (macdLine[i] !== null && signalLine[i] !== null) {
                histogram.push(macdLine[i] - signalLine[i]);
            } else {
                histogram.push(null);
            }
        }
        
        return { macdLine, signalLine, histogram };
    }
    
    /**
     * Calculate Bollinger Bands
     * @param {Array} data - Candle data
     * @param {number} period - Period for SMA
     * @param {number} stdDev - Standard deviation multiplier
     * @returns {Object} - Bollinger bands data
     */
    calculateBollingerBands(data, period = 20, stdDev = 2) {
        // Calculate middle band (SMA)
        const middle = this.calculateSMA(data, period);
        
        // Calculate upper and lower bands
        const upper = [];
        const lower = [];
        
        // Fill with null for the first 'period-1' points
        for (let i = 0; i < period - 1; i++) {
            upper.push(null);
            lower.push(null);
        }
        
        // Calculate standard deviation and bands for the rest of the points
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += Math.pow(data[i - j].close - middle[i], 2);
            }
            const standardDeviation = Math.sqrt(sum / period);
            
            upper.push(middle[i] + (stdDev * standardDeviation));
            lower.push(middle[i] - (stdDev * standardDeviation));
        }
        
        return { upper, middle, lower };
    }
    
    /**
     * Load mock data when API fails
     */
    async loadMockData() {
        console.log("Loading mock data as fallback from server API");
        
        try {
            // Build mock data URL - using our server-side generator for consistent data
            const url = `/api/exchange/klines/binance?symbol=${this.currentSymbol}&interval=${this.currentTimeframe}&limit=100`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error when fetching mock data! Status: ${response.status}`);
            }
            
            const data = await response.json();
            const mockData = data.data || [];
            
            console.log(`Received ${mockData.length} mock candles from API`);
            
            if (mockData.length > 0) {
                this.updateChartData(mockData);
                
                // Update UI to show it's mock data
                document.querySelector('.symbol-name').textContent = `${this.currentSymbol} (DEMO)`;
                
                console.log("Mock data loaded successfully");
            } else {
                console.error("Empty mock data received");
                
                // Generate fallback data locally if server fails
                this.generateLocalMockData();
            }
            
        } catch (error) {
            console.error("Error fetching mock data:", error);
            
            // Generate fallback data locally if server fails
            this.generateLocalMockData();
        }
    },
    
    generateLocalMockData() {
        console.log("Generating local mock data as ultimate fallback");
        
        // Generate 100 candlesticks of mock data
        const mockData = [];
        const now = new Date();
        let price = 45000 + Math.random() * 10000;
        
        // Get interval in milliseconds based on timeframe
        let interval = 3600000; // Default 1h
        switch (this.currentTimeframe) {
            case '1m': interval = 60000; break;
            case '5m': interval = 300000; break;
            case '15m': interval = 900000; break;
            case '30m': interval = 1800000; break;
            case '1h': interval = 3600000; break;
            case '4h': interval = 14400000; break;
            case '1d': interval = 86400000; break;
            case '1w': interval = 604800000; break;
        }
        
        // Generate data
        for (let i = 100; i >= 0; i--) {
            const timestamp = now.getTime() - (i * interval);
            
            // Random price movement
            const change = (Math.random() - 0.5) * price * 0.02;
            const open = price;
            const close = price + change;
            const high = Math.max(open, close) + (Math.random() * Math.abs(change));
            const low = Math.min(open, close) - (Math.random() * Math.abs(change));
            
            // Random volume
            const volume = 1000 + Math.random() * 5000;
            
            mockData.push({
                timestamp,
                open,
                high,
                low,
                close,
                volume
            });
            
            // Update price for next candle
            price = close;
        }
        
        this.updateChartData(mockData);
        
        // Update UI to show it's mock data
        document.querySelector('.symbol-name').textContent = `${this.currentSymbol} (DEMO LOCAL)`;
        
        console.log("Local mock data generated successfully");
    }
    
    /**
     * Update market information with mock data
     */
    updateMarketInfoMock() {
        console.log("Updating market info with mock data");
        
        // Update symbol and exchange info
        document.querySelector('.symbol-name').textContent = this.currentSymbol;
        document.querySelector('.exchange-tag').textContent = this.currentExchange;
        
        // Generate mock price based on symbol
        let basePrice = 0;
        switch (this.currentSymbol) {
            case 'BTCUSDT': basePrice = 65000; break;
            case 'ETHUSDT': basePrice = 3500; break;
            case 'SOLUSDT': basePrice = 150; break;
            default: basePrice = 1000; break;
        }
        
        // Random price variation (±2%)
        const change = basePrice * (Math.random() * 0.04 - 0.02);
        const changePercent = (change / basePrice) * 100;
        const price = basePrice + change;
        
        // Format values
        const formattedPrice = price.toLocaleString('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
        
        const formattedChange = `${change >= 0 ? '+' : ''}${change.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })} (${changePercent >= 0 ? '+' : ''}${changePercent.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}%)`;
        
        // Update UI
        const priceElement = document.querySelector('.current-price');
        const changeElement = document.querySelector('.price-change');
        
        if (priceElement && changeElement) {
            priceElement.textContent = formattedPrice;
            changeElement.textContent = formattedChange;
            
            // Set color based on change
            changeElement.classList.remove('positive', 'negative');
            changeElement.classList.add(change >= 0 ? 'positive' : 'negative');
        }
        
        // Update stats
        const elements = document.querySelectorAll('.stat-value');
        if (elements.length >= 3) {
            // Volume (in billions)
            elements[0].textContent = `${(Math.random() * 5 + 1).toFixed(1)}B`;
            
            // High
            elements[1].textContent = (price * (1 + Math.random() * 0.01)).toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
            
            // Low
            elements[2].textContent = (price * (1 - Math.random() * 0.01)).toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
        }
    },
    
    /**
     * Format date for chart X-axis
     * @param {Date} date - Date to format
     * @returns {string} - Formatted date
     */
    formatDate(date) {
        const month = date.getMonth() + 1;
        const day = date.getDate();
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        
        // Format based on timeframe
        if (this.currentTimeframe === '1d' || this.currentTimeframe === '1w') {
            return `${month}/${day}`;
        } else {
            return `${month}/${day} ${hours}:${minutes}`;
        }
    }
    
    /**
     * Format price
     * @param {number} price - Price to format
     * @returns {string} - Formatted price
     */
    formatPrice(price) {
        if (price >= 1000) {
            return price.toLocaleString(undefined, { maximumFractionDigits: 2 });
        } else if (price >= 1) {
            return price.toLocaleString(undefined, { maximumFractionDigits: 2 });
        } else {
            return price.toLocaleString(undefined, { maximumFractionDigits: 6 });
        }
    }
    
    /**
     * Format volume
     * @param {number} volume - Volume to format
     * @returns {string} - Formatted volume
     */
    formatVolume(volume) {
        if (volume >= 1000000000) {
            return `${(volume / 1000000000).toFixed(2)}B`;
        } else if (volume >= 1000000) {
            return `${(volume / 1000000).toFixed(2)}M`;
        } else if (volume >= 1000) {
            return `${(volume / 1000).toFixed(2)}K`;
        } else {
            return volume.toFixed(2);
        }
    }
    
    /**
     * Show notification
     * @param {string} message - Message to show
     * @param {string} type - Type of notification (success, error, warning, info)
     */
    showNotification(message, type = 'info') {
        console.log(`[${type}] ${message}`);
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add styles if not exist
        if (!document.getElementById('notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
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
            document.head.appendChild(style);
        }
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideIn 0.3s ease-out reverse';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }
    
    /**
     * Get icon for notification type
     * @param {string} type - Notification type
     * @returns {string} - Icon class
     */
    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }
}

// Initialize chart when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tealStreetChart = new TealStreetChart();
});
