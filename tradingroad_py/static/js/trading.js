// Configuración para el módulo de trading
document.addEventListener('DOMContentLoaded', function () {
    // Marcar que estamos en la sección de trading
    document.body.classList.add('section-trading');
    document.body.classList.remove('section-analysis');

    // Variables globales
    window.chart = null;
    let candleSeries;
    let currentSource = 'binance';
    let currentSymbol = 'BTC/USDT';
    let currentTimeframe = '1h';
    let marketData = [];
    let drawingMode = null;
    let drawingObjects = [];

    // Inicializar el gráfico
    const chartContainer = document.getElementById('tradingViewContainer');

    // Obtener la zona horaria seleccionada
    const timezoneSelect = document.getElementById('timezone-select');
    const selectedTimezone = timezoneSelect ? timezoneSelect.value : 'UTC';

    // Configuración del gráfico con estilo TradingView
    chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
        layout: {
            background: { type: 'solid', color: '#131722' },
            textColor: '#d9d9d9',
            fontSize: 12,
            fontFamily: "'Trebuchet MS', Roboto, Ubuntu, sans-serif",
        },
        grid: {
            vertLines: { color: '#1e222d' },
            horzLines: { color: '#1e222d' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {
                labelBackgroundColor: '#9B7DFF',
            },
            horzLine: {
                labelBackgroundColor: '#9B7DFF',
            },
        },
        rightPriceScale: {
            borderColor: '#1e222d',
            scaleMargins: {
                top: 0.1,
                bottom: 0.2,
            },
        },
        timeScale: {
            borderColor: '#1e222d',
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
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
    });

    // Función para cargar datos del mercado
    function loadMarketData() {
        // Mostrar indicador de carga
        document.querySelector('.chart-loading').style.display = 'flex';

        // Construir la URL de la API
        const url = `/api/market/data?source=${currentSource}&symbol=${currentSymbol}&timeframe=${currentTimeframe}&limit=500`;

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

                    // Actualizar información en la leyenda
                    updateChartLegend();

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

    // Función para actualizar la leyenda del gráfico
    function updateChartLegend() {
        if (marketData.length === 0) return;

        const lastCandle = marketData[marketData.length - 1];
        const prevCandle = marketData.length > 1 ? marketData[marketData.length - 2] : null;

        document.getElementById('chart-symbol').textContent = currentSymbol;
        document.getElementById('chart-price').textContent = lastCandle.close.toFixed(2);

        if (prevCandle) {
            const change = ((lastCandle.close - prevCandle.close) / prevCandle.close) * 100;
            const changeElement = document.getElementById('chart-change');
            changeElement.textContent = change.toFixed(2) + '%';
            changeElement.classList.toggle('positive', change >= 0);
            changeElement.classList.toggle('negative', change < 0);
        }

        document.getElementById('chart-timeframe').textContent = currentTimeframe.toUpperCase();

        // Actualizar máximos y mínimos
        const maxPrice = Math.max(...marketData.map(c => c.high));
        const minPrice = Math.min(...marketData.map(c => c.low));

        document.getElementById('chart-max-price').textContent = `Máximo: ${maxPrice.toFixed(2)}`;
        document.getElementById('chart-min-price').textContent = `Mínimo: ${minPrice.toFixed(2)}`;
    }

    // Función para iniciar el contador de vela
    function startCandleCountdown() {
        // Implementación del contador de vela
        const countdownElement = document.getElementById('chart-countdown');

        // Detener cualquier contador existente
        if (window.candleCountdownInterval) {
            clearInterval(window.candleCountdownInterval);
        }

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

        const interval = timeframeMap[currentTimeframe];

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

    // Inicializar búsqueda de símbolos
    initSymbolSearch();

    // Función para búsqueda de símbolos y favoritos
    function initSymbolSearch() {
        const symbolSearch = document.getElementById('symbolSearch');
        const symbolResults = document.getElementById('symbolResults');
        const favoriteToggle = document.getElementById('favorite-toggle');
        const favoriteSymbols = JSON.parse(localStorage.getItem('favoriteSymbols')) || [];

        // Cargar favoritos guardados
        loadFavoriteSymbols();

        // Configurar búsqueda
        symbolSearch.addEventListener('focus', function () {
            loadSymbols();
        });

        symbolSearch.addEventListener('input', function () {
            const searchTerm = this.value.toUpperCase();
            if (searchTerm.length < 1) {
                symbolResults.style.display = 'none';
                return;
            }

            fetch(`/api/market/symbols?source=${currentSource}`)
                .then(response => response.json())
                .then(symbols => {
                    // Filtrar símbolos por término de búsqueda
                    const filteredSymbols = symbols.filter(symbol =>
                        symbol.toUpperCase().includes(searchTerm)
                    ).slice(0, 20); // Limitar a 20 resultados

                    displaySymbolResults(filteredSymbols);
                });
        });

        // Manejar clic en botón de favoritos
        favoriteToggle.addEventListener('click', function () {
            const symbol = symbolSearch.value.toUpperCase();
            if (!symbol) return;

            const favorites = JSON.parse(localStorage.getItem('favoriteSymbols')) || [];

            if (favorites.includes(symbol)) {
                // Quitar de favoritos
                const index = favorites.indexOf(symbol);
                favorites.splice(index, 1);
                favoriteToggle.classList.remove('active');
            } else {
                // Añadir a favoritos
                favorites.push(symbol);
                favoriteToggle.classList.add('active');
            }

            localStorage.setItem('favoriteSymbols', JSON.stringify(favorites));
            loadFavoriteSymbols();
        });

        // Función para mostrar resultados de búsqueda
        function displaySymbolResults(symbols) {
            symbolResults.innerHTML = '';

            if (symbols.length === 0) {
                symbolResults.innerHTML = '<div class="symbol-result-item">No se encontraron coincidencias</div>';
            } else {
                symbols.forEach(symbol => {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'symbol-result-item';
                    resultItem.textContent = symbol;
                    resultItem.addEventListener('click', function () {
                        symbolSearch.value = symbol;
                        currentSymbol = symbol;
                        symbolResults.style.display = 'none';
                        loadMarketData();

                        // Actualizar estado del botón de favorito
                        const favorites = JSON.parse(localStorage.getItem('favoriteSymbols')) || [];
                        favoriteToggle.classList.toggle('active', favorites.includes(symbol));
                    });
                    symbolResults.appendChild(resultItem);
                });
            }

            symbolResults.style.display = 'block';
        }

        // Función para cargar símbolos
        function loadSymbols() {
            fetch(`/api/market/symbols?source=${currentSource}`)
                .then(response => response.json())
                .then(symbols => {
                    // Guardar símbolos en el almacenamiento local
                    localStorage.setItem(`symbols_${currentSource}`, JSON.stringify(symbols));
                })
                .catch(error => {
                    console.error('Error al cargar símbolos:', error);
                });
        }

        // Función para cargar favoritos
        function loadFavoriteSymbols() {
            const favoritesList = document.getElementById('favoriteList');
            const favorites = JSON.parse(localStorage.getItem('favoriteSymbols')) || [];

            favoritesList.innerHTML = '';

            if (favorites.length === 0) {
                favoritesList.innerHTML = '<div class="favorite-empty">No hay favoritos</div>';
            } else {
                favorites.forEach(symbol => {
                    const favoriteItem = document.createElement('div');
                    favoriteItem.className = 'favorite-item';
                    favoriteItem.textContent = symbol;
                    favoriteItem.addEventListener('click', function () {
                        symbolSearch.value = symbol;
                        currentSymbol = symbol;
                        loadMarketData();

                        // Actualizar estado del botón de favorito
                        favoriteToggle.classList.add('active');
                    });
                    favoritesList.appendChild(favoriteItem);
                });
            }

            // Actualizar estado del botón de favorito para el símbolo actual
            const currentSymbolValue = symbolSearch.value.toUpperCase();
            if (currentSymbolValue) {
                favoriteToggle.classList.toggle('active', favorites.includes(currentSymbolValue));
            }
        }
    }

    // Event listeners para cambios de fuente de datos y timeframe
    document.getElementById('dataSource').addEventListener('change', function () {
        currentSource = this.value;
        loadMarketData();
    });

    document.getElementById('timeframeSelect').addEventListener('change', function () {
        currentTimeframe = this.value;
        loadMarketData();
    });

    // Event listener para cambio de zona horaria
    if (timezoneSelect) {
        timezoneSelect.addEventListener('change', function () {
            // Recargar datos con nueva zona horaria
            loadMarketData();
        });
    }

    // Event listener para el botón de ocultar
    document.getElementById('hideControls').addEventListener('click', function () {
        document.getElementById('sidebar-controls').classList.toggle('collapsed');
    });

    // Event listener para el botón de tema
    document.getElementById('toggleTheme').addEventListener('click', function () {
        document.body.classList.toggle('dark-theme');
        document.body.classList.toggle('light-theme');

        // Guardar preferencia en localStorage
        const isDarkTheme = document.body.classList.contains('dark-theme');
        localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');

        // Actualizar icono
        const themeIcon = this.querySelector('i');
        if (themeIcon) {
            themeIcon.className = isDarkTheme ? 'fas fa-sun' : 'fas fa-moon';
        }
    });

    // Inicializar herramientas de dibujo
    initDrawingTools();

    function initDrawingTools() {
        const drawingTools = document.querySelectorAll('.drawing-tool');

        drawingTools.forEach(tool => {
            tool.addEventListener('click', function () {
                const toolType = this.getAttribute('data-tool');

                // Desactivar todas las herramientas
                drawingTools.forEach(t => t.classList.remove('active'));

                // Si se hace clic en la herramienta activa, desactivarla
                if (drawingMode === toolType) {
                    drawingMode = null;
                    return;
                }

                // Activar la herramienta seleccionada
                this.classList.add('active');
                drawingMode = toolType;

                // Configurar el modo de dibujo
                switch (toolType) {
                    case 'line':
                        enableLineDrawing();
                        break;
                    case 'horizontal':
                        enableHorizontalLineDrawing();
                        break;
                    case 'vertical':
                        enableVerticalLineDrawing();
                        break;
                    case 'rectangle':
                        enableRectangleDrawing();
                        break;
                    case 'fibonacci':
                        enableFibonacciDrawing();
                        break;
                    case 'text':
                        enableTextDrawing();
                        break;
                    case 'erase':
                        enableErasing();
                        break;
                }
            });
        });
    }

    // Funciones para habilitar diferentes modos de dibujo
    function enableLineDrawing() {
        // Implementar dibujo de líneas
        console.log('Modo de dibujo: Línea');
    }

    function enableHorizontalLineDrawing() {
        // Implementar dibujo de líneas horizontales
        console.log('Modo de dibujo: Línea horizontal');
    }

    function enableVerticalLineDrawing() {
        // Implementar dibujo de líneas verticales
        console.log('Modo de dibujo: Línea vertical');
    }

    function enableRectangleDrawing() {
        // Implementar dibujo de rectángulos
        console.log('Modo de dibujo: Rectángulo');
    }

    function enableFibonacciDrawing() {
        // Implementar dibujo de retrocesos de Fibonacci
        console.log('Modo de dibujo: Fibonacci');
    }

    function enableTextDrawing() {
        // Implementar añadir texto
        console.log('Modo de dibujo: Texto');
    }

    function enableErasing() {
        // Implementar borrado de objetos de dibujo
        console.log('Modo de dibujo: Borrador');
    }

    // Cargar datos iniciales
    loadMarketData();

    // Evento de redimensionamiento de ventana
    window.addEventListener('resize', function () {
        if (chart) {
            chart.applyOptions({
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight
            });
        }
    });
});
