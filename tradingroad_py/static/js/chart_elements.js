/**
 * Script para manejar elementos adicionales del gráfico:
 * - Contador de tiempo restante de vela
 * - Marcadores de máximo/mínimo
 * - Control de zona horaria
 * - Mejoras para búsqueda de símbolos
 */

// Función para formatear el tiempo
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

// Función para actualizar el contador de tiempo restante
function updateCandleCountdown(timeframe) {
    const countdownElem = document.getElementById('chart-countdown');
    if (!countdownElem) return;

    // Obtener la duración de la vela en segundos
    let durationInSeconds;

    switch (timeframe) {
        case '30s': durationInSeconds = 30; break;
        case '1m': durationInSeconds = 60; break;
        case '5m': durationInSeconds = 5 * 60; break;
        case '15m': durationInSeconds = 15 * 60; break;
        case '30m': durationInSeconds = 30 * 60; break;
        case '1h': durationInSeconds = 60 * 60; break;
        case '4h': durationInSeconds = 4 * 60 * 60; break;
        case '1d': durationInSeconds = 24 * 60 * 60; break;
        default: durationInSeconds = 60; // Default to 1 minute
    }

    // Calcular tiempo restante hasta que termine la vela actual
    const now = new Date();
    let secondsPassed = 0;

    switch (timeframe) {
        case '30s':
            secondsPassed = now.getSeconds() % 30;
            break;
        case '1m':
            secondsPassed = now.getSeconds();
            break;
        case '5m':
            secondsPassed = (now.getMinutes() % 5) * 60 + now.getSeconds();
            break;
        case '15m':
            secondsPassed = (now.getMinutes() % 15) * 60 + now.getSeconds();
            break;
        case '30m':
            secondsPassed = (now.getMinutes() % 30) * 60 + now.getSeconds();
            break;
        case '1h':
            secondsPassed = now.getMinutes() * 60 + now.getSeconds();
            break;
        case '4h':
            secondsPassed = (now.getHours() % 4) * 3600 + now.getMinutes() * 60 + now.getSeconds();
            break;
        case '1d':
            secondsPassed = now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
            break;
        default:
            secondsPassed = now.getSeconds();
    }

    const secondsRemaining = durationInSeconds - secondsPassed;
    countdownElem.textContent = formatTime(secondsRemaining);

    // Cambiar color cuando quede poco tiempo
    if (secondsRemaining < 10) {
        countdownElem.style.color = '#F44336'; // Rojo
    } else if (secondsRemaining < 30) {
        countdownElem.style.color = '#FFC107'; // Amarillo
    } else {
        countdownElem.style.color = '#f8f8f8'; // Blanco
    }
}

// Función para actualizar los marcadores de máximo y mínimo
function updatePriceMarkers(data) {
    if (!data || data.length === 0) return;

    const maxPriceElem = document.getElementById('chart-max-price');
    const minPriceElem = document.getElementById('chart-min-price');

    if (!maxPriceElem || !minPriceElem) return;

    // Encontrar precios máximos y mínimos en los datos visibles
    let maxPrice = data[0].high;
    let minPrice = data[0].low;

    data.forEach(candle => {
        if (candle.high > maxPrice) maxPrice = candle.high;
        if (candle.low < minPrice) minPrice = candle.low;
    });

    // Formatear precios según el número de decimales apropiado
    const decimals = getAppropriateDecimals(maxPrice);
    maxPriceElem.textContent = `Máximo: ${maxPrice.toFixed(decimals)}`;
    minPriceElem.textContent = `Mínimo: ${minPrice.toFixed(decimals)}`;
}

// Determinar el número de decimales apropiado para mostrar
function getAppropriateDecimals(price) {
    if (price >= 10000) return 0;
    if (price >= 1000) return 1;
    if (price >= 100) return 2;
    if (price >= 10) return 3;
    if (price >= 1) return 4;
    if (price >= 0.1) return 5;
    return 6;
}

// Función para manejar el cambio de zona horaria
function handleTimezoneChange(chart, timezone) {
    if (!chart) return;

    // Extraer el offset de la zona horaria
    let utcOffset = 0;
    if (timezone.startsWith('UTC+')) {
        utcOffset = parseInt(timezone.substring(4));
    } else if (timezone.startsWith('UTC-')) {
        utcOffset = -parseInt(timezone.substring(4));
    }

    // Aplicar la zona horaria al gráfico
    chart.applyOptions({
        localization: {
            timeFormatter: (timestamp) => {
                const date = new Date((timestamp) * 1000);
                date.setHours(date.getHours() + utcOffset);
                return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            }
        }
    });
}

// Función para formatear el símbolo según el exchange
function formatSymbolForExchange(symbol, exchange) {
    // Asegurarse de que el símbolo esté en mayúsculas
    symbol = symbol.toUpperCase();

    // Separar la base y la cotización
    let [base, quote] = symbol.includes('/') ? symbol.split('/') :
        symbol.includes('-') ? symbol.split('-') :
            [symbol.substring(0, 3), symbol.substring(3)];

    // Si no hay una separación clara, intentar detectar pares comunes
    if (!quote) {
        const commonQuotes = ['USDT', 'USD', 'BTC', 'ETH', 'BNB', 'USDC'];
        for (const q of commonQuotes) {
            if (base.endsWith(q)) {
                quote = q;
                base = base.substring(0, base.length - q.length);
                break;
            }
        }
    }

    // Formatear según el exchange
    switch (exchange.toLowerCase()) {
        case 'bingx':
            return `${base}-${quote}`;
        case 'binance':
        case 'kucoin':
        case 'bybit':
        case 'coinbase':
        case 'kraken':
            return `${base}/${quote}`;
        default:
            return `${base}/${quote}`;
    }
}

// Mejorar la búsqueda de símbolos para convertir a mayúsculas y manejar el botón desplegable
function enhanceSymbolSearch() {
    const symbolSearch = document.getElementById('symbolSearch');
    const symbolResults = document.getElementById('symbolResults');
    const symbolDropdownBtn = document.getElementById('symbolDropdownBtn');
    if (!symbolSearch) return;

    // Evento para convertir a mayúsculas
    symbolSearch.addEventListener('input', function (e) {
        this.value = this.value.toUpperCase();

        // Mostrar resultados si hay texto
        if (this.value.length > 0) {
            // Esta función debería estar definida en el script principal
            if (window.loadSymbolSearchResults) {
                window.loadSymbolSearchResults(this.value);
            }
        }
    });

    // Configurar botón desplegable
    if (symbolDropdownBtn) {
        symbolDropdownBtn.addEventListener('click', function () {
            if (symbolResults.style.display === 'block') {
                symbolResults.style.display = 'none';
            } else {
                // Esta función debería estar definida en el script principal
                if (window.loadSymbols && window.currentSource) {
                    window.loadSymbols(window.currentSource);
                }
                symbolResults.style.display = 'block';
            }
        });
    }

    // Detectar el cambio de exchange
    const exchangeSelect = document.getElementById('dataSource');
    if (!exchangeSelect) return;

    // Al cambiar el exchange, reformatear el símbolo actual
    exchangeSelect.addEventListener('change', function () {
        if (symbolSearch.value) {
            const formattedSymbol = formatSymbolForExchange(
                symbolSearch.value,
                this.value
            );
            symbolSearch.value = formattedSymbol;
        }
    });
}

// Inicializar todos los elementos mejorados
function initEnhancedChartElements(chart) {
    // Inicializar contador de velas
    const timeframeSelect = document.getElementById('timeframeSelect');
    let currentTimeframe = '1h';

    if (timeframeSelect) {
        currentTimeframe = timeframeSelect.value;
        timeframeSelect.addEventListener('change', function () {
            currentTimeframe = this.value;
            // Reiniciar el contador
            updateCandleCountdown(currentTimeframe);
        });
    }

    // Iniciar contador de velas
    updateCandleCountdown(currentTimeframe);
    setInterval(() => updateCandleCountdown(currentTimeframe), 1000);

    // Inicializar selector de zona horaria
    const timezoneSelect = document.getElementById('timezone-select');
    if (timezoneSelect && chart) {
        timezoneSelect.addEventListener('change', function () {
            handleTimezoneChange(chart, this.value);
        });

        // Aplicar zona horaria inicial
        handleTimezoneChange(chart, timezoneSelect.value);
    }

    // Mejorar búsqueda de símbolos
    enhanceSymbolSearch();

    // Exponer funciones para su uso en otras partes del código
    window.chartElements = {
        updatePriceMarkers,
        updateCandleCountdown,
        handleTimezoneChange,
        formatSymbolForExchange
    };
}

// Inicializar cuando el DOM esté cargado
document.addEventListener('DOMContentLoaded', function () {
    // La inicialización completa se hará cuando el gráfico esté disponible
    setTimeout(() => {
        if (window.chart) {
            initEnhancedChartElements(window.chart);

            // Iniciar actualizaciones periódicas
            setInterval(() => {
                const timeframeSelect = document.getElementById('timeframeSelect');
                const currentTimeframe = timeframeSelect ? timeframeSelect.value : '1h';
                updateCandleCountdown(currentTimeframe);

                if (window.marketData && window.marketData.length) {
                    updatePriceMarkers(window.marketData);
                }
            }, 1000);
        }
    }, 1500);
});
