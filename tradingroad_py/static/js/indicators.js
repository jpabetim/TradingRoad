/**
 * Módulo para gestión de indicadores técnicos en TradingRoad
 */

// Objeto para almacenar todos los indicadores activos
let indicators = {};

// Lista para mantener todas las medias móviles activas
let movingAverages = [];

// Configuración predeterminada para medias móviles
const defaultMAs = [
    { type: 'ema', period: 12, color: '#00FF00', width: 2, opacity: 0.8 }, // EMA 12 en verde
    { type: 'ema', period: 20, color: '#FFFFFF', width: 2, opacity: 0.8 }, // EMA 20 en blanco
    { type: 'sma', period: 50, color: '#FF69B4', width: 2, opacity: 0.8 }, // SMA 50 en fucsia claro
    { type: 'sma', period: 200, color: '#FF0000', width: 2, opacity: 0.8 } // SMA 200 en rojo
];

/**
 * Añade o actualiza un indicador en el gráfico
 * @param {string} type - Tipo de indicador ('movingAverage', 'bollinger', 'rsi', 'macd', 'volume')
 * @param {Object} params - Parámetros específicos para el indicador
 * @param {Object} chart - Referencia al objeto chart de LightweightCharts
 * @param {Array} marketData - Datos de mercado en formato OHLCV
 */
function addIndicator(type, params = {}, chart, marketData) {
    switch (type) {
        case 'movingAverage':
            addMovingAverage(params, chart, marketData);
            break;

        case 'bollinger':
            addBollingerBands(params, chart, marketData);
            break;

        case 'rsi':
            addRSI(params, chart, marketData);
            break;

        case 'macd':
            addMACD(params, chart, marketData);
            break;

        case 'volume':
            addVolume(params, chart, marketData);
            break;
    }

    updateActiveIndicators();
}

/**
 * Añade o actualiza una media móvil en el gráfico
 */
function addMovingAverage(params, chart, marketData) {
    const maType = params.type || 'ema';
    const period = params.period || 20;
    const color = params.color || '#FFFFFF';
    const width = params.width || 2;
    const opacity = params.opacity || 0.8;
    const id = `${maType}-${period}`;

    // Verificar si ya existe esta media móvil
    const existingMA = movingAverages.find(ma => ma.id === id);
    if (existingMA) {
        chart.removeSeries(existingMA.series);
        movingAverages = movingAverages.filter(ma => ma.id !== id);
    }

    // Crear la nueva serie
    const maColor = color + Math.round(opacity * 255).toString(16).padStart(2, '0');
    const maSeries = chart.addLineSeries({
        color: maColor,
        lineWidth: width,
        priceScaleId: 'right',
        title: `${maType.toUpperCase()} ${period}`
    });

    // Calcular datos según el tipo
    let maData;
    if (maType === 'ema') {
        maData = calculateEMA(marketData, period);
    } else {
        maData = calculateSMA(marketData, period);
    }

    maSeries.setData(maData);

    // Guardar la media móvil
    movingAverages.push({
        id,
        type: maType,
        period,
        color,
        width,
        opacity,
        series: maSeries
    });

    // Actualizar la lista de medias activas en la UI
    updateActiveMAsList();
}

/**
 * Añade o actualiza las bandas de Bollinger en el gráfico
 */
function addBollingerBands(params, chart, marketData) {
    const bollingerPeriod = params.period || 20;
    const bollingerDevs = params.deviations || 2;
    const opacity = params.opacity || 0.7;

    // Eliminar bandas existentes
    if (indicators.bollingerUpper) {
        chart.removeSeries(indicators.bollingerUpper);
    }
    if (indicators.bollingerMiddle) {
        chart.removeSeries(indicators.bollingerMiddle);
    }
    if (indicators.bollingerLower) {
        chart.removeSeries(indicators.bollingerLower);
    }

    // Crear nuevas bandas con la opacidad definida
    const upperColor = `rgba(76, 175, 80, ${opacity})`;
    const middleColor = `rgba(255, 255, 255, ${opacity})`;
    const lowerColor = `rgba(76, 175, 80, ${opacity})`;

    indicators.bollingerUpper = chart.addLineSeries({
        color: upperColor,
        lineWidth: 1,
        priceScaleId: 'right',
    });

    indicators.bollingerMiddle = chart.addLineSeries({
        color: middleColor,
        lineWidth: 1,
        priceScaleId: 'right',
    });

    indicators.bollingerLower = chart.addLineSeries({
        color: lowerColor,
        lineWidth: 1,
        priceScaleId: 'right',
    });

    // Calcular bandas de Bollinger
    const bollinger = calculateBollinger(marketData, bollingerPeriod, bollingerDevs);
    indicators.bollingerUpper.setData(bollinger.upper);
    indicators.bollingerMiddle.setData(bollinger.middle);
    indicators.bollingerLower.setData(bollinger.lower);
}

/**
 * Añade o actualiza el indicador RSI en el gráfico
 */
function addRSI(params, chart, marketData) {
    const rsiPeriod = params.period || 14;
    const overbought = params.overbought || 80;
    const oversold = params.oversold || 20;
    const showOverbought = params.showOverbought !== undefined ? params.showOverbought : true;
    const showOversold = params.showOversold !== undefined ? params.showOversold : true;
    const showMidline = params.showMidline !== undefined ? params.showMidline : true;
    const rsiColor = params.color || '#9C27B0';
    const rsiWidth = params.width || 2;

    // Crear un nuevo pane para el RSI si no existe
    if (!indicators.rsiPane) {
        indicators.rsiPane = chart.addPane({
            height: 120,
        });
    }

    // Eliminar series existentes si las hay
    if (indicators.rsi) {
        indicators.rsiPane.removeSeries(indicators.rsi);
    }
    if (indicators.rsiOverbought) {
        indicators.rsiPane.removeSeries(indicators.rsiOverbought);
    }
    if (indicators.rsiOversold) {
        indicators.rsiPane.removeSeries(indicators.rsiOversold);
    }
    if (indicators.rsiMidline) {
        indicators.rsiPane.removeSeries(indicators.rsiMidline);
    }

    // Crear la serie RSI principal
    indicators.rsi = indicators.rsiPane.addLineSeries({
        color: rsiColor,
        lineWidth: rsiWidth,
        priceFormat: {
            type: 'custom',
            minMove: 0.01,
            formatter: value => value.toFixed(2),
        },
        title: 'RSI (' + rsiPeriod + ')',
    });

    // Calcular y mostrar el RSI
    const rsiData = calculateRSI(marketData, rsiPeriod);
    indicators.rsi.setData(rsiData);

    // Añadir línea de sobrecompra (nivel 80) con línea continua gris
    if (showOverbought) {
        indicators.rsiOverbought = indicators.rsiPane.addLineSeries({
            color: 'rgba(128, 128, 128, 0.8)',
            lineWidth: 1,
            lineStyle: 0, // Línea continua
            lastValueVisible: false,
        });

        indicators.rsiOverbought.setData([
            { time: marketData[0].time, value: overbought },
            { time: marketData[marketData.length - 1].time, value: overbought }
        ]);
    }

    // Añadir línea de sobreventa (nivel 20) con línea continua gris
    if (showOversold) {
        indicators.rsiOversold = indicators.rsiPane.addLineSeries({
            color: 'rgba(128, 128, 128, 0.8)',
            lineWidth: 1,
            lineStyle: 0, // Línea continua
            lastValueVisible: false,
        });

        indicators.rsiOversold.setData([
            { time: marketData[0].time, value: oversold },
            { time: marketData[marketData.length - 1].time, value: oversold }
        ]);
    }

    // Añadir línea del nivel 50 con línea discontinua gris claro
    if (showMidline) {
        indicators.rsiMidline = indicators.rsiPane.addLineSeries({
            color: 'rgba(169, 169, 169, 0.5)',
            lineWidth: 1,
            lineStyle: 1, // Línea discontinua
            lastValueVisible: false,
        });

        indicators.rsiMidline.setData([
            { time: marketData[0].time, value: 50 },
            { time: marketData[marketData.length - 1].time, value: 50 }
        ]);
    }
}

/**
 * Añade o actualiza el indicador MACD en el gráfico
 */
function addMACD(params, chart, marketData) {
    const fastPeriod = params.fast || 12;
    const slowPeriod = params.slow || 26;
    const signalPeriod = params.signal || 9;
    const opacity = params.opacity || 0.8;

    // Crear un nuevo pane para el MACD
    if (!indicators.macdPane) {
        indicators.macdPane = chart.addPane({
            height: 120,
        });
    }

    if (indicators.macdLine) {
        indicators.macdPane.removeSeries(indicators.macdLine);
    }

    if (indicators.signalLine) {
        indicators.macdPane.removeSeries(indicators.signalLine);
    }

    if (indicators.histogramSeries) {
        indicators.macdPane.removeSeries(indicators.histogramSeries);
    }

    // Crear las series del MACD con la opacidad definida
    indicators.macdLine = indicators.macdPane.addLineSeries({
        color: `rgba(33, 150, 243, ${opacity})`,
        lineWidth: 2,
        title: 'MACD',
    });

    indicators.signalLine = indicators.macdPane.addLineSeries({
        color: `rgba(255, 152, 0, ${opacity})`,
        lineWidth: 1,
        title: 'Señal',
    });

    indicators.histogramSeries = indicators.macdPane.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'custom',
            minMove: 0.01,
            formatter: value => value.toFixed(2),
        },
        title: 'Histograma',
    });

    // Calcular MACD
    const macd = calculateMACD(marketData, fastPeriod, slowPeriod, signalPeriod);

    indicators.macdLine.setData(macd.macdLine);
    indicators.signalLine.setData(macd.signalLine);
    indicators.histogramSeries.setData(macd.histogram);
}

/**
 * Añade o actualiza el indicador de volumen en el gráfico
 */
function addVolume(params, chart, marketData) {
    const showMA = params.showMA !== undefined ? params.showMA : true;
    const topPosition = params.topPosition !== undefined ? params.topPosition : false;
    const volumeOpacity = params.opacity || 0.7;
    const maPeriod = params.maPeriod || 55;
    const maColor = params.maColor || '#FF9800';

    // Eliminar el panel de volumen existente si hay uno
    if (indicators.volumePane) {
        chart.removePaneByName('volumePane');
        indicators.volumePane = null;
        indicators.volume = null;
        indicators.volumeMA = null;
    }

    // Crear un nuevo panel para el volumen
    let paneOptions = {
        name: 'volumePane',
        height: 100
    };

    // Si se debe mostrar en la parte superior, cambiar el orden
    if (topPosition) {
        // Primero creamos el pane de volumen
        indicators.volumePane = chart.addPane(paneOptions);

        // Luego movemos todos los otros panes
        if (indicators.rsiPane) {
            chart.movePaneToPosition(indicators.rsiPane, 1);
        }
        if (indicators.macdPane) {
            chart.movePaneToPosition(indicators.macdPane, indicators.rsiPane ? 2 : 1);
        }
    } else {
        // Añadirlo normalmente al final
        indicators.volumePane = chart.addPane(paneOptions);
    }

    // Añadir serie de volumen
    indicators.volume = indicators.volumePane.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        title: 'Volumen',
        opacity: volumeOpacity
    });

    // Preparar datos de volumen
    const volumeData = marketData.map(d => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? '#26a69a' : '#ef5350',
    }));

    indicators.volume.setData(volumeData);

    // Añadir media móvil del volumen si está activada
    if (showMA) {
        // Calcular MA del volumen
        const volumeValues = marketData.map(d => d.volume);
        const volumeMA = calculateSimpleMA(volumeValues, maPeriod);

        // Crear la serie para la media móvil
        indicators.volumeMA = indicators.volumePane.addLineSeries({
            color: maColor,
            lineWidth: 2,
            title: `Vol MA (${maPeriod})`,
            priceScaleId: 'right'
        });

        // Preparar datos para la MA
        const volumeMAData = volumeMA.map((value, index) => ({
            time: marketData[index + maPeriod - 1].time,
            value: value
        }));

        indicators.volumeMA.setData(volumeMAData);
    }
}

/**
 * Actualiza la lista de indicadores activos en la UI
 */
function updateActiveIndicators() {
    const activeIndicators = Object.keys(indicators).filter(key =>
        !['rsiPane', 'macdPane', 'volumePane', 'rsiOverbought', 'rsiOversold', 'rsiMidline', 'histogramSeries', 'volumeMA'].includes(key)
    );

    // Añadir las medias móviles activas
    const maNames = movingAverages.map(ma => `${ma.type.toUpperCase()}${ma.period}`);

    let allIndicators = [...activeIndicators, ...maNames];

    let indicatorText = allIndicators.length > 0
        ? 'Indicadores: ' + allIndicators.join(', ')
        : 'Indicadores: Ninguno';

    document.getElementById('chart-active-indicators').textContent = indicatorText;
}

/**
 * Actualiza la lista de medias móviles activas en la UI
 */
function updateActiveMAsList() {
    const container = document.getElementById('active-averages');
    container.innerHTML = '';

    // Si hay medias móviles activas, mostrarlas
    if (movingAverages.length > 0) {
        const list = document.createElement('div');
        list.className = 'ma-list';

        movingAverages.forEach(ma => {
            const item = document.createElement('div');
            item.className = 'ma-item';
            item.innerHTML = `
                <span class="ma-color" style="background-color: ${ma.color}"></span>
                <span class="ma-info">${ma.type.toUpperCase()} ${ma.period}</span>
                <button class="ma-remove" data-id="${ma.id}">×</button>
            `;
            list.appendChild(item);
        });

        container.appendChild(list);

        // Añadir event listeners para los botones de eliminar
        document.querySelectorAll('.ma-remove').forEach(btn => {
            btn.addEventListener('click', function () {
                const maId = this.getAttribute('data-id');
                const ma = movingAverages.find(ma => ma.id === maId);
                if (ma && window.chart) {
                    window.chart.removeSeries(ma.series);
                    movingAverages = movingAverages.filter(m => m.id !== maId);
                    updateActiveMAsList();
                    updateActiveIndicators();
                }
            });
        });
    } else {
        container.innerHTML = '<p>No hay medias móviles activas</p>';
    }
}

/**
 * Inicializa las medias móviles predeterminadas
 */
function initDefaultMAs(chart, marketData) {
    // Limpiar medias existentes
    movingAverages.forEach(ma => chart.removeSeries(ma.series));
    movingAverages = [];

    // Añadir medias predeterminadas
    defaultMAs.forEach(ma => {
        addMovingAverage(ma, chart, marketData);
    });

    updateActiveIndicators();
}

/**
 * Calcula una media móvil simple
 */
function calculateSimpleMA(values, period) {
    const result = [];
    for (let i = period - 1; i < values.length; i++) {
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += values[i - j];
        }
        result.push(sum / period);
    }
    return result;
}

/**
 * Calcula una media móvil simple para datos OHLCV
 */
function calculateSMA(data, period) {
    const result = [];
    for (let i = period - 1; i < data.length; i++) {
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += data[i - j].close;
        }
        result.push({
            time: data[i].time,
            value: sum / period
        });
    }
    return result;
}

/**
 * Calcula una media móvil exponencial para datos OHLCV
 */
function calculateEMA(data, period) {
    const result = [];
    const k = 2 / (period + 1);

    // Primera EMA es un SMA
    let ema = data.slice(0, period).reduce((sum, d) => sum + d.close, 0) / period;

    for (let i = period - 1; i < data.length; i++) {
        if (i === period - 1) {
            result.push({
                time: data[i].time,
                value: ema
            });
            continue;
        }

        ema = data[i].close * k + ema * (1 - k);
        result.push({
            time: data[i].time,
            value: ema
        });
    }
    return result;
}

/**
 * Calcula las bandas de Bollinger
 */
function calculateBollinger(data, period, deviations) {
    const upper = [];
    const middle = [];
    const lower = [];

    // Calcular SMA (middle band)
    const smaData = calculateSMA(data, period);

    for (let i = 0; i < smaData.length; i++) {
        const index = i + period - 1;
        const sma = smaData[i].value;

        // Calcular desviación estándar
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += Math.pow(data[index - j].close - sma, 2);
        }
        const stdDev = Math.sqrt(sum / period);

        middle.push({
            time: data[index].time,
            value: sma
        });

        upper.push({
            time: data[index].time,
            value: sma + stdDev * deviations
        });

        lower.push({
            time: data[index].time,
            value: sma - stdDev * deviations
        });
    }

    return { upper, middle, lower };
}

/**
 * Calcula el indicador RSI
 */
function calculateRSI(data, period) {
    const result = [];
    let avgGain = 0;
    let avgLoss = 0;

    // Calcular ganancia/pérdida inicial
    for (let i = 1; i <= period; i++) {
        const change = data[i].close - data[i - 1].close;
        if (change >= 0) {
            avgGain += change;
        } else {
            avgLoss += Math.abs(change);
        }
    }

    avgGain /= period;
    avgLoss /= period;

    for (let i = period + 1; i < data.length; i++) {
        // Calcular RS
        let rs = avgGain / avgLoss;
        if (avgLoss === 0) rs = 100;

        // Calcular RSI
        const rsi = 100 - (100 / (1 + rs));

        result.push({
            time: data[i - 1].time,
            value: rsi
        });

        // Actualizar avgGain y avgLoss
        const change = data[i].close - data[i - 1].close;
        avgGain = ((avgGain * (period - 1)) + (change >= 0 ? change : 0)) / period;
        avgLoss = ((avgLoss * (period - 1)) + (change < 0 ? Math.abs(change) : 0)) / period;
    }

    return result;
}

/**
 * Calcula el indicador MACD
 */
function calculateMACD(data, fastPeriod, slowPeriod, signalPeriod) {
    const macdLine = [];
    const signalLine = [];
    const histogram = [];

    // Calcular EMAs
    const fastEma = calculateEMA(data, fastPeriod);
    const slowEma = calculateEMA(data, slowPeriod);

    // Alinear EMAs (comenzar desde el punto donde ambos estén disponibles)
    const startIndex = Math.max(fastPeriod, slowPeriod) - 1;

    // Calcular línea MACD (diferencia entre EMAs)
    const macdData = [];
    for (let i = 0; i < slowEma.length; i++) {
        const fastIndex = i + (slowEma.length - fastEma.length);
        if (fastIndex >= 0) {
            macdData.push({
                time: slowEma[i].time,
                value: fastEma[fastIndex].value - slowEma[i].value
            });
        }
    }

    // Calcular línea de señal (EMA del MACD)
    let signalEma = 0;
    for (let i = 0; i < macdData.length; i++) {
        if (i < signalPeriod) {
            signalEma += macdData[i].value;
            if (i === signalPeriod - 1) {
                signalEma /= signalPeriod;
                signalLine.push({
                    time: macdData[i].time,
                    value: signalEma
                });
            }
            continue;
        }

        signalEma = macdData[i].value * (2 / (signalPeriod + 1)) + signalEma * (1 - 2 / (signalPeriod + 1));
        signalLine.push({
            time: macdData[i].time,
            value: signalEma
        });

        // Añadir línea MACD y histograma una vez que tenemos la señal
        if (i >= signalPeriod - 1) {
            macdLine.push({
                time: macdData[i].time,
                value: macdData[i].value
            });

            histogram.push({
                time: macdData[i].time,
                value: macdData[i].value - signalEma,
                color: macdData[i].value >= signalEma ? '#26a69a' : '#ef5350'
            });
        }
    }

    return { macdLine, signalLine, histogram };
}

// Exponer funciones globalmente
window.indicatorsModule = {
    indicators,
    movingAverages,
    defaultMAs,
    addIndicator,
    updateActiveIndicators,
    updateActiveMAsList,
    initDefaultMAs,
    addMovingAverage,
    addRSI,
    addVolume,
    addMACD,
    addBollingerBands
};
