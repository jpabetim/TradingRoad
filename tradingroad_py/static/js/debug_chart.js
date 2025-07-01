// Script de debugging para verificar por qué no se muestra el gráfico

console.log('Debug script loaded');

// Verificar que ECharts está disponible
console.log('ECharts available:', typeof echarts !== 'undefined');

// Verificar que el elemento del gráfico existe
console.log('Chart container element:', document.getElementById('tradingChart'));

// Verificar que SocketIO está disponible
console.log('SocketIO available:', typeof io !== 'undefined');

// Función simple para crear un gráfico de prueba
function createTestChart() {
    console.log('Creating test chart...');
    
    const chartContainer = document.getElementById('tradingChart');
    if (!chartContainer) {
        console.error('Chart container not found!');
        return;
    }
    
    console.log('Chart container found:', chartContainer);
    console.log('Chart container dimensions:', chartContainer.offsetWidth, 'x', chartContainer.offsetHeight);
    
    // Crear gráfico básico
    const chart = echarts.init(chartContainer);
    console.log('Chart initialized:', chart);
    
    // Datos de prueba
    const testData = [
        [100, 110, 95, 105],
        [105, 115, 100, 112],
        [112, 120, 108, 118],
        [118, 125, 115, 122],
        [122, 130, 120, 125]
    ];
    
    const option = {
        backgroundColor: '#0C111C',
        grid: {
            left: '10%',
            right: '8%',
            top: '8%',
            bottom: '15%'
        },
        xAxis: {
            type: 'category',
            data: ['1', '2', '3', '4', '5'],
            axisLine: { lineStyle: { color: '#2A2E39' } },
            axisLabel: { color: '#D1D4DC' }
        },
        yAxis: {
            type: 'value',
            scale: true,
            axisLine: { show: false },
            axisLabel: { color: '#D1D4DC' },
            splitLine: { lineStyle: { color: '#2A2E39' } }
        },
        series: [{
            name: 'Test Candlestick',
            type: 'candlestick',
            data: testData,
            itemStyle: {
                color: '#089981',
                color0: '#F23645',
                borderColor: '#089981',
                borderColor0: '#F23645'
            }
        }]
    };
    
    chart.setOption(option);
    console.log('Test chart option set');
    
    // Forzar redimensionado
    setTimeout(() => {
        chart.resize();
        console.log('Chart resized');
    }, 100);
}

// Probar obtener datos de la API
async function testAPI() {
    console.log('Testing API...');
    
    try {
        const response = await fetch('/api/exchange/klines/binance?symbol=BTCUSDT&interval=4h&limit=10');
        console.log('API response status:', response.status);
        
        const data = await response.json();
        console.log('API data:', data);
        console.log('Number of candles:', data.data ? data.data.length : 0);
        
        if (data.data && data.data.length > 0) {
            console.log('First candle:', data.data[0]);
        }
        
    } catch (error) {
        console.error('API test failed:', error);
    }
}

// Ejecutar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, running debug tests...');
    
    setTimeout(() => {
        createTestChart();
        testAPI();
    }, 1000);
});

// Exportar funciones para uso manual
window.debugChart = {
    createTestChart,
    testAPI
};
