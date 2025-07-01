// Funciones base para la aplicación TradingRoad

document.addEventListener('DOMContentLoaded', function () {
    // Actualizar indicadores de mercado cada 60 segundos
    if (document.getElementById('sp500')) {
        startMarketDataUpdates();
    }

    // Configurar manejadores de flash messages
    setupFlashMessages();
});

// Función para actualizar datos de mercado
function startMarketDataUpdates() {
    // Actualizar inmediatamente
    updateMarketData();

    // Configurar actualización periódica
    setInterval(updateMarketData, 60000); // Cada 60 segundos
}

// Función para actualizar datos de mercado (simulado)
function updateMarketData() {
    // En una aplicación real, esto haría una petición a una API
    // Para la demo, usamos datos aleatorios simulados

    const markets = {
        sp500: { current: 4500, change: getRandomChange() },
        nasdaq: { current: 14000, change: getRandomChange() },
        dow: { current: 34500, change: getRandomChange() },
        eurusd: { current: 1.10, change: getRandomChange(0.005) },
        btcusd: { current: 30000, change: getRandomChange(2) },
        gold: { current: 2100, change: getRandomChange(0.5) }
    };

    // Actualizar valores en la interfaz
    for (const [id, data] of Object.entries(markets)) {
        const element = document.getElementById(id);
        if (element) {
            const finalValue = data.current * (1 + data.change / 100);

            // Formatear según el tipo de valor
            let formattedValue;
            if (id === 'eurusd') {
                formattedValue = finalValue.toFixed(2);
            } else if (id === 'btcusd' || id === 'gold') {
                formattedValue = finalValue.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
            } else {
                formattedValue = finalValue.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
            }

            element.textContent = formattedValue;

            // Actualizar indicador de cambio
            const changeElement = element.nextElementSibling;
            if (changeElement) {
                changeElement.textContent = (data.change > 0 ? '+' : '') + data.change.toFixed(1) + '%';
                changeElement.className = 'change ' + (data.change >= 0 ? 'positive' : 'negative');
            }
        }
    }
}

// Función para generar un cambio aleatorio para la simulación
function getRandomChange(maxChange = 1.0) {
    return (Math.random() * 2 - 1) * maxChange;
}

// Función para manejar flash messages
function setupFlashMessages() {
    const flashMessages = document.querySelectorAll('.flash-message');

    flashMessages.forEach(message => {
        // Auto-ocultar después de 5 segundos
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => {
                message.style.display = 'none';
            }, 500);
        }, 5000);
    });
}
