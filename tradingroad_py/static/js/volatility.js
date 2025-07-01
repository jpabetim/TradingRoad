// Funciones para el módulo de volatilidad

document.addEventListener('DOMContentLoaded', function () {
    // Elementos del DOM
    const assetSelect = document.getElementById('asset-select');
    const periodSelect = document.getElementById('period-select');

    // Inicializar gráficos vacíos
    initializeVolatilityChart();
    initializeCorrelationMatrix();

    // Escuchar cambios en los selectores
    if (assetSelect && periodSelect) {
        assetSelect.addEventListener('change', updateCharts);
        periodSelect.addEventListener('change', updateCharts);
    }

    // Función para actualizar los gráficos
    function updateCharts() {
        const selectedAssets = Array.from(assetSelect.selectedOptions).map(option => option.value);
        const selectedPeriod = periodSelect.value;

        if (selectedAssets.length > 0) {
            // Aquí se implementaría la llamada a la API para obtener datos reales
            // Por ahora, mostramos datos de ejemplo
            updateVolatilityChart(selectedAssets, selectedPeriod);
            updateCorrelationMatrix(selectedAssets, selectedPeriod);
        }
    }

    // Inicializar con datos de ejemplo
    function initializeVolatilityChart() {
        const volatilityContainer = document.getElementById('volatility-chart-container');
        if (volatilityContainer) {
            volatilityContainer.innerHTML = '<div class="placeholder">Seleccione activos para visualizar la volatilidad</div>';
        }
    }

    function initializeCorrelationMatrix() {
        const correlationContainer = document.getElementById('correlation-matrix-container');
        if (correlationContainer) {
            correlationContainer.innerHTML = '<div class="placeholder">Seleccione activos para visualizar la correlación</div>';
        }
    }

    // Actualizar con datos de ejemplo
    function updateVolatilityChart(assets, period) {
        const volatilityContainer = document.getElementById('volatility-chart-container');
        if (volatilityContainer) {
            // Datos de ejemplo
            const data = {
                'SPY': 0.15,
                'QQQ': 0.20,
                'EURUSD': 0.08,
                'BTCUSD': 0.65,
                'GLD': 0.12,
                'USO': 0.30
            };

            // Crear una tabla simple como ejemplo
            let html = '<table class="volatility-table">';
            html += '<tr><th>Activo</th><th>Volatilidad Anualizada</th></tr>';

            assets.forEach(asset => {
                html += `<tr><td>${asset}</td><td>${(data[asset] * 100).toFixed(2)}%</td></tr>`;
            });

            html += '</table>';
            volatilityContainer.innerHTML = html;
        }
    }

    function updateCorrelationMatrix(assets, period) {
        const correlationContainer = document.getElementById('correlation-matrix-container');
        if (correlationContainer) {
            // Datos de ejemplo de correlación
            const correlationData = {
                'SPY': { 'SPY': 1.0, 'QQQ': 0.8, 'EURUSD': 0.2, 'BTCUSD': 0.3, 'GLD': -0.2, 'USO': 0.5 },
                'QQQ': { 'SPY': 0.8, 'QQQ': 1.0, 'EURUSD': 0.1, 'BTCUSD': 0.4, 'GLD': -0.3, 'USO': 0.4 },
                'EURUSD': { 'SPY': 0.2, 'QQQ': 0.1, 'EURUSD': 1.0, 'BTCUSD': 0.2, 'GLD': 0.3, 'USO': 0.1 },
                'BTCUSD': { 'SPY': 0.3, 'QQQ': 0.4, 'EURUSD': 0.2, 'BTCUSD': 1.0, 'GLD': 0.1, 'USO': 0.2 },
                'GLD': { 'SPY': -0.2, 'QQQ': -0.3, 'EURUSD': 0.3, 'BTCUSD': 0.1, 'GLD': 1.0, 'USO': -0.1 },
                'USO': { 'SPY': 0.5, 'QQQ': 0.4, 'EURUSD': 0.1, 'BTCUSD': 0.2, 'GLD': -0.1, 'USO': 1.0 }
            };

            // Crear una tabla de correlación
            let html = '<table class="correlation-table">';
            html += '<tr><th></th>';

            // Encabezados
            assets.forEach(asset => {
                html += `<th>${asset}</th>`;
            });
            html += '</tr>';

            // Filas de datos
            assets.forEach(asset1 => {
                html += `<tr><td><strong>${asset1}</strong></td>`;

                assets.forEach(asset2 => {
                    const correlation = correlationData[asset1][asset2];
                    const colorClass = correlation > 0.5 ? 'high-positive' :
                        correlation < -0.5 ? 'high-negative' :
                            correlation > 0 ? 'low-positive' : 'low-negative';

                    html += `<td class="${colorClass}">${correlation.toFixed(2)}</td>`;
                });

                html += '</tr>';
            });

            html += '</table>';
            correlationContainer.innerHTML = html;
        }
    }
});
