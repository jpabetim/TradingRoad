// main.js - VERSIÓN FINAL CORREGIDA Y PROBADA

let strikeChart=null, expirationChart=null, oiSentimentChart=null, lsRatioChart=null;
let fundingRateHistoryChart=null, volatilitySmileChart=null, dvolHistoryChart=null, volumeStrikeChart=null;
let modalChart = null; // Para el gráfico en la modal
let currentCurrency='BTC', currentExpiration='all', currentOrderBookStep=0;
let displayUnit = 'USD'; // 'USD' o 'ASSET'

// ==============================================================================
// FUNCIONES DE SPINNER
// ==============================================================================
function showAllSpinners() {
    document.querySelectorAll('.loading-spinner').forEach(spinner => spinner.classList.remove('hidden'));
}
function hideAllSpinners() {
    document.querySelectorAll('.loading-spinner').forEach(spinner => spinner.classList.add('hidden'));
}

// ==============================================================================
// FUNCIONES DE FETCH
// ==============================================================================
async function fetchDeribitDataForCharts() {
    let url = `http://127.0.0.1:5000/api/data/${currentCurrency}`;
    if (currentExpiration !== 'all') { url += `?expiration=${currentExpiration}`; }
    
    return fetch(url).then(res => res.json()).then(data => {
        drawStrikeChart(data.strike_chart_data, data.metrics);
        drawExpirationChart(data.expiration_chart_data);
        drawVolumeStrikeChart(data.volume_chart_data);
        
        const volatilityNote = document.querySelector('#volatility-smile-chart-container .chart-note');
        if (currentExpiration !== 'all' && data.volatility_smile_data && data.volatility_smile_data.length > 0) {
             if (volatilityNote) volatilityNote.style.display = 'none';
             drawVolatilitySmileChart(data.volatility_smile_data);
        } else {
             if (volatilityNote) volatilityNote.style.display = 'block';
             if(volatilitySmileChart) { volatilitySmileChart.destroy(); volatilitySmileChart = null; }
             const renderArea = document.getElementById('volatility-smile-render-area');
             if (renderArea) renderArea.innerHTML = '';
        }

    }).catch(err => console.error("Error en fetchDeribitDataForCharts:", err));
}

async function populateExpirations() {
    return fetch(`http://127.0.0.1:5000/api/expirations/${currentCurrency}`).then(res => res.json()).then(expirations => {
        const selector = document.getElementById('expiration-selector');
        selector.innerHTML = '<option value="all">Todos los Vencimientos</option>';
        if (expirations && expirations.length > 0) {
            expirations.forEach(date => {
                const option = document.createElement('option');
                option.value = date;
                option.textContent = new Date(date + 'T00:00:00Z').toLocaleDateString('es-ES', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' });
                selector.appendChild(option);
            });
        }
        selector.value = currentExpiration;
    }).catch(err => console.error("Error Vencimientos:", err));
}

async function fetchConsolidatedMetrics() {
    try {
        const response = await fetch(`http://127.0.0.1:5000/api/consolidated-metrics/${currentCurrency}`);
        if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);
        const data = await response.json();
        const oiFormat = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact', minimumFractionDigits: 1, maximumFractionDigits: 1 });
        const rateFormat = new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 4, maximumFractionDigits: 4 });
        const priceFormat = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });
        const oiElement = document.getElementById('metric-oi-total-average');
        if (oiElement && data.oi_total_average !== undefined) {
            oiElement.innerHTML = oiFormat.format(data.oi_total_average);
            if (data.oi_change_4h_percent !== null && data.oi_change_4h_percent !== undefined) {
                const trendIcon = data.oi_change_4h_percent >= 0 ? '▲' : '▼';
                const trendClass = data.oi_change_4h_percent >= 0 ? 'positive' : 'negative';
                oiElement.innerHTML += `<span class="${trendClass}">${trendIcon} ${Math.abs(data.oi_change_4h_percent).toFixed(2)}% (4h)</span>`;
            }
        }
        document.getElementById('metric-funding-rate-average').textContent = rateFormat.format(data.funding_rate_average || 0);
        const nextFrTimeElement = document.getElementById('metric-next-funding-time');
        if (nextFrTimeElement) {
            const timeLeftMs = data.next_funding_time_ms - Date.now();
            if (timeLeftMs > 0) {
                const hours = Math.floor(timeLeftMs / 3600000);
                const minutes = Math.floor((timeLeftMs % 3600000) / 60000);
                nextFrTimeElement.textContent = `${hours}h ${minutes}m`;
            } else { nextFrTimeElement.textContent = data.next_funding_time_ms ? 'Pronto...' : 'N/A'; }
        }
        document.getElementById('metric-current-price').textContent = priceFormat.format(data.current_price || 0);
        const weekHighElement = document.getElementById('metric-week-high');
        if (weekHighElement) weekHighElement.innerHTML = `${priceFormat.format(data.week_high || 0)}<span class="date-label">${data.week_high_date || ''}</span>`;
        const weekLowElement = document.getElementById('metric-week-low');
        if (weekLowElement) weekLowElement.innerHTML = `${priceFormat.format(data.week_low || 0)}<span class="date-label">${data.week_low_date || ''}</span>`;
        document.getElementById('update-time').textContent = new Date().toLocaleTimeString();
    } catch (error) { console.error("Error al obtener métricas consolidadas:", error); }
}

async function fetchBinanceSentimentDataForCharts() {
    return fetch(`http://127.0.0.1:5000/api/sentiment/${currentCurrency}`)
        .then(res => res.json())
        .then(data => renderSentimentCharts(data))
        .catch(err => console.error("Error Sentimiento Binance:", err));
}

async function fetchFundingRateHistory() {
     return fetch(`http://127.0.0.1:5000/api/funding-rate-history/${currentCurrency}`)
        .then(res => res.json())
        .then(data => drawFundingRateHistoryChart(data))
        .catch(err => console.error("Error Historial Funding Rate:", err));
}

async function fetchOrderBook() {
    return fetch(`http://127.0.0.1:5000/api/order-book/${currentCurrency}?step=${currentOrderBookStep}`)
        .then(res => res.json())
        .then(data => updateOrderBookUI(data))
        .catch(err => console.error("Error Libro de Órdenes:", err));
}

async function fetchDvolHistory() {
    return fetch(`http://127.0.0.1:5000/api/dvol-history/${currentCurrency}`)
        .then(res => res.json())
        .then(data => drawDvolHistoryChart(data))
        .catch(err => { console.error("Error Historial DVOL:", err); drawDvolHistoryChart(null); });
}

// ==============================================================================
// FUNCIONES PARA DIBUJAR GRÁFICOS (VISTA PRINCIPAL)
// ==============================================================================
function updateStrikeMetrics(metrics) {
    const safeUpdate = (id, value) => document.getElementById(id) ? document.getElementById(id).textContent = value : null;
    const metricsData = metrics || {};
    const currencyFormat = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact', compactDisplay: 'short', minimumFractionDigits: 1, maximumFractionDigits: 1 });
    const numberFormat = new Intl.NumberFormat('en-US', { notation: 'compact', compactDisplay: 'short', maximumFractionDigits: 1 });
    const priceFormat = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });
    
    // Obtener el precio actual directamente del DOM para los cálculos
    const currentPriceText = document.getElementById('metric-current-price')?.textContent || '$0';
    const currentPrice = parseFloat(currentPriceText.replace(/[^0-9.-]+/g,""));

    if (displayUnit === 'ASSET') {
        safeUpdate('metric-call-oi', `${numberFormat.format(metricsData.call_oi || 0)}`);
        safeUpdate('metric-put-oi', `${numberFormat.format(metricsData.put_oi || 0)}`);
        safeUpdate('metric-total-oi', `${numberFormat.format(metricsData.total_oi || 0)}`);
        safeUpdate('metric-notional-value', `${numberFormat.format(metricsData.notional_value_asset || 0)} ${currentCurrency}`);
    } else { // displayUnit === 'USD'
        // Calcular el valor nocional para cada métrica
        const callOiUsd = (metricsData.call_oi || 0) * currentPrice;
        const putOiUsd = (metricsData.put_oi || 0) * currentPrice;
        const totalOiUsd = (metricsData.total_oi || 0) * currentPrice;

        safeUpdate('metric-call-oi', currencyFormat.format(callOiUsd));
        safeUpdate('metric-put-oi', currencyFormat.format(putOiUsd));
        safeUpdate('metric-total-oi', currencyFormat.format(totalOiUsd));
        safeUpdate('metric-notional-value', currencyFormat.format(metricsData.notional_value_usd || 0));
    }

    safeUpdate('metric-pc-ratio', (metricsData.pc_ratio || 0).toFixed(2));
    safeUpdate('metric-pc-ratio-vol', (metricsData.pc_ratio_volume || 0).toFixed(2));
    safeUpdate('max-pain-display', priceFormat.format(metricsData.max_pain || 0));
}

function drawStrikeChart(chartData, metricsData) {
    updateStrikeMetrics(metricsData);
    const renderArea = document.getElementById('strike-chart-render-area');
    if (!renderArea || !Array.isArray(chartData) || chartData.length === 0) { 
        if (strikeChart) { strikeChart.destroy(); strikeChart = null; }
        renderArea.innerHTML = ""; return;
    }
    const options = { series: [{ name: 'Puts', data: chartData.map(item => Number(item.Puts) || 0) }, { name: 'Calls', data: chartData.map(item => Number(item.Calls) || 0) }], chart: { type: 'bar', height: 400, foreColor: '#FAFAFA', toolbar: { show: false }, stacked: false }, plotOptions: { bar: { horizontal: false } }, colors: ['#FF6384', '#4BC0C0'], dataLabels: { enabled: false }, xaxis: { categories: chartData.map(item => item.strike), labels: { style: { colors: '#FAFAFA' }, rotate: -45, hideOverlappingLabels: true }, tickAmount: 15 }, yaxis: { labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val >= 1000 ? (val/1000).toFixed(0)+'k' : val } }, grid: { borderColor: '#30363D' }, tooltip: { theme: 'dark' } };
    if (!strikeChart) { strikeChart = new ApexCharts(renderArea, options); strikeChart.render(); } else { strikeChart.updateOptions(options); }
}

function drawVolumeStrikeChart(data) {
    const renderArea = document.getElementById('volume-chart-render-area');
    if (!renderArea || !Array.isArray(data) || data.length === 0) { if (volumeStrikeChart) { volumeStrikeChart.destroy(); volumeStrikeChart = null; } renderArea.innerHTML = ""; return; }
    const options = { series: [{ name: 'Volumen Puts', data: data.map(item => Number(item.Puts_Volume) || 0) }, { name: 'Volumen Calls', data: data.map(item => Number(item.Calls_Volume) || 0) }], chart: { type: 'bar', height: 400, foreColor: '#FAFAFA', toolbar: { show: false }, stacked: true }, plotOptions: { bar: { horizontal: false } }, colors: ['#FF9800', '#2196F3'], dataLabels: { enabled: false }, xaxis: { categories: data.map(item => item.strike), labels: { style: { colors: '#FAFAFA' }, rotate: -45, hideOverlappingLabels: true }, tickAmount: 15 }, yaxis: { title: { text: 'Volumen (Nº Contratos)', style: { color: '#8B949E' } }, labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val >= 1000 ? (val/1000).toFixed(1)+'k' : val.toFixed(0) } }, grid: { borderColor: '#30363D' }, tooltip: { theme: 'dark', y: { formatter: (val) => val.toLocaleString() } } };
    if (!volumeStrikeChart) { volumeStrikeChart = new ApexCharts(renderArea, options); volumeStrikeChart.render(); } else { volumeStrikeChart.updateOptions(options); }
}

function drawExpirationChart(data) {
    const renderArea = document.getElementById('expiration-chart-render-area');
    if (!renderArea || !Array.isArray(data) || data.length === 0) { if (expirationChart) { expirationChart.destroy(); expirationChart = null; } renderArea.innerHTML = '<p class="chart-note">No hay datos.</p>'; return; }
    const options = { series: [{ name: 'Open Interest', data: data.map(item => Number(item.open_interest) || 0) }], chart: { type: 'bar', height: 350, foreColor: '#FAFAFA', toolbar: { show: false } }, colors: ['#36A2EB'], dataLabels: { enabled: false }, xaxis: { categories: data.map(item => item.date), tickAmount: Math.min(data.length, 10), labels: { style: { colors: '#FAFAFA' }, rotate: -45 } }, yaxis: { labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val >= 1000 ? (val/1000).toFixed(0)+'k' : val } }, grid: { borderColor: '#30363D' }, tooltip: { theme: 'dark' } };
    if (!expirationChart) { expirationChart = new ApexCharts(renderArea, options); expirationChart.render(); } else { expirationChart.updateOptions(options); }
}

function renderSentimentCharts(data) {
    const oiContainer = document.getElementById('open-interest-render-area');
    if (data && data.open_interest_history) {
        const options = getSentimentChartOptions(data.open_interest_history, '#008FFB');
        if (!oiSentimentChart) { oiSentimentChart = new ApexCharts(oiContainer, options); oiSentimentChart.render(); } else { oiSentimentChart.updateOptions(options); }
    } else {
        if (oiSentimentChart) { oiSentimentChart.destroy(); oiSentimentChart = null; }
        if (oiContainer) oiContainer.innerHTML = '<p class="chart-note">No hay datos.</p>';
    }
    const lsContainer = document.getElementById('long-short-render-area');
    if (data && data.long_short_ratio) {
        const options = getSentimentChartOptions(data.long_short_ratio, '#00E396', 'Ratio');
        if (!lsRatioChart) { lsRatioChart = new ApexCharts(lsContainer, options); lsRatioChart.render(); } else { lsRatioChart.updateOptions(options); }
    } else {
        if (lsRatioChart) { lsRatioChart.destroy(); lsRatioChart = null; }
        if (lsContainer) lsContainer.innerHTML = '<p class="chart-note">No hay datos.</p>';
    }
}

function getSentimentChartOptions(data, color, seriesName = 'OI (USD)') {
    return { series: [{name: seriesName, data: data.values}], chart: { type: 'area', height: 350, foreColor: '#FAFAFA', toolbar: { show: false } }, colors: [color], dataLabels: { enabled: false }, stroke: { curve: 'smooth', width: 2 }, xaxis: { type: 'category', categories: data.timestamps, labels: { style: { colors: '#FAFAFA' }, rotate: 0 } }, yaxis: { title: { text: '' }, labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val >= 1e9 ? (val / 1e9).toFixed(2) + 'B' : (val >= 1e6 ? (val / 1e6).toFixed(2) + 'M' : (val >= 1e3 ? (val / 1e3).toFixed(1) + 'K' : val.toFixed(2))) } }, grid: { borderColor: '#30363D' }, tooltip: { theme: 'dark', x: { format: 'dd MMM HH:mm' } } };
}

function drawFundingRateHistoryChart(data) {
    const renderArea = document.getElementById('funding-rate-history-render-area');
    if (!renderArea || !data || !data.timestamps) { if(fundingRateHistoryChart) { fundingRateHistoryChart.destroy(); fundingRateHistoryChart = null; } if(renderArea) renderArea.innerHTML = '<p class="chart-note">No hay datos.</p>'; return; }
    const options = { series: [{ name: 'Funding Rate (%)', data: data.funding_rates.map(r => r * 100) }], chart: { type: 'line', height: 350, foreColor: '#FAFAFA', toolbar: { show: false } }, colors: ['#FFC107'], stroke: { curve: 'smooth', width: 2 }, markers: { size: 0 }, xaxis: { categories: data.timestamps, labels: { style: { colors: '#FAFAFA' } }, tickAmount: 10 }, yaxis: { labels: { formatter: (v) => v.toFixed(4) + '%' } }, grid: { borderColor: '#30363D' }, tooltip: { theme: 'dark', y: { formatter: (v) => v.toFixed(4) + '%' } } };
    if (!fundingRateHistoryChart) { fundingRateHistoryChart = new ApexCharts(renderArea, options); fundingRateHistoryChart.render(); } else { fundingRateHistoryChart.updateOptions(options); }
}

function updateOrderBookUI(data) {
    const bidsBody = document.getElementById('order-book-bids');
    const asksBody = document.getElementById('order-book-asks');
    if (!bidsBody || !asksBody || !data) return;
    bidsBody.innerHTML = ''; asksBody.innerHTML = '';
    const priceFmt = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const qtyFmt = new Intl.NumberFormat('en-US', { notation: 'compact', compactDisplay: 'short', maximumFractionDigits: 1 });
    if (data.bids) { data.bids.forEach(order => { const row = bidsBody.insertRow(); row.insertCell().textContent = priceFmt.format(order.price); row.insertCell().textContent = qtyFmt.format(order.quantity); }); }
    if (data.asks) { data.asks.forEach(order => { const row = asksBody.insertRow(); row.insertCell().textContent = priceFmt.format(order.price); row.insertCell().textContent = qtyFmt.format(order.quantity); }); }
}

function drawVolatilitySmileChart(data) {
    const renderArea = document.getElementById('volatility-smile-render-area');
    if (!renderArea || !data) { if(volatilitySmileChart) { volatilitySmileChart.destroy(); volatilitySmileChart = null; } if(renderArea) renderArea.innerHTML = ""; return; }
    const options = { series: [{ name: 'IV de Calls', data: data.map(item => [item.strike, item.call_iv]) }, { name: 'IV de Puts', data: data.map(item => [item.strike, item.put_iv]) }], chart: { type: 'line', height: 400, foreColor: '#FAFAFA' }, colors: ['#4BC0C0', '#FF6384'], stroke: { width: 2, curve: 'smooth' }, markers: { size: 4 }, xaxis: { type: 'numeric', title: { text: 'Strike', style: { color: '#FAFAFA' } }, labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val.toLocaleString() } }, yaxis: { title: { text: 'IV (%)', style: { color: '#FAFAFA' } }, labels: { style: { colors: '#FAFAFA' }, formatter: (value) => value ? value.toFixed(1) + '%' : '' } }, grid: { borderColor: '#30363D' }, tooltip: { theme: 'dark', x: { formatter: (value) => `Strike: ${value.toLocaleString()}` }, y: { formatter: (value) => value ? value.toFixed(2) + '%' : 'N/A' } }, legend: { position: 'top' } };
    if (!volatilitySmileChart) { volatilitySmileChart = new ApexCharts(renderArea, options); volatilitySmileChart.render(); } else { volatilitySmileChart.updateOptions(options); }
}

function drawDvolHistoryChart(data) {
    const renderArea = document.getElementById('dvol-history-render-area');
    if (!renderArea) return;
    if (!data || !data.timestamps) { if (dvolHistoryChart) { dvolHistoryChart.destroy(); dvolHistoryChart = null; } renderArea.innerHTML = '<p class="chart-note">No hay datos.</p>'; return; }
    const options = { series: [{ name: 'DVOL (SMA 7D)', data: data.values }], chart: { type: 'area', height: 350, foreColor: '#FAFAFA', toolbar: { show: false } }, colors: ['#FEB019'], stroke: { curve: 'smooth', width: 2 }, xaxis: { type: 'datetime', categories: data.timestamps, labels: { style: { colors: '#FAFAFA' } } }, yaxis: { labels: { style: { colors: '#FAFAFA' }, formatter: (value) => value.toFixed(1) } }, grid: { borderColor: '#30363D' }, tooltip: { theme: 'dark', x: { format: 'dd MMM yy' }, y: { formatter: (value) => value ? value.toFixed(1) : 'N/A' } } };
    if (!dvolHistoryChart) { dvolHistoryChart = new ApexCharts(renderArea, options); dvolHistoryChart.render(); } else { dvolHistoryChart.updateOptions(options); }
}

// ==============================================================================
// LÓGICA DE LA MODAL
// ==============================================================================
async function openChartInModal(chartType) {
    const modal = document.getElementById('chart-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalRenderArea = document.getElementById('modal-chart-render-area');
    
    modal.classList.remove('hidden');
    modalRenderArea.innerHTML = '<div class="loading-spinner"></div>';

    let url, title, chartDrawer;

    // Asignar la URL, el título y la función de dibujado correcta para cada tipo de gráfico
    switch (chartType) {
        case 'oi-history':
            title = 'Historial Extendido - Interés Abierto (Binance)';
            url = `http://127.0.0.1:5000/api/sentiment/${currentCurrency}?limit=500`;
            chartDrawer = (data) => drawLargeOiChart(data.open_interest_history);
            break;
        case 'ls-ratio':
            title = 'Historial Extendido - Ratio Long/Short (Binance)';
            url = `http://127.0.0.1:5000/api/sentiment/${currentCurrency}?limit=500`;
            chartDrawer = (data) => drawLargeLsRatioChart(data.long_short_ratio);
            break;
        case 'dvol':
            title = 'Historial Extendido - Volatilidad DVOL (Deribit)';
            url = `http://127.0.0.1:5000/api/dvol-history/${currentCurrency}?days=365`;
            chartDrawer = (data) => drawLargeDvolChart(data);
            break;
        case 'funding-rate':
            title = 'Historial Extendido - Tasa de Financiación (Binance)';
            url = `http://127.0.0.1:5000/api/funding-rate-history/${currentCurrency}?limit=500`;
            chartDrawer = (data) => drawLargeFundingRateChart(data);
            break;
        case 'expirations':
            title = 'Interés Abierto Global por Vencimiento';
            url = `http://127.0.0.1:5000/api/data/${currentCurrency}`;
            chartDrawer = (data) => drawLargeBarChart(data.expiration_chart_data, 'Open Interest', '#36A2EB');
            break;
    }

    modalTitle.textContent = title;

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        chartDrawer(data);
    } catch (error) {
        console.error('Error fetching data for modal:', error);
        modalRenderArea.innerHTML = '<p class="chart-note">Error al cargar el historial extendido.</p>';
    }
}

// --- NUEVAS FUNCIONES DE DIBUJADO ESPECÍFICAS PARA LA MODAL ---

function getBaseLargeChartOptions(xaxisType = 'category') {
    return {
        chart: { type: 'area', height: '100%', foreColor: '#FAFAFA', toolbar: { show: true } },
        dataLabels: { enabled: false },
        stroke: { curve: 'smooth', width: 2 },
        xaxis: { type: xaxisType, labels: { style: { colors: '#FAFAFA' } } },
        grid: { borderColor: '#30363D' },
        tooltip: { theme: 'dark', x: { format: 'dd MMM HH:mm' } }
    };
}

function drawLargeOiChart(data) {
    if (modalChart) { modalChart.destroy(); }
    const modalRenderArea = document.getElementById('modal-chart-render-area');
    let options = getBaseLargeChartOptions();
    options.series = [{ name: 'OI (USD)', data: data.values }];
    options.colors = ['#008FFB'];
    options.xaxis.categories = data.timestamps;
    options.yaxis = { title: { text: '' }, labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val >= 1e9 ? (val / 1e9).toFixed(2) + 'B' : (val >= 1e6 ? (val / 1e6).toFixed(2) + 'M' : (val >= 1e3 ? (val / 1e3).toFixed(1) + 'K' : val.toFixed(2))) } };
    modalChart = new ApexCharts(modalRenderArea, options);
    modalChart.render();
}

function drawLargeLsRatioChart(data) {
    if (modalChart) { modalChart.destroy(); }
    const modalRenderArea = document.getElementById('modal-chart-render-area');
    let options = getBaseLargeChartOptions();
    options.series = [{ name: 'Ratio', data: data.values }];
    options.colors = ['#00E396'];
    options.xaxis.categories = data.timestamps;
    options.yaxis = { title: { text: '' }, labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val.toFixed(3) } };
    options.tooltip.y = { formatter: (val) => val.toFixed(4) };
    modalChart = new ApexCharts(modalRenderArea, options);
    modalChart.render();
}

function drawLargeFundingRateChart(data) {
    if (modalChart) { modalChart.destroy(); }
    const modalRenderArea = document.getElementById('modal-chart-render-area');
    let options = getBaseLargeChartOptions();
    options.series = [{ name: 'Funding Rate (%)', data: data.funding_rates.map(r => r * 100) }];
    options.colors = ['#FFC107'];
    options.xaxis.categories = data.timestamps;
    options.yaxis = { title: { text: '' }, labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val.toFixed(4) + '%' } };
    options.tooltip.y = { formatter: (val) => val.toFixed(4) + '%' };
    modalChart = new ApexCharts(modalRenderArea, options);
    modalChart.render();
}

function drawLargeDvolChart(data) {
    if (modalChart) { modalChart.destroy(); }
    const modalRenderArea = document.getElementById('modal-chart-render-area');
    let options = getBaseLargeChartOptions('datetime');
    options.series = [{ name: 'DVOL', data: data.values }];
    options.colors = ['#FEB019'];
    options.xaxis.categories = data.timestamps;
    options.yaxis = { title: { text: '' }, labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val.toFixed(1) } };
    options.tooltip.y = { formatter: (val) => val.toFixed(2) };
    modalChart = new ApexCharts(modalRenderArea, options);
    modalChart.render();
}

function drawLargeBarChart(data, seriesName, color) {
    if (modalChart) { modalChart.destroy(); }
    const modalRenderArea = document.getElementById('modal-chart-render-area');
    const options = {
        series: [{ name: seriesName, data: data.map(item => item.open_interest) }],
        chart: { type: 'bar', height: '100%', foreColor: '#FAFAFA', toolbar: { show: true } },
        colors: [color],
        xaxis: { categories: data.map(item => item.date), labels: { style: { colors: '#FAFAFA' } } },
        yaxis: { labels: { style: { colors: '#FAFAFA' }, formatter: (val) => val >= 1000 ? (val/1000).toFixed(0)+'k' : val } },
        grid: { borderColor: '#30363D' },
        tooltip: { theme: 'dark' }
    };
    modalChart = new ApexCharts(modalRenderArea, options);
    modalChart.render();
}

function closeModal() {
    const modal = document.getElementById('chart-modal');
    modal.classList.add('hidden');
    if (modalChart) {
        modalChart.destroy();
        modalChart = null;
    }
    document.getElementById('modal-chart-render-area').innerHTML = "";
}

// ==============================================================================
// LÓGICA DE INICIALIZACIÓN Y EVENTOS
// ==============================================================================
async function initialize() {
    // Selectores de control
    document.getElementById('btn-btc').addEventListener('click', () => handleCurrencyChange('BTC'));
    document.getElementById('btn-eth').addEventListener('click', () => handleCurrencyChange('ETH'));
    document.getElementById('expiration-selector').addEventListener('change', (e) => { currentExpiration = e.target.value; fetchDeribitDataForCharts(); });
    document.getElementById('order-book-step-selector').addEventListener('click', (e) => { if (e.target.tagName === 'BUTTON' && e.target.dataset.step) { handleOrderBookStepChange(parseFloat(e.target.dataset.step)); } });
    document.getElementById('unit-toggle-checkbox').addEventListener('change', (e) => { displayUnit = e.target.checked ? 'ASSET' : 'USD'; fetchDeribitDataForCharts(); });

    // Listeners para la Modal
    document.getElementById('modal-close-btn').addEventListener('click', closeModal);
    document.getElementById('chart-modal').addEventListener('click', (e) => { if (e.target.id === 'chart-modal') closeModal(); });
    
    // Listeners para los gráficos clicables
    document.getElementById('open-interest-chart-container').addEventListener('click', () => openChartInModal('oi-history'));
    document.getElementById('long-short-chart-container').addEventListener('click', () => openChartInModal('ls-ratio'));
    document.getElementById('dvol-history-chart-container').addEventListener('click', () => openChartInModal('dvol'));
    document.getElementById('funding-rate-history-chart-container').addEventListener('click', () => openChartInModal('funding-rate'));
    document.getElementById('expiration-chart-container').addEventListener('click', () => openChartInModal('expirations'));

    // Carga inicial de datos
    await refreshAllData();
    
    // Intervalos de refresco
    setInterval(refreshAllData, 60000);
    setInterval(fetchOrderBook, 15000);
}

function handleCurrencyChange(newCurrency) {
    if (currentCurrency === newCurrency) return;
    currentCurrency = newCurrency;
    currentExpiration = 'all';
    currentOrderBookStep = 0;
    document.getElementById('btn-btc').classList.toggle('active', newCurrency === 'BTC');
    document.getElementById('btn-eth').classList.toggle('active', newCurrency === 'ETH');
    document.getElementById('asset-unit-label').textContent = currentCurrency;
    const stepSelector = document.getElementById('order-book-step-selector');
    stepSelector.querySelector('button.active')?.classList.remove('active');
    stepSelector.querySelector(`button[data-step="0"]`)?.classList.add('active');
    refreshAllData();
}

function handleOrderBookStepChange(newStep) {
    if (currentOrderBookStep === newStep) return;
    currentOrderBookStep = newStep;
    const selector = document.getElementById('order-book-step-selector');
    selector.querySelector('button.active')?.classList.remove('active');
    selector.querySelector(`button[data-step="${newStep}"]`)?.classList.add('active');
    fetchOrderBook();
}

async function refreshAllData() {
    showAllSpinners();
    document.body.style.cursor = 'wait';
    try {
        const promises = [
            populateExpirations(),
            fetchConsolidatedMetrics(),
            fetchDeribitDataForCharts(),
            fetchBinanceSentimentDataForCharts(),
            fetchFundingRateHistory(),
            fetchDvolHistory()
        ];
        // Solo obtener el libro de órdenes si no hay datos
        if (!document.querySelector('#order-book-bids tr')) {
            promises.push(fetchOrderBook());
        }
        await Promise.all(promises);
    } catch (error) {
        console.error("Error en el refresco periódico de datos:", error);
    } finally {
        hideAllSpinners();
        document.body.style.cursor = 'default';
    }
}

document.addEventListener('DOMContentLoaded', initialize);