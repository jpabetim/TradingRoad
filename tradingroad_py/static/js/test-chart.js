// Simple test to verify ApexCharts is working
console.log('Test script loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, ApexCharts available:', typeof ApexCharts !== 'undefined');
    
    // Find the strike chart container
    const container = document.getElementById('strike-chart-render-area');
    if (!container) {
        console.log('Container not found!');
        return;
    }
    
    console.log('Container found:', container);
    
    // Add test text to verify we can manipulate the DOM
    container.innerHTML = '<div style="color: white; padding: 20px; text-align: center;">🚀 JavaScript is working! ApexCharts: ' + (typeof ApexCharts !== 'undefined' ? '✅ Loaded' : '❌ Not loaded') + '</div>';
    
    // Create a simple test chart
    const options = {
        series: [{
            name: 'Puts',
            data: [10, 20, 30, 40, 50]
        }, {
            name: 'Calls', 
            data: [15, 25, 35, 45, 55]
        }],
        chart: {
            type: 'bar',
            height: 400,
            foreColor: '#FAFAFA',
            background: 'transparent'
        },
        colors: ['#FF6384', '#4BC0C0'],
        plotOptions: {
            bar: {
                horizontal: false,
                columnWidth: '60%'
            }
        },
        dataLabels: {
            enabled: false
        },
        xaxis: {
            categories: ['85K', '90K', '95K', '100K', '105K'],
            labels: {
                style: { colors: '#FAFAFA' }
            }
        },
        yaxis: {
            labels: {
                style: { colors: '#FAFAFA' }
            }
        },
        grid: {
            borderColor: '#30363D'
        },
        theme: {
            mode: 'dark'
        }
    };

    try {
        const chart = new ApexCharts(container, options);
        chart.render();
        console.log('Test chart rendered successfully!');
    } catch (error) {
        console.error('Error rendering chart:', error);
    }
});
