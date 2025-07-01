// Simple test script para debugging
console.log('🚀 Simple test script loaded');

document.addEventListener('DOMContentLoaded', async function() {
    console.log('📍 DOM loaded, starting tests...');
    
    // Test básico de APIs
    try {
        console.log('🔍 Testing DVOL API...');
        const response = await fetch('http://127.0.0.1:8088/api/dvol-history/BTC');
        console.log('📡 DVOL Response status:', response.status);
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ DVOL Data received:', data.length, 'records');
            console.log('📊 First record:', data[0]);
            
            // Mostrar en el HTML
            const renderArea = document.getElementById('dvol-history-render-area');
            if (renderArea) {
                renderArea.innerHTML = `
                    <div style="color: #4CAF50; padding: 20px; background: #1e1e1e; border-radius: 8px;">
                        <h4>✅ DVOL Data Loaded Successfully!</h4>
                        <p><strong>Records:</strong> ${data.length}</p>
                        <p><strong>First Record:</strong></p>
                        <pre>${JSON.stringify(data[0], null, 2)}</pre>
                    </div>
                `;
                console.log('✅ DVOL display updated');
            }
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('❌ DVOL API Error:', error);
        const renderArea = document.getElementById('dvol-history-render-area');
        if (renderArea) {
            renderArea.innerHTML = `
                <div style="color: #f44336; padding: 20px; background: #1e1e1e; border-radius: 8px;">
                    <h4>❌ DVOL API Error</h4>
                    <p>${error.message}</p>
                </div>
            `;
        }
    }
    
    // Test API de Sentiment
    try {
        console.log('🔍 Testing Sentiment API...');
        const response = await fetch('http://127.0.0.1:8088/api/sentiment/BTC');
        console.log('📡 Sentiment Response status:', response.status);
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Sentiment Data received');
            console.log('📊 Long/Short timestamps:', data.long_short_ratio?.timestamps?.length);
            
            // Mostrar en el HTML
            const renderArea = document.getElementById('long-short-render-area');
            if (renderArea) {
                renderArea.innerHTML = `
                    <div style="color: #4CAF50; padding: 20px; background: #1e1e1e; border-radius: 8px;">
                        <h4>✅ Sentiment Data Loaded!</h4>
                        <p><strong>L/S Ratio Points:</strong> ${data.long_short_ratio?.timestamps?.length || 0}</p>
                        <p><strong>Current OI:</strong> $${(data.current_oi_binance / 1e9).toFixed(2)}B</p>
                    </div>
                `;
                console.log('✅ Sentiment display updated');
            }
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('❌ Sentiment API Error:', error);
    }
    
    console.log('🏁 Simple test completed');
});
