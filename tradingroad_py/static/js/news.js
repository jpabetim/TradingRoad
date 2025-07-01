// JavaScript para el módulo de noticias financieras

document.addEventListener('DOMContentLoaded', function () {
    // Elementos del DOM
    const categorySelect = document.getElementById('news-category');
    const sentimentSelect = document.getElementById('news-sentiment');
    const newsList = document.getElementById('news-list');
    const sentimentChart = document.getElementById('sentiment-chart');

    // Cargar noticias iniciales
    loadNews();

    // Escuchar cambios en los filtros
    if (categorySelect && sentimentSelect) {
        categorySelect.addEventListener('change', loadNews);
        sentimentSelect.addEventListener('change', loadNews);
    }

    // Función para cargar noticias según los filtros
    function loadNews() {
        const category = categorySelect ? categorySelect.value : 'all';
        const sentiment = sentimentSelect ? sentimentSelect.value : 'all';

        // Mostrar estado de carga
        if (newsList) {
            newsList.innerHTML = '<div class="loading">Cargando noticias...</div>';
        }

        // En una aplicación real, haríamos una petición AJAX
        // Para el prototipo, simulamos una carga con setTimeout
        setTimeout(() => {
            fetchNewsFromAPI(category, sentiment);
        }, 500);
    }

    // Simular una llamada a la API
    function fetchNewsFromAPI(category, sentiment) {
        // En una aplicación real, esta sería una llamada fetch() a la API
        fetch(`/api/news?category=${category}&sentiment=${sentiment}`)
            .then(response => response.json())
            .then(data => {
                displayNews(data.news);
                displaySentimentChart(data.sentiment);
            })
            .catch(error => {
                console.error('Error al cargar las noticias:', error);
                // Fallback a datos de ejemplo en caso de error
                displayMockNews(category, sentiment);
            });
    }

    // Mostrar noticias en la interfaz
    function displayNews(newsItems) {
        if (!newsList) return;

        if (!newsItems || newsItems.length === 0) {
            newsList.innerHTML = '<div class="no-news">No se encontraron noticias con los filtros seleccionados</div>';
            return;
        }

        let html = '';

        newsItems.forEach(news => {
            const sentimentClass = `sentiment-${news.sentiment}`;

            html += `
                <div class="news-item ${sentimentClass}">
                    <h3 class="news-title">
                        <a href="${news.url}" target="_blank">${news.title}</a>
                    </h3>
                    <p class="news-description">${news.description}</p>
                    <div class="news-meta">
                        <span class="news-source">${news.source}</span>
                        <span class="news-date">${formatDate(news.date)}</span>
                        <span class="news-sentiment ${sentimentClass}">${capitalizeFirstLetter(news.sentiment)}</span>
                    </div>
                </div>
            `;
        });

        newsList.innerHTML = html;
    }

    // Mostrar el gráfico de sentimiento
    function displaySentimentChart(sentimentData) {
        if (!sentimentChart) return;

        const total = sentimentData.positive + sentimentData.neutral + sentimentData.negative;

        if (total === 0) {
            sentimentChart.innerHTML = '<div class="no-data">No hay datos de sentimiento disponibles</div>';
            return;
        }

        const positivePercent = Math.round((sentimentData.positive / total) * 100);
        const neutralPercent = Math.round((sentimentData.neutral / total) * 100);
        const negativePercent = Math.round((sentimentData.negative / total) * 100);

        const html = `
            <div class="sentiment-summary">
                <div class="sentiment-bar">
                    <div class="sentiment-positive" style="width: ${positivePercent}%"></div>
                    <div class="sentiment-neutral" style="width: ${neutralPercent}%"></div>
                    <div class="sentiment-negative" style="width: ${negativePercent}%"></div>
                </div>
                <div class="sentiment-legend">
                    <div class="legend-item">
                        <span class="legend-color positive"></span>
                        <span class="legend-label">Positivo (${positivePercent}%)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color neutral"></span>
                        <span class="legend-label">Neutral (${neutralPercent}%)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color negative"></span>
                        <span class="legend-label">Negativo (${negativePercent}%)</span>
                    </div>
                </div>
            </div>
        `;

        sentimentChart.innerHTML = html;
    }

    // Función para mostrar noticias de ejemplo (fallback)
    function displayMockNews(category, sentiment) {
        // Datos de ejemplo para demostración
        const mockNews = getMockNewsData(category, sentiment);
        const mockSentiment = analyzeMockSentiment(mockNews);

        displayNews(mockNews);
        displaySentimentChart(mockSentiment);
    }

    // Función para obtener datos de ejemplo
    function getMockNewsData(category, sentiment) {
        // Datos similares a los del backend, pero para fallback cliente
        const allNews = [
            {
                title: "Los mercados globales cierran al alza tras la decisión de la Fed",
                description: "Los índices bursátiles mundiales registraron ganancias significativas después de que la Reserva Federal mantuviera las tasas de interés sin cambios.",
                url: "https://finance.yahoo.com/news/markets-rise-fed-decision",
                sentiment: "positive",
                source: "Yahoo Finance",
                date: "2025-06-27 14:30:00",
                category: "markets"
            },
            {
                title: "El Dow Jones alcanza nuevo máximo histórico",
                description: "El índice industrial superó los 40,000 puntos por primera vez en su historia, impulsado por resultados corporativos mejores de lo esperado.",
                url: "https://finance.yahoo.com/news/dow-record-high",
                sentiment: "positive",
                source: "Yahoo Finance",
                date: "2025-06-27 10:15:00",
                category: "markets"
            },
            {
                title: "Inflación se mantiene estable en 2.8% en junio",
                description: "El índice de precios al consumidor mostró una estabilización en junio, aliviando temores de presiones inflacionarias descontroladas.",
                url: "https://finance.yahoo.com/news/inflation-stable-june",
                sentiment: "neutral",
                source: "Yahoo Finance",
                date: "2025-06-27 09:45:00",
                category: "economy"
            },
            {
                title: "Bitcoin supera los $80,000 por primera vez",
                description: "La principal criptomoneda alcanzó un nuevo récord histórico, superando la barrera de los $80,000 por primera vez.",
                url: "https://finance.yahoo.com/news/bitcoin-new-ath-80k",
                sentiment: "positive",
                source: "Yahoo Finance",
                date: "2025-06-26 22:10:00",
                category: "crypto"
            },
            {
                title: "Reguladores proponen nuevas normas para criptomonedas",
                description: "Varias agencias reguladoras están coordinando nuevas normativas que podrían afectar significativamente al mercado de criptomonedas.",
                url: "https://finance.yahoo.com/news/crypto-regulations-proposed",
                sentiment: "negative",
                source: "Yahoo Finance",
                date: "2025-06-26 16:30:00",
                category: "crypto"
            },
            {
                title: "Apple supera expectativas con nuevos lanzamientos",
                description: "Las acciones de Apple suben un 5% tras anunciar nuevos productos que superaron las expectativas de los analistas.",
                url: "https://finance.yahoo.com/news/apple-exceeds-expectations",
                sentiment: "positive",
                source: "Yahoo Finance",
                date: "2025-06-26 11:20:00",
                category: "stocks"
            }
        ];

        // Filtrar por categoría
        let filtered = allNews;
        if (category !== 'all') {
            filtered = filtered.filter(news => news.category === category);
        }

        // Filtrar por sentimiento
        if (sentiment !== 'all') {
            filtered = filtered.filter(news => news.sentiment === sentiment);
        }

        return filtered;
    }

    // Analizar el sentimiento de las noticias de ejemplo
    function analyzeMockSentiment(newsList) {
        const sentiment = { positive: 0, neutral: 0, negative: 0 };

        newsList.forEach(news => {
            sentiment[news.sentiment]++;
        });

        return sentiment;
    }

    // Formateador de fechas
    function formatDate(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleDateString('es-ES', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    // Capitalizar primera letra
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});
