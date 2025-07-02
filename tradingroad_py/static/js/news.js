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

    // Obtener noticias reales de las APIs
    async function fetchNewsFromAPI(category, sentiment) {
        try {
            console.log('Cargando noticias reales desde APIs...');

            // Usar el endpoint real de noticias que combina Finnhub y FMP
            const response = await fetch(`/api/news?category=${category}&sentiment=${sentiment}`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const responseData = await response.json();
            console.log('Noticias obtenidas:', responseData);

            if (responseData.status === 'ok' && responseData.news && Array.isArray(responseData.news) && responseData.news.length > 0) {
                // Las noticias ya vienen filtradas del backend
                const newsData = responseData.news;
                const sentimentData = responseData.sentiment || {};

                displayNews(newsData);
                displaySentimentChart(sentimentData);
            } else {
                throw new Error('No se recibieron noticias válidas');
            }

        } catch (error) {
            console.error('Error al cargar noticias reales:', error);

            // Mostrar mensaje de error más informativo
            if (newsList) {
                newsList.innerHTML = `
                    <div class="error-message">
                        <h3>Error al cargar noticias</h3>
                        <p>No se pudieron cargar las noticias en tiempo real.</p>
                        <p>Error: ${error.message}</p>
                        <button onclick="loadNews()" class="retry-btn">Reintentar</button>
                    </div>
                `;
            }

            // Fallback a datos de ejemplo solo como último recurso
            console.log('Usando noticias de respaldo...');
            displayMockNews(category, sentiment);
        }
    }

    // Filtrar noticias por categoría
    function filterNewsByCategory(news, category) {
        if (category === 'all') return news;

        // Mapeo básico de categorías
        const categoryKeywords = {
            'markets': ['market', 'trading', 'index', 'dow', 'nasdaq', 's&p'],
            'economy': ['economic', 'gdp', 'inflation', 'federal', 'fed', 'economy'],
            'stocks': ['stock', 'equity', 'share', 'earnings', 'dividend'],
            'crypto': ['bitcoin', 'crypto', 'blockchain', 'ethereum', 'btc', 'eth']
        };

        if (!categoryKeywords[category]) return news;

        return news.filter(article => {
            const text = (article.title + ' ' + article.description).toLowerCase();
            return categoryKeywords[category].some(keyword => text.includes(keyword));
        });
    }

    // Análisis básico de sentimiento para noticias reales
    function analyzeSentimentFromRealNews(news) {
        const positiveWords = ['gain', 'rise', 'up', 'growth', 'profit', 'surge', 'boost', 'positive', 'strong'];
        const negativeWords = ['fall', 'drop', 'down', 'loss', 'decline', 'crash', 'negative', 'weak', 'concern'];

        let positive = 0;
        let negative = 0;
        let neutral = 0;

        news.forEach(article => {
            const text = (article.title + ' ' + article.description).toLowerCase();

            const positiveCount = positiveWords.reduce((count, word) =>
                count + (text.match(new RegExp(word, 'g')) || []).length, 0);
            const negativeCount = negativeWords.reduce((count, word) =>
                count + (text.match(new RegExp(word, 'g')) || []).length, 0);

            if (positiveCount > negativeCount) {
                positive++;
            } else if (negativeCount > positiveCount) {
                negative++;
            } else {
                neutral++;
            }
        });

        return {
            positive: positive,
            neutral: neutral,
            negative: negative
        };
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
                        <span class="news-date">${news.time || news.date || 'Fecha no disponible'}</span>
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
        if (!dateStr) return 'Fecha no disponible';

        try {
            let date;

            // Si es un timestamp Unix (número)
            if (typeof dateStr === 'number') {
                date = new Date(dateStr * 1000);
            } else {
                // String de fecha
                date = new Date(dateStr);
            }

            // Verificar si la fecha es válida
            if (isNaN(date.getTime())) {
                return 'Fecha inválida';
            }

            return date.toLocaleDateString('es-ES', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (error) {
            console.error('Error formateando fecha:', dateStr, error);
            return 'Fecha no disponible';
        }
    }

    // Capitalizar primera letra
    function capitalizeFirstLetter(string) {
        if (!string || typeof string !== 'string') {
            return 'Neutral';
        }
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});
