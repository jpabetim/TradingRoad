// Cargador de noticias reales para el dashboard
class NewsLoader {
    constructor() {
        this.newsContainer = null;
        // Intentar cargar inmediatamente y cada segundo hasta que funcione
        this.init();
    }

    init() {
        console.log('NewsLoader inicializando...');
        this.tryInitialize();
    }

    tryInitialize() {
        this.newsContainer = document.getElementById('news-container');
        console.log('Buscando news-container:', this.newsContainer);
        
        if (this.newsContainer) {
            console.log('Contenedor encontrado, cargando noticias...');
            this.loadNews();
        } else {
            this.retryCount = (this.retryCount || 0) + 1;
            if (this.retryCount < 20) {
                console.warn(`news-container no encontrado, reintentando (${this.retryCount}/20) en 500ms...`);
                setTimeout(() => this.tryInitialize(), 500);
            } else {
                console.error('news-container no encontrado después de 20 intentos');
            }
        }
    }

    async loadNews() {
        console.log('Iniciando carga de noticias...');
        try {
            const response = await fetch('/api/news?category=general&sentiment=all');
            console.log('Response status:', response.status);
            const data = await response.json();
            console.log('Datos recibidos:', data);
            
            if (data.news && data.news.length > 0) {
                console.log(`Renderizando ${data.news.length} noticias`);
                this.renderNews(data.news);
            } else {
                console.warn('No hay noticias para mostrar');
                this.renderError('No se pudieron cargar las noticias en este momento.');
            }
        } catch (error) {
            console.error('Error cargando noticias:', error);
            this.renderError('Error de conexión al cargar las noticias.');
        }
    }

    renderNews(newsArray) {
        if (!this.newsContainer) return;

        // Limpiar container
        this.newsContainer.innerHTML = '';

        // Crear elementos de noticias
        newsArray.forEach(newsItem => {
            const newsElement = this.createNewsElement(newsItem);
            this.newsContainer.appendChild(newsElement);
        });
    }

    createNewsElement(newsItem) {
        const newsDiv = document.createElement('div');
        newsDiv.className = 'news-item';
        
        newsDiv.innerHTML = `
            <h3>${this.escapeHtml(newsItem.title)}</h3>
            <p>${this.escapeHtml(newsItem.description)}</p>
            <span class="news-source">${this.escapeHtml(newsItem.source)} - ${this.escapeHtml(newsItem.time)}</span>
        `;

        // Añadir evento click si hay URL
        if (newsItem.url && newsItem.url !== '#') {
            newsDiv.style.cursor = 'pointer';
            newsDiv.addEventListener('click', () => {
                window.open(newsItem.url, '_blank');
            });
        }

        return newsDiv;
    }

    renderError(message) {
        if (!this.newsContainer) return;

        this.newsContainer.innerHTML = `
            <div class="news-error">
                <h3>⚠️ Error al cargar noticias</h3>
                <p>${this.escapeHtml(message)}</p>
                <button onclick="newsLoader.loadNews()" style="
                    background: rgba(142, 79, 255, 0.8);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    cursor: pointer;
                    margin-top: 15px;
                ">Reintentar</button>
            </div>
        `;
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
}

// Inicializar el cargador de noticias
console.log('Creando instancia de NewsLoader...');
const newsLoader = new NewsLoader();
console.log('NewsLoader creado:', newsLoader);
