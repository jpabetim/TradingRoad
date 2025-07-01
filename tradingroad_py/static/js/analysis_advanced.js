document.addEventListener('DOMContentLoaded', function () {
    // Marcar que estamos en la sección de análisis
    document.body.classList.add('section-analysis');
    document.body.classList.remove('section-trading');
});

// Funcionalidad para búsqueda de símbolos y favoritos
function initSymbolSearch() {
    const symbolSearch = document.getElementById('symbolSearch');
    const symbolResults = document.getElementById('symbolResults');
    const favoriteSymbols = JSON.parse(localStorage.getItem('favoriteSymbols')) || [];

    // Cargar favoritos guardados
    loadFavoriteSymbols();

    // Configurar búsqueda
    symbolSearch.addEventListener('focus', function () {
        loadSymbols(currentSource);
    });

    symbolSearch.addEventListener('input', function () {
        const searchTerm = this.value.toUpperCase();
        if (searchTerm.length < 1) {
            symbolResults.style.display = 'none';
            return;
        }

        fetch(`/api/market/symbols?source=${currentSource}`)
            .then(response => response.json())
            .then(symbols => {
                // Filtrar símbolos por término de búsqueda
                const filteredSymbols = symbols.filter(symbol =>
                    symbol.toUpperCase().includes(searchTerm)
                ).slice(0, 20); // Limitar a 20 resultados

                if (filteredSymbols.length > 0) {
                    displaySymbolResults(filteredSymbols);
                } else {
                    symbolResults.innerHTML = '<div class="symbol-result-item">No se encontraron coincidencias</div>';
                    symbolResults.style.display = 'block';
                }
            });
    });

    // Cerrar resultados al hacer clic fuera
    document.addEventListener('click', function (event) {
        if (!symbolSearch.contains(event.target) && !symbolResults.contains(event.target)) {
            symbolResults.style.display = 'none';
        }
    });

    // Función para mostrar resultados de búsqueda
    function displaySymbolResults(symbols) {
        symbolResults.innerHTML = '';

        symbols.forEach(symbol => {
            const resultItem = document.createElement('div');
            resultItem.className = 'symbol-result-item';

            const isFavorite = favoriteSymbols.some(fav => fav.symbol === symbol && fav.source === currentSource);

            resultItem.innerHTML = `
                <div class="symbol-name">${symbol}</div>
                <div>
                    <span class="symbol-exchange">${currentSource}</span>
                    <i class="fas ${isFavorite ? 'fa-star' : 'fa-star-o'} favorite-toggle ${isFavorite ? 'active' : ''}"></i>
                </div>
            `;

            // Evento para seleccionar símbolo
            resultItem.addEventListener('click', function (e) {
                if (e.target.classList.contains('favorite-toggle')) {
                    // Alternar favorito
                    toggleFavorite(symbol, currentSource);
                    e.target.classList.toggle('fa-star-o');
                    e.target.classList.toggle('fa-star');
                    e.target.classList.toggle('active');
                } else {
                    // Seleccionar símbolo
                    symbolSearch.value = symbol;
                    currentSymbol = symbol;
                    symbolResults.style.display = 'none';
                    loadMarketData();
                }
            });

            symbolResults.appendChild(resultItem);
        });

        symbolResults.style.display = 'block';
    }

    // Funciones para gestionar favoritos
    function toggleFavorite(symbol, source) {
        const index = favoriteSymbols.findIndex(fav => fav.symbol === symbol && fav.source === source);

        if (index === -1) {
            // Añadir a favoritos
            favoriteSymbols.push({ symbol, source });
        } else {
            // Quitar de favoritos
            favoriteSymbols.splice(index, 1);
        }

        // Guardar en localStorage
        localStorage.setItem('favoriteSymbols', JSON.stringify(favoriteSymbols));

        // Actualizar lista de favoritos
        loadFavoriteSymbols();
    }

    function loadFavoriteSymbols() {
        const favoriteList = document.getElementById('favoriteList');
        favoriteList.innerHTML = '';

        if (favoriteSymbols.length === 0) {
            favoriteList.innerHTML = '<div class="favorite-item">No hay favoritos guardados</div>';
            return;
        }

        favoriteSymbols.forEach(fav => {
            const favoriteItem = document.createElement('div');
            favoriteItem.className = 'favorite-item';
            favoriteItem.innerHTML = `
                <div class="favorite-name">${fav.symbol}</div>
                <div>
                    <span class="favorite-source">${fav.source}</span>
                    <i class="fas fa-times favorite-remove"></i>
                </div>
            `;

            // Evento para seleccionar favorito
            favoriteItem.addEventListener('click', function (e) {
                if (e.target.classList.contains('favorite-remove')) {
                    // Eliminar favorito
                    toggleFavorite(fav.symbol, fav.source);
                } else {
                    // Seleccionar símbolo
                    symbolSearch.value = fav.symbol;
                    if (currentSource !== fav.source) {
                        // Cambiar fuente si es diferente
                        document.getElementById('dataSource').value = fav.source;
                        currentSource = fav.source;
                    }
                    currentSymbol = fav.symbol;
                    loadMarketData();
                }
            });

            favoriteList.appendChild(favoriteItem);
        });
    }
}

// Funcionalidad para el acordeón de indicadores
function initIndicatorAccordion() {
    document.querySelectorAll('.indicator-header').forEach(header => {
        header.addEventListener('click', function () {
            // Cerrar todos los paneles abiertos
            document.querySelectorAll('.indicator-content').forEach(content => {
                content.style.display = 'none';
            });

            // Abrir el panel actual
            const indicatorType = this.getAttribute('data-indicator');
            const contentPanel = document.getElementById(`${indicatorType}-settings`);
            contentPanel.style.display = 'block';
        });
    });
}

// Función para cambiar el tema (claro/oscuro)
function initThemeToggle() {
    const toggleBtn = document.getElementById('toggleTheme');
    const themeIcon = document.getElementById('themeIcon');
    const body = document.body;
    const isDarkTheme = localStorage.getItem('darkTheme') !== 'false'; // Por defecto tema oscuro

    // Aplicar tema guardado
    setTheme(isDarkTheme);

    toggleBtn.addEventListener('click', function () {
        const currentIsDark = body.classList.contains('dark-theme');
        setTheme(!currentIsDark);
        localStorage.setItem('darkTheme', String(!currentIsDark));
    });

    function setTheme(isDark) {
        if (isDark) {
            body.classList.remove('light-theme');
            body.classList.add('dark-theme');
            themeIcon.className = 'fas fa-sun';

            // Actualizar configuración del gráfico para tema oscuro
            if (chart) {
                chart.applyOptions({
                    layout: {
                        background: { type: 'solid', color: '#131722' },
                        textColor: '#d9d9d9',
                    },
                    grid: {
                        vertLines: { color: '#1e222d' },
                        horzLines: { color: '#1e222d' },
                    },
                });
            }
        } else {
            body.classList.remove('dark-theme');
            body.classList.add('light-theme');
            themeIcon.className = 'fas fa-moon';

            // Actualizar configuración del gráfico para tema claro
            if (chart) {
                chart.applyOptions({
                    layout: {
                        background: { type: 'solid', color: '#f5f5f5' },
                        textColor: '#333333',
                    },
                    grid: {
                        vertLines: { color: '#dddddd' },
                        horzLines: { color: '#dddddd' },
                    },
                });
            }
        }

        // Redibujar el gráfico si existe
        if (chart) {
            chart.timeScale().fitContent();
        }
    }
}
