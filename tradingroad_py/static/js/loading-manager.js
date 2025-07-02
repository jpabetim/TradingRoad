// Sistema global de spinners de carga
class LoadingManager {
    constructor() {
        this.activeLoaders = new Set();
        this.init();
    }

    init() {
        this.createGlobalStyles();
        this.interceptNavigation();
        this.interceptForms();
        this.interceptButtons();
    }

    createGlobalStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                backdrop-filter: blur(2px);
            }
            
            .loading-spinner-global {
                width: 50px;
                height: 50px;
                border: 4px solid rgba(255, 255, 255, 0.1);
                border-top: 4px solid #3498db;
                border-radius: 50%;
                animation: spin-global 1s linear infinite;
            }
            
            @keyframes spin-global {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .btn-loading-state {
                position: relative;
                color: transparent !important;
                pointer-events: none;
            }
            
            .btn-loading-state::after {
                content: "";
                position: absolute;
                width: 20px;
                height: 20px;
                top: 50%;
                left: 50%;
                margin-left: -10px;
                margin-top: -10px;
                border: 2px solid transparent;
                border-top-color: currentColor;
                border-radius: 50%;
                animation: spin-global 1s linear infinite;
            }
            
            .section-loading-state {
                position: relative;
                opacity: 0.6;
                pointer-events: none;
            }
            
            .section-loading-state::before {
                content: "";
                position: absolute;
                top: 50%;
                left: 50%;
                width: 30px;
                height: 30px;
                margin: -15px 0 0 -15px;
                border: 3px solid rgba(255, 255, 255, 0.1);
                border-top-color: #3498db;
                border-radius: 50%;
                animation: spin-global 1s linear infinite;
                z-index: 1000;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 50%;
            }
        `;
        document.head.appendChild(style);
    }

    interceptNavigation() {
        // Interceptar clics en enlaces de navegación
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href]');
            if (link && !link.getAttribute('href').startsWith('#') &&
                !link.getAttribute('href').startsWith('javascript:') &&
                link.getAttribute('href') !== '#') {

                // Solo mostrar loading para navegación interna
                if (link.getAttribute('href').startsWith('/') ||
                    link.getAttribute('href').includes(window.location.hostname)) {
                    this.showGlobalLoader('Cargando página...');
                }
            }
        });
    }

    interceptForms() {
        // Interceptar envío de formularios
        document.addEventListener('submit', (e) => {
            const form = e.target;
            if (form.tagName === 'FORM') {
                this.showGlobalLoader('Procesando...');

                // Auto-ocultar después de 10 segundos como fallback
                setTimeout(() => {
                    this.hideGlobalLoader();
                }, 10000);
            }
        });
    }

    interceptButtons() {
        // Interceptar clics en botones
        document.addEventListener('click', (e) => {
            const button = e.target.closest('button');
            if (button && !button.classList.contains('no-loading')) {
                this.showButtonLoader(button);

                // Auto-restaurar después de 3 segundos como fallback
                setTimeout(() => {
                    this.hideButtonLoader(button);
                }, 3000);
            }
        });
    }

    showGlobalLoader(message = 'Cargando...') {
        this.hideGlobalLoader(); // Ocultar loader anterior si existe

        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.id = 'global-loading-overlay';

        overlay.innerHTML = `
            <div style="text-align: center; color: white;">
                <div class="loading-spinner-global"></div>
                <p style="margin-top: 20px; font-size: 16px;">${message}</p>
            </div>
        `;

        document.body.appendChild(overlay);
        this.activeLoaders.add('global');
    }

    hideGlobalLoader() {
        const overlay = document.getElementById('global-loading-overlay');
        if (overlay) {
            overlay.remove();
            this.activeLoaders.delete('global');
        }
    }

    showButtonLoader(button) {
        if (!button.dataset.originalText) {
            button.dataset.originalText = button.textContent;
        }
        button.classList.add('btn-loading-state');
        button.disabled = true;
    }

    hideButtonLoader(button) {
        button.classList.remove('btn-loading-state');
        button.disabled = false;
        if (button.dataset.originalText) {
            button.textContent = button.dataset.originalText;
        }
    }

    showSectionLoader(sectionElement) {
        sectionElement.classList.add('section-loading-state');
    }

    hideSectionLoader(sectionElement) {
        sectionElement.classList.remove('section-loading-state');
    }

    // Métodos públicos para uso manual
    show(message) {
        this.showGlobalLoader(message);
    }

    hide() {
        this.hideGlobalLoader();
    }

    showForSection(selector, message) {
        const section = document.querySelector(selector);
        if (section) {
            this.showSectionLoader(section);
        }
    }

    hideForSection(selector) {
        const section = document.querySelector(selector);
        if (section) {
            this.hideSectionLoader(section);
        }
    }
}

// Inicializar el gestor de carga global
let loadingManager;
document.addEventListener('DOMContentLoaded', function () {
    loadingManager = new LoadingManager();

    // Hacer disponible globalmente
    window.loadingManager = loadingManager;

    // Ocultar loading al cargar la página
    window.addEventListener('load', () => {
        loadingManager.hide();
    });

    // Funciones de conveniencia globales
    window.showLoading = (message) => loadingManager.show(message);
    window.hideLoading = () => loadingManager.hide();
});
