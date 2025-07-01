// Funciones para el módulo de configuración

document.addEventListener('DOMContentLoaded', function () {
    // Referencias a elementos del DOM
    const tabs = document.querySelectorAll('.tab-item');
    const tabContents = document.querySelectorAll('.config-tab-content');
    const saveButton = document.getElementById('save-config');
    const resetButton = document.getElementById('reset-config');
    const themeOptions = document.querySelectorAll('.theme-option');

    // Cargar configuración actual
    loadConfig();

    // Event Listener para cambio de pestañas
    tabs.forEach(tab => {
        tab.addEventListener('click', function () {
            // Remover clase active de todas las pestañas
            tabs.forEach(t => t.classList.remove('active'));
            // Añadir clase active a la pestaña seleccionada
            this.classList.add('active');

            // Ocultar todos los contenidos
            tabContents.forEach(content => content.classList.remove('active'));

            // Mostrar el contenido correspondiente a la pestaña seleccionada
            const tabId = this.dataset.tab;
            document.getElementById(`${tabId}-config`).classList.add('active');
        });
    });

    // Event Listener para selección de tema
    themeOptions.forEach(option => {
        option.addEventListener('click', function () {
            // Remover clase selected de todas las opciones
            themeOptions.forEach(opt => opt.classList.remove('selected'));
            // Añadir clase selected a la opción seleccionada
            this.classList.add('selected');
        });
    });

    // Event Listener para guardar configuración
    saveButton.addEventListener('click', saveConfig);

    // Event Listener para restablecer configuración
    resetButton.addEventListener('click', resetConfig);

    // Función para cargar configuración
    function loadConfig() {
        fetch('/api/config')
            .then(response => response.json())
            .then(config => {
                // Aplicar tema
                const themeOption = document.querySelector(`.theme-option[data-theme="${config.theme}"]`);
                if (themeOption) {
                    themeOptions.forEach(opt => opt.classList.remove('selected'));
                    themeOption.classList.add('selected');
                }

                // Aplicar API keys
                document.getElementById('yahoo-finance-key').value = config.api_keys.yahoo_finance || '';
                document.getElementById('alpha-vantage-key').value = config.api_keys.alpha_vantage || '';
                document.getElementById('investing-key').value = config.api_keys.investing || '';

                // Aplicar configuración de visualización
                document.getElementById('chart-type').value = config.display_settings.default_chart_type;
                document.getElementById('time-frame').value = config.display_settings.default_time_frame;
                document.getElementById('show-indicators').checked = config.display_settings.show_indicators;
                document.getElementById('show-volume').checked = config.display_settings.show_volume;

                // Aplicar configuración de notificaciones
                document.getElementById('email-alerts').checked = config.notifications.email_alerts;
                document.getElementById('email').value = config.notifications.email || '';
                document.getElementById('price-alerts').checked = config.notifications.price_alerts;
                document.getElementById('news-alerts').checked = config.notifications.news_alerts;
            })
            .catch(error => {
                console.error('Error al cargar la configuración:', error);
            });
    }

    // Función para guardar configuración
    function saveConfig() {
        // Recopilar datos de los formularios
        const selectedTheme = document.querySelector('.theme-option.selected').dataset.theme;

        const apiKeys = {
            yahoo_finance: document.getElementById('yahoo-finance-key').value,
            alpha_vantage: document.getElementById('alpha-vantage-key').value,
            investing: document.getElementById('investing-key').value
        };

        const displaySettings = {
            default_chart_type: document.getElementById('chart-type').value,
            default_time_frame: document.getElementById('time-frame').value,
            show_indicators: document.getElementById('show-indicators').checked,
            show_volume: document.getElementById('show-volume').checked
        };

        const notifications = {
            email_alerts: document.getElementById('email-alerts').checked,
            email: document.getElementById('email').value,
            price_alerts: document.getElementById('price-alerts').checked,
            news_alerts: document.getElementById('news-alerts').checked
        };

        // Crear objeto de configuración
        const config = {
            theme: selectedTheme,
            api_keys: apiKeys,
            display_settings: displaySettings,
            notifications: notifications
        };

        // Enviar configuración al servidor
        fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        })
            .then(response => response.json())
            .then(data => {
                alert('Configuración guardada con éxito');
            })
            .catch(error => {
                console.error('Error al guardar la configuración:', error);
                alert('Error al guardar la configuración');
            });
    }

    // Función para restablecer configuración
    function resetConfig() {
        if (confirm('¿Está seguro de que desea restablecer la configuración a los valores por defecto?')) {
            fetch('/api/config/reset', {
                method: 'POST'
            })
                .then(response => response.json())
                .then(data => {
                    alert('Configuración restablecida con éxito');
                    loadConfig(); // Recargar configuración
                })
                .catch(error => {
                    console.error('Error al restablecer la configuración:', error);
                    alert('Error al restablecer la configuración');
                });
        }
    }
});
