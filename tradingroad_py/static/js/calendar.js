// Funciones para el módulo de calendario económico

document.addEventListener('DOMContentLoaded', function () {
    // Referencias a elementos del DOM
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    const todayBtn = document.getElementById('today-btn');
    const weekBtn = document.getElementById('week-btn');
    const monthBtn = document.getElementById('month-btn');
    const impactFilters = document.querySelectorAll('.impact-filter');
    const countrySelect = document.getElementById('country-select');
    const applyFiltersBtn = document.getElementById('apply-filters');
    const calendarData = document.getElementById('calendar-data');
    const timezoneSelect = document.getElementById('timezone-select');
    const currentTimeDisplay = document.getElementById('current-time');
    
    // Variable global para zona horaria actual
    let currentTimezone = +2; // UTC+2 por defecto

    // Inicializar fechas
    const today = new Date();
    const oneWeekLater = new Date();
    oneWeekLater.setDate(today.getDate() + 7);

    startDateInput.valueAsDate = today;
    endDateInput.valueAsDate = oneWeekLater;

    // Event Listeners para botones de fecha
    todayBtn.addEventListener('click', function () {
        const today = new Date();
        const tomorrow = new Date();
        tomorrow.setDate(today.getDate() + 1);

        startDateInput.valueAsDate = today;
        endDateInput.valueAsDate = tomorrow;
    });

    weekBtn.addEventListener('click', function () {
        const today = new Date();
        const oneWeekLater = new Date();
        oneWeekLater.setDate(today.getDate() + 7);

        startDateInput.valueAsDate = today;
        endDateInput.valueAsDate = oneWeekLater;
    });

    monthBtn.addEventListener('click', function () {
        const today = new Date();
        const oneMonthLater = new Date();
        oneMonthLater.setMonth(today.getMonth() + 1);

        startDateInput.valueAsDate = today;
        endDateInput.valueAsDate = oneMonthLater;
    });

    // Event Listeners para filtros
    applyFiltersBtn.addEventListener('click', loadCalendarData);
    
    // Event Listener para cambio de zona horaria
    timezoneSelect.addEventListener('change', function() {
        currentTimezone = parseInt(this.value);
        updateCurrentTime();
        loadCalendarData(); // Recargar datos con nueva zona horaria
    });
    
    // Función para actualizar la hora actual
    function updateCurrentTime() {
        const now = new Date();
        const utcTime = now.getTime() + (now.getTimezoneOffset() * 60000);
        const targetTime = new Date(utcTime + (currentTimezone * 3600000));
        
        const timeString = targetTime.toLocaleTimeString('es-ES', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        const timezoneText = currentTimezone >= 0 ? `+${currentTimezone}` : `${currentTimezone}`;
        currentTimeDisplay.textContent = `Hora actual (UTC${timezoneText}): ${timeString}`;
    }
    
    // Función para convertir fecha a zona horaria seleccionada
    function convertToTimezone(dateString) {
        const date = new Date(dateString);
        const utcTime = date.getTime() + (date.getTimezoneOffset() * 60000);
        const targetTime = new Date(utcTime + (currentTimezone * 3600000));
        return targetTime;
    }
    
    // Inicializar reloj y actualizarlo cada segundo
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);

    // Función para cargar datos del calendario
    function loadCalendarData() {
        // Mostrar mensaje de carga
        calendarData.innerHTML = '<tr><td colspan="7" class="loading-message">Cargando eventos económicos...</td></tr>';

        // Obtener valores de los filtros
        const startDate = startDateInput.value;
        const endDate = endDateInput.value;

        // Obtener impactos seleccionados
        const selectedImpacts = [];
        impactFilters.forEach(checkbox => {
            if (checkbox.checked) {
                selectedImpacts.push(checkbox.value);
            }
        });

        // Obtener países seleccionados
        const selectedCountries = Array.from(countrySelect.selectedOptions).map(option => option.value);

        // Realizar petición a la API
        fetch(`/api/calendar?start_date=${startDate}&end_date=${endDate}`)
            .then(response => response.json())
            .then(data => {
                displayCalendarData(data, selectedImpacts, selectedCountries);
            })
            .catch(error => {
                console.error('Error al cargar datos del calendario:', error);
                calendarData.innerHTML = '<tr><td colspan="7" class="error-message">Error al cargar datos. Por favor, intente nuevamente.</td></tr>';
            });
    }

    // Función para mostrar datos en la tabla
    function displayCalendarData(data, selectedImpacts, selectedCountries) {
        if (!data || data.length === 0) {
            calendarData.innerHTML = '<tr><td colspan="7" class="no-data-message">No hay eventos para mostrar en el período seleccionado.</td></tr>';
            return;
        }

        // Filtrar por impacto y país
        const filteredData = data.filter(item => {
            return (
                (selectedImpacts.length === 0 || selectedImpacts.includes(item.impact)) &&
                (selectedCountries.length === 0 || selectedCountries.includes(item.country.toLowerCase()))
            );
        });

        if (filteredData.length === 0) {
            calendarData.innerHTML = '<tr><td colspan="7" class="no-data-message">No hay eventos que coincidan con los filtros seleccionados.</td></tr>';
            return;
        }

        // Generar filas de la tabla
        let html = '';
        filteredData.forEach(item => {
            const formattedDate = formatDate(item.event_datetime);

            const impactClass = `impact-${item.impact.toLowerCase()}`;

            html += `
                <tr class="${impactClass}">
                    <td>${formattedDate}</td>
                    <td>${getCountryName(item.country)}</td>
                    <td>${item.event_name}</td>
                    <td class="impact ${impactClass}">${item.impact}</td>
                    <td>${item.actual_value || '-'}</td>
                    <td>${item.forecast_value || '-'}</td>
                    <td>${item.previous_value || '-'}</td>
                </tr>
            `;
        });

        calendarData.innerHTML = html;
    }

    // Función para formatear fecha con zona horaria
    function formatDate(dateString) {
        // Convertir a la zona horaria seleccionada
        const targetDate = convertToTimezone(dateString);
        
        const options = {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        
        const formattedDate = targetDate.toLocaleDateString('es-ES', options);
        const timezoneText = currentTimezone >= 0 ? `+${currentTimezone}` : `${currentTimezone}`;
        return `${formattedDate} (UTC${timezoneText})`;
    }

    // Función para obtener nombre de país en español
    function getCountryName(countryCode) {
        const countryMap = {
            'united states': 'EE.UU.',
            'euro zone': 'Eurozona',
            'united kingdom': 'Reino Unido',
            'japan': 'Japón',
            'china': 'China',
            'germany': 'Alemania',
            'france': 'Francia',
            'italy': 'Italia',
            'spain': 'España',
            'canada': 'Canadá',
            'australia': 'Australia',
            'switzerland': 'Suiza',
            'new zealand': 'Nueva Zelanda'
        };

        return countryMap[countryCode.toLowerCase()] || countryCode;
    }

    // Cargar datos iniciales
    loadCalendarData();
});
