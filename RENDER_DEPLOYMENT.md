# Guía de Despliegue en Render.com

## 📋 Prerrequisitos

1. Cuenta en [Render.com](https://render.com)
2. Repositorio Git con el código (GitHub, GitLab, etc.)
3. Claves de API configuradas

## 🚀 Pasos para el Despliegue

### 1. Conectar Repositorio

1. Ve a tu dashboard de Render
2. Conecta tu repositorio Git que contiene este proyecto
3. Render detectará automáticamente el archivo `render.yaml`

### 2. Configurar Variables de Entorno

En el dashboard de Render, configura las siguientes variables para cada servicio:

#### Backend (tradingroad-backend)
```
GEMINI_API_KEY=tu_clave_api_gemini
FINNHUB_API_KEY=tu_clave_api_finnhub
FMP_API_KEY=tu_clave_api_fmp
FRONTEND_URL=https://tradingroad-frontend.onrender.com
```

#### Frontend (tradingroad-frontend)
```
VITE_BACKEND_URL=https://tradingroad-backend.onrender.com
VITE_GEMINI_API_KEY=tu_clave_api_gemini
VITE_FINNHUB_API_KEY=tu_clave_api_finnhub
VITE_FMP_API_KEY=tu_clave_api_fmp
```

### 3. URLs de los Servicios

Una vez desplegado, tendrás:

- **Backend**: `https://tradingroad-backend.onrender.com`
- **Frontend**: `https://tradingroad-frontend.onrender.com`

### 4. Verificar el Despliegue

1. Accede al frontend en: `https://tradingroad-frontend.onrender.com`
2. Verifica que puedes navegar a:
   - `/dashboard` - Panel principal
   - `/analysis` - Análisis técnico
   - `/news` - Noticias financieras
3. Comprueba que las APIs funcionan correctamente

## 🔧 Configuración Técnica

### Arquitectura del Despliegue

```
┌─────────────────┐    API calls    ┌─────────────────┐
│                 │ ───────────────> │                 │
│   Frontend      │                  │   Backend       │
│   (React/Vite)  │ <─────────────── │   (Flask)       │
│                 │    responses     │                 │
└─────────────────┘                  └─────────────────┘
```

### Variables de Entorno por Servicio

#### Backend Flask
- `FLASK_ENV=production`
- `DEBUG=false`
- `PORT` (auto-generado por Render)
- `FRONTEND_URL` - URL del frontend para CORS y redirecciones
- APIs keys para servicios externos

#### Frontend React
- `VITE_BACKEND_URL` - URL del backend para llamadas API
- `VITE_*` variables para APIs que se usan en el cliente

### Comandos de Build y Start

#### Backend
- **Build**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start**: `python run.py`

#### Frontend
- **Build**: `npm install && npm run build`
- **Start**: `npm run preview --host 0.0.0.0 --port $PORT`

## 🔍 Troubleshooting

### Error de CORS
Si hay errores de CORS, verifica que `FRONTEND_URL` esté configurado correctamente en el backend.

### API Keys no funcionan
1. Verifica que las claves estén configuradas en el dashboard de Render
2. Las variables del frontend deben tener prefijo `VITE_`
3. Reinicia los servicios después de cambiar variables

### Servicios no se comunican
1. Verifica que `VITE_BACKEND_URL` apunte al backend correcto
2. Comprueba que ambos servicios estén desplegados y funcionando
3. Revisa los logs en el dashboard de Render

### Cambios de URL
Si cambias los nombres de los servicios, actualiza:
1. Las variables `FRONTEND_URL` y `VITE_BACKEND_URL`
2. El archivo `render.yaml` si es necesario

## 📊 Monitoreo

- **Logs**: Disponibles en el dashboard de Render para cada servicio
- **Health Checks**: El backend tiene health check en `/dashboard`
- **Métricas**: Render proporciona métricas básicas de uso

## 🔄 Actualizaciones

Para desplegar cambios:
1. Haz push a tu repositorio Git
2. Render automáticamente redespliegará (si `autoDeploy: true`)
3. O puedes redesplegar manualmente desde el dashboard

## 💡 Consejos

1. **Plan Free**: Render tiene límites en el plan gratuito
2. **Sleep Mode**: Los servicios gratuitos entran en "sleep" tras inactividad
3. **Cold Start**: El primer acceso puede ser lento tras el "sleep"
4. **SSL**: Render proporciona SSL automático (HTTPS)
