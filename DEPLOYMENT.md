# 🚀 Despliegue en Render - TradingRoad Platform

## 📋 Pasos para Desplegar

### 1. **Conectar Repositorio**
- Ve a [Render Dashboard](https://dashboard.render.com)
- Clic en "New" → "Web Service"
- Conecta tu repositorio: `https://github.com/jpabetim/TradingRoad.git`

### 2. **Configuración del Servicio**
```yaml
Name: tradingroad-platform
Runtime: Python 3
Build Command: cd tradingroad_py && pip install -r requirements.txt
Start Command: cd tradingroad_py && python start.py
```

### 3. **Variables de Entorno Requeridas**
En Render Dashboard → Environment:
```
PORT=10000 (Render lo asigna automáticamente)
FLASK_ENV=production
DEBUG=false
FINNHUB_API_KEY=d1hi1h9r01qsvr2aace0d1hi1h9r01qsvr2aaceg
FMP_API_KEY=XtUErGGxXn3UOuGKmn3y6h6OWKFuoZcN
GEMINI_API_KEY=tu_clave_gemini_aqui
```

### 4. **Plan Recomendado**
- **Starter Plan**: $7/mes (recomendado para producción)
- **Free Plan**: Disponible pero con limitaciones

### 5. **Health Check**
- URL: `/dashboard`
- Render verificará automáticamente que la app esté funcionando

## 🔧 Archivos de Configuración Creados

- ✅ `render.yaml` - Configuración automática
- ✅ `Procfile` - Comando de inicio
- ✅ `runtime.txt` - Versión Python 3.12.8
- ✅ `start.py` - Script optimizado para producción
- ✅ `requirements.txt` - Dependencias actualizadas

## 🎯 Funcionalidades Desplegadas

- ✅ **Sistema de noticias reales** (Finnhub + FMP APIs)
- ✅ **Modal IA con scroll robusto**
- ✅ **TraderAlpha Platform completa**
- ✅ **WebSocket support** para tiempo real
- ✅ **APIs de exchanges integradas**

## 🚨 Notas Importantes

1. **APIs Keys**: Configurar en variables de entorno de Render
2. **Puerto**: Render asigna automáticamente el puerto
3. **Primer despliegue**: Puede tardar 10-15 minutos
4. **Logs**: Revisar en Render Dashboard para debug

¡Listo para producción! 🎉
