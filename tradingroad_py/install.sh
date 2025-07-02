#!/bin/bash

echo "=== Instalación de dependencias para TradeRoad AI ==="

# Detectar el comando Python correcto
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "❌ Error: Python no está instalado o no está en el PATH"
    exit 1
fi

echo "✅ Usando $PYTHON_CMD y $PIP_CMD"

# Instalar dependencias básicas
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias básicas..."
    $PIP_CMD install -r requirements.txt
else
    echo "⚠️  requirements.txt no encontrado, instalando dependencias básicas manualmente..."
    $PIP_CMD install flask flask-cors flask-socketio requests pandas numpy python-dotenv
fi

# Instalar dependencias adicionales para IA
if [ -f "requirements_ai.txt" ]; then
    echo "Instalando dependencias para análisis con IA..."
    $PIP_CMD install -r requirements_ai.txt
else
    echo "⚠️  requirements_ai.txt no encontrado, instalando dependencias de IA manualmente..."
    $PIP_CMD install google-generativeai Pillow websockets
fi

echo ""
echo "=== Configuración de variables de entorno ==="
echo "1. Copia el archivo .env.example a .env"
echo "2. Agrega tu API key de Google Gemini en VITE_GEMINI_API_KEY"
echo "3. Verifica las API keys de Finnhub y FMP si las tienes"
echo ""
echo "Para ejecutar la aplicación:"
echo "$PYTHON_CMD app.py"
echo ""
echo "¡Instalación completada!"
