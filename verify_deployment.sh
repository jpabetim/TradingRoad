#!/bin/bash
# Script de verificación pre-despliegue

echo "🔍 VERIFICACIÓN PRE-DESPLIEGUE RENDER"
echo "======================================"

# Verificar archivos necesarios
echo "📁 Verificando archivos necesarios..."

files=(
    "render.yaml"
    "package.json"
    "requirements.txt"
    "run.py"
    "tradingroad_py/app.py"
    "tradingroad_py/start.py"
    ".env.example"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file - OK"
    else
        echo "❌ $file - FALTA"
    fi
done

echo ""
echo "🔧 Verificando configuración del package.json..."
if grep -q "preview.*host.*port" package.json; then
    echo "✅ Comando preview configurado correctamente"
else
    echo "❌ Comando preview necesita configuración"
fi

echo ""
echo "🐍 Verificando configuración del backend..."
if [ -f "tradingroad_py/requirements.txt" ]; then
    echo "✅ requirements.txt encontrado"
else
    echo "❌ requirements.txt no encontrado"
fi

echo ""
echo "🌐 Verificando variables de entorno..."
if grep -q "VITE_BACKEND_URL" .env.example; then
    echo "✅ Variables frontend configuradas"
else
    echo "❌ Variables frontend falta configurar"
fi

if grep -q "FRONTEND_URL" .env.example; then
    echo "✅ Variables backend configuradas"
else
    echo "❌ Variables backend falta configurar"
fi

echo ""
echo "🚀 ESTADO DEL PROYECTO:"
echo "======================="

if [ -f "render.yaml" ] && [ -f "package.json" ] && [ -f "run.py" ]; then
    echo "✅ LISTO PARA DESPLIEGUE EN RENDER"
    echo ""
    echo "📋 PRÓXIMOS PASOS:"
    echo "1. Conectar repositorio en Render.com"
    echo "2. Configurar variables de entorno en dashboard"
    echo "3. Desplegar servicios"
    echo ""
    echo "📖 Ver RENDER_DEPLOYMENT.md para instrucciones detalladas"
else
    echo "❌ FALTAN ARCHIVOS PARA EL DESPLIEGUE"
fi
