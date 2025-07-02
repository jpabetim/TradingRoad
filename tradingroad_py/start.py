#!/usr/bin/env python3
"""
Script de inicio optimizado para Render
"""
import os
import sys
from app import app

if __name__ == '__main__':
    # Configuración para producción
    port = int(os.environ.get('PORT', 8088))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting TradingRoad Platform on port {port}")
    print(f"📊 Debug mode: {debug}")
    print(f"🌍 Environment: {os.environ.get('FLASK_ENV', 'development')}")
    
    # Configurar app para producción
    app.config['DEBUG'] = debug
    
    try:
        # Ejecutar la aplicación Flask
        app.run(
            debug=debug,
            port=port,
            host='0.0.0.0'
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
