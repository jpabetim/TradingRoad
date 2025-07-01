#!/usr/bin/env python3
"""
Script de inicio optimizado para Render
"""
import os
import sys
from app import app, socketio

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
        # Ejecutar con socketio para WebSocket support
        socketio.run(
            app, 
            debug=debug, 
            port=port, 
            host='0.0.0.0',
            allow_unsafe_werkzeug=True,
            log_output=True
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        # Fallback: ejecutar sin socketio
        app.run(
            debug=debug,
            port=port,
            host='0.0.0.0'
        )
