#!/usr/bin/env python3
"""
Script de inicio principal para Render
Detecta automáticamente si se ejecuta desde la raíz o desde tradingroad_py
"""
import os
import sys
import subprocess

def main():
    print("🚀 TradingRoad Platform - Starting...")
    
    # Detectar si estamos en la raíz o en el subdirectorio
    current_dir = os.getcwd()
    print(f"📂 Current directory: {current_dir}")
    
    # Buscar el archivo app.py
    if os.path.exists("tradingroad_py/app.py"):
        # Estamos en la raíz
        print("📍 Running from root directory")
        os.chdir("tradingroad_py")
        print(f"📂 Changed to: {os.getcwd()}")
    elif os.path.exists("app.py"):
        # Ya estamos en tradingroad_py
        print("📍 Already in tradingroad_py directory")
    else:
        print("❌ Error: app.py not found!")
        sys.exit(1)
    
    # Verificar que app.py existe
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory!")
        sys.exit(1)
    
    # Obtener variables de entorno
    port = os.environ.get('PORT', '8088')
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"🌐 Port: {port}")
    print(f"🔧 Debug: {debug}")
    print(f"🌍 Environment: {os.environ.get('FLASK_ENV', 'development')}")
    
    # Ejecutar la aplicación
    try:
        print("🚀 Starting Flask application...")
        if os.path.exists("start.py"):
            subprocess.run([sys.executable, "start.py"], check=True)
        else:
            # Fallback directo a app.py
            subprocess.run([sys.executable, "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()
