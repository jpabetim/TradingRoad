import os

# Configuración de Gunicorn optimizada para Render (512MB RAM)
bind = f"0.0.0.0:{os.getenv('PORT', '5007')}"
workers = 1  # Solo 1 worker para ahorrar memoria
threads = 2  # 2 threads por worker
worker_class = "eventlet"  # Async worker para mejor rendimiento
worker_connections = 1000
max_requests = 1000  # Reciclar workers para evitar memory leaks
max_requests_jitter = 50
timeout = 120
keepalive = 2
preload_app = True  # Precargar la app para ahorrar memoria

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Memory optimization
max_worker_memory = 400  # MB por worker
worker_memory_limit = 400 * 1024 * 1024  # bytes
