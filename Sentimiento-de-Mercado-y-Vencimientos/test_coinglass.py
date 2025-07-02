# test_coinglass.py
import os
import requests

# Leemos la clave de la misma forma que lo hace la aplicación
api_key = os.environ.get("COINGLASS_API_KEY")

if not api_key:
    print("--- RESULTADO ---")
    print("ERROR: La variable de entorno COINGLASS_API_KEY no está establecida en esta terminal.")
    exit()

# Imprimimos una versión parcial de la clave para confirmar que se está leyendo
print("--- RESULTADO ---")
print(f"Clave de API encontrada: ...{api_key[-4:]}") # Muestra solo los últimos 4 caracteres

headers = {"coinglass-api-key": api_key}
url = "https://open-api.coinglass.com/public/v2/liquidation_map?symbol=BTC"

try:
    print("Intentando conectar con Coinglass...")
    response = requests.get(url, headers=headers, timeout=10)

    # Imprimimos el estado de la respuesta
    print(f"Coinglass respondió con el código de estado: {response.status_code}")

    # Forzamos un error si el código no es 200 (éxito)
    response.raise_for_status()

    data = response.json()
    print("¡ÉXITO! Conexión y autenticación con Coinglass funcionan correctamente.")
    print("Datos recibidos:", data)

except requests.exceptions.RequestException as e:
    print("ERROR en la petición a Coinglass:")
    print(e)