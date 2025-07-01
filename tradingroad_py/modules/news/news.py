import requests
from bs4 import BeautifulSoup
import json
import random
from datetime import datetime, timedelta

class NewsService:
    def __init__(self):
        self.base_url = "https://finance.yahoo.com/news"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_yahoo_finance_news(self, category="all", limit=10):
        """
        Obtiene noticias financieras de Yahoo Finance.
        
        Args:
            category: Categoría de noticias ("markets", "economy", "stocks", "crypto" o "all")
            limit: Número máximo de noticias a devolver
            
        Returns:
            Una lista de noticias con título, descripción, enlace, fecha y sentimiento
        """
        try:
            # En una aplicación real, haríamos web scraping o usaríamos una API
            # Por ahora, simulamos datos para el prototipo
            return self._get_mock_news(category, limit)
        except Exception as e:
            print(f"Error al obtener noticias: {e}")
            return []
    
    def _get_mock_news(self, category="all", limit=10):
        """
        Genera noticias de ejemplo para demostración
        """
        # Datos de ejemplo para diferentes categorías
        news_by_category = {
            "markets": [
                {
                    "title": "Los mercados globales cierran al alza tras la decisión de la Fed",
                    "description": "Los índices bursátiles mundiales registraron ganancias significativas después de que la Reserva Federal mantuviera las tasas de interés sin cambios.",
                    "url": "https://finance.yahoo.com/news/markets-rise-fed-decision",
                    "sentiment": "positive",
                    "source": "Yahoo Finance"
                },
                {
                    "title": "El Dow Jones alcanza nuevo máximo histórico",
                    "description": "El índice industrial superó los 40,000 puntos por primera vez en su historia, impulsado por resultados corporativos mejores de lo esperado.",
                    "url": "https://finance.yahoo.com/news/dow-record-high",
                    "sentiment": "positive",
                    "source": "Yahoo Finance"
                },
                {
                    "title": "Volatilidad en mercados emergentes preocupa a inversores",
                    "description": "La incertidumbre política y económica en varios mercados emergentes está generando nerviosismo entre inversores institucionales.",
                    "url": "https://finance.yahoo.com/news/emerging-markets-volatility",
                    "sentiment": "negative",
                    "source": "Yahoo Finance"
                }
            ],
            "economy": [
                {
                    "title": "Inflación se mantiene estable en 2.8% en junio",
                    "description": "El índice de precios al consumidor mostró una estabilización en junio, aliviando temores de presiones inflacionarias descontroladas.",
                    "url": "https://finance.yahoo.com/news/inflation-stable-june",
                    "sentiment": "neutral",
                    "source": "Yahoo Finance"
                },
                {
                    "title": "Desempleo cae a mínimos de 50 años",
                    "description": "La tasa de desempleo bajó al 3.2%, el nivel más bajo desde 1975, superando todas las expectativas de los economistas.",
                    "url": "https://finance.yahoo.com/news/unemployment-50-year-low",
                    "sentiment": "positive",
                    "source": "Yahoo Finance"
                },
                {
                    "title": "PIB del segundo trimestre decepciona con crecimiento del 1.8%",
                    "description": "El crecimiento económico fue menor de lo esperado en el segundo trimestre, encendiendo alarmas sobre una posible desaceleración.",
                    "url": "https://finance.yahoo.com/news/gdp-disappoints-q2",
                    "sentiment": "negative",
                    "source": "Yahoo Finance"
                }
            ],
            "stocks": [
                {
                    "title": "Apple supera expectativas con nuevos lanzamientos",
                    "description": "Las acciones de Apple suben un 5% tras anunciar nuevos productos que superaron las expectativas de los analistas.",
                    "url": "https://finance.yahoo.com/news/apple-exceeds-expectations",
                    "sentiment": "positive",
                    "source": "Yahoo Finance"
                },
                {
                    "title": "Tesla enfrenta problemas en su cadena de suministro",
                    "description": "El fabricante de vehículos eléctricos reportó dificultades en su cadena de suministro que podrían afectar la producción del próximo trimestre.",
                    "url": "https://finance.yahoo.com/news/tesla-supply-chain-issues",
                    "sentiment": "negative",
                    "source": "Yahoo Finance"
                },
                {
                    "title": "Microsoft anuncia nueva estrategia de IA",
                    "description": "La compañía presentó su nueva estrategia de inteligencia artificial para integrar capacidades avanzadas en todos sus productos.",
                    "url": "https://finance.yahoo.com/news/microsoft-ai-strategy",
                    "sentiment": "neutral",
                    "source": "Yahoo Finance"
                }
            ],
            "crypto": [
                {
                    "title": "Bitcoin supera los $80,000 por primera vez",
                    "description": "La principal criptomoneda alcanzó un nuevo récord histórico, superando la barrera de los $80,000 por primera vez.",
                    "url": "https://finance.yahoo.com/news/bitcoin-new-ath-80k",
                    "sentiment": "positive",
                    "source": "Yahoo Finance"
                },
                {
                    "title": "Ethereum completa actualización importante",
                    "description": "La blockchain de Ethereum implementó con éxito una importante actualización que promete mejorar la escalabilidad y reducir comisiones.",
                    "url": "https://finance.yahoo.com/news/ethereum-upgrade-success",
                    "sentiment": "positive",
                    "source": "Yahoo Finance"
                },
                {
                    "title": "Reguladores proponen nuevas normas para criptomonedas",
                    "description": "Varias agencias reguladoras están coordinando nuevas normativas que podrían afectar significativamente al mercado de criptomonedas.",
                    "url": "https://finance.yahoo.com/news/crypto-regulations-proposed",
                    "sentiment": "negative",
                    "source": "Yahoo Finance"
                }
            ]
        }
        
        # Obtener noticias según la categoría
        if category == "all":
            all_news = []
            for cat_news in news_by_category.values():
                all_news.extend(cat_news)
            news_list = all_news
        else:
            news_list = news_by_category.get(category, [])
        
        # Limitar el número de noticias
        result = news_list[:limit]
        
        # Añadir fechas aleatorias recientes
        now = datetime.now()
        for news in result:
            random_hours = random.randint(1, 48)
            news_date = now - timedelta(hours=random_hours)
            news["date"] = news_date.strftime("%Y-%m-%d %H:%M:%S")
        
        return result
    
    def analyze_sentiment(self, news_list):
        """
        Analiza el sentimiento de una lista de noticias
        """
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for news in news_list:
            sentiment = news.get("sentiment", "neutral")
            sentiment_counts[sentiment] += 1
        
        return sentiment_counts
