import requests
from bs4 import BeautifulSoup
import json
import random
from datetime import datetime, timedelta
import feedparser
import os
import google.generativeai as genai

class NewsService:
    def __init__(self):
        self.base_url = "https://finance.yahoo.com/news"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # API Keys desde variables de entorno
        self.traderalpha_api_key = os.getenv('VITE_TRADERALPHA_API_KEY') or os.getenv('TRADERALPHA_API_KEY')
        self.translate_api_key = os.getenv('VITE_TRANSLATE_API_KEY') or os.getenv('TRANSLATE_API_KEY')
        
        # Configurar Gemini para traducción
        if self.translate_api_key:
            try:
                genai.configure(api_key=self.translate_api_key)
                self.translate_model = genai.GenerativeModel('gemini-1.5-flash')
                print(f"✅ Modelo de traducción configurado correctamente")
            except Exception as e:
                print(f"Error configurando Gemini para traducción: {e}")
                self.translate_model = None
        else:
            print("⚠️ API key de traducción no encontrada")
            self.translate_model = None
        
        # URLs de RSS feeds gratuitos
        self.rss_feeds = {
            "markets": [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://www.marketwatch.com/rss/topstories",
                "https://www.investing.com/rss/news.rss"
            ],
            "crypto": [
                "https://cointelegraph.com/rss",
                "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "https://decrypt.co/feed"
            ],
            "economy": [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://www.marketwatch.com/rss/economy-politics"
            ]
        }
    
    def get_yahoo_finance_news(self, category="all", limit=10):
        """
        Obtiene noticias financieras reales de múltiples fuentes RSS.
        
        Args:
            category: Categoría de noticias ("markets", "economy", "stocks", "crypto" o "all")
            limit: Número máximo de noticias a devolver
            
        Returns:
            Una lista de noticias con título, descripción, enlace, fecha y sentimiento
        """
        try:
            # Intentar obtener noticias reales primero
            real_news = self._get_real_news_from_rss(category, limit)
            if real_news:
                return real_news
            
            # Si falla, usar noticias mock como fallback
            print("Usando noticias mock como fallback")
            return self._get_mock_news(category, limit)
        except Exception as e:
            print(f"Error al obtener noticias: {e}")
            return self._get_mock_news(category, limit)
    
    def _get_real_news_from_rss(self, category="all", limit=10):
        """
        Obtiene noticias reales de fuentes RSS
        """
        all_news = []
        
        try:
            # Determinar qué feeds usar según la categoría
            feeds_to_use = []
            if category == "all":
                for feed_list in self.rss_feeds.values():
                    feeds_to_use.extend(feed_list)
            elif category in self.rss_feeds:
                feeds_to_use = self.rss_feeds[category]
            else:
                feeds_to_use = self.rss_feeds["markets"]  # Fallback
            
            # Obtener noticias de cada feed
            for feed_url in feeds_to_use[:3]:  # Limitar a 3 feeds para no sobrecargar
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:5]:  # Máximo 5 noticias por feed
                        # Extraer información básica
                        title = getattr(entry, 'title', 'Sin título')
                        description = getattr(entry, 'summary', getattr(entry, 'description', 'Sin descripción'))
                        url = getattr(entry, 'link', '#')
                        
                        # Parsear fecha
                        published = getattr(entry, 'published_parsed', None)
                        if published:
                            pub_date = datetime(*published[:6])
                        else:
                            pub_date = datetime.now()
                        
                        # Limpiar HTML de la descripción
                        if description:
                            soup = BeautifulSoup(description, 'html.parser')
                            description = soup.get_text().strip()[:200] + "..."
                        
                        # Determinar sentimiento básico (mejorar esto más tarde)
                        sentiment = self._analyze_sentiment_basic(title + " " + description)
                        
                        # Traducir si es necesario y tenemos API key
                        if self._is_english_text(title) and self.translate_model:
                            print(f"🔄 Traduciendo: {title[:50]}...")
                            translated_title = self._translate_text(title)
                            if translated_title != title:
                                print(f"✅ Traducido: {translated_title[:50]}...")
                                title = translated_title
                            
                            translated_desc = self._translate_text(description)
                            if translated_desc != description:
                                description = translated_desc
                        
                        # Calcular tiempo relativo para mostrar
                        time_diff = datetime.now() - pub_date
                        if time_diff.days > 0:
                            time_display = f"Hace {time_diff.days} días"
                        elif time_diff.seconds > 3600:
                            hours = time_diff.seconds // 3600
                            time_display = f"Hace {hours} horas"
                        elif time_diff.seconds > 60:
                            minutes = time_diff.seconds // 60
                            time_display = f"Hace {minutes} minutos"
                        else:
                            time_display = "Hace unos segundos"
                        
                        news_item = {
                            "title": title,
                            "description": description,
                            "url": url,
                            "sentiment": sentiment,
                            "source": feed_url.split('/')[2] if '/' in feed_url else "RSS Feed",
                            "date": pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                            "time": time_display,  # Añadir campo time para compatibilidad con frontend
                            "timestamp": pub_date.timestamp()
                        }
                        
                        all_news.append(news_item)
                
                except Exception as feed_error:
                    print(f"Error procesando feed {feed_url}: {feed_error}")
                    continue
            
            # Ordenar por fecha (más recientes primero) y limitar
            all_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            return all_news[:limit]
            
        except Exception as e:
            print(f"Error en _get_real_news_from_rss: {e}")
            return []
    
    def _analyze_sentiment_basic(self, text):
        """
        Análisis básico de sentimiento usando palabras clave
        """
        positive_words = ['up', 'rise', 'gain', 'increase', 'high', 'strong', 'positive', 'growth', 'bull', 'rally', 'surge', 'boom']
        negative_words = ['down', 'fall', 'loss', 'decrease', 'low', 'weak', 'negative', 'decline', 'bear', 'crash', 'drop', 'recession']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _is_english_text(self, text):
        """
        Detecta si el texto está en inglés (básico)
        """
        english_indicators = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'that']
        text_lower = text.lower()
        return any(word in text_lower for word in english_indicators)
    
    def _translate_text(self, text):
        """
        Traduce texto del inglés al español usando Gemini
        """
        if not self.translate_model or not text or len(text.strip()) < 3:
            return text
        
        try:
            # Limpiar texto de caracteres extraños
            clean_text = text.strip()
            if len(clean_text) > 1000:  # Limitar longitud para evitar errores
                clean_text = clean_text[:1000] + "..."
            
            prompt = f"""Traduce este texto financiero del inglés al español. Mantén un tono profesional y usa terminología financiera apropiada. Solo devuelve la traducción sin explicaciones:

{clean_text}"""
            
            response = self.translate_model.generate_content(prompt)
            translated = response.text.strip()
            
            # Verificar que la traducción sea válida
            if translated and len(translated) > 5 and not translated.lower().startswith('error'):
                return translated
            else:
                return text  # Devolver original si la traducción parece inválida
                
        except Exception as e:
            print(f"Error traduciendo texto: {e}")
            return text  # Devolver texto original si falla la traducción
    
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
