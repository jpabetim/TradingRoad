class ConfigService:
    def __init__(self):
        self.default_config = {
            "theme": "dark",
            "api_keys": {
                "yahoo_finance": "",
                "alpha_vantage": "",
                "investing": ""
            },
            "display_settings": {
                "default_chart_type": "candlestick",
                "default_time_frame": "1d",
                "show_indicators": True,
                "show_volume": True
            },
            "notifications": {
                "email_alerts": False,
                "email": "",
                "price_alerts": False,
                "news_alerts": False
            }
        }
        self.config = self.default_config.copy()
    
    def get_config(self):
        """Retorna la configuración actual"""
        return self.config
    
    def update_config(self, new_config):
        """Actualiza la configuración con los nuevos valores"""
        # Actualizar solo los valores proporcionados
        if "theme" in new_config:
            self.config["theme"] = new_config["theme"]
            
        if "api_keys" in new_config:
            for key, value in new_config["api_keys"].items():
                if key in self.config["api_keys"]:
                    self.config["api_keys"][key] = value
                    
        if "display_settings" in new_config:
            for key, value in new_config["display_settings"].items():
                if key in self.config["display_settings"]:
                    self.config["display_settings"][key] = value
                    
        if "notifications" in new_config:
            for key, value in new_config["notifications"].items():
                if key in self.config["notifications"]:
                    self.config["notifications"][key] = value
        
        return self.config
    
    def reset_config(self):
        """Restablece la configuración a los valores por defecto"""
        self.config = self.default_config.copy()
        return self.config
