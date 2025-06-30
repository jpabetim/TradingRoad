


export interface MarketDataPoint {
  time: number; // Unix timestamp (seconds) for Lightweight Charts
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export enum AnalysisPointType {
  POI_OFERTA = "poi_oferta", // Bearish Order Block or Supply Zone
  POI_DEMANDA = "poi_demanda", // Bullish Order Block or Demand Zone
  LIQUIDEZ_VENDEDORA = "liquidez_vendedora", // Sell-Side Liquidity (e.g., below old lows)
  LIQUIDEZ_COMPRADORA = "liquidez_compradora", // Buy-Side Liquidity (e.g., above old highs)
  FVG_BAJISTA = "fvg_bajista", // Bearish Fair Value Gap / Imbalance
  FVG_ALCISTA = "fvg_alcista", // Bullish Fair Value Gap / Imbalance
  BOS_ALCISTA = "bos_alcista", // Break of Structure (Bullish)
  BOS_BAJISTA = "bos_bajista", // Break of Structure (Bearish)
  CHOCH_ALCISTA = "choch_alcista", // Change of Character (Bullish)
  CHOCH_BAJISTA = "choch_bajista", // Change of Character (Bearish)
  SWEEP_LIQUIDEZ = "sweep_liquidez", // Liquidity Sweep
  EQUILIBRIUM = "equilibrium", // 50% level of a range
  WEAK_HIGH = "weak_high", // A high that is likely to be taken
  WEAK_LOW = "weak_low", // A low that is likely to be taken
  STRONG_HIGH = "strong_high", // A high that is less likely to be taken soon
  STRONG_LOW = "strong_low", // A low that is less likely to be taken soon
  ENTRADA_LARGO = "entrada_largo",
  ENTRADA_CORTO = "entrada_corto",
  STOP_LOSS = "stop_loss",
  TAKE_PROFIT = "take_profit",
  RANGO_OPERATIVO = "rango_operativo", // Trading range
  NOTA_GENERAL = "nota_general", // General annotation on chart
  AI_W_SIGNAL_BULLISH = "ai_w_signal_bullish", // AI identified W-Signal (Bullish POI confirmation)
  AI_W_SIGNAL_BEARISH = "ai_w_signal_bearish", // AI identified W-Signal (Bearish POI confirmation)
}


export interface AnalysisPoint {
  tipo: AnalysisPointType | string;
  zona?: [number, number]; // For POIs, FVGs, Ranges -> [minPrice, maxPrice]
  nivel?: number;          // For liquidity levels, BOS, Choch, SL/TP -> price level
  label: string;
  descripcion?: string; // Optional fuller description
  temporalidad?: string; // e.g., "15M", "1H", "4H", "1D"
  mitigado?: boolean; // For POIs
  importancia?: "alta" | "media" | "baja";

  // Optional fields for direct chart markers
  marker_time?: number; // Unix timestamp (seconds) for the bar the marker should be on
  marker_position?: "aboveBar" | "belowBar" | "inBar";
  marker_shape?: "arrowUp" | "arrowDown" | "circle" | "square"; // Removed "diamond"
  marker_text?: string; // Short text for the marker (e.g., "W", "R")
  // marker_color?: string; // Color might be better handled by global settings per signal type
}

export interface TradeSetup {
  tipo: "largo" | "corto" | "ninguno";
  descripcion_entrada?: string; // Detailed entry condition: "Esperar mitigación de POI Demanda y ChoCh en LTF"
  punto_entrada_ideal?: number; // Specific entry price, if applicable
  zona_entrada?: [number, number]; // Entry zone [min, max]
  stop_loss: number;
  take_profit_1: number;
  take_profit_2?: number;
  take_profit_3?: number;
  razon_fundamental: string; // Why this setup is good
  confirmaciones_adicionales?: string[]; // e.g., ["RSI Divergence", "Volume Spike"]
  ratio_riesgo_beneficio?: string; // e.g., "1:3"
  calificacion_confianza?: "alta" | "media" | "baja"; // Confidence in the setup
}

export interface ScenarioAnalysis {
  nombre_escenario: string; // e.g., "Escenario Principal: Continuación Bajista"
  probabilidad: "alta" | "media" | "baja";
  descripcion_detallada: string;
  trade_setup_asociado?: TradeSetup;
  niveles_clave_de_invalidacion?: string; // e.g., "Cierre de 4H por encima de $2600"
}

export interface FibonacciLevel {
  level: number; // e.g., 0.618, 1.618 (as decimal) - Ensured this is number
  price: number;
  label: string; // e.g., "Retracement 61.8% (0.618)", "Extension 161.8% (1.618)"
}

export interface FibonacciAnalysis {
  descripcion_impulso: string; // e.g., "Impulso alcista desde $100 a $150 tras BOS en 1H."
  precio_inicio_impulso: number; // Price at point A
  precio_fin_impulso: number;    // Price at point B
  precio_fin_retroceso?: number;   // Price at point C (end of retracement from B, for extension calculation)
  niveles_retroceso: FibonacciLevel[];
  niveles_extension?: FibonacciLevel[]; // Optional if no clear retracement C identified for extension
}

export interface GeminiAnalysisResult {
  analisis_general: {
    simbolo: string;
    temporalidad_principal_analisis: string;
    fecha_analisis: string; // e.g., "YYYY-MM-DD HH:MM UTC"
    estructura_mercado_resumen: { // Summaries for different timeframes
      htf_1W?: string; // Higher Time Frame (e.g., Semanal)
      htf_1D?: string; // Daily
      mtf_4H?: string; // Medium Time Frame
      ltf_1H?: string; // Lower Time Frame
      ltf_15M?: string;
    };
    fase_wyckoff_actual?: string; // e.g., "Acumulación", "Distribución", "Redistribución"
    sesgo_direccional_general: "alcista" | "bajista" | "lateral" | "indefinido";
    comentario_volumen?: string; // e.g., "Volumen decreciente en el retroceso, sugiere debilidad vendedora."
    interpretacion_volumen_detallada?: string; // Added for detailed volume analysis by AI
    comentario_funding_rate_oi?: string; // New: For FR/OI conceptual analysis
  };
  puntos_clave_grafico: AnalysisPoint[];
  liquidez_importante: {
    buy_side: AnalysisPoint[];
    sell_side: AnalysisPoint[];
  };
  zonas_criticas_oferta_demanda: {
    oferta_clave: AnalysisPoint[];
    demanda_clave: AnalysisPoint[];
    fvg_importantes: AnalysisPoint[];
  };
  analisis_fibonacci?: FibonacciAnalysis; // Added for Fibonacci analysis
  escenarios_probables: ScenarioAnalysis[];
  conclusion_recomendacion: {
    resumen_ejecutivo: string;
    proximo_movimiento_esperado: string;
    mejor_oportunidad_actual?: TradeSetup;
    advertencias_riesgos?: string;
    oportunidades_reentrada_detectadas?: string; // New: For re-entry signal concepts
    consideraciones_salida_trade?: string; // New: For exit signal concepts
    senales_confluencia_avanzada?: string; // New: For "Yellow" signal type concepts
  };
}


// For Gemini API interaction
export interface GeminiRequestPayload {
  symbol: string;
  timeframe: string; // Primary timeframe for analysis request
  currentPrice: number;
  marketContextPrompt: string; // The detailed context constructed for the prompt
  historicalDataSummary?: string; // Optional: a brief summary of recent price action if available
}

export type DataSource = 'binance' | 'bingx';

export interface TickerData {
    provider: DataSource;
    symbol: string;
    price?: number;
    changePercent?: number;
    volume?: number; // Volume in base currency
    quoteVolume?: number; // Volume in quote currency
    lastPriceChange?: 'up' | 'down' | 'none';
}

// For Lightweight Charts
export interface ChartCandlestickData {
  time: number; // UTCTimestamp
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number; // Optional volume data for candlestick series
}

export interface ChartLineData {
  time: number; // UTCTimestamp
  value: number;
}

export interface ChartHistogramData {
  time: number; // UTCTimestamp
  value: number;
  color?: string;
}

// Configuration for Moving Averages
export interface MovingAverageConfig {
  id: string; // Unique ID, e.g., 'ma1', 'ema2'
  type: 'MA' | 'EMA'; // Type of moving average
  period: number;
  color: string;
  visible: boolean;
}