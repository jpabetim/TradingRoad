import React, { useState, useEffect, useCallback, useRef } from 'react';
import RealTimeTradingChart from './components/RealTimeTradingChart';
import AnalysisPanel from './components/AnalysisPanel';
import ApiKeyMessage from './components/ApiKeyMessage';
import DisplaySettingsDialog from './components/DisplaySettingsDialog';
import MarketDashboard from './components/MarketDashboard';
import NewsPanel from './components/NewsPanel';
import TestConnectivity from './components/TestConnectivity';
import { GeminiAnalysisResult, DataSource, MovingAverageConfig } from './types';
import { analyzeChartWithGemini, ExtendedGeminiRequestPayload } from './services/geminiService';
import { DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_DATA_SOURCE, AVAILABLE_DATA_SOURCES, AVAILABLE_TIMEFRAMES, AVAILABLE_SYMBOLS_BINANCE, AVAILABLE_SYMBOLS_BINGX } from './constants';

// Helper for debouncing
function debounce<T extends (...args: any[]) => void>(func: T, delay: number): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  return function (this: ThisParameterType<T>, ...args: Parameters<T>) {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
}

interface LatestChartInfo {
  price: number | null;
  volume?: number | null;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  timestamp: number;
}

type Theme = 'dark' | 'light';
export type AnalysisPanelMode = 'initial' | 'analysis' | 'chat';
type AppView = 'trading' | 'dashboard' | 'test';

const initialMAs: MovingAverageConfig[] = [
  { id: 'ma1', type: 'EMA', period: 12, color: '#34D399', visible: true },
  { id: 'ma2', type: 'EMA', period: 20, color: '#F472B6', visible: true },
  { id: 'ma3', type: 'MA', period: 50, color: '#CBD5E1', visible: true },
  { id: 'ma4', type: 'MA', period: 200, color: '#FF0000', visible: true },
];

const INITIAL_DARK_CHART_PANE_BACKGROUND_COLOR = '#18191B';
const INITIAL_LIGHT_CHART_PANE_BACKGROUND_COLOR = '#FFFFFF';
const INITIAL_VOLUME_PANE_HEIGHT = 0;
const INITIAL_W_SIGNAL_COLOR = '#243EA8';
const INITIAL_W_SIGNAL_OPACITY = 70;
const INITIAL_SHOW_W_SIGNALS = true;

const getLocalStorageItem = <T,>(key: string, defaultValue: T): T => {
  if (typeof window !== 'undefined' && window.localStorage) {
    const storedValue = localStorage.getItem(key);
    if (storedValue) {
      try {
        return JSON.parse(storedValue) as T;
      } catch (e) {
        console.error(`Error parsing localStorage item ${key}:`, e);
        return defaultValue;
      }
    }
  }
  return defaultValue;
};

const getConsistentSymbolForDataSource = (symbol: string, ds: DataSource): string => {
  let consistentSymbol = symbol.toUpperCase();
  if (ds === 'bingx') {
    if (consistentSymbol === 'BTCUSDT') return 'BTC-USDT';
    if (consistentSymbol === 'ETHUSDT') return 'ETH-USDT';
    if (consistentSymbol === 'SOLUSDT') return 'SOL-USDT';
  } else if (ds === 'binance') {
    if (consistentSymbol === 'BTC-USDT') return 'BTCUSDT';
    if (consistentSymbol === 'ETH-USDT') return 'ETHUSDT';
    if (consistentSymbol === 'SOL-USDT') return 'SOLUSDT';
  }
  return consistentSymbol;
};


const App: React.FC = () => {
  // Navigation state
  const [currentView, setCurrentView] = useState<AppView>('dashboard'); // Empezar con dashboard

  const initialRawSymbol = getLocalStorageItem('traderoad_actualSymbol', DEFAULT_SYMBOL);
  const initialDataSource = getLocalStorageItem('traderoad_dataSource', DEFAULT_DATA_SOURCE);
  const consistentInitialSymbol = getConsistentSymbolForDataSource(initialRawSymbol, initialDataSource);

  const [dataSource, setDataSource] = useState<DataSource>(initialDataSource);
  const [actualSymbol, setActualSymbol] = useState<string>(consistentInitialSymbol);
  const [symbolInput, setSymbolInput] = useState<string>(consistentInitialSymbol);
  const [timeframe, setTimeframe] = useState<string>(() => getLocalStorageItem('traderoad_timeframe', DEFAULT_TIMEFRAME));
  const [theme, setTheme] = useState<Theme>(() => getLocalStorageItem('traderoad_theme', 'dark'));
  const [movingAverages, setMovingAverages] = useState<MovingAverageConfig[]>(() => getLocalStorageItem('traderoad_movingAverages', initialMAs));

  const initialBgColorBasedOnTheme = theme === 'dark' ? INITIAL_DARK_CHART_PANE_BACKGROUND_COLOR : INITIAL_LIGHT_CHART_PANE_BACKGROUND_COLOR;
  const [chartPaneBackgroundColor, setChartPaneBackgroundColor] = useState<string>(() =>
    getLocalStorageItem('traderoad_chartPaneBackgroundColor', initialBgColorBasedOnTheme)
  );

  const [volumePaneHeight, setVolumePaneHeight] = useState<number>(() => getLocalStorageItem('traderoad_volumePaneHeight', INITIAL_VOLUME_PANE_HEIGHT));
  const [showAiAnalysisDrawings, setShowAiAnalysisDrawings] = useState<boolean>(() => getLocalStorageItem('traderoad_showAiAnalysisDrawings', true));
  const [isPanelVisible, setIsPanelVisible] = useState<boolean>(() => getLocalStorageItem('traderoad_isPanelVisible', true));
  const [wSignalColor, setWSignalColor] = useState<string>(() => getLocalStorageItem('traderoad_wSignalColor', INITIAL_W_SIGNAL_COLOR));
  const [wSignalOpacity, setWSignalOpacity] = useState<number>(() => getLocalStorageItem('traderoad_wSignalOpacity', INITIAL_W_SIGNAL_OPACITY));
  const [showWSignals, setShowWSignals] = useState<boolean>(() => getLocalStorageItem('traderoad_showWSignals', INITIAL_SHOW_W_SIGNALS));

  const [apiKey, setApiKey] = useState<string | null>(null);
  const [apiKeyPresent, setApiKeyPresent] = useState<boolean>(false);
  const [displaySettingsDialogOpen, setDisplaySettingsDialogOpen] = useState<boolean>(false);

  const [analysisResult, setAnalysisResult] = useState<GeminiAnalysisResult | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const [latestChartInfo, setLatestChartInfo] = useState<LatestChartInfo>({ price: null, volume: null });
  const [isChartLoading, setIsChartLoading] = useState<boolean>(true);
  const [chartOhlcData, setChartOhlcData] = useState<any[]>([]); // Store OHLC data from chart
  const [isMobile, setIsMobile] = useState<boolean>(false);

  const [analysisPanelMode, setAnalysisPanelMode] = useState<AnalysisPanelMode>('initial');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string | null>(null);

  // Add a new state and ref for the dropdown visibility
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const symbolDropdownRef = useRef<HTMLDivElement>(null);
  const symbolInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem('traderoad_dataSource', JSON.stringify(dataSource));
      localStorage.setItem('traderoad_actualSymbol', JSON.stringify(actualSymbol));
      localStorage.setItem('traderoad_timeframe', JSON.stringify(timeframe));
      localStorage.setItem('traderoad_theme', JSON.stringify(theme));
      localStorage.setItem('traderoad_movingAverages', JSON.stringify(movingAverages));
      localStorage.setItem('traderoad_chartPaneBackgroundColor', JSON.stringify(chartPaneBackgroundColor));
      localStorage.setItem('traderoad_volumePaneHeight', JSON.stringify(volumePaneHeight));
      localStorage.setItem('traderoad_showAiAnalysisDrawings', JSON.stringify(showAiAnalysisDrawings));
      localStorage.setItem('traderoad_isPanelVisible', JSON.stringify(isPanelVisible));
      localStorage.setItem('traderoad_wSignalColor', JSON.stringify(wSignalColor));
      localStorage.setItem('traderoad_wSignalOpacity', JSON.stringify(wSignalOpacity));
      localStorage.setItem('traderoad_showWSignals', JSON.stringify(showWSignals));
    }
  }, [
    dataSource, actualSymbol, timeframe, theme, movingAverages,
    chartPaneBackgroundColor, volumePaneHeight, showAiAnalysisDrawings, isPanelVisible,
    wSignalColor, wSignalOpacity, showWSignals
  ]);

  useEffect(() => {
    setIsMobile(typeof navigator !== 'undefined' && /Mobi|Android/i.test(navigator.userAgent));
    let keyFromEnv: string | undefined = undefined;

    // Debug: mostrar todas las variables de entorno disponibles
    console.log("🔑 Debug API Keys:", {
      VITE_GEMINI_API_KEY: import.meta.env.VITE_GEMINI_API_KEY,
      allEnv: import.meta.env
    });

    if (typeof import.meta.env.VITE_GEMINI_API_KEY === 'string') {
      keyFromEnv = import.meta.env.VITE_GEMINI_API_KEY;
    }
    if (keyFromEnv && keyFromEnv !== "TU_CLAVE_API_DE_GEMINI_AQUI") {
      setApiKey(keyFromEnv);
      setApiKeyPresent(true);
      console.log("✅ API Key configurada correctamente");
    } else {
      setApiKey(null);
      setApiKeyPresent(false);
      console.warn("❌ Gemini API Key no configurada o es valor placeholder. Análisis IA deshabilitado.");
      console.warn("Valor recibido:", keyFromEnv);
    }
  }, []);

  const getSymbolSuggestions = () => {
    if (dataSource === 'bingx') return AVAILABLE_SYMBOLS_BINGX;
    return AVAILABLE_SYMBOLS_BINANCE;
  };

  const getSymbolPlaceholder = () => {
    if (dataSource === 'bingx') return 'Ej: BTC-USDT';
    return 'Ej: BTCUSDT';
  };

  const debouncedSetActualSymbol = useCallback(
    debounce((newSymbol: string) => {
      const consistentTypedSymbol = getConsistentSymbolForDataSource(newSymbol.trim(), dataSource);
      setActualSymbol(consistentTypedSymbol);
      if (consistentTypedSymbol !== newSymbol.trim()) {
        setSymbolInput(consistentTypedSymbol);
      }
    }, 750),
    [dataSource]
  );

  const handleSymbolInputChange = (newInputValue: string) => {
    // Convertir a mayúsculas para consistencia
    const upperValue = newInputValue.toUpperCase();
    setSymbolInput(upperValue);

    // Mantener el dropdown visible mientras se está escribiendo
    setDropdownVisible(true);

    // Si el valor está vacío, no cambiar el símbolo
    if (!upperValue.trim()) return;

    // No debounce para permitir que el usuario escriba libremente
    // La actualización del símbolo se hará al hacer clic en un elemento de la lista
    // o al presionar el botón de confirmar en la sección de símbolos personalizados
  };

  useEffect(() => {
    if (symbolInput !== actualSymbol) {
      setSymbolInput(actualSymbol);
    }
  }, [actualSymbol]);


  useEffect(() => {
    setAnalysisResult(null);
    setAnalysisError(null);
    setAnalysisPanelMode('initial'); // Reset to initial to avoid showing stale analysis for new symbol
  }, [actualSymbol, dataSource]);


  useEffect(() => {
    setLatestChartInfo({ price: null, volume: null });
  }, [actualSymbol, timeframe, dataSource]);

  useEffect(() => {
    const newThemeDefaultBgColor = theme === 'dark' ? INITIAL_DARK_CHART_PANE_BACKGROUND_COLOR : INITIAL_LIGHT_CHART_PANE_BACKGROUND_COLOR;
    const isCurrentBgThemeDefault =
      chartPaneBackgroundColor === INITIAL_DARK_CHART_PANE_BACKGROUND_COLOR ||
      chartPaneBackgroundColor === INITIAL_LIGHT_CHART_PANE_BACKGROUND_COLOR;

    if (isCurrentBgThemeDefault && chartPaneBackgroundColor !== newThemeDefaultBgColor) {
      setChartPaneBackgroundColor(newThemeDefaultBgColor);
    }
  }, [theme, chartPaneBackgroundColor]);

  const handleLatestChartInfoUpdate = useCallback((info: LatestChartInfo) => setLatestChartInfo(info), []);
  const handleChartLoadingStateChange = useCallback((chartLoading: boolean) => setIsChartLoading(chartLoading), []);
  const handleHistoricalDataUpdate = useCallback((data: any[]) => {
    console.log('📈 Datos históricos recibidos del gráfico:', { count: data.length, sample: data.slice(-2) });
    setChartOhlcData(data);
  }, []);

  const handleRequestAnalysis = useCallback(async () => {
    if (!apiKey) {
      setAnalysisError("Clave API no configurada. El análisis no puede proceder.");
      setAnalysisPanelMode('analysis');
      return;
    }
    if (isChartLoading || latestChartInfo.price === null || latestChartInfo.price === 0) {
      setAnalysisError("Datos del gráfico cargando o precio actual no disponible.");
      setAnalysisPanelMode('analysis');
      return;
    }

    // If switching TO analysis mode AND a result exists for the current context, just show it.
    if (analysisPanelMode !== 'analysis' && analysisResult) {
      setAnalysisPanelMode('analysis');
      setAnalysisLoading(false); // Ensure loading is off if we're just switching views
      setAnalysisError(null);
      return;
    }

    // Otherwise (already in analysis mode OR no result exists), fetch new analysis.
    setAnalysisLoading(true);
    setAnalysisError(null);
    setAnalysisResult(null); // Clear previous result before fetching new one
    setAnalysisPanelMode('analysis'); // Ensure mode is set

    try {
      const displaySymbolForAI = actualSymbol.includes('-') ? actualSymbol.replace('-', '/') : (actualSymbol.endsWith('USDT') ? actualSymbol.replace(/USDT$/, '/USDT') : actualSymbol);
      const payload: ExtendedGeminiRequestPayload = {
        symbol: displaySymbolForAI, timeframe: timeframe.toUpperCase(), currentPrice: latestChartInfo.price,
        marketContextPrompt: "Context will be generated by getFullAnalysisPrompt",
        latestVolume: latestChartInfo.volume, apiKey: apiKey
      };
      const result = await analyzeChartWithGemini(payload);
      setAnalysisResult(result);
    } catch (err) {
      let userErrorMessage = (err instanceof Error) ? err.message : "Ocurrió un error desconocido.";
      setAnalysisError(`${userErrorMessage} --- Revisa la consola para más detalles.`);
    } finally {
      setAnalysisLoading(false);
    }
  }, [apiKey, actualSymbol, timeframe, latestChartInfo, isChartLoading, isMobile, analysisResult, analysisPanelMode]);

  const handleShowChat = () => {
    setAnalysisPanelMode('chat');
    setChatError(null);
    setIsPanelVisible(true); // Asegurar que el panel esté visible
    if (!apiKeyPresent) {
      setChatError("Clave API no configurada. El Chat IA no está disponible.");
    }
  };

  const handleSendMessageToChat = async (messageText: string) => {
    if (!messageText.trim() || chatLoading) return;

    if (!apiKeyPresent) {
      setChatError("API key no configurada. El Chat IA no está disponible.");
      return;
    }

    const displaySymbolForAI = actualSymbol.includes('-') ? actualSymbol.replace('-', '/') : (actualSymbol.endsWith('USDT') ? actualSymbol.replace(/USDT$/, '/USDT') : actualSymbol);

    // Debug: verificar los datos OHLC disponibles
    console.log('📊 Debug OHLC Data:', {
      totalCandles: chartOhlcData.length,
      sampleData: chartOhlcData.slice(-3), // últimas 3 velas para debug
      isLoading: isChartLoading
    });

    // Preparar datos OHLC para la IA (últimas 50-100 velas para no saturar)
    const recentOhlcData = chartOhlcData.slice(-50).map(candle => {
      try {
        // Asegurar que el tiempo esté en formato ISO string correcto
        const timeInMs = typeof candle.time === 'number' ?
          (candle.time > 1e12 ? candle.time : candle.time * 1000) : // Si ya está en ms, usar tal como está
          Date.now();

        return {
          time: new Date(timeInMs).toISOString(),
          open: parseFloat(candle.open) || 0,
          high: parseFloat(candle.high) || 0,
          low: parseFloat(candle.low) || 0,
          close: parseFloat(candle.close) || 0,
          volume: parseFloat(candle.volume) || 0
        };
      } catch (error) {
        console.error('Error procesando vela:', candle, error);
        return null;
      }
    }).filter(Boolean); // Filtrar valores null

    console.log('🤖 Datos enviados a IA:', {
      symbol: displaySymbolForAI,
      candlesCount: recentOhlcData.length,
      latestPrice: latestChartInfo.price,
      sampleOhlc: recentOhlcData.slice(-2) // últimas 2 velas procesadas
    });

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      sender: 'user',
      text: messageText.trim(),
      timestamp: Date.now(),
    };
    setChatMessages((prevMessages) => [...prevMessages, userMessage]);
    setChatLoading(true);
    setChatError(null);

    const currentAiMessageId = crypto.randomUUID();
    setChatMessages((prevMessages) => [
      ...prevMessages,
      { id: currentAiMessageId, sender: 'ai', text: "Escribiendo...", timestamp: Date.now() },
    ]);

    try {
      // ========= CORRECCIÓN IMPORTANTE AQUÍ =========
      // Usamos la variable de entorno para la URL del backend en producción.
      const backendUrl = import.meta.env.VITE_BACKEND_URL;
      if (!backendUrl) {
        throw new Error("La URL del backend no está configurada. Revisa VITE_BACKEND_URL.");
      }

      // La llamada fetch ahora usa la URL completa.
      // NOTA: He quitado el prefijo "/api" asumiendo que tu backend en Flask no lo usa.
      // Si tus rutas en Flask SÍ empiezan por /api, la URL debería ser: `${backendUrl}/api/ai/assistant`
      const response = await fetch(`${backendUrl}/ai/assistant`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageText.trim(), // Solo la pregunta del usuario
          section: 'analysis',
          context: {
            symbol: displaySymbolForAI,
            timeframe: timeframe.toUpperCase(),
            price: latestChartInfo.price,
            volume: latestChartInfo.volume,
            isLoading: isChartLoading,
            dataSource: dataSource.toUpperCase(),
            ohlcData: recentOhlcData, // Los datos OHLC van aquí
            candlesCount: recentOhlcData.length,
            analysis: analysisResult
          }
        }),
      });
      // ===============================================

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setChatMessages((prevMessages) =>
        prevMessages.map((msg) =>
          msg.id === currentAiMessageId ? { ...msg, text: data.response } : msg
        )
      );

    } catch (e: any) {
      console.error("Error sending message to AI Assistant:", e);
      const errorMessage = `Error comunicándose con la IA: ${e.message}`;
      setChatError(errorMessage);
      setChatMessages((prevMessages) =>
        prevMessages.map((msg) =>
          msg.id === currentAiMessageId ? { ...msg, text: `Error: ${e.message}` } : msg
        )
      );
    } finally {
      setChatLoading(false);
    }
  };

  const handleClearChatHistory = () => {
    setChatMessages([]);
    setChatError(null);
  };


  const handleDataSourceChange = (newDataSource: DataSource) => {
    setDataSource(newDataSource);
    const symbolToConvert = symbolInput || actualSymbol;
    const consistentNewSymbol = getConsistentSymbolForDataSource(symbolToConvert, newDataSource);
    setActualSymbol(consistentNewSymbol);
    setSymbolInput(consistentNewSymbol);
  };

  const toggleAllMAsVisibility = (forceVisible?: boolean) => {
    const newVisibility = typeof forceVisible === 'boolean'
      ? forceVisible
      : !movingAverages.every(ma => ma.visible);
    setMovingAverages(prevMAs => prevMAs.map(ma => ({ ...ma, visible: newVisibility })));
  };

  // Add a click outside handler for the dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        symbolDropdownRef.current &&
        symbolInputRef.current &&
        !symbolDropdownRef.current.contains(event.target as Node) &&
        !symbolInputRef.current.contains(event.target as Node)
      ) {
        setDropdownVisible(false);
      }
    }

    // Close dropdown on Escape key
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setDropdownVisible(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  // Add keyboard navigation support for the dropdown
  const handleKeyboardNavigation = (event: React.KeyboardEvent) => {
    if (!dropdownVisible) return;

    if (event.key === 'Enter') {
      // If the input matches exactly one of the filtered symbols, select it
      const filtered = filteredSymbols();
      if (filtered.length === 1) {
        handleSelectSymbol(filtered[0]);
      } else if (symbolInput) {
        // Otherwise just use the current input
        debouncedSetActualSymbol(symbolInput);
        setDropdownVisible(false);
      }
    }
  };

  // Filter symbols based on input - always show all symbols initially
  const filteredSymbols = () => {
    const suggestions = getSymbolSuggestions();
    // Don't filter when dropdown is first opened
    if (!symbolInput || !symbolInput.trim() || !initialFilterApplied) return suggestions;
    return suggestions.filter(s => s.toLowerCase().includes(symbolInput.toLowerCase()));
  };

  // Track if initial filter has been applied
  const [initialFilterApplied, setInitialFilterApplied] = useState(false);

  // Handle symbol selection from dropdown
  const handleSelectSymbol = (symbol: string) => {
    setSymbolInput(symbol);
    debouncedSetActualSymbol(symbol);
    setDropdownVisible(false);
  };

  return (
    <div className={`flex flex-col h-screen antialiased ${theme === 'dark' ? 'bg-slate-900 text-slate-100' : 'bg-gray-100 text-gray-900'}`}>
      <header className={`p-2 sm:p-3 shadow-md flex justify-between items-center flex-nowrap gap-4 ${theme === 'dark' ? 'bg-slate-800' : 'bg-white border-b border-gray-200'}`}>
        {/* Left side: Logo, Title and market controls */}
        <div className="flex items-center gap-2 sm:gap-4 flex-wrap">
          <div
            className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            // ========= CORRECCIÓN IMPORTANTE AQUÍ =========
            // En lugar de redirigir a otra página, cambiamos la vista interna de React
            onClick={() => setCurrentView('dashboard')}
            // ===============================================
          >
            <img
              src="/logo-tradingroad.png"
              alt="TradingRoad Logo"
              className="w-8 h-8 sm:w-10 sm:h-10"
            />
            <h1 className={`text-lg sm:text-xl font-bold ${theme === 'dark' ? 'text-sky-400' : 'text-sky-600'} flex-shrink-0`}>TradingRoad</h1>
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            {/* Data Source */}
            <div className="w-32 sm:w-36">
              <label htmlFor="dataSource-header" className="sr-only">Fuente de Datos</label>
              <select
                id="dataSource-header"
                value={dataSource}
                onChange={(e) => handleDataSourceChange(e.target.value as DataSource)}
                className={`w-full text-xs rounded-md p-1.5 border focus:ring-1 focus:outline-none transition-colors ${theme === 'dark'
                  ? 'bg-slate-700 border-slate-600 text-slate-100 focus:ring-sky-500'
                  : 'bg-gray-100 border-gray-300 text-gray-800 focus:ring-sky-500'
                  }`}
                aria-label="Fuente de Datos"
              >
                {AVAILABLE_DATA_SOURCES.map(ds => <option key={ds.value} value={ds.value}>{ds.label}</option>)}
              </select>
            </div>

            {/* Symbol - Selector rediseñado con dropdown y campo de texto separados */}
            <div className="w-36 sm:w-40 relative">
              <label htmlFor="symbol-input-header" className="sr-only">Símbolo</label>

              {/* Panel con simbolo actual y botones de acción */}
              <div className={`flex rounded-md border overflow-hidden ${theme === 'dark'
                ? 'bg-slate-700 border-slate-600'
                : 'bg-gray-100 border-gray-300'}`}>

                {/* Visualización del símbolo actual */}
                <div className={`flex items-center justify-between px-2 py-1.5 flex-grow ${theme === 'dark'
                  ? 'text-white font-medium'
                  : 'text-gray-900 font-medium'}`}>
                  <span className="text-xs truncate">
                    {symbolInput || getSymbolPlaceholder()}
                  </span>
                </div>

                {/* Botón desplegable */}
                <button
                  onClick={() => {
                    setDropdownVisible(!dropdownVisible);
                    if (!dropdownVisible) {
                      // When opening the dropdown, always show all symbols
                      setInitialFilterApplied(false);
                    }
                  }}
                  className={`flex items-center justify-center px-2 border-l ${theme === 'dark'
                    ? 'bg-slate-800 border-slate-600 text-white hover:bg-slate-900'
                    : 'bg-gray-200 border-gray-300 text-gray-800 hover:bg-gray-300'
                    }`}
                  aria-label="Ver lista de símbolos"
                  aria-expanded={dropdownVisible}
                  aria-haspopup="listbox"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3 h-3 flex-shrink-0">
                    <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>

              {/* Dropdown panel con búsqueda y opciones */}
              {dropdownVisible && (
                <div
                  ref={symbolDropdownRef}
                  className={`absolute z-10 w-64 sm:w-72 left-0 shadow-lg rounded-md mt-1 overflow-hidden focus:outline-none ${theme === 'dark'
                    ? 'bg-slate-900 border border-slate-700'
                    : 'bg-white border border-gray-300'
                    }`}
                  role="listbox"
                >
                  {/* Encabezado del dropdown */}
                  <div className={`p-2 ${theme === 'dark' ? 'bg-slate-700' : 'bg-gray-100'}`}>
                    <h3 className={`text-xs font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                      Seleccionar símbolo
                    </h3>
                  </div>

                  {/* Campo de búsqueda dentro del dropdown */}
                  <div className={`p-2 border-b ${theme === 'dark' ? 'border-slate-700' : 'border-gray-200'}`}>
                    <input
                      type="text"
                      ref={symbolInputRef}
                      value={symbolInput}
                      onChange={(e) => {
                        handleSymbolInputChange(e.target.value);
                        setInitialFilterApplied(true);
                      }}
                      onKeyDown={handleKeyboardNavigation}
                      onClick={(e) => e.stopPropagation()}
                      placeholder={getSymbolPlaceholder()}
                      className={`w-full text-xs p-1.5 border rounded-md focus:ring-1 focus:outline-none ${theme === 'dark'
                        ? 'bg-slate-700 border-slate-600 text-white focus:ring-sky-500 placeholder-slate-400'
                        : 'bg-white border-gray-300 text-gray-800 focus:ring-sky-500 placeholder-gray-400'
                        }`}
                      autoComplete="off"
                    />
                  </div>

                  {/* Lista de símbolos */}
                  <div className={`overflow-y-auto max-h-60 ${theme === 'dark' ? 'bg-slate-800' : 'bg-white'}`}>
                    {filteredSymbols().length > 0 ? (
                      filteredSymbols().map(symbol => (
                        <div
                          key={symbol}
                          onClick={() => handleSelectSymbol(symbol)}
                          className={`cursor-pointer select-none py-2 px-3 transition-colors ${symbol === symbolInput
                              ? theme === 'dark' ? 'bg-sky-700 text-white font-bold' : 'bg-sky-500 text-white font-bold'
                              : theme === 'dark' ? 'bg-slate-800 text-white hover:bg-slate-700' : 'bg-white text-gray-900 hover:bg-gray-100'
                            }`}
                          role="option"
                          aria-selected={symbol === symbolInput}
                        >
                          <span className="block truncate text-sm font-medium">
                            {symbol}
                          </span>
                        </div>
                      ))
                    ) : (
                      <div className={`py-3 px-3 text-center ${theme === 'dark' ? 'text-slate-300 bg-slate-800' : 'text-gray-700 bg-gray-100'}`}>
                        No hay coincidencias
                      </div>
                    )}
                  </div>

                  {/* Sección para símbolos personalizados */}
                  <div className={`p-2 border-t ${theme === 'dark' ? 'border-slate-700 bg-slate-700' : 'border-gray-200 bg-gray-100'}`}>
                    <h3 className={`text-xs font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                      Símbolo personalizado
                    </h3>
                    <div className="flex gap-1">
                      <input
                        type="text"
                        value={symbolInput}
                        onChange={(e) => {
                          handleSymbolInputChange(e.target.value);
                          setInitialFilterApplied(true);
                        }}
                        onKeyDown={handleKeyboardNavigation}
                        placeholder="Escribir símbolo..."
                        className={`flex-grow text-xs p-1.5 border rounded-md focus:ring-1 focus:outline-none ${theme === 'dark'
                          ? 'bg-slate-800 border-slate-600 text-white focus:ring-sky-500 placeholder-slate-400'
                          : 'bg-white border-gray-300 text-gray-800 focus:ring-sky-500 placeholder-gray-400'
                          }`}
                        autoComplete="off"
                      />
                      <button
                        onClick={() => {
                          if (symbolInput) {
                            debouncedSetActualSymbol(symbolInput);
                            setDropdownVisible(false);
                          }
                        }}
                        className={`px-2 py-1 rounded ${theme === 'dark'
                          ? 'bg-sky-600 text-white hover:bg-sky-700'
                          : 'bg-sky-500 text-white hover:bg-sky-600'
                          }`}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3 h-3">
                          <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Timeframe */}
            <div className="w-16 sm:w-20">
              <label htmlFor="timeframe-header" className="sr-only">Temporalidad</label>
              <select
                id="timeframe-header"
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                className={`w-full text-xs rounded-md p-1.5 border focus:ring-1 focus:outline-none transition-colors ${theme === 'dark'
                  ? 'bg-slate-700 border-slate-600 text-slate-100 focus:ring-sky-500'
                  : 'bg-gray-100 border-gray-300 text-gray-800 focus:ring-sky-500'
                  }`}
                aria-label="Temporalidad"
              >
                {AVAILABLE_TIMEFRAMES.map(tf => (
                  <option key={tf} value={tf}>{tf.toUpperCase()}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Right side: Navigation, AI buttons and control buttons */}
        <div className="flex items-center gap-1 sm:gap-2">
          {/* Navigation buttons */}
          <button
            onClick={() => setCurrentView('dashboard')}
            className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded text-xs font-medium transition-colors ${
              currentView === 'dashboard'
                ? (theme === 'dark' ? 'bg-sky-600 text-white' : 'bg-sky-500 text-white')
                : (theme === 'dark' ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-700')
            }`}
          >
            Dashboard
          </button>
          
          <button
            onClick={() => setCurrentView('trading')}
            className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded text-xs font-medium transition-colors ${
              currentView === 'trading'
                ? (theme === 'dark' ? 'bg-sky-600 text-white' : 'bg-sky-500 text-white')
                : (theme === 'dark' ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-700')
            }`}
          >
            Trading
          </button>

          <button
            onClick={() => setCurrentView('test')}
            className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded text-xs font-medium transition-colors ${
              currentView === 'test'
                ? (theme === 'dark' ? 'bg-orange-600 text-white' : 'bg-orange-500 text-white')
                : (theme === 'dark' ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-700')
            }`}
          >
            Test API
          </button>

          {/* Separator */}
          <div className={`w-px h-6 ${theme === 'dark' ? 'bg-slate-600' : 'bg-gray-300'}`}></div>

          {/* AI Analysis Button - Solo visible en vista Trading */}
          {currentView === 'trading' && (
            <>
              <button
                onClick={handleRequestAnalysis}
                disabled={analysisLoading || chatLoading || !apiKeyPresent || isChartLoading}
                title="Análisis IA del gráfico"
                className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded text-xs font-medium transition-colors ${theme === 'dark'
                  ? 'bg-blue-600 hover:bg-blue-700 text-white disabled:bg-slate-600 disabled:text-slate-400'
                  : 'bg-blue-500 hover:bg-blue-600 text-white disabled:bg-gray-300 disabled:text-gray-500'
                  }`}
              >
                {analysisLoading ? 'Analizando...' : 'Análisis IA'}
              </button>

              {/* AI Assistant Button */}
              <button
                onClick={handleShowChat}
                disabled={analysisLoading || chatLoading || !apiKeyPresent}
                title="Asistente IA de trading"
                className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded text-xs font-medium transition-colors ${theme === 'dark'
                  ? 'bg-green-600 hover:bg-green-700 text-white disabled:bg-slate-600 disabled:text-slate-400'
                  : 'bg-green-500 hover:bg-green-600 text-white disabled:bg-gray-300 disabled:text-gray-500'
                  }`}
              >
                {chatLoading ? 'Pensando...' : 'Asistente IA'}
              </button>

              {/* Separator */}
              <div className={`w-px h-6 ${theme === 'dark' ? 'bg-slate-600' : 'bg-gray-300'}`}></div>
            </>
          )}

          <button
            onClick={() => setIsPanelVisible(!isPanelVisible)}
            aria-label={isPanelVisible ? 'Ocultar panel de controles' : 'Mostrar panel de controles'}
            className={`p-1.5 sm:p-2 rounded text-xs transition-colors ${theme === 'dark' ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-700'}`}
          >
            {isPanelVisible ? 'Ocultar Panel' : 'Mostrar Panel'}
          </button>

          <button
            onClick={() => setDisplaySettingsDialogOpen(true)}
            title="Configuración de Visualización"
            aria-label="Abrir Configuración de Visualización"
            className={`p-1.5 sm:p-2 rounded text-xs transition-colors ${theme === 'dark' ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-700'}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4 sm:w-5 sm:h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
            </svg>
          </button>
          <button
            onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            title={`Cambiar a tema ${theme === 'light' ? 'oscuro' : 'claro'}`}
            aria-label={`Cambiar a tema ${theme === 'light' ? 'oscuro' : 'claro'}`}
            className={`p-1.5 sm:p-2 rounded text-xs transition-colors ${theme === 'dark' ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-700'}`}
          >
            {theme === 'light' ? (
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4 sm:w-5 sm:h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4 sm:w-5 sm:h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
              </svg>
            )}
          </button>

          {/* Chart Tools Toggle */}
          <button
            onClick={() => setShowAiAnalysisDrawings(!showAiAnalysisDrawings)}
            title={showAiAnalysisDrawings ? "Ocultar herramientas de análisis" : "Mostrar herramientas de análisis"}
            className={`p-1.5 sm:p-2 rounded text-xs transition-colors ${showAiAnalysisDrawings
              ? (theme === 'dark' ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-blue-500 hover:bg-blue-600 text-white')
              : (theme === 'dark' ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-700')
              }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4 sm:w-5 sm:h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.53 16.122a3 3 0 00-5.78 1.128 2.25 2.25 0 01-2.4 0 3 3 0 00-5.78-1.128 2.25 2.25 0 010-4.244 3 3 0 005.78-1.128 2.25 2.25 0 012.4 0 3 3 0 005.78 1.128 2.25 2.25 0 010 4.244A3 3 0 009.53 16.122zm0 0V11m0 0A2.25 2.25 0 007.28 8.75m2.25 2.25A2.25 2.25 0 0112 13.5m0 0V7.28A2.25 2.25 0 0114.22 5m0 0a2.25 2.25 0 013.5 2.25m0 0V11a2.25 2.25 0 01-2.25 2.25m0 0h-3.75M14.22 5h3.75" />
            </svg>
          </button>
        </div>
      </header>

      <ApiKeyMessage apiKeyPresent={apiKeyPresent} />

      <main className="flex-1 flex flex-col gap-2 sm:gap-4 md:gap-6 overflow-hidden min-h-0">
        {/* VISTA DASHBOARD: Solo muestra el dashboard de mercado y noticias */}
        {currentView === 'dashboard' && (
          <div className="flex flex-col gap-4 h-full w-full">
            <h1 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              Dashboard de Mercado
            </h1>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Panel principal - Dashboard de mercado */}
              <div className="lg:col-span-2 space-y-6">
                <div className={`p-6 rounded-lg shadow-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-white'}`}>
                  <MarketDashboard />
                </div>
              </div>
              
              {/* Panel lateral - Noticias */}
              <div className="space-y-6">
                <div className={`p-6 rounded-lg shadow-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-white'}`}>
                  <h2 className={`text-xl font-bold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                    Noticias del Mercado
                  </h2>
                  <NewsPanel />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VISTA TRADING: Solo muestra el gráfico y controles de trading */}
        {currentView === 'trading' && (
          <div className="flex flex-col md:flex-row gap-4 h-full">
            {/* Contenido principal - Gráfico y controles */}
            <div className="flex-1 flex flex-col gap-4">
              <div className={`flex-grow min-h-[400px] sm:min-h-[500px] md:min-h-[600px] shadow-lg rounded-lg overflow-hidden ${theme === 'dark' ? 'bg-slate-800' : 'bg-white'}`}>
                <RealTimeTradingChart
                  dataSource={dataSource} symbol={actualSymbol} timeframe={timeframe}
                  analysisResult={analysisResult} onLatestChartInfoUpdate={handleLatestChartInfoUpdate}
                  onChartLoadingStateChange={handleChartLoadingStateChange}
                  onHistoricalDataUpdate={handleHistoricalDataUpdate}
                  movingAverages={movingAverages}
                  theme={theme} chartPaneBackgroundColor={chartPaneBackgroundColor}
                  volumePaneHeight={volumePaneHeight} showAiAnalysisDrawings={showAiAnalysisDrawings}
                  wSignalColor={wSignalColor} wSignalOpacity={wSignalOpacity / 100}
                  showWSignals={showWSignals}
                />
              </div>
            </div>
            
            {/* Panel de análisis - Solo visible cuando hay análisis o chat activo */}
            {(analysisPanelMode !== 'initial') && (
              <div
                id="controls-analysis-panel"
                className={`w-full md:w-80 lg:w-[360px] xl:w-[400px] flex-none flex flex-col gap-2 sm:gap-4 overflow-y-auto ${!isPanelVisible ? 'hidden' : ''}`}
              >
                <div className={`${theme === 'dark' ? 'bg-slate-800' : 'bg-white'} rounded-lg shadow-md flex-grow flex flex-col`}>
                  <AnalysisPanel
                    panelMode={analysisPanelMode}
                    analysisResult={analysisResult}
                    analysisLoading={analysisLoading}
                    analysisError={analysisError}
                    chatMessages={chatMessages}
                    chatLoading={chatLoading}
                    chatError={chatError}
                    onSendMessage={handleSendMessageToChat}
                    onClearChatHistory={handleClearChatHistory}
                    theme={theme}
                    apiKeyPresent={apiKeyPresent}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {currentView === 'test' && (
          <div className="flex flex-col gap-4 h-full">
            <div className={`p-6 rounded-lg shadow-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-white'}`}>
              <h2 className={`text-xl font-bold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                Test de Conectividad API
              </h2>
              <TestConnectivity />
            </div>
          </div>
        )}
      </main>

      {displaySettingsDialogOpen && (
        <DisplaySettingsDialog
          isOpen={displaySettingsDialogOpen}
          onClose={() => setDisplaySettingsDialogOpen(false)}
          theme={theme}
          movingAverages={movingAverages}
          setMovingAverages={setMovingAverages}
          onToggleAllMAs={toggleAllMAsVisibility}
          chartPaneBackgroundColor={chartPaneBackgroundColor}
          setChartPaneBackgroundColor={setChartPaneBackgroundColor}
          volumePaneHeight={volumePaneHeight}
          setVolumePaneHeight={setVolumePaneHeight}
          wSignalColor={wSignalColor}
          setWSignalColor={setWSignalColor}
          wSignalOpacity={wSignalOpacity}
          setWSignalOpacity={setWSignalOpacity}
          showWSignals={showWSignals}
          setShowWSignals={setShowWSignals}
        />
      )}
    </div>
  );
};

export default App;
