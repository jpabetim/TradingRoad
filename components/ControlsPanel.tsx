

import React from 'react';
import { AnalysisPanelMode } from '../App';

interface ControlsPanelProps {
  onRequestAnalysis: () => void;
  onRequestChat: () => void;
  isLoading: boolean;
  apiKeyPresent: boolean;
  isChartLoading: boolean;
  showAiAnalysisDrawings: boolean;
  setShowAiAnalysisDrawings: (show: boolean) => void;
  analysisPanelMode: AnalysisPanelMode;
  hasAnalysisResult: boolean;
}

const ControlsPanel: React.FC<ControlsPanelProps> = ({
  onRequestAnalysis,
  onRequestChat,
  isLoading,
  apiKeyPresent,
  isChartLoading,
  showAiAnalysisDrawings,
  setShowAiAnalysisDrawings,
  analysisPanelMode,
  hasAnalysisResult,
}) => {

  const analysisButtonText = 
    analysisPanelMode === 'analysis' && hasAnalysisResult
    ? 'Refrescar Análisis' 
    : 'Análisis IA';

  return (
    <div className="p-3 sm:p-4 bg-slate-800 rounded-lg shadow">
      <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 text-sky-400">Acciones de Análisis</h2>
      
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5 sm:gap-2">
        <button
          onClick={() => setShowAiAnalysisDrawings(!showAiAnalysisDrawings)}
          className="bg-teal-600 hover:bg-teal-700 text-white font-medium py-1.5 sm:py-2 px-2 sm:px-3 rounded-lg focus:outline-none focus:shadow-outline transition-colors text-xs sm:text-sm"
          aria-label="Alternar Dibujos en Gráfico"
        >
          {showAiAnalysisDrawings ? 'Ocultar Dibujos' : 'Mostrar Dibujos'}
        </button>
        <button
          onClick={onRequestAnalysis}
          disabled={isLoading || isChartLoading || !apiKeyPresent}
          className="bg-sky-600 hover:bg-sky-700 disabled:bg-slate-600 disabled:text-slate-400 disabled:cursor-not-allowed text-white font-bold py-1.5 sm:py-2 px-2 sm:px-3 rounded-lg focus:outline-none focus:shadow-outline transition-colors text-xs sm:text-sm"
          aria-label="Analizar Gráfico con IA"
        >
          {isLoading && analysisPanelMode === 'analysis' ? 'Analizando...' : (isChartLoading ? 'Cargando Gráfico...' : analysisButtonText)}
        </button>
         <button
          onClick={onRequestChat}
          disabled={isLoading && analysisPanelMode === 'chat' || !apiKeyPresent} 
          className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-600 disabled:text-slate-400 disabled:cursor-not-allowed text-white font-bold py-1.5 sm:py-2 px-2 sm:px-3 rounded-lg focus:outline-none focus:shadow-outline transition-colors text-xs sm:text-sm"
          aria-label="Abrir Asistente de Chat IA"
        >
          {isLoading && analysisPanelMode === 'chat' ? 'Procesando...' : 'Asistente IA'}
        </button>
      </div>
      {!apiKeyPresent && <p className="text-xs text-yellow-400 mt-1.5 sm:mt-2 text-center">Funciones IA deshabilitadas: Clave API no configurada.</p>}
    </div>
  );
};

export default ControlsPanel;
