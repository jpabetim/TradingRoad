
import React from 'react';
import { MovingAverageConfig } from '../types';
import MovingAverageControls from './MovingAverageControls';

interface DisplaySettingsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  theme: 'dark' | 'light';
  movingAverages: MovingAverageConfig[];
  setMovingAverages: (configs: MovingAverageConfig[]) => void;
  onToggleAllMAs: (forceVisible?: boolean) => void;
  chartPaneBackgroundColor: string;
  setChartPaneBackgroundColor: (color: string) => void;
  volumePaneHeight: number; // For visibility toggle based on 0 or >0
  setVolumePaneHeight: (height: number) => void;
  wSignalColor: string;
  setWSignalColor: (color: string) => void;
  wSignalOpacity: number; // 0-100
  setWSignalOpacity: (opacity: number) => void;
  showWSignals: boolean;
  setShowWSignals: (show: boolean) => void;
}

const DisplaySettingsDialog: React.FC<DisplaySettingsDialogProps> = ({
  isOpen,
  onClose,
  theme,
  movingAverages,
  setMovingAverages,
  onToggleAllMAs,
  chartPaneBackgroundColor,
  setChartPaneBackgroundColor,
  volumePaneHeight,
  setVolumePaneHeight,
  wSignalColor,
  setWSignalColor,
  wSignalOpacity,
  setWSignalOpacity,
  showWSignals,
  setShowWSignals,
}) => {
  if (!isOpen) {
    return null;
  }

  const allMAsVisible = movingAverages.every(ma => ma.visible);

  const dialogBgColor = theme === 'dark' ? 'bg-slate-800' : 'bg-white';
  const textColor = theme === 'dark' ? 'text-slate-100' : 'text-slate-900';
  const inputBgColor = theme === 'dark' ? 'bg-slate-700' : 'bg-gray-100';
  const inputBorderColor = theme === 'dark' ? 'border-slate-600' : 'border-gray-300';
  const sectionBorderColor = theme === 'dark' ? 'border-slate-700' : 'border-gray-300';
  const labelColor = theme === 'dark' ? 'text-slate-300' : 'text-slate-700';

  return (
    <div 
        className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50"
        onClick={onClose}
        role="dialog"
        aria-modal="true"
        aria-labelledby="display-settings-dialog-title"
    >
      <div 
        className={`flex flex-col w-full max-w-md shadow-xl rounded-lg ${dialogBgColor} ${textColor}`}
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside dialog
      >
        <header className={`p-3 sm:p-4 border-b ${sectionBorderColor} flex justify-between items-center flex-shrink-0`}>
          <h2 id="display-settings-dialog-title" className="text-md sm:text-lg font-semibold text-sky-400">Configuración de Visualización</h2>
          <button
            onClick={onClose}
            className={`p-1 rounded-md ${theme === 'dark' ? 'hover:bg-slate-700' : 'hover:bg-gray-200'}`}
            aria-label="Cerrar diálogo de configuración"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 sm:w-6 sm:h-6">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </header>

        <div className="flex-grow p-3 sm:p-4 space-y-3 sm:space-y-4 overflow-y-auto max-h-[70vh]">
            <div className={`p-2 sm:p-3 rounded-md border ${sectionBorderColor} ${inputBgColor}`}>
                <h4 className="text-sm sm:text-md font-semibold text-sky-300 mb-2">Medias Móviles</h4>
                <MovingAverageControls
                    movingAverages={movingAverages}
                    setMovingAverages={setMovingAverages}
                />
                <div className="mt-2">
                    <label htmlFor="toggle-all-mas-dialog" className={`flex items-center cursor-pointer text-xs sm:text-sm ${labelColor}`}>
                        <input
                        type="checkbox"
                        id="toggle-all-mas-dialog"
                        checked={allMAsVisible}
                        onChange={(e) => onToggleAllMAs(e.target.checked)}
                        className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-sky-600 bg-slate-100 border-slate-300 rounded focus:ring-sky-500 dark:focus:ring-sky-600 dark:ring-offset-slate-800 focus:ring-2 dark:bg-slate-700 dark:border-slate-600 mr-2"
                        />
                        Mostrar Todas las Medias Móviles
                    </label>
                </div>
            </div>
            
            <div className={`p-2 sm:p-3 rounded-md border ${sectionBorderColor} ${inputBgColor}`}>
                <h4 className="text-sm sm:text-md font-semibold text-sky-300 mb-2">Apariencia del Gráfico</h4>
                <div className="mb-2 sm:mb-3">
                    <label htmlFor="chart-bg-color-picker-dialog" className={`block mb-1 text-xs sm:text-sm font-medium ${labelColor}`}>Color de Fondo del Gráfico</label>
                    <input
                    type="color"
                    id="chart-bg-color-picker-dialog"
                    value={chartPaneBackgroundColor}
                    onChange={(e) => setChartPaneBackgroundColor(e.target.value)}
                    className={`w-full h-7 sm:h-8 p-0 border-none rounded cursor-pointer ${theme === 'dark' ? 'bg-slate-600' : 'bg-gray-200'}`}
                    />
                </div>

                <div className="mb-2 sm:mb-3">
                    <label htmlFor="volume-pane-height-dialog" className={`block mb-1 text-xs sm:text-sm font-medium ${labelColor}`}>
                        {volumePaneHeight > 0 ? `Altura Panel Volumen: ${volumePaneHeight}px (Referencial)` : "Panel de Volumen Oculto"}
                    </label>
                     <input
                        type="range"
                        id="volume-pane-height-dialog"
                        min="0" max="300" step="10" // 0 means hidden for lightweight-charts visibility toggle
                        value={volumePaneHeight}
                        onChange={(e) => setVolumePaneHeight(Number(e.target.value))}
                        className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-sky-500"
                        title={volumePaneHeight > 0 ? "Ajustar altura (referencial, visibilidad si >0)" : "Mostrar panel de volumen (establecer >0)"}
                    />
                    <p className="text-xs text-slate-400 mt-1">Nota: 0 oculta el panel de volumen. La altura es referencial para la visualización.</p>
                </div>
            </div>

            <div className={`p-2 sm:p-3 rounded-md border ${sectionBorderColor} ${inputBgColor}`}>
              <h4 className="text-sm sm:text-md font-semibold text-sky-300 mb-2">Apariencia Señal W</h4>
              <div className="flex items-center mb-2 sm:mb-3">
                <input
                  type="checkbox"
                  id="show-w-signals-checkbox-dialog"
                  checked={showWSignals}
                  onChange={(e) => setShowWSignals(e.target.checked)}
                  className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-sky-600 bg-slate-100 border-slate-300 rounded focus:ring-sky-500 dark:focus:ring-sky-600 dark:ring-offset-slate-800 focus:ring-2 dark:bg-slate-700 dark:border-slate-600 mr-2"
                />
                <label htmlFor="show-w-signals-checkbox-dialog" className={`text-xs sm:text-sm font-medium ${labelColor}`}>Mostrar Señales W</label>
              </div>
              <div className="mb-2 sm:mb-3">
                  <label htmlFor="w-signal-color-picker-dialog" className={`block mb-1 text-xs sm:text-sm font-medium ${labelColor}`}>Color Marcador Señal W</label>
                  <input
                      type="color"
                      id="w-signal-color-picker-dialog"
                      value={wSignalColor}
                      onChange={(e) => setWSignalColor(e.target.value)}
                      className={`w-full h-7 sm:h-8 p-0 border-none rounded cursor-pointer ${theme === 'dark' ? 'bg-slate-600' : 'bg-gray-200'}`}
                  />
              </div>
              <div className="mb-2 sm:mb-3">
                  <label htmlFor="w-signal-opacity-slider-dialog" className={`block mb-1 text-xs sm:text-sm font-medium ${labelColor}`}>Opacidad Marcador Señal W: {wSignalOpacity}%</label>
                  <input
                      type="range"
                      id="w-signal-opacity-slider-dialog"
                      min="0" max="100" step="1"
                      value={wSignalOpacity}
                      onChange={(e) => setWSignalOpacity(Number(e.target.value))}
                      className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-sky-500"
                  />
              </div>
            </div>
        </div>
      </div>
    </div>
  );
};

export default DisplaySettingsDialog;
