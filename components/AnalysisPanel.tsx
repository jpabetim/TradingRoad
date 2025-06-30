


import React, { useRef, useEffect, useState } from 'react';
import { GeminiAnalysisResult, TradeSetup, ScenarioAnalysis, AnalysisPointType, FibonacciAnalysis, FibonacciLevel } from '../types';
import { AnalysisPanelMode, ChatMessage } from '../App'; 

interface AnalysisPanelProps {
  panelMode: AnalysisPanelMode;
  analysisResult: GeminiAnalysisResult | null;
  analysisLoading: boolean;
  analysisError: string | null;
  chatMessages: ChatMessage[];
  chatLoading: boolean;
  chatError: string | null;
  onSendMessage: (message: string) => void;
  onClearChatHistory: () => void; // Nueva prop
  theme: 'dark' | 'light';
  isMobile: boolean;
  apiKeyPresent: boolean;
}

const SectionTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h3 className="text-md sm:text-lg font-semibold mt-3 sm:mt-4 mb-1.5 sm:mb-2 text-sky-400 border-b border-slate-700 pb-1">{children}</h3>
);

const DetailItem: React.FC<{ label: string; value?: string | number | null; isCode?: boolean; valueClassName?: string }> = ({ label, value, isCode = false, valueClassName = "" }) => (
  value || value === 0 ? ( 
    <p className="text-xs sm:text-sm text-slate-300">
      <span className="font-medium text-slate-100">{label}:</span>{' '}
      {isCode ? <code className="text-xs bg-slate-600 p-0.5 sm:p-1 rounded">{value}</code> : <span className={valueClassName}>{value}</span>}
    </p>
  ) : null
);


const TradeSetupDisplay: React.FC<{ setup: TradeSetup | undefined, isMobile: boolean }> = ({ setup, isMobile }) => {
  if (!setup || setup.tipo === "ninguno") {
    return <p className="text-xs sm:text-sm text-slate-400 italic">No se identificó una configuración de trade específica.</p>;
  }
  return (
    <div className="space-y-1 mt-1 p-2 sm:p-3 bg-slate-700/50 rounded-md">
      <DetailItem label="Tipo" value={setup.tipo?.toUpperCase()} />
      {!isMobile && <DetailItem label="Condición de Entrada" value={setup.descripcion_entrada} />}
      <DetailItem label="Precio de Entrada Ideal" value={setup.punto_entrada_ideal ? `$${setup.punto_entrada_ideal.toFixed(Math.abs(setup.punto_entrada_ideal) < 1 ? 4 : 2)}` : undefined} />
      {!isMobile && <DetailItem label="Zona de Entrada" value={setup.zona_entrada ? `[$${setup.zona_entrada[0].toFixed(Math.abs(setup.zona_entrada[0]) < 1 ? 4 : 2)} - $${setup.zona_entrada[1].toFixed(Math.abs(setup.zona_entrada[1]) < 1 ? 4 : 2)}]` : undefined} />}
      <DetailItem label="Stop Loss" value={setup.stop_loss ? `$${setup.stop_loss.toFixed(Math.abs(setup.stop_loss) < 1 ? 4 : 2)}` : undefined} />
      <DetailItem label="Take Profit 1" value={setup.take_profit_1 ? `$${setup.take_profit_1.toFixed(Math.abs(setup.take_profit_1) < 1 ? 4 : 2)}` : undefined} />
      {!isMobile && setup.take_profit_2 && <DetailItem label="Take Profit 2" value={`$${setup.take_profit_2.toFixed(Math.abs(setup.take_profit_2) < 1 ? 4 : 2)}`} />}
      {!isMobile && setup.take_profit_3 && <DetailItem label="Take Profit 3" value={`$${setup.take_profit_3.toFixed(Math.abs(setup.take_profit_3) < 1 ? 4 : 2)}`} />}
      {setup.razon_fundamental && <p className="text-xs sm:text-sm text-slate-300 mt-1"><span className="font-medium text-slate-100">Razón:</span> {setup.razon_fundamental}</p>}
      {!isMobile && setup.confirmaciones_adicionales && setup.confirmaciones_adicionales.length > 0 && (
        <DetailItem label="Confirmaciones" value={setup.confirmaciones_adicionales.join(', ')} />
      )}
      {!isMobile && <DetailItem label="Riesgo/Beneficio" value={setup.ratio_riesgo_beneficio} />}
      <DetailItem label="Confianza" value={setup.calificacion_confianza} />
    </div>
  );
};

const FibonacciLevelDisplay: React.FC<{ level: FibonacciLevel }> = ({ level }) => (
  <li className="text-xs">
    <span className="font-medium text-slate-200">{level.label}:</span> ${level.price.toFixed(Math.abs(level.price) < 1 ? 4 : 2)}
  </li>
);

const FibonacciAnalysisDisplay: React.FC<{ fiboAnalysis: FibonacciAnalysis | undefined }> = ({ fiboAnalysis }) => {
  if (!fiboAnalysis) {
    return <p className="text-xs sm:text-sm text-slate-400 italic">Análisis Fibonacci no disponible.</p>;
  }

  const sortedRetracementLevels = [...(fiboAnalysis.niveles_retroceso || [])].sort((a, b) => {
    const isUpwardImpulse = fiboAnalysis.precio_fin_impulso > fiboAnalysis.precio_inicio_impulso;
    return isUpwardImpulse ? b.price - a.price : a.price - b.price;
  });

  const sortedExtensionLevels = [...(fiboAnalysis.niveles_extension || [])].sort((a, b) => {
    return a.level - b.level;
  });

  return (
    <div className="space-y-1.5 sm:space-y-2 mt-1.5 sm:mt-2 p-2 sm:p-3 bg-slate-700 rounded-md">
      <p className="text-xs sm:text-sm text-slate-300"><span className="font-medium text-slate-100">Impulso:</span> {fiboAnalysis.descripcion_impulso}</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-2 text-xs">
        <DetailItem label="Inicio Impulso (A)" value={fiboAnalysis.precio_inicio_impulso ? `$${fiboAnalysis.precio_inicio_impulso.toFixed(Math.abs(fiboAnalysis.precio_inicio_impulso) < 1 ? 4 : 2)}` : 'N/A'} />
        <DetailItem label="Fin Impulso (B)" value={fiboAnalysis.precio_fin_impulso ? `$${fiboAnalysis.precio_fin_impulso.toFixed(Math.abs(fiboAnalysis.precio_fin_impulso) < 1 ? 4 : 2)}` : 'N/A'} />
        {fiboAnalysis.precio_fin_retroceso != null && (
             <DetailItem label="Fin Retroceso (C)" value={`$${fiboAnalysis.precio_fin_retroceso.toFixed(Math.abs(fiboAnalysis.precio_fin_retroceso) < 1 ? 4 : 2)}`} />
        )}
      </div>

      {sortedRetracementLevels.length > 0 && (
        <div>
          <h4 className="text-xs sm:text-sm font-semibold text-slate-200 mt-1.5 sm:mt-2 mb-1">Niveles de Retroceso (A-B):</h4>
          <ul className="list-disc list-inside space-y-0.5 pl-2">
            {sortedRetracementLevels.map((level, index) => (
              <FibonacciLevelDisplay key={`retracement-${index}`} level={level} />
            ))}
          </ul>
        </div>
      )}

      {sortedExtensionLevels.length > 0 && (
        <div>
          <h4 className="text-xs sm:text-sm font-semibold text-slate-200 mt-1.5 sm:mt-2 mb-1">Niveles de Extensión (A-B-C):</h4>
          <ul className="list-disc list-inside space-y-0.5 pl-2">
            {sortedExtensionLevels.map((level, index) => (
              <FibonacciLevelDisplay key={`extension-${index}`} level={level} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

const AnalysisPanel: React.FC<AnalysisPanelProps> = ({
  panelMode,
  analysisResult,
  analysisLoading,
  analysisError,
  chatMessages,
  chatLoading,
  chatError,
  onSendMessage,
  onClearChatHistory,
  theme,
  isMobile,
  apiKeyPresent,
}) => {
  const [chatInputValue, setChatInputValue] = useState('');
  const chatMessagesEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (panelMode === 'chat') {
      chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, panelMode]);

  useEffect(() => {
    if (panelMode === 'chat' && chatInputRef.current) {
      chatInputRef.current.focus();
    }
  }, [panelMode]);

  const handleChatSend = () => {
    if (chatInputValue.trim()) {
      onSendMessage(chatInputValue.trim());
      setChatInputValue('');
    }
  };
  
  const StatusDisplayWrapper: React.FC<{ title: string; children: React.ReactNode; titleColor?: string }> = ({ title, children, titleColor = "text-sky-400" }) => (
    <div className="p-3 sm:p-4 bg-slate-800 rounded-lg shadow h-full">
      <h2 className={`text-lg sm:text-xl font-semibold mb-2 ${titleColor}`}>{title}</h2>
      <div className="text-xs sm:text-sm text-slate-300">{children}</div>
    </div>
  );

  const renderAnalysisContent = () => {
    if (analysisLoading) {
      return <div className="p-3 sm:p-4 text-center">Cargando análisis...</div>;
    }
    if (analysisError) {
      return <div className="p-3 sm:p-4 text-red-400">Error en Análisis: {analysisError}</div>;
    }
    if (!analysisResult) {
      return <div className="p-3 sm:p-4 text-center">Selecciona "Análisis IA" para obtener un análisis técnico.</div>;
    }

    const primaryScenario = analysisResult.escenarios_probables?.find(s => s.probabilidad === 'alta') || analysisResult.escenarios_probables?.[0];
    const alternativeScenarios = analysisResult.escenarios_probables?.filter(s => s !== primaryScenario) || [];
    const activeSignalType = analysisResult.conclusion_recomendacion?.mejor_oportunidad_actual?.tipo;
    const overallBias = analysisResult.analisis_general?.sesgo_direccional_general;
    const biasColorMap: Record<string, string> = { alcista: "text-green-400", bajista: "text-red-400" };
    const biasColorClass = biasColorMap[overallBias?.toLowerCase() || ''] || "text-slate-300";
    const activeSignalColorMap: Record<string, string> = { largo: 'text-green-400', corto: 'text-red-400' };
    const activeSignalColorClass = activeSignalColorMap[activeSignalType?.toLowerCase() || ''] || 'text-slate-300';

    return (
      <div className="p-3 sm:p-4"> 
        <h2 className="text-lg sm:text-xl font-semibold mb-2 sm:mb-3 text-sky-400">Resultados del Análisis IA</h2>
         {!isMobile && primaryScenario && (
          <>
            <SectionTitle>Escenario Principal</SectionTitle>
            <div className="p-2 sm:p-3 bg-slate-700 rounded-md">
              <p className="text-xs sm:text-sm font-semibold text-slate-100">{primaryScenario.nombre_escenario}</p>
              <p className="text-xs text-slate-300 mt-1">{primaryScenario.descripcion_detallada}</p>
              {primaryScenario.niveles_clave_de_invalidacion && (
                <p className="text-xs text-slate-400 mt-1">Invalidación: {primaryScenario.niveles_clave_de_invalidacion}</p>
              )}
              {primaryScenario.trade_setup_asociado && primaryScenario.trade_setup_asociado.tipo !== "ninguno" && (
                   <>
                      <p className="text-xs font-semibold text-slate-100 mt-1.5 sm:mt-2">
                        Configuración Asociada: <span className={primaryScenario.trade_setup_asociado.tipo === 'largo' ? 'text-green-400' : 'text-red-400'}>{primaryScenario.trade_setup_asociado.tipo.toUpperCase()}</span>
                      </p>
                      <TradeSetupDisplay setup={primaryScenario.trade_setup_asociado} isMobile={isMobile} />
                   </>
              )}
            </div>
          </>
        )}
        {!isMobile && !primaryScenario && (
             <p className="text-xs sm:text-sm text-slate-400 italic">No se identificó un escenario principal.</p>
        )}
        {(alternativeScenarios.length > 0 || (isMobile && primaryScenario)) && (
          <>
            <SectionTitle>Escenarios Alternativos</SectionTitle>
            {isMobile && primaryScenario && !alternativeScenarios.find(s => s.nombre_escenario === primaryScenario.nombre_escenario) && (
                 <div className="mt-1.5 sm:mt-2 p-2 sm:p-3 bg-slate-700/70 rounded-md">
                    <p className="text-xs sm:text-sm font-semibold text-slate-200">{primaryScenario.nombre_escenario} <span className="text-xs text-slate-400">(Prob: {primaryScenario.probabilidad})</span></p>
                    <p className="text-xs text-slate-300 mt-1">{primaryScenario.descripcion_detallada}</p>
                    {primaryScenario.niveles_clave_de_invalidacion && (
                      <p className="text-xs text-slate-400 mt-1">Invalidación: {primaryScenario.niveles_clave_de_invalidacion}</p>
                    )}
                    {primaryScenario.trade_setup_asociado && primaryScenario.trade_setup_asociado.tipo !== "ninguno" && (
                      <>
                        <p className="text-xs font-semibold text-slate-100 mt-1.5 sm:mt-2">
                          Trade Potencial: <span className={primaryScenario.trade_setup_asociado.tipo === 'largo' ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>{primaryScenario.trade_setup_asociado.tipo.toUpperCase()}</span>
                        </p>
                        <TradeSetupDisplay setup={primaryScenario.trade_setup_asociado} isMobile={isMobile}/>
                      </>
                    )}
                </div>
            )}
            {alternativeScenarios.map((scenario, index) => (
              <div key={index} className="mt-1.5 sm:mt-2 p-2 sm:p-3 bg-slate-700/70 rounded-md">
                <p className="text-xs sm:text-sm font-semibold text-slate-200">{scenario.nombre_escenario} <span className="text-xs text-slate-400">(Prob: {scenario.probabilidad})</span></p>
                <p className="text-xs text-slate-300 mt-1">{scenario.descripcion_detallada}</p>
                {scenario.niveles_clave_de_invalidacion && (
                  <p className="text-xs text-slate-400 mt-1">Invalidación: {scenario.niveles_clave_de_invalidacion}</p>
                )}
                {scenario.trade_setup_asociado && scenario.trade_setup_asociado.tipo !== "ninguno" && (
                  <>
                    <p className="text-xs font-semibold text-slate-100 mt-1.5 sm:mt-2">
                      Trade Potencial: <span className={scenario.trade_setup_asociado.tipo === 'largo' ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>{scenario.trade_setup_asociado.tipo.toUpperCase()}</span>
                    </p>
                    <TradeSetupDisplay setup={scenario.trade_setup_asociado} isMobile={isMobile} />
                  </>
                )}
              </div>
            ))}
          </>
        )}
        {!isMobile && (
          <>
            <SectionTitle>Señal Activa Sugerida</SectionTitle>
            <p className={`text-xs sm:text-sm font-semibold ${activeSignalColorClass}`}>
              {activeSignalType && activeSignalType !== "ninguno" ? activeSignalType.toUpperCase() : "NINGUNA"}
            </p>
            <SectionTitle>Configuración de Trade Recomendada</SectionTitle>
            <TradeSetupDisplay setup={analysisResult.conclusion_recomendacion?.mejor_oportunidad_actual} isMobile={isMobile} />
            <SectionTitle>Análisis Fibonacci</SectionTitle>
            <FibonacciAnalysisDisplay fiboAnalysis={analysisResult.analisis_fibonacci} />
          </>)}
        {analysisResult.puntos_clave_grafico && analysisResult.puntos_clave_grafico.length > 0 && (
          <> <SectionTitle>Niveles y Zonas Clave Identificados</SectionTitle>
            <ul className="list-disc list-inside space-y-1 text-xs sm:text-sm text-slate-300 mt-1.5 sm:mt-2 p-1.5 sm:p-2 bg-slate-700 rounded-md">
              {analysisResult.puntos_clave_grafico.map((point, index) => (
                <li key={index}>
                  <span className="font-medium text-slate-100">{point.label}</span>
                  {!isMobile && point.tipo && <span className="text-xs text-slate-400"> ({point.tipo.replace(/_/g, ' ')})</span>}
                  {point.zona && <span className="text-xs text-slate-400"> [$${point.zona[0].toFixed(Math.abs(point.zona[0]) < 1 ? 4 : 2)} - $${point.zona[1].toFixed(Math.abs(point.zona[1]) < 1 ? 4 : 2)}]</span>}
                  {point.nivel != null && <span className="text-xs text-slate-400"> @ ${typeof point.nivel === 'number' ? point.nivel.toFixed(Math.abs(point.nivel) < 1 ? 4 : 2) : point.nivel}</span>}
                  {!isMobile && point.temporalidad && <span className="text-xs text-slate-500"> ({point.temporalidad})</span>}
                  {(point.tipo === AnalysisPointType.POI_DEMANDA || point.tipo === AnalysisPointType.POI_OFERTA) && point.mitigado && <span className="text-xs text-sky-300"> (Mitigado)</span>}
                  {!isMobile && point.importancia && <span className="text-xs text-yellow-400"> Importancia: {point.importancia}</span>}
                </li>))} </ul> </>)}
        <SectionTitle>Estructura General de Mercado y Volumen</SectionTitle>
        {analysisResult.analisis_general?.estructura_mercado_resumen && Object.entries(analysisResult.analisis_general.estructura_mercado_resumen).map(([tf, desc]) =>
          desc && <DetailItem key={tf} label={`Estructura (${tf.replace('htf_', '').replace('mtf_', '').replace('ltf_', '')})`} value={desc as string} />)}
        {!isMobile && <DetailItem label="Fase Wyckoff" value={analysisResult.analisis_general?.fase_wyckoff_actual} />}
        <DetailItem label="Sesgo General" value={analysisResult.analisis_general?.sesgo_direccional_general?.toUpperCase()} valueClassName={biasColorClass + " font-semibold"}/>
        {!isMobile && analysisResult.analisis_general?.comentario_volumen && (
          <p className="text-xs sm:text-sm text-slate-300 mt-1"><span className="font-medium text-slate-100">Comentario Breve de Volumen:</span> {analysisResult.analisis_general.comentario_volumen}</p>)}
        {!isMobile && analysisResult.analisis_general?.interpretacion_volumen_detallada && (
          <p className="text-xs sm:text-sm text-slate-300 mt-1"><span className="font-medium text-slate-100">Análisis Detallado de Volumen:</span> {analysisResult.analisis_general.interpretacion_volumen_detallada}</p>)}
         {!isMobile && analysisResult.analisis_general?.comentario_funding_rate_oi && (
           <> <SectionTitle>Análisis Conceptual FR/OI</SectionTitle>
            <p className="text-xs sm:text-sm text-slate-300 mt-1 mb-1.5 sm:mb-2 p-2 sm:p-3 bg-slate-700 rounded-md">
              {analysisResult.analisis_general.comentario_funding_rate_oi} </p> </>)}
        <SectionTitle>Conclusión y Recomendaciones</SectionTitle>
        {analysisResult.conclusion_recomendacion?.resumen_ejecutivo && (
          <p className="text-xs sm:text-sm text-slate-300 mt-1 mb-1.5 sm:mb-2 p-2 sm:p-3 bg-slate-700 rounded-md">{analysisResult.conclusion_recomendacion.resumen_ejecutivo}</p>)}
        {!isMobile && <DetailItem label="Próximo Movimiento Esperado" value={analysisResult.conclusion_recomendacion?.proximo_movimiento_esperado} />}
        {!isMobile && analysisResult.conclusion_recomendacion?.oportunidades_reentrada_detectadas && (
           <> <h4 className="text-sm font-semibold mt-2 mb-1 text-sky-300">Oportunidades de Reentrada:</h4>
            <p className="text-xs sm:text-sm text-slate-300 p-2 bg-slate-700/70 rounded-md">{analysisResult.conclusion_recomendacion.oportunidades_reentrada_detectadas}</p> </>)}
        {!isMobile && analysisResult.conclusion_recomendacion?.consideraciones_salida_trade && (
           <> <h4 className="text-sm font-semibold mt-2 mb-1 text-sky-300">Consideraciones de Salida:</h4>
            <p className="text-xs sm:text-sm text-slate-300 p-2 bg-slate-700/70 rounded-md">{analysisResult.conclusion_recomendacion.consideraciones_salida_trade}</p> </>)}
        {!isMobile && analysisResult.conclusion_recomendacion?.senales_confluencia_avanzada && (
           <> <h4 className="text-sm font-semibold mt-2 mb-1 text-sky-300">Señales Avanzadas de Confluencia (Conceptual):</h4>
            <p className="text-xs sm:text-sm text-slate-300 p-2 bg-slate-700/70 rounded-md">{analysisResult.conclusion_recomendacion.senales_confluencia_avanzada}</p> </>)}
        {!isMobile && <DetailItem label="Advertencias/Riesgos" value={analysisResult.conclusion_recomendacion?.advertencias_riesgos} />}
      </div>
    );
  };

  const renderChatContent = () => {
    const chatContainerBg = theme === 'dark' ? 'bg-slate-800' : 'bg-white';
    const chatTextColor = theme === 'dark' ? 'text-slate-100' : 'text-slate-900';
    const inputBgColor = theme === 'dark' ? 'bg-slate-700' : 'bg-gray-100';
    const inputBorderColor = theme === 'dark' ? 'border-slate-600' : 'border-gray-300';
    const buttonBgColor = theme === 'dark' ? 'bg-sky-600 hover:bg-sky-700' : 'bg-sky-500 hover:bg-sky-600';
    const userMessageBg = theme === 'dark' ? 'bg-sky-700' : 'bg-sky-500';
    const aiMessageBg = theme === 'dark' ? 'bg-slate-600' : 'bg-gray-200';
    const aiMessageText = theme === 'dark' ? 'text-slate-50' : 'text-slate-800';

    return (
      <div className={`flex flex-col h-full ${chatContainerBg} ${chatTextColor}`}>
        <div className={`flex justify-between items-center p-3 sm:p-4 border-b ${inputBorderColor} flex-shrink-0`}>
          <h2 className="text-lg sm:text-xl font-semibold text-sky-400">Asistente de Trading IA</h2>
          {apiKeyPresent && chatMessages.length > 0 && (
            <button
              onClick={onClearChatHistory}
              className={`p-1 rounded-md transition-colors ${
                theme === 'dark' ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-200'
              }`}
              title="Borrar historial del chat"
              aria-label="Borrar historial del chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4 sm:w-5 sm:h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12.56 0c1.153 0 2.243.032 3.223.096M15 5.25a3 3 0 00-3-3m-3 3a3 3 0 01-3-3m-3 3a3 3 0 003 3m3 3a3 3 0 013 3m3 3a3 3 0 003-3" />
              </svg>
            </button>
          )}
        </div>
        <div className="flex-grow p-3 sm:p-4 space-y-3 overflow-y-auto">
          {!apiKeyPresent && (
             <div className="p-3 text-sm text-yellow-300 bg-yellow-800 bg-opacity-50 rounded-lg border border-yellow-600">
                Clave API no configurada. Las funciones de Chat IA están deshabilitadas.
            </div>
          )}
          {apiKeyPresent && chatError && (
            <div className="p-3 text-sm text-red-300 bg-red-800 bg-opacity-50 rounded-lg border border-red-600">
              Error en Chat: {chatError}
            </div>
          )}
          {chatMessages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] p-2 sm:p-3 rounded-lg text-xs sm:text-sm shadow ${
                  msg.sender === 'user'
                    ? `${userMessageBg} text-white`
                    : `${aiMessageBg} ${aiMessageText}`
                }`}
                dangerouslySetInnerHTML={{ __html: msg.text.replace(/\n/g, '<br/>').replace(/```json\s*\n?(.*?)\n?\s*```/gs, (match, p1) => `<pre class="bg-slate-900 text-slate-100 p-2 rounded overflow-x-auto text-xs">${p1.replace(/</g, '&lt;').replace(/>/g, '&gt;').trim()}</pre>`).replace(/`([^`]+)`/g, '<code class="bg-opacity-50 bg-black text-white px-1 py-0.5 rounded text-xs">$1</code>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>') }}
              />
            </div>
          ))}
          <div ref={chatMessagesEndRef} />
        </div>
        
        {chatLoading && (
            <div className={`px-3 sm:px-4 pb-1 text-xs text-center ${theme === 'dark' ? 'text-sky-300' : 'text-sky-600'}`}>
                TradeGuru IA está pensando...
            </div>
        )}

        <div className={`p-3 sm:p-4 border-t ${inputBorderColor} flex-shrink-0`}>
          <div className="flex items-start space-x-2">
            <textarea
              ref={chatInputRef}
              value={chatInputValue}
              onChange={(e) => setChatInputValue(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleChatSend();
                }
              }}
              placeholder={apiKeyPresent ? "Pregunta a TradeGuru AI..." : "Se requiere Clave API para el chat"}
              className={`flex-grow p-2 sm:p-2.5 border rounded-md shadow-sm text-xs sm:text-sm resize-none ${inputBgColor} ${chatTextColor} ${inputBorderColor} focus:ring-sky-500 focus:border-sky-500`}
              rows={2}
              disabled={chatLoading || !apiKeyPresent || (!!chatError && chatMessages.length === 0)}
              aria-label="Entrada de mensaje de chat"
            />
            <button
              onClick={handleChatSend}
              disabled={chatLoading || !chatInputValue.trim() || !apiKeyPresent || (!!chatError && chatMessages.length === 0)}
              className={`px-3 py-2 sm:px-4 sm:py-2.5 text-white font-semibold rounded-md shadow-sm text-xs sm:text-sm ${buttonBgColor} disabled:bg-slate-500 disabled:cursor-not-allowed`}
              aria-label="Enviar mensaje"
            >
              Enviar
            </button>
          </div>
        </div>
      </div>
    );
  };
  
  if (panelMode === 'initial') {
     return (
        <StatusDisplayWrapper title="Panel de IA">
          Selecciona "Análisis IA" para un análisis técnico o "Asistente IA" para chatear.
        </StatusDisplayWrapper>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {panelMode === 'analysis' ? renderAnalysisContent() : renderChatContent()}
    </div>
  );
};

export default AnalysisPanel;