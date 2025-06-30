
import React from 'react';

interface ApiKeyMessageProps {
  apiKeyPresent: boolean;
}

const ApiKeyMessage: React.FC<ApiKeyMessageProps> = ({ apiKeyPresent }) => {
  if (apiKeyPresent) {
    return null;
  }

  return (
    <div className="p-4 mb-4 text-sm text-yellow-300 bg-yellow-800 bg-opacity-30 rounded-lg border border-yellow-700" role="alert">
      <span className="font-medium">Clave API No Configurada:</span> La clave API de Gemini (process.env.API_KEY) no está disponible. Las funciones de análisis IA estarán deshabilitadas. Asegúrate de que la clave API esté configurada correctamente en tu entorno. Este mensaje es para desarrolladores.
    </div>
  );
};

export default ApiKeyMessage;