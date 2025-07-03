import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(() => {
  // URL dinámica del backend para desarrollo y producción
  const backendUrl = process.env.VITE_BACKEND_URL || 'http://localhost:5010';

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      }
    },
    server: {
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
          secure: false,
        }
      },
    },
    // Esta configuración permite que el enrutamiento de React funcione correctamente
    build: {
      rollupOptions: {
        output: {
          manualChunks: undefined,
        },
      },
    },
    // Configuración optimizada para el servidor de vista previa en Render
    preview: {
      host: '0.0.0.0', // Permite acceso desde cualquier dirección IP
      port: parseInt(process.env.PORT || '3000'), // Usa el puerto asignado por Render o 3000 como fallback
      strictPort: true, // Falla si no puede usar el puerto especificado
      allowedHosts: ['*'], // Permite todos los hosts (Render maneja la seguridad)
    }
  };
});

