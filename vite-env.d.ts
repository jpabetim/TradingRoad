/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_GEMINI_API_KEY: string
    readonly VITE_ANALYSIS_API_KEY: string
    readonly VITE_TRADERALPHA_API_KEY: string
    readonly VITE_TRANSLATE_API_KEY: string
    readonly VITE_BINGX_API_KEY: string
    readonly VITE_TWELVE_DATA_API_KEY: string
    readonly VITE_FMP_API_KEY: string
    readonly VITE_FINNHUB_API_KEY: string
    readonly VITE_ALPHA_VANTAGE_API_KEY: string
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
