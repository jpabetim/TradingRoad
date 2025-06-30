
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { GeminiAnalysisResult, GeminiRequestPayload } from "../types";
import { GEMINI_MODEL_NAME, getFullAnalysisPrompt } from "../constants";

// getApiKey function is removed as apiKey will be passed directly.

export interface ExtendedGeminiRequestPayload extends GeminiRequestPayload {
  latestVolume?: number | null;
  apiKey: string; // API key is now part of the payload
}

export const analyzeChartWithGemini = async (
  payload: ExtendedGeminiRequestPayload
): Promise<GeminiAnalysisResult> => {
  const { apiKey, ...restOfPayload } = payload; // Extract apiKey from payload

  if (!apiKey || apiKey === "TU_CLAVE_API_DE_GEMINI_AQUI") {
    console.error("API_KEY is not configured or is a placeholder. It was passed to analyzeChartWithGemini.");
    throw new Error("API Key is not configured or is a placeholder. AI analysis disabled.");
  }

  const ai = new GoogleGenAI({ apiKey }); // Use the apiKey from payload

  const fullPrompt = getFullAnalysisPrompt(
    restOfPayload.symbol,
    restOfPayload.timeframe,
    restOfPayload.currentPrice,
    restOfPayload.latestVolume
  );

  const finalPromptWithTimestamp = fullPrompt.replace("AUTO_GENERATED_TIMESTAMP_ISO8601", new Date().toISOString());

  let genAIResponse: GenerateContentResponse | undefined;

  try {
    genAIResponse = await ai.models.generateContent({
      model: GEMINI_MODEL_NAME,
      contents: finalPromptWithTimestamp,
      config: {
        responseMimeType: "application/json",
      },
    });

    // Log the raw text response from Gemini for debugging
    console.log("Raw text response from Gemini API:", genAIResponse?.text);

    let jsonStr = genAIResponse.text.trim();
    
    // New, more robust JSON extraction logic
    // 1. First, try to extract from a markdown fence. The regex is now unanchored.
    const fenceRegex = /```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/;
    const match = jsonStr.match(fenceRegex);
    
    if (match && match[1]) {
      jsonStr = match[1].trim();
    } else {
      // 2. If no fence is found, or if it's empty, try to find the first '{' and last '}'
      // This is a fallback for when the model doesn't use a fence or includes text outside of it.
      const firstBrace = jsonStr.indexOf('{');
      const lastBrace = jsonStr.lastIndexOf('}');
      if (firstBrace !== -1 && lastBrace > firstBrace) {
        jsonStr = jsonStr.substring(firstBrace, lastBrace + 1).trim();
      }
    }

    const parsedData = JSON.parse(jsonStr) as GeminiAnalysisResult;

    if (!parsedData.analisis_general || !parsedData.escenarios_probables) {
        console.warn("Parsed Gemini response seems to be missing key fields.", parsedData);
    }

    return parsedData;

  } catch (error: any) {
    console.error("Error calling Gemini API or parsing response. Full error object:", error); 

    let errorMessage = "Failed to get analysis from Gemini. An unknown error occurred during the API call or response processing."; 

    if (error.message) {
        if (error.message.includes("API_KEY_INVALID") || error.message.includes("API key not valid")) {
             errorMessage = "Gemini API Key is invalid. Please check your API_KEY configuration in index.html.";
        } else if (error.message.includes("quota") || error.message.includes("Quota")) {
            errorMessage = "Gemini API quota exceeded. Please check your quota or try again later.";
        } else if (error.message.toLowerCase().includes("json") || error instanceof SyntaxError) { // Catch SyntaxError explicitly
            errorMessage = "Failed to parse the analysis from Gemini. The response was not valid JSON.";
            if (genAIResponse && typeof genAIResponse.text === 'string') {
                console.error("Problematic JSON string from Gemini (leading to parsing error):", genAIResponse.text);
            }
        } else {
            errorMessage = `Gemini API error: ${error.message}`;
        }
    } else if (typeof error === 'string' && error.includes("```")) {
        errorMessage = "Received a malformed response from Gemini (likely unparsed markdown/JSON).";
         if (genAIResponse && typeof genAIResponse.text === 'string') { // Log it here too
            console.error("Malformed (markdown/JSON) string from Gemini:", genAIResponse.text);
        }
    } else if (error && typeof error.toString === 'function') { 
        const errorString = error.toString();
        errorMessage = `Gemini API call failed: ${errorString.startsWith('[object Object]') ? 'Non-descriptive error object received.' : errorString}`;
    }

    // Include raw response text in error if available and not already logged by specific conditions
    if (genAIResponse && typeof genAIResponse.text === 'string' && !errorMessage.toLowerCase().includes("json")) {
       // Avoid re-logging if already handled by the SyntaxError/json message condition
       // but ensure it's logged if some other error occurred after receiving text.
       console.error("Gemini raw response text during error:", genAIResponse.text);
    }


    throw new Error(errorMessage);
  }
};
