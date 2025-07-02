import { FibonacciLevel } from '../types';

// Niveles estándar de retroceso de Fibonacci
export const FIBONACCI_RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786];

// Niveles estándar de extensión de Fibonacci
export const FIBONACCI_EXTENSION_LEVELS = [1.272, 1.414, 1.618, 2.0, 2.618];

/**
 * Calcula los niveles de retroceso de Fibonacci
 * @param pointA - Precio de inicio del impulso
 * @param pointB - Precio de fin del impulso
 * @param isUptrend - Si es true, es un impulso alcista; si es false, es bajista
 * @returns Array de niveles de retroceso de Fibonacci
 */
export function calculateFibonacciRetracements(
    pointA: number,
    pointB: number,
    isUptrend: boolean
): FibonacciLevel[] {
    const levels: FibonacciLevel[] = [];

    // Para impulso alcista: retrocesos van hacia abajo desde B hacia A
    // Para impulso bajista: retrocesos van hacia arriba desde B hacia A
    const range = Math.abs(pointB - pointA);

    FIBONACCI_RETRACEMENT_LEVELS.forEach(level => {
        let price: number;

        if (isUptrend) {
            // En tendencia alcista, los retrocesos van desde el máximo (B) hacia abajo
            price = pointB - (range * level);
        } else {
            // En tendencia bajista, los retrocesos van desde el mínimo (B) hacia arriba
            price = pointB + (range * level);
        }

        levels.push({
            level: level,
            price: parseFloat(price.toFixed(2)),
            label: `Retroceso ${(level * 100).toFixed(1)}%`
        });
    });

    // Ordenar los niveles apropiadamente
    if (isUptrend) {
        // Para tendencia alcista, ordenar de mayor a menor precio (38.2%, 50%, 61.8%)
        levels.sort((a, b) => b.price - a.price);
    } else {
        // Para tendencia bajista, ordenar de menor a mayor precio
        levels.sort((a, b) => a.price - b.price);
    }

    return levels;
}

/**
 * Calcula los niveles de extensión de Fibonacci
 * @param pointA - Precio de inicio del impulso
 * @param pointB - Precio de fin del impulso
 * @param pointC - Precio de fin del retroceso
 * @param isUptrend - Si es true, es un impulso alcista; si es false, es bajista
 * @returns Array de niveles de extensión de Fibonacci
 */
export function calculateFibonacciExtensions(
    pointA: number,
    pointB: number,
    pointC: number,
    isUptrend: boolean
): FibonacciLevel[] {
    const levels: FibonacciLevel[] = [];

    // Calcular el rango del impulso A-B
    const impulseRange = Math.abs(pointB - pointA);

    FIBONACCI_EXTENSION_LEVELS.forEach(level => {
        let price: number;

        if (isUptrend) {
            // En tendencia alcista, las extensiones van hacia arriba desde C
            price = pointC + (impulseRange * level);
        } else {
            // En tendencia bajista, las extensiones van hacia abajo desde C
            price = pointC - (impulseRange * level);
        }

        levels.push({
            level: level,
            price: parseFloat(price.toFixed(2)),
            label: `Extensión ${(level * 100).toFixed(1)}%`
        });
    });

    // Ordenar los niveles apropiadamente
    if (isUptrend) {
        // Para tendencia alcista, ordenar de menor a mayor precio
        levels.sort((a, b) => a.price - b.price);
    } else {
        // Para tendencia bajista, ordenar de mayor a menor precio
        levels.sort((a, b) => b.price - a.price);
    }

    return levels;
}

/**
 * Determina si un impulso es alcista o bajista
 * @param pointA - Precio de inicio
 * @param pointB - Precio de fin
 * @returns true si es alcista, false si es bajista
 */
export function isUptrendImpulse(pointA: number, pointB: number): boolean {
    return pointB > pointA;
}
