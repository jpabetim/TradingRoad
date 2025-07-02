import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi } from 'lightweight-charts';

const TestChart: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: '#0f172a' },
        textColor: '#ffffff',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      timeScale: {
        borderColor: '#334155',
      },
      rightPriceScale: {
        borderColor: '#334155',
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    // Datos de prueba
    const testData = [
      { time: '2023-12-01', open: 100, high: 110, low: 95, close: 105 },
      { time: '2023-12-02', open: 105, high: 115, low: 100, close: 108 },
      { time: '2023-12-03', open: 108, high: 120, low: 106, close: 118 },
      { time: '2023-12-04', open: 118, high: 125, low: 115, close: 122 },
      { time: '2023-12-05', open: 122, high: 130, low: 120, close: 128 },
    ];

    candlestickSeries.setData(testData);
    chartRef.current = chart;

    const handleResize = () => {
      if (chart && chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: 400,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  return (
    <div style={{ width: '100%', height: '400px', background: '#0f172a', borderRadius: '8px' }}>
      <h3 style={{ color: 'white', padding: '10px', margin: 0 }}>Test Chart - Should show 5 candles</h3>
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: '360px',
          background: '#0f172a'
        }}
      />
    </div>
  );
};

export default TestChart;
