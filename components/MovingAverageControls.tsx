

import React from 'react';
import { MovingAverageConfig } from '../types';

interface MovingAverageControlsProps {
  movingAverages: MovingAverageConfig[];
  setMovingAverages: (configs: MovingAverageConfig[]) => void;
}

const MovingAverageControls: React.FC<MovingAverageControlsProps> = ({
  movingAverages,
  setMovingAverages,
}) => {
  const handleMAChange = (index: number, field: keyof MovingAverageConfig, value: any) => {
    const updatedMAs = movingAverages.map((ma, i) => {
      if (i === index) {
        if (field === 'period') {
          const numValue = parseInt(value, 10);
          return { ...ma, [field]: numValue > 0 ? numValue : 1 };
        }
        if (field === 'visible') {
            return { ...ma, [field]: !!value };
        }
        return { ...ma, [field]: value };
      }
      return ma;
    });
    setMovingAverages(updatedMAs);
  };

  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 p-0.5 sm:p-1 bg-slate-800 rounded-md border border-slate-700">
      <h3 className="text-[11px] sm:text-xs font-semibold text-sky-400 mr-1 sm:mr-1.5 whitespace-nowrap">MMs:</h3>
      {movingAverages.map((ma, index) => (
        <div key={ma.id} className="flex items-center gap-x-1 sm:gap-x-1.5 p-0.5 sm:p-1 bg-slate-700 rounded">
          <input
            type="checkbox"
            id={`ma-visible-${index}`}
            checked={!!ma.visible}
            onChange={(e) => handleMAChange(index, 'visible', e.target.checked)}
            className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-sky-600 bg-slate-600 border-slate-500 rounded focus:ring-sky-500 focus:ring-offset-slate-800"
            title={`Alternar visibilidad MM ${index + 1}`}
          />
          <select
            id={`ma-type-${index}`}
            value={ma.type}
            onChange={(e) => handleMAChange(index, 'type', e.target.value as 'MA' | 'EMA')}
            className="bg-slate-600 border-slate-500 text-slate-100 text-xs rounded px-1 py-0 sm:py-0.5 focus:ring-sky-500 focus:border-sky-500"
            title={`Tipo MM ${index + 1}`}
          >
            <option value="MA">MA</option>
            <option value="EMA">EMA</option>
          </select>
          <input
            type="number"
            id={`ma-period-${index}`}
            value={ma.period}
            min="1"
            max="500"
            onChange={(e) => handleMAChange(index, 'period', e.target.value)}
            className="bg-slate-600 border-slate-500 text-slate-100 text-xs rounded px-1 py-0 sm:py-0.5 w-10 sm:w-12 focus:ring-sky-500 focus:border-sky-500 appearance-none text-center"
            title={`Período MM ${index + 1}`}
          />
          <input
            type="color"
            id={`ma-color-${index}`}
            value={ma.color}
            onChange={(e) => handleMAChange(index, 'color', e.target.value)}
            className="w-4 h-4 sm:w-5 sm:h-5 p-0 border-none rounded cursor-pointer bg-slate-600"
            title={`Color MM ${index + 1}`}
          />
        </div>
      ))}
    </div>
  );
};

export default MovingAverageControls;