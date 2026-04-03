import React, { useEffect, useState } from 'react';
import './PercentageRing.css';

const PercentageRing = ({ percentage = 0, size = 180, strokeWidth = 14 }) => {
  const [animatedPct, setAnimatedPct] = useState(0);

  useEffect(() => {
    // Small animation delay for visual appeal
    const timeout = setTimeout(() => {
      setAnimatedPct(percentage);
    }, 100);
    return () => clearTimeout(timeout);
  }, [percentage]);

  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (animatedPct / 100) * circumference;

  // Determine color based on percentage
  const getColor = () => {
    if (animatedPct < 20) return '#00E676'; // --accent-green
    if (animatedPct < 60) return '#FFB300'; // --color-mixed (keeping for variety)
    return '#FF4D4D'; // --accent-red
  };

  const ringColor = getColor();

  return (
    <div className="ring-container" style={{ width: size, height: size }}>
      <svg className="ring-svg" width={size} height={size}>
        {/* Background Ring */}
        <circle
          className="ring-bg"
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth={strokeWidth}
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        {/* Progress Ring */}
        <circle
          className="ring-progress"
          stroke={ringColor}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: strokeDashoffset,
            filter: `drop-shadow(0 0 10px ${ringColor})`
          }}
        />
      </svg>
      <div className="ring-text">
        <span className="ring-number" style={{ color: ringColor }}>
          {animatedPct}%
        </span>
        <span className="ring-label">AI DETECTED</span>
      </div>
    </div>
  );
};

export default PercentageRing;
