import React from 'react';
import './HighlightText.css';

const HighlightText = ({ segments }) => {
  return (
    <div className="highlighted-text-container">
      {segments.map((segment, index) => {
        let spanClass = "segment";
        let tooltip = "Likely Human generated";
        
        if (segment.type === 'ai') {
          spanClass += " ai";
          tooltip = 'Highly likely AI generated';
        } else if (segment.type === 'ai-refined') {
          spanClass += " ai-refined";
          tooltip = 'AI base text but modified';
        } else if (segment.type === 'human-ai') {
          spanClass += " human-ai";
          tooltip = 'Human text heavily polished by AI';
        } else {
          spanClass += " human";
        }

        return (
          <span 
            key={index} 
            className={spanClass}
            title={tooltip}
          >
            {segment.text}{" "}
          </span>
        );
      })}
    </div>
  );
};

export default HighlightText;
