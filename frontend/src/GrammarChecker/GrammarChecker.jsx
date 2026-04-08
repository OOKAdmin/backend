import React, { useState } from 'react';
import axios from 'axios';
import { AlertCircle, Check, X, RotateCcw, Play } from 'lucide-react';
import './GrammarChecker.css';

const API_BASE = process.env.REACT_APP_API_URL || 'https://backend-7cv7.onrender.com';

function GrammarChecker() {
  const [text, setText] = useState('');
  const [errors, setErrors] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setIsAnalyzing(true);
    setAnalyzed(false);

    try {
      const response = await axios.post(`${API_BASE}/api/grammar`, { text });
      setErrors(response.data.errors);
      setAnalyzed(true);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Failed to connect to the analysis server. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAccept = (errorIndex, replacementIndex) => {
    const error = errors[errorIndex];
    if (error.replacements && error.replacements.length > replacementIndex) {
      const replacement = error.replacements[replacementIndex];
      const lengthDiff = replacement.length - error.errorLength;
      const newText = text.substring(0, error.offset) + replacement + text.substring(error.offset + error.errorLength);
      setText(newText);
      const newErrors = errors
        .filter((_, i) => i !== errorIndex)
        .map(err => {
          if (err.offset > error.offset) {
            return { ...err, offset: err.offset + lengthDiff };
          }
          return err;
        });
      setErrors(newErrors);
    }
  };

  const handleReject = (errorIndex) => {
    setErrors(errors.filter((_, i) => i !== errorIndex));
  };

  const handleReset = () => {
    setText('');
    setErrors([]);
    setAnalyzed(false);
  };

  return (
    <div className="gc-container">
      <header className="gc-header">
        <h1>Advanced Grammar &amp; Intelligence Checker</h1>
        <p className="gc-subtitle">AI-Powered Text Analysis</p>
      </header>

      <main className="gc-main">
        <div className="gc-panel gc-left">
          <div className="gc-panel-header">
            <h2>ANALYSIS TERMINAL</h2>
            <div className="gc-stats">
              <span>{text.split(/\s+/).filter(w => w.length > 0).length} Words</span>
              <span>{text.length} Characters</span>
            </div>
          </div>

          <textarea
            className="gc-textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter your text here for analysis..."
          />

          <div className="gc-actions">
            <button className="gc-btn-secondary" onClick={handleReset}>
              <RotateCcw size={16} /> Reset
            </button>
            <button
              className="gc-btn-primary"
              onClick={handleAnalyze}
              disabled={isAnalyzing || text.length === 0}
            >
              {isAnalyzing ? (
                <>Analyzing...</>
              ) : (
                <><Play size={16} /> Run Analysis</>
              )}
            </button>
          </div>
        </div>

        <div className="gc-panel gc-right">
          <div className="gc-panel-header">
            <h2>Results</h2>
            <span className="gc-error-count">{errors.length} Issues Found</span>
          </div>

          <div className="gc-results">
            {!analyzed && !isAnalyzing && (
              <div className="gc-empty-state">
                <AlertCircle size={48} />
                <p>Waiting for Input</p>
                <span>Run analysis to see grammar suggestions</span>
              </div>
            )}

            {isAnalyzing && (
              <div className="gc-loading-state">
                <div className="gc-spinner"></div>
                <p>Processing text with AI models...</p>
              </div>
            )}

            {analyzed && errors.length === 0 && (
              <div className="gc-success-state">
                <Check size={48} />
                <p>No issues found!</p>
                <span>Your text looks perfect.</span>
              </div>
            )}

            {analyzed && errors.length > 0 && (
              <div className="gc-error-list">
                {errors.map((error, index) => (
                  <div key={index} className="gc-error-card">
                    <div className="gc-error-header">
                      <span className="gc-error-category">{error.category}</span>
                      <p className="gc-error-message">{error.message}</p>
                    </div>

                    <div className="gc-error-context">
                      "...{error.context}..."
                    </div>

                    {error.replacements && error.replacements.length > 0 ? (
                      <div className="gc-suggestions">
                        <h4>Suggestions:</h4>
                        <div className="gc-suggestion-actions">
                          <button
                            className="gc-btn-accept"
                            onClick={() => handleAccept(index, 0)}
                          >
                            <Check size={14} /> Accept "{error.replacements[0]}"
                          </button>
                          <button
                            className="gc-btn-reject"
                            onClick={() => handleReject(index)}
                          >
                            <X size={14} /> Reject
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="gc-suggestions">
                        <button
                          className="gc-btn-reject"
                          onClick={() => handleReject(index)}
                        >
                          <X size={14} /> Dismiss
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default GrammarChecker;
