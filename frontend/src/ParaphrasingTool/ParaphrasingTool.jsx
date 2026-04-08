import React, { useState, useEffect } from 'react';
import './ParaphrasingTool.css';

const API_BASE = process.env.REACT_APP_API_URL || 'https://backend-7cv7.onrender.com';

function ParaphrasingTool() {
  const [inputText, setInputText] = useState('');
  const [paraphrasedText, setParaphrasedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('standard');
  const [wordCount, setWordCount] = useState({ words: 0, chars: 0 });
  const [copyStatus, setCopyStatus] = useState('Copy');

  useEffect(() => {
    const words = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
    const chars = inputText.length;
    setWordCount({ words, chars });
  }, [inputText]);

  const handleParaphrase = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    setParaphrasedText('');

    try {
      const response = await fetch(`${API_BASE}/api/paraphrase`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, mode }),
      });

      const data = await response.json();
      if (data.paraphrased) {
        setParaphrasedText(data.paraphrased);
      } else if (data.error) {
        setParaphrasedText(`Error: ${data.error}`);
      }
    } catch (err) {
      setParaphrasedText('Error: Could not connect to the backend. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setInputText('');
    setParaphrasedText('');
    setCopyStatus('Copy');
  };

  const handleCopy = () => {
    if (!paraphrasedText) return;
    navigator.clipboard.writeText(paraphrasedText).then(() => {
      setCopyStatus('Copied!');
      setTimeout(() => setCopyStatus('Copy'), 2000);
    });
  };

  return (
    <div className="pt-app">
      <header className="pt-header">
        <span className="pt-status-badge">
          &#9679; System Online: AI Paraphrasing Engine v1.0
        </span>
        <h1 className="pt-header-title">Advanced AI Paraphrasing Tool</h1>
        <p className="pt-header-subtitle">
          Reimagine your text with precision. Enhance clarity, tone, and delivery
          using our advanced neural rewriting model.
        </p>
      </header>

      <main className="pt-tool-container">
        {/* Left Side: Input Terminal */}
        <section className="pt-input-side">
          <div className="pt-panel-title">
            &#9654; Analysis Terminal
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '10px' }}>
              <span className="pt-status-badge" style={{ margin: 0, opacity: 0.8 }}>{wordCount.words} Words</span>
              <span className="pt-status-badge" style={{ margin: 0, opacity: 0.8 }}>{wordCount.chars} Characters</span>
            </div>
          </div>

          <div className="pt-textarea-container">
            <textarea
              className="pt-textarea"
              placeholder="Initialize rewriting: Paste content to paraphrase for AI insights..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
          </div>

          <div className="pt-options-bar">
            <div className="pt-mode-options">
              {['Standard', 'Fluency', 'Creative', 'Formal'].map((m) => (
                <button
                  key={m}
                  className={`pt-option-btn ${mode === m.toLowerCase() ? 'active' : ''}`}
                  onClick={() => setMode(m.toLowerCase())}
                >
                  {m}
                </button>
              ))}
            </div>

            <div className="pt-action-buttons">
              <button className="pt-btn-reset" onClick={handleReset}>Reset</button>
              <button
                className="pt-btn-paraphrase"
                onClick={handleParaphrase}
                disabled={loading || !inputText.trim()}
              >
                {loading ? <div className="pt-loading-spinner"></div> : 'Run Paraphrase Analysis'}
              </button>
            </div>
          </div>
        </section>

        {/* Right Side: Results */}
        <section className="pt-results-side">
          <div className="pt-results-content">
            {!paraphrasedText && !loading ? (
              <div className="pt-empty-state">
                <div className="pt-empty-icon">&#8987;</div>
                <h3 style={{ marginBottom: '10px' }}>Waiting for Input</h3>
                <p style={{ fontSize: '12px', opacity: 0.6 }}>
                  Neural engine standby. Paste text and initiate scan to see detailed AI rewriting analytics.
                </p>
              </div>
            ) : (
              <div className="pt-output-container">
                <div className="pt-paraphrase-output">
                  {paraphrasedText}
                </div>
                {paraphrasedText && !loading && (
                  <button className="pt-copy-btn" onClick={handleCopy}>
                    &#128203; {copyStatus}
                  </button>
                )}
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Footer Info Cards */}
      <section className="pt-info-section">
        <div className="pt-info-card">
          <div className="pt-info-icon">&#128737;</div>
          <h4>How to Use</h4>
          <p>Paste your text, select a rewriting mode (Standard, Fluency, Creative, or Formal), and click "Run Paraphrase Analysis" for instant results.</p>
        </div>
        <div className="pt-info-card">
          <div className="pt-info-icon">&#128301;</div>
          <h4>Model Methodology</h4>
          <p>Our tool utilizes a specialized AI model fine-tuned for high-fidelity paraphrasing and semantic preservation.</p>
        </div>
        <div className="pt-info-card">
          <div className="pt-info-icon">&#128200;</div>
          <h4>Rewriting Precision</h4>
          <p>Optimized for natural flow and grammatical integrity, delivering human-like rewrites with forensic precision.</p>
        </div>
      </section>
    </div>
  );
}

export default ParaphrasingTool;
