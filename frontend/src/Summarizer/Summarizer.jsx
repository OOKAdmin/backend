import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet';
import './Summarizer.css';

import API_BASE from '../apiConfig';

function Summarizer() {
  const [inputText, setInputText] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [format, setFormat] = useState('paragraph');
  const [length, setLength] = useState('medium');
  const [wordCount, setWordCount] = useState({ words: 0, chars: 0 });
  const [copyStatus, setCopyStatus] = useState('Copy');

  useEffect(() => {
    const words = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
    const chars = inputText.length;
    setWordCount({ words, chars });
  }, [inputText]);

  const handleSummarize = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    setSummary('');

    try {
      const response = await fetch(`${API_BASE}/api/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, type: format, length }),
      });

      const data = await response.json();
      if (data.summary) {
        setSummary(data.summary);
      } else if (data.error) {
        setSummary(`Error: ${data.error}`);
      }
    } catch (err) {
      setSummary('Error: Could not connect to the backend. The server may be waking up; please try again in 30 seconds.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setInputText('');
    setSummary('');
    setCopyStatus('Copy');
  };

  const handleCopy = () => {
    if (!summary) return;
    navigator.clipboard.writeText(summary).then(() => {
      setCopyStatus('Copied!');
      setTimeout(() => setCopyStatus('Copy'), 2000);
    });
  };

  const handleSliderChange = (e) => {
    const val = parseInt(e.target.value);
    if (val === 1) setLength('short');
    else if (val === 2) setLength('medium');
    else setLength('long');
  };

  const getSliderValue = () => {
    if (length === 'short') return 1;
    if (length === 'medium') return 2;
    return 3;
  };

  return (
    <div className="sm-app">
      <Helmet>
        <title>AI Text Summarizer – OOK | Extract Key Insights</title>
        <meta
          name="description"
          content="Condense complex information into clear, actionable insights with the OOK AI Text Summarizer. Perfect for researchers and pros, our tool extracts key themes and critical data points from lengthy articles and documents, allowing you to master any subject in a fraction of the time."
        />
        <link rel="canonical" href="https://www.ook-calculator.com/Summarizer" />
      </Helmet>
      <br /><br /><br />
      <header className="sm-header">
        <span className="sm-status-badge">
          &#9679; System Online: AI Summarization Engine v2.4
        </span>
        <h1 className="sm-header-title">Advanced AI Text Summarizer</h1>
        <p className="sm-header-subtitle">
          Decode the essence of your content. Identify key insights and verify context
          with forensic precision using our neural language processing model.
        </p>
      </header>

      <main className="sm-tool-container">
        {/* Left Side: Input Terminal */}
        <section className="sm-input-side">
          <div className="sm-panel-title">
            &#9654; Analysis Terminal
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '10px' }}>
              <span className="sm-status-badge" style={{ margin: 0, opacity: 0.8 }}>{wordCount.words} Words</span>
              <span className="sm-status-badge" style={{ margin: 0, opacity: 0.8 }}>{wordCount.chars} Characters</span>
            </div>
          </div>

          <textarea
            className="sm-textarea"
            placeholder="Initialize scanning: Paste content to summarize for AI insights..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />

          <div className="sm-options-bar">
            <div className="sm-format-options">
              <button
                className={`sm-option-btn ${format === 'paragraph' ? 'active' : ''}`}
                onClick={() => setFormat('paragraph')}
              >
                Paragraph
              </button>
              <button
                className={`sm-option-btn ${format === 'bullets' ? 'active' : ''}`}
                onClick={() => setFormat('bullets')}
              >
                Bullet Points
              </button>
            </div>

            <div className="sm-slider-group">
              <span>Short</span>
              <input
                type="range"
                min="1"
                max="3"
                step="1"
                className="sm-length-slider"
                value={getSliderValue()}
                onChange={handleSliderChange}
              />
              <span>Long</span>
            </div>

            <div className="sm-action-buttons">
              <button className="sm-btn-reset" onClick={handleReset}>Reset</button>
              <button
                className="sm-btn-summarize"
                onClick={handleSummarize}
                disabled={loading || !inputText.trim()}
              >
                {loading ? <div className="sm-loading-spinner"></div> : 'Run Summary Analysis'}
              </button>
            </div>
          </div>
        </section>

        {/* Right Side: Results */}
        <section className="sm-results-side">
          <div className="sm-results-content">
            {!summary && !loading ? (
              <div className="sm-empty-state">
                <div className="sm-empty-icon">&#8987;</div>
                <h3 style={{ marginBottom: '10px' }}>Waiting for Input</h3>
                <p style={{ fontSize: '12px', opacity: 0.6 }}>
                  Neural engine standby. Paste text and initiate scan to see detailed AI probability analytics.
                </p>
              </div>
            ) : (
              <div className="sm-output-container">
                <div className="sm-summary-output">
                  {summary}
                </div>
                {summary && !loading && (
                  <button className="sm-copy-btn" onClick={handleCopy}>
                    &#128203; {copyStatus}
                  </button>
                )}
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Footer Info Cards */}
      <section className="sm-info-section">
        <div className="sm-info-card">
          <div className="sm-info-icon">&#128737;</div>
          <h4>How to Use</h4>
          <p>Simply paste your text in the analysis terminal, select your desired length and format, and click "Run Summary Analysis" to begin.</p>
        </div>
        <div className="sm-info-card">
          <div className="sm-info-icon">&#128301;</div>
          <h4>Model Methodology</h4>
          <p>Our system uses an advanced AI model fine-tuned on millions of documents for supreme summarization accuracy.</p>
        </div>
        <div className="sm-info-card">
          <div className="sm-info-icon">&#128200;</div>
          <h4>Summary Precision</h4>
          <p>Achieve up to 99% accuracy in capturing the main intent while maintaining perfect grammatical integrity.</p>
        </div>
      </section>
    </div>
  );
}

export default Summarizer;
