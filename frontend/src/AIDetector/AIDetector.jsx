import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ShieldCheck, Binary, Zap, Award, Info, RefreshCw, BarChart3, AlertCircle } from 'lucide-react';
import './AIDetector.css';
import PercentageRing from './components/PercentageRing';
import HighlightText from './components/HighlightText';

export default function AIDetectorUI() {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setIsAnalyzing(true);
    setResults(null);

    try {
      const token = localStorage.getItem('token');
      const headers = {
        'Content-Type': 'application/json'
      };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${process.env.REACT_APP_API_URL || 'https://backend-7cv7.onrender.com'}/api/analyze`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ text: text })
      });

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error("You have reached your 8 free uses for today! Please Log In or Sign Up to unlock unlimited access.");
        }
        throw new Error(data.error || "Neural engine encountered an error.");
      }

      setResults(data);
    } catch (error) {
      console.error("Failed to analyze:", error);
      alert(error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearText = () => {
    setText('');
    setResults(null);
  };

  return (
    <div className="ai-detector-app dashboard-theme">
      {/* ── AMBIENT BACKGROUND GLOWS ── */}
      <div className="glow-container">
        <div className="glow glow-1"></div>
        <div className="glow glow-2"></div>
        <div className="glow glow-3"></div>
      </div>

      <br/><br/><br/><br/><br/>

      {/* ── HERO COMMAND CENTER ── */}
      <motion.header 
        className="hero-dashboard"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <div className="hero-status-pill">
          <span className="pulse-dot"></span> System Online: AI Prediction Engine v2.4
        </div>
        <h1 className="hero-title-main">
          Advanced <span>AI Intelligence</span> Detector
        </h1>
        <p className="hero-subtitle-main">
          Decode the DNA of your content. Identify algorithmic signatures and verify human authenticity with forensic precision.
        </p>
      </motion.header>

      {/* ── MAIN DASHBOARD INTERFACE ── */}
      <main className="dashboard-layout">
        <div className="dashboard-grid glass-pane">
          
          {/* ── INPUT SECTION ── */}
          <section className="input-workstation">
            <div className="workstation-header">
              <div className="header-meta">
                <Search size={18} className="icon-pulse" />
                <span className="header-label">Analysis Terminal</span>
              </div>
              <div className="header-stats">
                <span className="stat-pill">{text.split(/\s+/).filter(w => w).length} Words</span>
                <span className="stat-pill">{text.length} Characters</span>
              </div>
            </div>

            <div className="editor-container">
              <textarea
                className="dashboard-textarea"
                placeholder="Initialize scanning: Paste content to analyze for AI signatures..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>

            <div className="workstation-actions">
              <button 
                className="secondary-action-btn"
                onClick={clearText}
                disabled={!text && !results}
              >
                Reset
              </button>
              <button 
                className={`primary-scan-btn ${isAnalyzing ? 'scanning' : ''}`}
                onClick={handleAnalyze}
                disabled={!text.trim() || isAnalyzing}
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="icon-spin" size={20} />
                    Analyzing Neural Patterns...
                  </>
                ) : (
                  <>
                    <Zap size={20} />
                    Run Forensic Analysis
                  </>
                )}
              </button>
            </div>
          </section>

          {/* ── RESULTS SECTION ── */}
          <section className="results-workstation">
            <AnimatePresence mode="wait">
              {!isAnalyzing && !results && (
                <motion.div 
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="empty-dashboard-state"
                >
                  <div className="empty-visual">
                    <Binary size={64} className="icon-flicker" />
                  </div>
                  <h3>Waiting for Input</h3>
                  <p>Neural engine standby. Paste text and initiate scan to see detailed AI probability analytics.</p>
                </motion.div>
              )}

              {isAnalyzing && (
                <motion.div 
                  key="loading"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="scanning-state"
                >
                  <div className="scan-line"></div>
                  <div className="loading-grid">
                    {[...Array(9)].map((_, i) => (
                      <div key={i} className="loading-pixel"></div>
                    ))}
                  </div>
                  <h3>Scanning Linguistic DNA</h3>
                  <p>Cross-referencing RoBERTa datasets and perplexity markers...</p>
                </motion.div>
              )}

              {results && !isAnalyzing && (
                <motion.div 
                  key="results"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="results-dashboard fade-in"
                >
                  <div className="result-header-row">
                    <BarChart3 size={20} />
                    <span>Scan Verdict</span>
                  </div>

                  <div className="main-score-display">
                    <PercentageRing percentage={results.aiPercentage} />
                    <div className="verdict-label-wrap">
                      <span className="verdict-tag">{results.mood}</span>
                    </div>
                  </div>
                  
                  <div className="sentence-breakdown-area">
                    <div className="breakdown-header">
                      <span className="title">Linguistic Highlighting</span>
                      <div className="dashboard-legend">
                        <span className="legend-p"><i className="dot d-human"></i> Human</span>
                        <span className="legend-p"><i className="dot d-ai"></i> AI</span>
                      </div>
                    </div>
                    <div className="highlight-scrollpane">
                      <HighlightText segments={results.segments} />
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </div>
      </main>

      {/* ── EDUCATIONAL DASHBOARD SECTIONS ── */}
      <section className="dashboard-info-center">
        <motion.div 
          className="info-box glass-card"
          whileHover={{ y: -5 }}
        >
          <div className="box-icon"><ShieldCheck size={32} /></div>
          <h3>Why Forensic Analysis?</h3>
          <p>Maintain academic integrity, protect SEO rankings, and ensure your brand voice remains authentic and human-centric.</p>
        </motion.div>

        <motion.div 
          className="info-box glass-card"
          whileHover={{ y: -5 }}
        >
          <div className="box-icon"><Binary size={32} /></div>
          <h3>Model Methodology</h3>
          <p>Our 24-marker engine analyzes perplexity and burstiness to identify patterns unique to LLMs like GPT-4 and Claude.</p>
        </motion.div>

        <motion.div 
          className="info-box glass-card"
          whileHover={{ y: -5 }}
        >
          <div className="box-icon"><BarChart3 size={32} /></div>
          <h3>Scan Accuracy</h3>
          <p>Tested against millions of human and AI outputs, providing industry-leading precision in text classification.</p>
        </motion.div>
      </section>

      <footer className="dashboard-footer">
        <p>&copy; 2026 OOK-Calculators AI Division | Advanced Neural Detection Engine</p>
      </footer>
    </div>
  );
}