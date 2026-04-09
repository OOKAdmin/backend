import React, { useState, useCallback, useMemo } from 'react';
import { Helmet } from 'react-helmet';
import './Humanizer.css';

import API_URL from '../apiConfig';

// ── Small helpers ──────────────────────────────────────────────────────────
const wordCount = (text) => (text.trim() ? text.trim().split(/\s+/).length : 0);
const charCount = (text) => text.length;

const sentimentLabel = (val) => {
  if (val > 0.3) return { label: 'Positive 😊', color: '#10b981' };
  if (val < -0.3) return { label: 'Negative 😟', color: '#ef4444' };
  return { label: 'Neutral 😐', color: '#a78bfa' };
};

const aiScoreColor = (score) => {
  if (score >= 70) return '#ef4444';
  if (score >= 40) return '#f59e0b';
  return '#10b981';
};

const readabilityLabel = (score) => {
  if (score >= 70) return 'Easy';
  if (score >= 50) return 'Medium';
  return 'Hard';
};

// ── MetricCard component ───────────────────────────────────────────────────
const MetricCard = ({ title, icon, before, after, unit = '', lowerBetter = false, single, singleColor }) => {
  const improved = lowerBetter ? after < before : after > before;
  const diff = after !== undefined ? Math.abs(after - before) : null;

  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div className="metric-title">{title}</div>
      {single !== undefined ? (
        <div className="metric-single" style={{ color: singleColor || '#a78bfa' }}>{single}</div>
      ) : (
        <div className="metric-values">
          <span className="metric-before">{before}{unit}</span>
          <span className="metric-arrow">→</span>
          <span className="metric-after" style={{ color: improved ? '#10b981' : '#ef4444' }}>
            {after}{unit}
          </span>
          {diff > 0 && (
            <span className="metric-diff" style={{ color: improved ? '#10b981' : '#ef4444' }}>
              {improved ? '▲' : '▼'} {diff}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

// ── LoadingDots ────────────────────────────────────────────────────────────
const LoadingDots = () => (
  <div className="loading-dots">
    <span /><span /><span />
  </div>
);

// ── Main App ───────────────────────────────────────────────────────────────
export default function HumanizerUI() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [strength, setStrength] = useState('medium');
  const [analysis, setAnalysis] = useState(null);
  const [copied, setCopied] = useState(false);

  const words = useMemo(() => wordCount(input), [input]);
  const chars = useMemo(() => charCount(input), [input]);

  const humanize = useCallback(async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError('');
    setAnalysis(null);
    setOutput('');

    try {
      const token = localStorage.getItem('token');
      const headers = {
        'Content-Type': 'application/json'
      };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const res = await fetch(`${API_URL}/api/humanize`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ text: input, strength }),
      });
      const data = await res.json();

      if (res.ok && data.success) {
        setOutput(data.humanized);
        setAnalysis({ before: data.analysis_before, after: data.analysis_after });
      } else {
        if (res.status === 429) {
          setError("You have reached your 8 free uses for today! Please Log In or Sign Up to unlock unlimited access.");
        } else {
          setError(data.error || 'Humanization failed or engine is warming up.');
        }
      }
    } catch (err) {
      console.error("Humanization error:", err);
      setError('Cannot connect to neural engine. Please check your internet or try again in a moment.');
    } finally {
      setLoading(false);
    }
  }, [input, strength]);

  const clear = () => {
    setInput('');
    setOutput('');
    setError('');
    setAnalysis(null);
    setCopied(false);
  };

  const copyToClipboard = () => {
    if (!output) return;
    navigator.clipboard.writeText(output).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const strengthOptions = [
    { id: 'light', label: '🪶 Light', desc: 'Subtle tweaks' },
    { id: 'medium', label: '⚖️ Medium', desc: 'Balanced rewrite' },
    { id: 'strong', label: '🔥 Strong', desc: 'Deep transform' },
  ];

  return (
    <div className="humanizer-app">
      <Helmet>
        <title>AI Text Humanizer – OOK | Undetectable AI Content</title>
        <meta
          name="description"
          content="Convert sterile AI-generated text into natural, engaging human writing with the OOK AI Humanizer. Our tool restructures machine-output to bypass even the most advanced AI detectors while maintaining your unique intent and emotional resonance. Achieve undetectable, high-quality content."
        />
        <link rel="canonical" href="https://www.ook-calculator.com/Humanizer" />
      </Helmet>
      <br/><br/><br/><br/>
      {/* ── HEADER ── */}
      <header className="header">
        <div className="header-bg" />
        <div className="header-content">
          <div className="header-badge">AI-Powered</div>
          <h1 className="header-title">✨ AI Humanizer – Transform AI Text Into Natural Human Writing</h1>
          <p className="header-sub">
            Paste any AI-generated content and get back writing that reads, flows, and feels genuinely human. No detection. No flags.
          </p>
        </div>
      </header>

      <main className="main">
        {/* ── STRENGTH SELECTOR ── */}
        <section className="controls-section">
          <span className="controls-label">Humanization Strength</span>
          <div className="strength-pills">
            {strengthOptions.map(({ id, label, desc }) => (
              <button
                key={id}
                id={`strength-${id}`}
                className={`strength-pill ${strength === id ? 'active' : ''}`}
                onClick={() => setStrength(id)}
                title={desc}
              >
                {label}
                <span className="pill-desc">{desc}</span>
              </button>
            ))}
          </div>
        </section>

        {/* ── EDITOR GRID ── */}
        <section className="editor-grid">
          {/* Input Panel */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">📝 Original Text</span>
              <div className="panel-meta">
                <span>{words} words</span>
                <span>{chars} chars</span>
              </div>
            </div>
            <textarea
              id="input-textarea"
              className="editor-textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Paste your AI-generated text here..."
              disabled={loading}
            />
          </div>

          {/* Output Panel */}
          <div className="panel output-panel">
            <div className="panel-header">
              <span className="panel-title">🧠 Humanized Text</span>
              {output && (
                <button id="copy-btn" className="copy-btn" onClick={copyToClipboard}>
                  {copied ? '✅ Copied!' : '📋 Copy'}
                </button>
              )}
            </div>
            <div className="output-area">
              {loading ? (
                <div className="output-loading">
                  <LoadingDots />
                  <p>Humanizing your text...</p>
                </div>
              ) : output ? (
                <div className="output-text">{output}</div>
              ) : (
                <div className="output-placeholder">
                  <span className="placeholder-icon">✨</span>
                  <p>Your humanized text will appear here</p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── ACTION ROW ── */}
        <section className="action-row">
          <button
            id="humanize-btn"
            className="btn-primary"
            onClick={humanize}
            disabled={loading || !input.trim()}
          >
            {loading ? '🔄 Humanizing...' : '✨ Humanize Text'}
          </button>
          <button id="clear-btn" className="btn-secondary" onClick={clear} disabled={loading}>
            🗑️ Clear
          </button>
        </section>

        {/* ── ERROR ── */}
        {error && (
          <div className="error-banner" id="error-banner">
            ❌ {error}
          </div>
        )}

        {/* ── ANALYSIS PANEL ── */}
        {analysis && (
          <section className="analysis-section" id="analysis-panel">
            <h2 className="analysis-title">📊 Text Analysis</h2>
            <div className="metrics-grid">
              <MetricCard
                title="AI Detection Score"
                icon="🤖"
                before={analysis.before.ai_score}
                after={analysis.after.ai_score}
                unit="%"
                lowerBetter
              />
              <MetricCard
                title="Readability"
                icon="📖"
                before={analysis.before.readability}
                after={analysis.after.readability}
                unit="/100"
              />
              <MetricCard
                title="Formality"
                icon="🎩"
                before={analysis.before.formality}
                after={analysis.after.formality}
                unit="%"
                lowerBetter
              />
              <MetricCard
                title="Sentiment"
                icon="💬"
                single={sentimentLabel(analysis.before.sentiment).label}
                singleColor={sentimentLabel(analysis.before.sentiment).color}
              />
            </div>

            {/* AI Score bar */}
            <div className="ai-score-bar-wrap">
              <div className="ai-score-row">
                <span>AI Score Before</span>
                <strong style={{ color: aiScoreColor(analysis.before.ai_score) }}>
                  {analysis.before.ai_score}%
                </strong>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width: `${analysis.before.ai_score}%`,
                    background: aiScoreColor(analysis.before.ai_score),
                  }}
                />
              </div>
              <div className="ai-score-row" style={{ marginTop: 8 }}>
                <span>AI Score After</span>
                <strong style={{ color: aiScoreColor(analysis.after.ai_score) }}>
                  {analysis.after.ai_score}%
                </strong>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width: `${analysis.after.ai_score}%`,
                    background: aiScoreColor(analysis.after.ai_score),
                  }}
                />
              </div>
            </div>
          </section>
        )}

        {/* ── RICH CONTENT SECTIONS ── */}
        <section className="info-content-section">
          <div className="info-card fade-in-up">
            <h2 className="info-heading">3 Easy Steps to Humanize AI Text</h2>
            <div className="steps-grid">
              <div className="step-item">
                <h3>Step 1 — Paste Your AI Content</h3>
                <p>Copy text generated by ChatGPT, Claude, Gemini or another AI-writing tool, and paste the text in the input box.</p>
              </div>
              <div className="step-item">
                <h3>Step 2 — Click on “Humanize”</h3>
                <p>With our AI Humanizer, your content is re-cast instantly in human language patterns with variations in tone, rhythm, vocabulary and sentence structure.</p>
              </div>
              <div className="step-item">
                <h3>Step 3 — Copy and Use</h3>
                <p>Get back polished, human-sounding text in seconds. Use it in essays, blogs, emails, reports, or any content — with complete confidence.</p>
              </div>
            </div>
          </div>

          <div className="info-card fade-in-up">
            <h2 className="info-heading">What Is an AI Humanizer?</h2>
            <p>It is an intelligent rewriting tool that transforms the AI-generated text into writing that sounds genuine, real, and indistinguishable from content written by a human itself.</p>
            <p>Using AI writing assistants like ChatGPT, Claude, or Gemini leads to outputs that leave patterns your detection tools can find. These consist of the same rephrases used again, phrasing that is too rigorously formal words that are not avoided like the plague, as well as algorithmically artificial uniformity.</p>
          </div>

          <div className="info-card fade-in-up">
            <h2 className="info-heading">Why Choose Our AI Humanizer?</h2>
            <ul className="info-list">
              <li><strong>Avoid AI Detectors:</strong> Our humanizer is designed to help your text pass all the top AI detector tools such as GPTZero, Turnitin, Originality.ai, Winston AI and Copyleaks—so your content never gets flagged.</li>
              <li><strong>Retain Your Original Meaning:</strong> Your thoughts remain unchanged. While we rewrite what you say, the way it sounds remains intact. The message, arguments and key conclusions are correct and entirely unchanged.</li>
              <li><strong>Immediate and Totally Free:</strong> Receive results in less than five seconds. No account required. No subscription. No credit card. Just paste, click, and copy.</li>
            </ul>
          </div>

          <div className="info-card fade-in-up">
            <h2 className="info-heading">SEO-Friendly Output & Benefits</h2>
            <p>Humanized text is helpful for ranking purposes, assisting in improving the SEO of your web pages. Humanized text mirrors natural language patterns, a key focus of Google's Helpful Content guidelines.</p>
            <ul className="info-list">
              <li><strong>Improve Readability:</strong> AI-generated text can feel mechanical and repetitive. The content we put out is the flow made naturally by our tool.</li>
              <li><strong>Increase Engagement:</strong> Speak as we do, and the reader is more likely to keep reading and not bounce.</li>
              <li><strong>SEO Advantage:</strong> Authentic, user friendly content is favoured by search engines.</li>
              <li><strong>Save Efforts:</strong> No need to rewrite each word yourself — the tool will rephrase your text immediately.</li>
            </ul>
          </div>

          <div className="cta-banner fade-in-up">
            <h2 className="cta-heading">📣 Ready to transform your AI text?</h2>
            <p>Paste your content into the Humanizer tool and click “Humanize” to get natural, polished writing in seconds.</p>
            <button className="btn-primary" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
              Back to Tool 🚀
            </button>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>✨ AI Text Humanizer — Powered by NLTK WordNet & TextBlob</p>
      </footer>
    </div>
  );
}