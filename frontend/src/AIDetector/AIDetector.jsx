// ================= FILE 1: AIDetectorUI.jsx =================
import React from "react";
import "./AIDetector.css";

export default function AIDetectorUI() {
  return (
    <div className="AIDetectorcontainer">

      {/* Navbar */}
      <br/>
      <br/>
      <br/>
      <br/>
      <br/>

      {/* Hero Section */}
      <div className="hero">
        <h2>Detect AI Generated Content Instantly</h2>
        <p>
          Paste your text below and get instant AI detection results with confidence score and detailed analysis.
        </p>
      </div>

      {/* Main Panel */}
      <div className="main-panel">

        {/* Input Section */}
        <div className="input-section">
          <h3>Your Text</h3>

          <textarea
            placeholder="Paste or write your content here..."
            className="textarea"
          />

          <div className="btn-group">
            <button className="btn primary">Analyze</button>
            <button className="btn secondary">Reset</button>
          </div>
        </div>

        {/* Side Panel */}
        <div className="side-panel">

          <div className="card">
            <h4>AI Score</h4>
            <div className="score">72%</div>

            <div className="progress">
              <div className="progress-bar" />
            </div>
          </div>

          <div className="card">
            <h4>Result</h4>
            <p>
              This text appears to be partially AI-generated with moderate confidence.
            </p>
          </div>

          <div className="card">
            <h4>Details</h4>
            <p>Words: 120</p>
            <p>Sentences: 8</p>
            <p>Complexity: Medium</p>
          </div>

        </div>

      </div>

    </div>
  );
}