import React, { useState } from "react";
import { Sparkles, Copy } from "lucide-react";
import "./Humanizer.css";

export default function HumanizerUI() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleHumanize = () => {
    setLoading(true);
    setTimeout(() => {
      setOutput("✨ This is your clean, natural, human-like rewritten text.");
      setLoading(false);
    }, 1200);
  };

  return (
    <div className="Humanizercontainer">

      {/* NAVBAR */}
      <div className="navbar">
<br/>
<br/>
<br/>
<br/>
      </div>

      {/* HERO */}
      <div className="hero">
        <h2>
          Make AI Text Feel <span className="highlight">Human</span>
        </h2>
        <p>
          Instantly rewrite robotic content into smooth, natural writing that passes detection and feels real.
        </p>
      </div>

      {/* MAIN TOOL */}
      <div className="toolBox">

        {/* INPUT */}
        <div className="inputBox">
          <div className="boxHeader">
            <span>Input</span>
            <span>{input.length} chars</span>
          </div>

          <textarea
            placeholder="Paste your AI text..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="textarea"
          />
        </div>

        {/* CENTER ACTION */}
        <div className="centerAction">
          <button
            onClick={handleHumanize}
            disabled={!input || loading}
            className="button"
          >
            <Sparkles />
          </button>

          <span>
            {loading ? "Processing..." : "Click to Humanize"}
          </span>
        </div>

        {/* OUTPUT */}
        <div className="outputBox">
          <div className="boxHeader">
            <span>Output</span>
            <button
              onClick={() => navigator.clipboard.writeText(output)}
              className="copyBtn"
            >
              <Copy size={16} />
            </button>
          </div>

          <div className="outputText">
            {output || "Your humanized text will appear here..."}
          </div>
        </div>

      </div>

    </div>
  );
}