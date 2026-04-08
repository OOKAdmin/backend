import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Mail } from "lucide-react";
import "./ForgotPassword.css";

export default function ForgotPasswordPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState(null); // 'success' | 'error' | null
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setStatus(null);
    setMessage("");

    try {
      const res = await fetch(
        `${process.env.REACT_APP_API_URL || 'https://backend-7cv7.onrender.com'}/api/forgot-password`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email }),
        }
      );

      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Invalid server response");
      }

      if (data.success) {
        setStatus("success");
        setMessage(
          "If that email is registered, a password reset link has been sent. Please check your inbox (and spam folder)."
        );
      } else {
        setStatus("error");
        setMessage(data.error || "Something went wrong. Please try again.");
      }
    } catch (err) {
      console.error(err);
      setStatus("error");
      setMessage("Server error. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <br /><br /><br /><br />
      <div className="fp-container">
        <div className="fp-card">

          <div className="fp-glow fp-glow-top" />
          <div className="fp-glow fp-glow-bottom" />

          <div className="fp-top-bar" />

          {/* Icon */}
          <div className="fp-icon-wrap">
            <div className="fp-icon-circle">
              <Mail size={24} />
            </div>
          </div>

          <h1 className="fp-title">Forgot Password?</h1>
          <p className="fp-subtitle">
            Enter your email and we'll send you a reset link 🔑
          </p>

          {/* Success / Error feedback */}
          {status === "success" && (
            <div className="fp-success">✅ {message}</div>
          )}
          {status === "error" && (
            <div className="fp-error">❌ {message}</div>
          )}

          {/* Only show form if not yet successful */}
          {status !== "success" && (
            <form className="fp-form" onSubmit={handleSubmit}>
              <div className="fp-input-group">
                <Mail className="fp-icon" size={18} />
                <input
                  type="email"
                  placeholder="Enter your email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              <button className="fp-btn" type="submit" disabled={loading}>
                {loading ? "Sending..." : "Send Reset Link"}
              </button>
            </form>
          )}

          <div className="fp-back">
            <span onClick={() => navigate("/login")}>← Back to Login</span>
          </div>

        </div>
      </div>
    </>
  );
}
