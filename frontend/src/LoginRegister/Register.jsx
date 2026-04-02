import React, { useState } from "react";
import { Mail, Lock, User } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { GoogleLogin } from "@react-oauth/google";
import "./Register.css";

export default function RegisterPage({ setUser }) {

  const navigate = useNavigate();

  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: ""
  });

  // ==========================
  // NORMAL REGISTER
  // ==========================
  const handleRegister = async (e) => {
    e.preventDefault();

    if (form.password !== form.confirmPassword) {
      alert("Passwords do not match");
      return;
    }

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(form)
      });

      const data = await res.json();

      if (data.success) {
        alert("Registration successful!");
        navigate("/login");
      } else {
        alert(data.error || "Registration failed");
      }

    } catch (err) {
      console.error(err);
      alert("Server error");
    }
  };

  // ==========================
  // GOOGLE REGISTER / LOGIN
  // ==========================
const googleSuccess = async (response) => {
  try {
    const res = await fetch(`${process.env.REACT_APP_API_URL}/api/auth/google`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        token: response.credential
      })
    });

    const data = await res.json();

    if (data.success) {
      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));

      setUser(data.user);   // 🔥 IMPORTANT
      navigate("/");        // 🔥 IMPORTANT
    } else {
      alert(data.error);
    }

  } catch (err) {
    console.error(err);
  }
};
  return (
    <div className="register-container">

      <div className="register-card">

        <div className="reg-glow reg-glow-top"></div>
        <div className="reg-glow reg-glow-bottom"></div>

        <div className="reg-top-bar"></div>

        <h1 className="reg-title">OOK Calculator</h1>
        <p className="reg-subtitle">Create your account ✨</p>

        <form className="reg-form" onSubmit={handleRegister}>

          <div className="reg-input-group">
            <User className="reg-icon" />
            <input
              type="text"
              placeholder="Full Name"
              value={form.name}
              onChange={(e) => setForm({...form, name: e.target.value})}
              required
            />
          </div>

          <div className="reg-input-group">
            <Mail className="reg-icon" />
            <input
              type="email"
              placeholder="Email"
              value={form.email}
              onChange={(e) => setForm({...form, email: e.target.value})}
              required
            />
          </div>

          <div className="reg-input-group">
            <Lock className="reg-icon" />
            <input
              type="password"
              placeholder="Password"
              value={form.password}
              onChange={(e) => setForm({...form, password: e.target.value})}
              required
            />
          </div>

          <div className="reg-input-group">
            <Lock className="reg-icon" />
            <input
              type="password"
              placeholder="Confirm Password"
              value={form.confirmPassword}
              onChange={(e) => setForm({...form, confirmPassword: e.target.value})}
              required
            />
          </div>

          <button className="reg-btn">Register</button>

        </form>

        <div className="reg-divider">
          <span>OR</span>
        </div>

        <div className="google-btn">
          <GoogleLogin
  onSuccess={googleSuccess}
  onError={() => console.log("Google Register Failed")}
/>
        </div>

        <div className="reg-login">
          <p>Already have an account?</p>
          <span onClick={() => navigate("/login")}>Login</span>
        </div>

      </div>

    </div>
  );
}