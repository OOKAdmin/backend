import { GoogleLogin } from "@react-oauth/google";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";
export default function LoginPage({ setUser }) {

  const navigate = useNavigate();

  const [form, setForm] = useState({
    email: "",
    password: ""
  });

  // ==========================
// NORMAL LOGIN
// ==========================
const handleLogin = async (e) => {
  e.preventDefault();

  try {
    const res = await fetch(`${process.env.REACT_APP_API_URL}/api/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(form)
    });

    // 🔥 Fix crash (if backend sends HTML)
    const text = await res.text();
    let data;

    try {
      data = JSON.parse(text);
    } catch {
      throw new Error("Invalid server response");
    }

    if (data.success) {

      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));

      setUser(data.user);

      navigate("/");   // ✅ go to main page

    } else {

      if (data.error?.toLowerCase().includes("not found")) {
        alert("User not found. Kindly sign up first.");
        navigate("/register");
      }

      else if (data.error?.toLowerCase().includes("password")) {
        alert("Incorrect password. Please try again.");
      }

      else {
        alert(data.error || "Login failed");
      }
    }

  } catch (err) {
    console.error(err);
    alert("Server error. Please try again later.");
  }
};


// ==========================
// GOOGLE LOGIN
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

      setUser(data.user);

      navigate("/");   // ✅ go to main page

    } else {
      alert("Google login failed");
    }

  } catch (err) {
    console.error(err);
    alert("Google login error");
  }
};

  // ==========================
  // UI
  // ==========================
  return (
    <>
    <br/>
    <br/>
    <br/>
    <br/>
  <div className="login-container">

    <div className="login-card">

      <div className="glow glow-top"></div>
      <div className="glow glow-bottom"></div>

      <div className="top-bar"></div>

      <h1 className="title">OOK Calculator</h1>
      <p className="subtitle">Welcome back 👋</p>

      <form className="form" onSubmit={handleLogin}>

        <div className="input-group">
          <input
            type="email"
            placeholder="Email address"
            value={form.email}
            onChange={(e) =>
              setForm({ ...form, email: e.target.value })
            }
            required
          />
        </div>

        <div className="input-group">
          <input
            type="password"
            placeholder="Password"
            value={form.password}
            onChange={(e) =>
              setForm({ ...form, password: e.target.value })
            }
            required
          />
        </div>

        <button className="login-btn" type="submit">
          Login
        </button>

      </form>

      <div className="forgot">
        <span onClick={() => navigate("/forgot-password")}>
          Forgot Password?
        </span>
      </div>

      <div className="divider">
        <span>OR CONTINUE WITH</span>
      </div>

      <div className="google-btn" style={{ display: "flex", justifyContent: "center" }}>
        <GoogleLogin onSuccess={googleSuccess} />
      </div>

      <div className="signup">
        <p>Don’t have an account?</p>
        <span onClick={() => navigate("/register")}>Create Account</span>
      </div>

    </div>

  </div>
  </>
  );
}