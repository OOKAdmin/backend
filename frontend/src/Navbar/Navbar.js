import { Link, useNavigate } from 'react-router-dom';
import OOKLogo from '../images/OOK-Logo.png';
import React, { useState, useEffect } from 'react';
import '../Css/Navbar.css';

export default function Navbar() {
  const [isActive, setIsActive] = useState(false);
  const [user, setUser] = useState(null);

  const navigate = useNavigate();

  const handleToggle = () => {
    setIsActive((prev) => !prev);
  };

  // ✅ Get user from localStorage
  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem("user"));
    setUser(storedUser);
  }, []);

  // ✅ Logout function
const handleLogout = () => {
  localStorage.clear();   // ✅ clears everything
  setUser(null);
  navigate("/");
};

  return (
    <>
      <nav className="ME-navbar">
        <img className="MeNavlogo" src={OOKLogo} alt="" width="180" height="36" loading="lazy"/>
        
        <div className="Background-Black"></div>

        <div className="ME-navbar-links">
          <div
            className={`ME-navbar-links-toggle ${isActive ? 'active' : ''}`}
            onClick={handleToggle}
          >
            <span className={`ME-navbar-links-toggle-bar ${isActive ? 'active' : ''}`}></span>
            <span className={`ME-navbar-links-toggle-bar ${isActive ? 'active' : ''}`}></span>
            <span className={`ME-navbar-links-toggle-bar ${isActive ? 'active' : ''}`}></span>
          </div>

          <ul className={`ME-navbar-links-ul ${isActive ? 'active' : ''}`}>
            
            <li className="ME-navbar-links-li dropdown">
              <span className="dropdown-title">Calculator ▾</span>
              <div className="mega-menu">
                <ul>
                  <li><Link to="/PadEye">Pad eye</Link></li>
                  <li><Link to="/BeamProperties">Beam Properties</Link></li>
                  <li><Link to="/BeamDeflection">Beam Deflection</Link></li>
                  <li><Link to="/NetForce">Net Force</Link></li>
                </ul>
              </div>
            </li>

            <li className="ME-navbar-links-li dropdown">
              <span className="dropdown-title">Tools ▾</span>
              <div className="mega-menu">
                <ul>
                  <li><Link to="/Plagiarism">Plagiarism Checker</Link></li>
                  <li><Link to="/Humanizer">Humanizer</Link></li>
                  <li><Link to="/AIDetector">AI Detector</Link></li>
                </ul>
              </div>
            </li>

            <li className="ME-navbar-links-li">
              <Link to="/AboutUs">About Us</Link>
            </li>

            {/* ✅ AUTH SECTION */}
            {!user ? (
  <>
    <li className="ME-navbar-links-li">
      <Link to="/login">Login</Link>
    </li>
    <li className="ME-navbar-links-li">
      <Link to="/register">Register</Link>
    </li>
  </>
) : (
  <>
    {/* ✅ LOGOUT FIRST */}
    <li className="ME-navbar-links-li">
      <span onClick={handleLogout} style={{ cursor: "pointer" }}>
        Logout
      </span>
    </li>

    {/* ✅ PROFILE ICON */}
    <li className="ME-navbar-links-li">
      {user.photo ? (
        <img
          src={user.photo}
          alt="profile"
          style={{
            width: "32px",
            height: "32px",
            borderRadius: "50%",
            objectFit: "cover"
          }}
        />
      ) : (
        <div
          style={{
            width: "32px",
            height: "32px",
            borderRadius: "50%",
            background: "#6c5ce7",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: "bold"
          }}
        >
          {user.name?.charAt(0).toUpperCase()}
        </div>
      )}
    </li>
  </>
)}
          </ul>
        </div>
      </nav>
    </>
  );
}