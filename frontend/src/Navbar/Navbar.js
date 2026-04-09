import { Link, useNavigate } from 'react-router-dom';
import OOKLogo from '../images/OOK-Logo.png';
import React, { useState, useEffect } from 'react';
import { ChevronDown, User, LogOut, LayoutGrid, Wrench } from 'lucide-react';
import '../Css/Navbar.css';

export default function Navbar({ user, setUser }) {
  const [isActive, setIsActive] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [expandedDropdown, setExpandedDropdown] = useState(null);

  const navigate = useNavigate();

  const handleToggle = () => {
    setIsActive((prev) => !prev);
    setExpandedDropdown(null); // Reset when closing menu
  };

  const toggleDropdown = (name) => {
    setExpandedDropdown(expandedDropdown === name ? null : name);
  };

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleLogout = () => {
    localStorage.clear();
    setUser(null);
    navigate("/");
  };

  return (
    <nav className={`premium-navbar ${isScrolled ? 'scrolled' : ''}`}>
      <div className="nav-container">
        <div className="nav-logo-section">
          <Link to="/" className="logo-link">
            <img className="nav-logo" src={OOKLogo} alt="OOK" />
          </Link>
        </div>

        <div className={`nav-menu ${isActive ? 'mobile-active' : ''}`}>
          <ul className="nav-links">
            <li className={`nav-item dropdown ${expandedDropdown === 'calculators' ? 'expanded' : ''}`}>
              <span className="nav-link-title" onClick={() => toggleDropdown('calculators')}>
                <LayoutGrid size={18} /> Calculators <ChevronDown size={14} className={`chevron ${expandedDropdown === 'calculators' ? 'rotate' : ''}`} />
              </span>
              <div className="mega-dropdown">
                <div className="mega-content">
                  <div className="mega-column">
                    <h4>Engineering</h4>
                    <Link to="/PadEye" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>Pad Eye</Link>
                    <Link to="/BeamProperties" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>Beam Properties</Link>
                    <Link to="/BeamDeflection" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>Beam Deflection</Link>
                  </div>
                  <div className="mega-column">
                    <h4>Physics</h4>
                    <Link to="/NetForce" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>Net Force</Link>
                  </div>
                </div>
              </div>
            </li>

            <li className={`nav-item dropdown ${expandedDropdown === 'tools' ? 'expanded' : ''}`}>
              <span className="nav-link-title" onClick={() => toggleDropdown('tools')}>
                <Wrench size={18} /> Tools <ChevronDown size={14} className={`chevron ${expandedDropdown === 'tools' ? 'rotate' : ''}`} />
              </span>
              <div className="mega-dropdown">
                <div className="mega-content">
                  <div className="mega-column">
                    <h4>AI Writing</h4>
                    <Link to="/AIDetector" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>AI Detector</Link>
                    <Link to="/Humanizer" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>AI Humanizer</Link>
                    <Link to="/Paraphraser" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>Paraphraser</Link>
                    <Link to="/Summarizer" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>Summarizer</Link>
                  </div>
                  <div className="mega-column">
                    <h4>Text Analysis</h4>
                    <Link to="/Plagiarism" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>Plagiarism Checker</Link>
                    <Link to="/GrammarChecker" onClick={() => { setIsActive(false); setExpandedDropdown(null); }}>Grammar Checker</Link>
                  </div>
                </div>
              </div>
            </li>

            <li className="nav-item">
              <Link to="/Blogs" className="nav-link-standard" onClick={() => setIsActive(false)}>Blogs</Link>
            </li>
            
            <li className="nav-item">
              <Link to="/AboutUs" className="nav-link-standard" onClick={() => setIsActive(false)}>About Us</Link>
            </li>
          </ul>
        </div>

        <div className="nav-auth-section">
          {!user ? (
            <Link to="/login" className="login-trigger-btn">
              <User size={18} /> <span>Sign In</span>
            </Link>
          ) : (
            <div className="user-profile-dropdown">
              <div className="profile-trigger">
                {user.photo ? (
                  <img src={user.photo} alt="profile" className="avatar-img" />
                ) : (
                  <div className="avatar-placeholder">{user.name?.charAt(0).toUpperCase()}</div>
                )}
                <span className="user-name">{user.name?.split(' ')[0]}</span>
              </div>
              <div className="profile-menu">
                <div className="menu-header">
                  <p className="full-name">{user.name}</p>
                  <p className="email">{user.email}</p>
                </div>
                <button onClick={handleLogout} className="logout-btn">
                  <LogOut size={16} /> Logout
                </button>
              </div>
            </div>
          )}

          <button className="mobile-toggle" onClick={handleToggle} aria-label="Toggle Navigation">
            <div className={`hamburger ${isActive ? 'open' : ''}`}>
              <span></span>
              <span></span>
              <span></span>
            </div>
          </button>
        </div>
      </div>
    </nav>
  );
}