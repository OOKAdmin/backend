import React from "react";
import { Link } from "react-router-dom";
import './Footer.css';
import { FaInstagram, FaLinkedinIn, FaYoutube } from "react-icons/fa";
import { FaXTwitter } from "react-icons/fa6";
import OOKLogo from '../images/OOK-Logo.png';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="modern-footer">
      <div className="footer-glow"></div>
      <div className="footer-container">
        
        {/* BRAND SECTION */}
        <div className="footer-col brand-col">
          <Link to="/" className="footer-logo" onClick={scrollToTop}>
            <img src={OOKLogo} alt="OOK Logo" />
            <span>OOK Calculator</span>
          </Link>
          <p className="brand-desc">
            Empowering engineers and creators with precision-engineered calculators and advanced AI detection tools.
          </p>
          <div className="social-links">
            <a href="https://instagram.com/ookcalculator" target="_blank" rel="noreferrer"><FaInstagram /></a>
            <a href="https://www.linkedin.com/in/ook-calculator-4223a7331/" target="_blank" rel="noreferrer"><FaLinkedinIn /></a>
            <a href="https://x.com/ookcalculator" target="_blank" rel="noreferrer"><FaXTwitter /></a>
            <a href="https://www.youtube.com/@OceanofknowledgeOOK" target="_blank" rel="noreferrer"><FaYoutube /></a>
          </div>
        </div>

        {/* CALCULATORS SECTION */}
        <div className="footer-col">
          <h4>Calculators</h4>
          <ul className="footer-links">
            <li><Link to="/PadEye" onClick={scrollToTop}>Pad Eye Calculator</Link></li>
            <li><Link to="/BeamProperties" onClick={scrollToTop}>Beam Properties</Link></li>
            <li><Link to="/BeamDeflection" onClick={scrollToTop}>Beam Deflection</Link></li>
            <li><Link to="/NetForce" onClick={scrollToTop}>Net Force Calculator</Link></li>
            <li><Link to="/FourPointRiggingCalculator" onClick={scrollToTop}>4-Point Rigging</Link></li>
            <li><Link to="/FourPointRiggingwithSpreaderBarCalculator" onClick={scrollToTop}>Rigging w/ Spreader Bar</Link></li>
          </ul>
        </div>

        {/* AI TOOLS SECTION */}
        <div className="footer-col">
          <h4>AI Tools</h4>
          <ul className="footer-links">
            <li><Link to="/AIDetector" onClick={scrollToTop}>AI Content Detector</Link></li>
            <li><Link to="/Humanizer" onClick={scrollToTop}>AI Text Humanizer</Link></li>
            <li><Link to="/Paraphraser" onClick={scrollToTop}>Paraphraser</Link></li>
            <li><Link to="/Summarizer" onClick={scrollToTop}>Summarizer</Link></li>
            <li><Link to="/Plagiarism" onClick={scrollToTop}>Plagiarism Checker</Link></li>
            <li><Link to="/GrammarChecker" onClick={scrollToTop}>Grammar Checker</Link></li>
          </ul>
        </div>

        {/* RESOURCES SECTION */}
        <div className="footer-col">
          <h4>Resources</h4>
          <ul className="footer-links">
            <li><Link to="/Blogs" onClick={scrollToTop}>Technical Blogs</Link></li>
            <li><Link to="/AboutUs" onClick={scrollToTop}>About Our Mission</Link></li>
            <li><Link to="/Policy" onClick={scrollToTop}>Privacy & Terms</Link></li>
          </ul>
        </div>

      </div>

      <div className="footer-bottom">
        <div className="bottom-content">
          <p>© {currentYear} OOK Calculator. Built for precision.</p>
          <div className="status-indicator">
            <span className="pulse"></span> System Operational
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
