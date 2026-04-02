import React from "react";
import { Link } from "react-router-dom";
import './Footer.css'; // Import the CSS file for styling
import { FaInstagram, FaLinkedinIn, FaYoutube } from "react-icons/fa";
import { FaXTwitter } from "react-icons/fa6";
import OOKLogo from '../images/OOK-Logo.png';


const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-inner">

        {/* Site title or logo */}
        <div className="footer-section logo-section">
          <h3><img src={OOKLogo} alt="OOK Calculator Logo" width='60px' height='50px' style={{ marginRight: '10px' }} />OOK Calculator</h3>
          <br />
          <p>© {new Date().getFullYear()} OOK Calculator. All rights reserved.</p>
        </div>

        {/* Tools / Features */}
        <div className="footer-section links-section">
          <h4>Tools</h4>
          <br />
          <ul>
            <li><Link to="/PadEye">Pad eye</Link></li>
            <li><Link to="/BeamProperties">Beam Properties</Link></li>
            <li><Link to="/BeamDeflection">Beam Deflection</Link></li>
            <li><Link to="/NetForce">Net Force</Link></li>
            <li><Link to="/Plagiarism">Plagiarism Checker</Link></li>
            {/* Add other tools pages here */}
          </ul>
        </div>

        {/* About / Policy */}
        <div className="footer-section links-section">
          <h4>About</h4>
          <br />
          <ul>
            <li><Link to="/about">About Us</Link></li>
            <li><Link to="/Policy">Privacy Policy</Link></li>
            <li><Link to="/Blogs">Blogs</Link></li>
          </ul>
        </div>

        {/* Social Media */}
        <div className="footer-section social-section">
          <h4>Follow Us</h4>
          <br />
          <div className="social-icons">
            <a href="https://instagram.com/ookcalculator" target="_blank" rel="noopener noreferrer" aria-label="Instagram"><FaInstagram /></a>
            <a href="https://www.linkedin.com/in/ook-calculator-4223a7331/?originalSubdomain=in" aria-label="LinkedIn" target="_blank" rel="noopener noreferrer"><FaLinkedinIn /></a>
            <a href="https://x.com/ookcalculator" target="_blank" rel="noopener noreferrer" aria-label="Twitter"><FaXTwitter /></a>
            <a href="https://www.youtube.com/@OceanofknowledgeOOK" target="_blank" rel="noopener noreferrer" aria-label="YouTube"><FaYoutube /></a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
