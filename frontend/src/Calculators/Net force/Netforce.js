import React, { useState, useEffect } from "react";
import { Helmet } from 'react-helmet';
import backgroundPNG from '../../images/Netforce.jpg';   // Replace with your actual image path
import './Netforce.css';
import WhatisNetForce from "./Topics/WhatisNetForce";
import Howtocalculateforces from "./Topics/Howtocalculateforces";
// modules
import { Link } from 'react-router-dom';
// CSS
import '../../Css/BeamProperties.css'
import '../../Css/BeamDeflection.css'
import '../../Css/NumberLine.css'
import '../../Css/AboutUS.css'
import '../../Css/Navbar.css'
import '../../Css/Padeye.css'

// icons
import { GrLinkTop } from "react-icons/gr";
export default function Netforce() {
    const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };
  
  const [isActive3, setIsActive3] = useState(false);

  const toggleClass3 = () => {
    setIsActive3(prev => !prev);
  };

  const [forces, setForces] = useState([
    { magnitude: 50, angle: 30, label: "F₁", unit: "N" },
    { magnitude: 70, angle: 60, label: "F₂", unit: "N" },
  ]);

  const ForceUnits = ["N", "kN"];

  // ✅ Correct conversion helpers
  const toNewton = (value, unit) => {
    switch (unit) {
      case "kN": return value * 1000;
      case "N":
      default: return value;
    }
  };

  const fromNewton = (valueInN, unit) => {
    switch (unit) {
      case "kN": return valueInN / 1000;
      case "N":
      default: return valueInN;
    }
  };

  const handleForceChange = (index, field, value) => {
    const newForces = [...forces];

    if (field === "magnitude") {
      const unit = newForces[index].unit;
      newForces[index].magnitude = toNewton(parseFloat(value) || 0, unit);
    } else {
      newForces[index][field] = parseFloat(value) || 0;
    }

    setForces(newForces);
  };

  const handleForceUnitChange = (index, newUnit) => {
    const newForces = [...forces];
    const valueInN = newForces[index].magnitude;

    newForces[index].unit = newUnit;
    newForces[index].magnitude = valueInN; // keep stored in N
    setForces(newForces);
  };

  // --- Diagram Logic ---
  const [forceCount, setForceCount] = useState(2);
  const [customCount, setCustomCount] = useState("");

  const polarToCartesian = (magnitude, angle, scaleFactor) => {
    const rad = (angle * Math.PI) / 180;
    return {
      x: magnitude * Math.cos(rad) * scaleFactor,
      y: -magnitude * Math.sin(rad) * scaleFactor,
    };
  };

  const updateForces = (count) => {
    setForceCount(count);
    setForces(
      Array.from({ length: count }, (_, i) => ({
        magnitude: forces[i]?.magnitude || 50,
        angle: forces[i]?.angle || 0,
        label: `F${i + 1}`,
        unit: forces[i]?.unit || "N",
      }))
    );
  };

  const handleForceCountChange = (e) => {
    const value = e.target.value;
    if (value === "custom") {
      setCustomCount("");
    } else {
      updateForces(parseInt(value));
      setCustomCount("");
    }
  };

  const handleCustomCountChange = (e) => {
    const value = parseInt(e.target.value) || 0;
    setCustomCount(e.target.value);
    if (value > 0) {
      updateForces(value);
    }
  };

  const maxMagnitude = Math.max(...forces.map((f) => f.magnitude), 1);
  const maxDrawableLength = 200;
  const scaleFactor = maxDrawableLength / maxMagnitude;

  // --- Force Calculations ---
  const horizontalComponents = forces.map(f => f.magnitude * Math.cos((f.angle * Math.PI) / 180));
  const verticalComponents = forces.map(f => f.magnitude * Math.sin((f.angle * Math.PI) / 180));

  const Fx_internal = horizontalComponents.reduce((sum, val) => sum + val, 0);
  const Fy_internal = verticalComponents.reduce((sum, val) => sum + val, 0);

  const resultant_internal = Math.sqrt(Fx_internal ** 2 + Fy_internal ** 2);
  const direction_internal = Math.atan2(Fy_internal, Fx_internal) * (180 / Math.PI);

  const [FxUnit, setFxUnit] = useState('N');
  const [FyUnit, setFyUnit] = useState('N');
  const [ResultantUnit, setResultantUnit] = useState('N');
  const [DirectionUnit, setDirectionUnit] = useState('deg');

  const convertForce = (valueInN, unit) => fromNewton(valueInN, unit);

  // animation on scroll
  const [expanded, setExpanded] = useState(false);
  const [Sectionthird, setSectionthird] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 100 && !expanded) setExpanded(true);
      if (window.scrollY > 350 && !Sectionthird) setSectionthird(true);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [expanded, Sectionthird]);

  const [showfirstDiv, setShowFirstDiv] = useState(false);
  const [showSecondDiv, setShowSecondDiv] = useState(false);
  const [showThirdDiv, setShowThirdDiv] = useState(false);

  const handleToggleWithDelay = () => {
    if (!showfirstDiv && !showSecondDiv && !showThirdDiv) {
      setShowFirstDiv(true);
      setTimeout(() => {
        setShowSecondDiv(true);
        setTimeout(() => {
          setShowThirdDiv(true);
        }, 2000);
      }, 2000);
    } else {
      setShowThirdDiv(false);
      setTimeout(() => {
        setShowSecondDiv(false);
        setTimeout(() => {
          setShowFirstDiv(false);
        }, 2000);
      }, 2000);
    }
  };

  const handleCombinedClick = () => {
    handleToggleWithDelay();
    toggleClass3();
  };

  return (
    
    <>
    <Helmet>
        <title>Net Force Calculator - Structural Engineering</title>
        <meta
          name="description"
          content="Calculate net force in mechanical and structural applications with precision."
        />
        <link rel="canonical" href="https://www.ookcalculator.com/NetForce" />
      </Helmet>
      <div className='Background-Black'></div>
      <section className='background-white'>
        <div className="position-relative">
          {/* Image section */}
          <div className="position-relative overflow-hidden" style={{ height: '85vh' }}>
            <picture>
              <source type="image/webp" srcSet={backgroundPNG} />
              <img
                loading="lazy"
                src={backgroundPNG}
                alt="Background"
                className="h-100"
                width="600"
                height="400"
                style={{
                  objectFit: 'cover',
                  objectPosition: 'center',
                  width: '100%',
                  transform: 'translateX(0%)'
                }}
  fetchpriority="high"
  decoding="async"
              />
            </picture>
            <div className="Overlay-Black-header deflection"></div>
          </div>
          {/* Text section */}
          <div className="container text-left text-white position-absolute top-50 translate-middle Padeye"
            style={{
              left: '48%'
            }}
          >
            <h1 className="display-4" style={{ fontWeight: '600' }}>Netforce Calculator</h1>
            {/* <br/> */}
            <p className="fs-5">
              A Net Force Calculator is an advanced tool that accurately determines the resultant force acting on an object. It accounts for both magnitude and direction because it resolves each input into components and applies vector addition in order to deliver precise results.
              <br />

            </p>
          </div>
        </div>

        <section className="calculator-definition-section text-center py-5">
          <div className="container information-section-of-BeamProperties">
            <hr className="mb-4 Beam-properties-calculator-hr" />
            <br />
            <h2
              className={`display-4 mb-4 ${expanded ? "expanded" : ""}`}
              style={{ fontWeight: '600' }}
            >
              OOK Netforce Calculator
            </h2>

            <div className={`content ${expanded ? "expanded" : ""}`}>
              <p className="first-important lead mb-4" style={{ fontWeight: "500" }}>
                By simplifying complex calculations, this calculator helps users to accurately track and
                <br />
                analyse motion, equilibrium, and the resulting dynamics in various scenarios,
                <br />
                from academic physics problems to real-world engineering applications.
              </p>
              <p className="second-important lead mb-4" style={{ fontWeight: "500" }}>
                It serves as an essential resource for ensuring clarity and accuracy
                <br />
                in force-related computations.
              </p>

            </div>

            <hr className="Beam-properties-calculator-hr" />
            <br />
          </div>
        </section>
        <section className='container-fluid py-4 justify-content-center align-items-center d-flex'>
          <div className="p-6 space-y-6 NetforceCalculatorMain row structure-analysis-calculator-calculator">
            <div>
              {/* Input controls */}
                <br />
              <div className="d-flex items-center justify-content-around align-items-center space-x-4 ps-2 pe-2 pb-0 pt-4 rounded-lg" >
                <p className="font-semibold text-white">Number of forces: </p>
                <br />
                <div className="input-and-select-div netforce" style={{ width: '25%' }}>

                  <select
                    value={customCount ? "custom" : forceCount}
                    onChange={handleForceCountChange}
                    className="border p-1 rounded ms-2"
                  >
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((num) => (
                      <option key={num} value={num}>
                        {num}
                      </option>
                    ))}
                  </select>
                </div>

                {customCount !== "" && (
                  <input
                    type="number"
                    min="1"
                    value={customCount}
                    onChange={handleCustomCountChange}
                    className="border p-1 w-24 rounded"
                    placeholder="Enter number"
                  />
                )}
              </div>
            </div>
            {/* Diagram */}
            <div className="bg-white p-4">
              <svg width="500" height="500" className="bg-white">
                {/* Axes */}
                <line x1="250" y1="20" x2="250" y2="480" stroke="black" strokeWidth="2" />
                <line x1="20" y1="250" x2="480" y2="250" stroke="black" strokeWidth="2" />

                {/* Axis arrows (all 4 directions) */}
                <polygon points="250,20 245,35 255,35" fill="black" />
                <polygon points="250,480 245,465 255,465" fill="black" />
                <polygon points="480,250 465,245 465,255" fill="black" />
                <polygon points="20,250 35,245 35,255" fill="black" />

                {/* Axis labels */}
                <text x="470" y="240" fontSize="18" fontWeight="bold">x</text>
                <text x="30" y="240" fontSize="18" fontWeight="bold">-x</text>
                <text x="260" y="30" fontSize="18" fontWeight="bold">y</text>
                <text x="260" y="475" fontSize="18" fontWeight="bold">-y</text>
                <text x="260" y="265" fontSize="18" fontWeight="bold">O</text>

                {/* Origin circle */}
                <circle cx="250" cy="250" r="8" fill="black" stroke="black" />

                {/* Forces */}
                {forces.map((force, index) => {
                  const { x, y } = polarToCartesian(force.magnitude, force.angle, scaleFactor);
                  const endX = 250 + x;
                  const endY = 250 + y;

                  return (


                    <g key={index}>
                      {/* Force line */}
                      <line
                        x1="250"
                        y1="250"
                        x2={endX}
                        y2={endY}
                        stroke="black"
                        strokeWidth="3"
                        markerEnd="url(#arrowhead-black)"
                      />

                      {/* Projection lines */}
                      <line
                        x1={endX}
                        y1="250"
                        x2={endX}
                        y2={endY}
                        stroke="black"
                        strokeWidth="1.5"
                        strokeDasharray="5,5"
                      />
                      <line
                        x1="250"
                        y1={endY}
                        x2={endX}
                        y2={endY}
                        stroke="black"
                        strokeWidth="1.5"
                        strokeDasharray="5,5"
                      />

                      {/* Force label */}
                      <text
                        x={endX + (x > 0 ? 15 : -30)}
                        y={endY + (y > 0 ? 20 : -10)}
                        fontSize="18"
                        fontWeight="bold"
                      >
                        {force.label}
                      </text>
                    </g>
                  );
                })}

                {/* Arrowhead definition */}
                <defs>
                  <marker
                    id="arrowhead-black"
                    markerWidth="6"
                    markerHeight="4"
                    refX="6"
                    refY="2"
                    orient="auto" >
                    <polygon points="0 0, 6 2, 0 4" fill="black" />
                  </marker>
                </defs>
              </svg>
            </div>


            {/* Force inputs */}
            <div className="space-y-4 d-flex flex-column force-inputs ps-4 pe-4 pb-0 pt-4">
              <p className="font-semibold text-white mt-3 text-left ">Forces: </p>
              <br />
              {forces.map((force, index) => (
                <div key={index} className="d-grid items-center space-x-4" style={{ gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div className="input-and-select-div netforce">
                    <label className="font-semibold text-white" style={{ marginRight: '10px' }}> {force.label} </label>
                    {/* Magnitude input */}
                    <input
                      type="number"
                      value={fromNewton(force.magnitude, force.unit)}
                      onChange={(e) => handleForceChange(index, "magnitude", e.target.value)}
                      className="border p-1 w-24 rounded"
                      style={{ width: '80%', border: "none", outline: "none" }}
                      placeholder="Magnitude"
                      aria-label="Force"
                    />
                    <select
                      value={force.unit}
                      onChange={(e) => handleForceUnitChange(index, e.target.value)}
                      className="border p-1 rounded"
                    >
                      {ForceUnits.map((unit) => (
                        <option key={unit} value={unit}>{unit}</option>
                      ))}
                    </select>
                  </div>
                  {/* Angle input */}
                  {/* Angle input */}
                  <div className="input-and-select-div netforce">
                    <input
                      type="number"
                      value={force.angle}
                      onChange={(e) => handleForceChange(index, "angle", e.target.value)}
                      className="border p-1 w-24 rounded"
                      placeholder="Angle"
                      style={{ width: "80%", border: "none", outline: "none" }}
                      aria-label="Angle"
                    />
                    <select
                      value="deg"
                      className="border p-1 rounded ms-2"
                      disabled
                    >
                      <option value="deg">deg</option>
                    </select>
                  </div>

                  <br />
                </div>
              ))}
              <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div netforce'>
                <button className='structure-analysis-calculator-calculator-right-show-hidden-btn  netforce' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
              </div>
            </div>
          </div>

        </section>
        <br />
        <br />
        <div className={isActive3 ? 'show Sectionmodules' : 'hidden Sectionmodules'} style={{ height: '40vw' }}>
          <br />
          <br />
          <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw' }}>Resultant Force</h2>
          <br />
          <br />
          <div className='Section-properties-Solutions' style={{ borderRadius: '10px', width: '90%', margin: 'auto' }}>

            {/* Fx */}
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.3vw', fontWeight: '600' }}>
                Horizontal component (Fx):
              </p>
              <div className='Calculator-Side-A d-flex Netforce'>
                <input
                  className='calculator-input'
                  type="number"
                  value={convertForce(Fx_internal, FxUnit).toFixed(2)}
                  readOnly
                  aria-label="Horizontal component (Fx)"
                />
                <select value={FxUnit} onChange={(e) => setFxUnit(e.target.value)} className="ms-2">
                  {ForceUnits.map((unit) => (
                    <option key={unit} value={unit}>{unit}</option>
                  ))}
                </select>
              </div>
            </div>

            <br />

            {/* Fy */}
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.3vw', fontWeight: '600' }}>
                Vertical component (Fy):
              </p>
              <div className='Calculator-Side-A d-flex Netforce'>
                <input
                  className='calculator-input'
                  type="number"
                  value={convertForce(Fy_internal, FyUnit).toFixed(2)}
                  readOnly
                  aria-label="Vertical component (Fy)"
                />
                <select value={FyUnit} onChange={(e) => setFyUnit(e.target.value)} className="ms-2">
                  {ForceUnits.map((unit) => (
                    <option key={unit} value={unit}>{unit}</option>
                  ))}
                </select>
              </div>
            </div>

            <br />

            {/* Resultant */}
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.3vw', fontWeight: '600' }}>
                Magnitude of resultant force (F):
              </p>
              <div className='Calculator-Side-A d-flex Netforce'>
                <input
                  className='calculator-input'
                  type="number"
                  value={convertForce(resultant_internal, ResultantUnit).toFixed(2)}
                  readOnly
                  aria-label="Magnitude of resultant force (F)"
                />
                <select value={ResultantUnit} onChange={(e) => setResultantUnit(e.target.value)} className="ms-2">
                  {ForceUnits.map((unit) => (
                    <option key={unit} value={unit}>{unit}</option>
                  ))}
                </select>
              </div>
            </div>

            <br />

            {/* Direction */}
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.3vw', fontWeight: '600' }}>
                Direction of resultant force (θ):
              </p>
              <div className='Calculator-Side-A d-flex Netforce'>
                <input
                  className='calculator-input'
                  type="number"
                  value={direction_internal.toFixed(2)}
                  readOnly
                  aria-label="Direction of resultant force (θ)"
                />
                <select value={DirectionUnit} onChange={(e) => setDirectionUnit(e.target.value)} className="ms-2">
                  <option value="deg">deg</option>
                  {/* If later you want radians, just add: <option value="rad">rad</option> */}
                </select>
              </div>
            </div>


          </div>
        </div>
        <div className={showfirstDiv ? ' padeye height110 ' : ' padeye  height0 '} ></div>
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <WhatisNetForce />
        <br />
        <br />
        <br />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <Howtocalculateforces />
        <br />
        <br />
        <br />
                <section className='cse-header-top'>
                  <Link smooth='true' duration={500} offset={-70} onClick={scrollToTop} aria-label="Scroll to top">
                    <GrLinkTop className='' />
                  </Link>
                </section>
      </section>
    </>
  );
}
