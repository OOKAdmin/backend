import React, { useState, useEffect } from 'react';

import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet';

// images
// import Background from '../../images/BeamDeflaction-FirstSection-Background.jpg';

import backgroundJPG from '../../images/BeamDeflaction-FirstSection-Background.jpg';   // Replace with your actual image path
import backgroundWebP from '../../images/BeamDeflaction-FirstSection-Background.webp'; // WebP version of the image

import CrossSectionImg from '../../images/beam deflection image/cross section.png';
import YoungsModulusImg from '../../images/beam deflection image/youngs modulus.jpg';
import FixedSupportSymbolsImg from '../../images/beam deflection image/fixed support symbol.png';
import PinnedSupportSymbolsImg from '../../images/beam deflection image/BeamDeflaction-Blackbox-Pined-support.png';
import RollerSupportSymbolsImg from '../../images/beam deflection image/BeamDeflaction-Blackbox-roller-support.png';
import PointLoadSymbolsImg from '../../images/beam deflection image/point load symbol.png';
import PointLoadSymbolsForReactionsImg from '../../images/beam deflection image/point load symbol for reactions.png';
import DistributedLoadSymbolsImg from '../../images/beam deflection image/distributed support symbol.png';
import Non_DistributedLoadSymbolsImg1 from '../../images/beam deflection image/non distributed support symbol 01.png';
import Non_DistributedLoadSymbolsImg2 from '../../images/beam deflection image/non distributed support symbol 02.png';

// Files
import OutputParameterFile from './FormulaSections/OutputParameterFile';
import InputsParametersFile from './FormulaSections/InputsParametersFile';

// CSS
import '../../Css/BeamProperties.css'
import '../../Css/BeamDeflection.css'
import '../../Css/NumberLine.css'
import '../../Css/AboutUS.css'
import '../../Css/Navbar.css'
import '../../Css/Padeye.css'

// icons
import { MdDelete } from "react-icons/md";
import { GrLinkTop } from "react-icons/gr";
// import VideoPlayerSection from './VideoPlayer/VideoPlayerSection';

import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

const VideoPlayerSection = React.lazy(() => import("./VideoPlayer/VideoPlayerSection"));

export default function BeamDeflection() {
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };
  const [selectedOption, setSelectedOption] = useState("option1");
  const handleOptionChange = (option) => {
    setSelectedOption(option)
  }

  const [isActive3, setIsActive3] = useState(false);

  const toggleClass3 = () => {

    setIsActive3(true);
  };

  const [expanded, setExpanded] = useState(false);
  const [Sectionthird, setSectionthird] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 100 && !expanded) {
        setExpanded(true);
      }
      if (window.scrollY > 350 && !Sectionthird) {
        setSectionthird(true);
      }
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [expanded, Sectionthird]);

  const [showfirstDiv, setShowFirstDiv] = useState(false);
  const [showSecondDiv, setShowSecondDiv] = useState(false);
  const [showThirdDiv, setShowThirdDiv] = useState(false);

  // Toggle with delay function
  const handleToggleWithDelay = () => {
    if (!showfirstDiv && !showSecondDiv && !showThirdDiv) {
      // Open in sequence: First, Second, then Third
      setShowFirstDiv(true);
      setTimeout(() => {
        setShowSecondDiv(true);
        setTimeout(() => {
          setShowThirdDiv(true);
        }, 2000);
      }, 2000);
    }
  };
  const handleCombinedClick = () => {
    handleToggleWithDelay();
    toggleClass3();
    sendData()
  };

  const DropDowmOneMain = `
    rightBeamDeflactionDropDown
    reactiongraph
      
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hiddenBeamDeflection' : 'showBeamDeflection'}

      ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  `;
  const DropDowmOnerightMain = `
    rightBeamDeflactionDropDown rightsideBeamDeflactionDropDown
    
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hiddenBeamDeflection' : 'showBeamDeflection'}

    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  `;
  const DropDowmTwoMain = `
   ScrollTransactionTwoBeamDeflactionDropDown
    LoadBendingDeflectionclasses
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hiddenBeamDeflection' : 'showBeamDeflection'}

    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
    ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
  `;
  const DropDowmTworightMain = `
    ScrollTransactionTwoBeamDeflactionDropDown rightsideTwoBeamDeflactionDropDown
  LoadBendingDeflectionclasses
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hiddenBeamDeflection' : 'showBeamDeflection'}

    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
  `;
  // const DropDowmthirdMain = `
  //   ScrollTransactionTwoBeamDeflactionDropDown rightsideTwoBeamDeflactionDropDown top405
  // LoadBendingDeflectionclasses
  //   ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hiddenBeamDeflection' : 'showBeamDeflection'}

  //   ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  // ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
  // ${showThirdDiv ? 'ScrollTransactionThree' : ''}
  // `;

  // Deflection function code

  const YoungModulessunits = ['Mpa', 'Pa'];
  const YoungModulessConversionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };
  const [youngModules, setYoungModulessValue] = useState(200000);
  const [YoungModulesselectedUnit, setYoungModulessSelectedUnit] = useState('Mpa');
  const [YoungModulessInMPa, setYoungModulessInMPa] = useState(200000);

  const handleYoungModulesChange = (value) => {
    setYoungModulessValue(value);
    const factor = YoungModulessConversionFactors[YoungModulesselectedUnit][0];
    setYoungModulessInMPa(parseFloat(value) * factor);
  };

  const handleYoungModuleschange = (unit) => {
    let newValue = parseFloat(youngModules);
    if (unit === 'Pa' && YoungModulesselectedUnit === 'Mpa') {
      newValue *= 1e6;
      if (newValue >= 1000) {
        newValue = newValue.toExponential(3);
      }
    } else if (unit === 'Mpa' && YoungModulesselectedUnit === 'Pa') {
      newValue /= 1e6;
    }
    setYoungModulessSelectedUnit(unit);
    setYoungModulessValue(isNaN(newValue) ? 0 : newValue);
  };


  const [area, setArea] = useState(1780);
  const [areaSelectedUnit, setareaSelectedUnit] = useState('mm²');
  const [internalareaValue, setInternalareaValue] = useState(1780); // Always in mm
  const areaUnits = ['mm²', 'm²'];
  const areaMetricConversionFactors = {
    'mm²': [1, 1e-6], // 1 mm² = 1e-6 m²
    'm²': [1e6, 1],   // 1 m² = 1e6 mm²
  };

  const handleareaInputChange = (value) => {
    setArea(value);
    const factor = areaMetricConversionFactors[areaSelectedUnit][0];
    setInternalareaValue(parseFloat(value) * factor);
  };

  const handleareaUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(area) * areaMetricConversionFactors[areaSelectedUnit][0];
    const convertedValue = newMetricValueInMM / areaMetricConversionFactors[unit][0];
    setareaSelectedUnit(unit);
    setArea(isNaN(convertedValue) ? 0 : convertedValue);
  };

  const [inertia, setInertia] = useState(6660000);
  const [InertiaSelectedUnit, setInertiaSelectedUnit] = useState('mm⁴');
  const [internalInertiaValue, setInternalInertiaValue] = useState(6660000); // Always in mm
  const InertiaUnits = ['mm⁴', 'm⁴'];
  const InertiaMetricConversionFactors = {
    'mm⁴': [1, 1e-12], // 1 mm⁴ = 1e-12 m⁴
    'm⁴': [1e12, 1],   // 1 m⁴ = 1e12 mm⁴
  };

  const handleInertiaInputChange = (value) => {
    setInertia(value);
    const factor = InertiaMetricConversionFactors[InertiaSelectedUnit][0];
    setInternalInertiaValue(parseFloat(value) * factor);
  };

  const handleInertiaUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(inertia) * InertiaMetricConversionFactors[InertiaSelectedUnit][0];
    const convertedValue = newMetricValueInMM / InertiaMetricConversionFactors[unit][0];
    setInertiaSelectedUnit(unit);
    setInertia(isNaN(convertedValue) ? 0 : convertedValue);
  };

  const [supports, setSupports] = useState([{ type: 'fixed', position: 0 }]);
  const [pointLoads, setPointLoads] = useState([{ position: 0, magnitude: 10000 }]);
  const [pointLoadSelectedUnit, setPointLoadSelectedUnit] = useState('KN'); // Default unit is Newton



  const [distributedLoads, setDistributedLoads] = useState([
    { start_position: 0, end_position: 0, start_magnitude: 0, end_magnitude: 0 },
  ]);
  const [distributedLoadSelectedUnit, setDistributedLoadSelectedUnit] = useState('KN/m'); // Default unit is Newton/meter


  const [reactionData, setReactionData] = useState([]);
  const [deflectionData, setDeflectionData] = useState([]);
  const [shearForceData, setShearForceData] = useState([]);
  const [bendingMomentData, setBendingMomentData] = useState([]);
  const handlePointLoadChange = (index, field, value) => {
    const newPointLoads = [...pointLoads];
    newPointLoads[index][field] = Number(value);
    setPointLoads(newPointLoads);
  };

  const handleDistributedLoadChange = (index, field, value) => {
    const newDistributedLoads = [...distributedLoads];
    newDistributedLoads[index][field] = Number(value);
    setDistributedLoads(newDistributedLoads);
  };

  const handleSupportChange = (index, field, value) => {
    const newSupports = [...supports];
    newSupports[index][field] = value;
    setSupports(newSupports);
  };

  const addSupport = () => {
    setSupports([...supports, { type: 'fixed', position: 0 }]);
  };
  const deleteSupport = (index) => {
    const newSupports = supports.filter((_, i) => i !== index);
    setSupports(newSupports);
  };
  const addPointLoad = () => {
    setPointLoads([...pointLoads, { position: 0, magnitude: 0 }]);
  };
  const deletePointLoad = (index) => {
    const newPointLoads = pointLoads.filter((_, i) => i !== index);
    setPointLoads(newPointLoads);
  };

  const addDistributedLoad = () => {
    setDistributedLoads([...distributedLoads, { start_position: 0, end_position: 0, start_magnitude: 0, end_magnitude: 0 }]);
  };
  const deleteDistributedLoad = (index) => {
    const newDistributedLoads = distributedLoads.filter((_, i) => i !== index);
    setDistributedLoads(newDistributedLoads);
  };

  const [length, setLength] = useState(2000); // Length in selected unit
  const [internalLengthValue, setInternalLengthValue] = useState(0); // Always in mm
  const [lengthSelectedUnit, setLengthSelectedUnit] = useState('mm');
  const LengthUnits = ['mm', 'm'];
  const LengthMetricConversionFactors = {
    mm: [1, 0.001],
    m: [1000, 1],
  };

  // Handle input change for length
  const handleLengthInputChange = (value) => {
    setLength(value);
    const factor = LengthMetricConversionFactors[lengthSelectedUnit][0];
    setInternalLengthValue(parseFloat(value) * factor);
  };

  // Handle unit change for length
  const handleLengthUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(length) * LengthMetricConversionFactors[lengthSelectedUnit][0];
    const convertedValue = newMetricValueInMM / LengthMetricConversionFactors[unit][0];
    setLengthSelectedUnit(unit);
    setLength(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Dynamically calculate step size
  const calculateStepSize = () => {
    const desiredLabels = 5; // Target number of labels
    const range = length; // Use length in the selected unit for display
    if (range < 10) {
      return 0.5; // Show points for small ranges
    }
    return Math.ceil(range / desiredLabels / 5) * 5; // Round to nearest 5 for clean steps
  };

  const step = calculateStepSize(); // Calculate step size
  const values = Array.from({ length: Math.ceil(length / step) + 1 }, (_, i) => i * step);

  // Get the image source based on the support type
  // Function to dynamically get the support image
  const getSupportImage = (type) => {
    switch (type) {
      case "fixed":
        return FixedSupportSymbolsImg; // Replace with your image path
      case "roller":
        return RollerSupportSymbolsImg; // Replace with your image path
      case "pinned":
        return PinnedSupportSymbolsImg; // Replace with your image path
      default:
        return "";
    }
  };


  // Send data to the backend
  const sendData = () => {
    axios
      .post(
        'https://ook10-2.onrender.com', // Ensure this matches your Flask host and port
        {
          length: Number(length), // Use internalLengthValue in mm for the backend
          youngmodules: Number(youngModules),
          area: Number(area),
          inertia: Number(inertia),
          supports: supports.map((s) => ({ type: s.type, position: Number(s.position) })),
          point_loads: pointLoads,
          distributed_loads: distributedLoads,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      )
      .then((response) => {
        setReactionData(response.data.reactions);
        setDeflectionData(response.data.deflection_data);
        setShearForceData(response.data.shear_force_data);
        setBendingMomentData(response.data.bending_moment_data);
        console.log('Response from Flask:', response.data);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };


  const deflectionChartData = {
    labels: deflectionData.map(item => item.position),
    datasets: [
      {
        fill: true,
        borderColor: '#1f77b4',
        backgroundColor: 'rgba(31, 119, 180, 0.1)',
        pointBackgroundColor: '#1f77b4',
        pointBorderColor: '#fff',
        data: deflectionData.map(item => item.deflection),
        pointRadius: 0,
      },
    ],
  };

  const shearForceChartData = {
    labels: shearForceData.map(item => item.position),
    datasets: [
      {
        fill: true,
        borderColor: '#1f77b4',
        backgroundColor: 'rgba(31, 119, 180, 0.1)',
        pointBackgroundColor: '#1f77b4',
        pointBorderColor: '#fff',
        pointRadius: 0,
        data: shearForceData.map(item => item.shear_force),
      },
    ],
  };

  const bendingMomentChartData = {
    labels: bendingMomentData.map(item => item.position),
    datasets: [
      {
        fill: true,
        borderColor: '#1f77b4',
        backgroundColor: 'rgba(31, 119, 180, 0.1)',
        pointBackgroundColor: '#1f77b4',
        pointBorderColor: '#fff',
        pointRadius: 0,
        data: bendingMomentData.map(item => item.bending_moment),
      },
    ],
  };

  // const beamLength = 1000; // Example beam length in mm, replace with your actual beam length

  const shearForceChartDataoptions = {
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Length',
        },
        ticks: {
          maxTicksLimit: 8,

        },
      },
      y: {
        title: {
          display: true,
          text: 'Shear Force ',
        },
        ticks: {
          maxTicksLimit: 5,
          callback: function (value) {
            // Format ticks for the y-axis
            return value >= 1_000_000
              ? `${value / 1_000_000}M`
              : value >= 1_000
                ? `${value / 1_000}k`
                : value <= -1_000_000
                  ? `${value / 1_000_000}M`
                  : value <= -1_000
                    ? `${value / 1_000}k`
                    : value;
          },
        },
      },
    },
  };

  const bendingMomentChartDataoptions = {
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Length',
        },
        ticks: {
          maxTicksLimit: 8,
        },
      },
      y: {
        title: {
          display: true,
          text: 'Bending Moment',
        },
        ticks: {
          maxTicksLimit: 5,
          callback: function (value) {
            // Format ticks for the y-axis
            return value >= 1_000_000
              ? `${value / 1_000_000}M`
              : value >= 1_000
                ? `${value / 1_000}k`
                : value <= -1_000_000
                  ? `${value / 1_000_000}M`
                  : value <= -1_000
                    ? `${value / 1_000}k`
                    : value;
          },
        },
      },
    },
  };
  const deflectionChartDataoptions = {
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Length',
        },
        ticks: {
          maxTicksLimit: 8,
        },
        max: length, // Ensure the last tick aligns with the beam length
      },
      y: {
        title: {
          display: true,
          text: 'Deflection '
        },
        ticks: {
          maxTicksLimit: 5,
          callback: function (value) {
            // Format ticks for the y-axis
            return value >= 1_000_000
              ? `${value / 1_000_000}M`
              : value >= 1_000
                ? `${value / 1_000}k`
                : value <= -1_000_000
                  ? `${value / 1_000_000}M`
                  : value <= -1_000
                    ? `${value / 1_000}k`
                    : value;
          },
        },
      },
    },
  };


  return (
    <>
      <Helmet>
        <title>Beam Deflection Calculator – OOK | Structural Analysis Tool</title>
        <meta
          name="description"
          content="Analyze beam deflection under various loading conditions with OOK's Beam Deflection Calculator. Ideal for engineers and architects."
        />
        <link rel="canonical" href="https://www.ookcalculator.com/BeamDeflection" />

      </Helmet>


      <div className='Background-Black'></div>
      <section className='background-white'>
        <div className="position-relative">
          {/* Image section */}
          <div className="position-relative overflow-hidden" style={{ height: '85vh' }}>
            <picture>
              <source type="image/webp" srcSet={backgroundWebP} />
              <img
                loading="lazy"
                src={backgroundJPG}
                alt="Background"
                className="h-100 BeamDeflectionHeaderImage"
                width="600"
                height="400"
                fetchpriority="high"
                decoding="async"
                style={{
                  objectFit: 'cover',
                  objectPosition: 'center',
                  width: '120%',
                  transform: 'translateX(-12%)'
                }}
              />
            </picture>
            <div className="Overlay-Black-header deflection"></div>
          </div>

          {/* Text section */}
          <div className="container text-left text-white position-absolute top-50 translate-middle BeamDeflection"
            style={{
              left: '48%'
            }}
          >
            <h1 className="display-4" style={{ fontWeight: '600' }}>Beam Deflection Calculator</h1>
            <p className="fs-5">
              Beam Deflection Calculator is a powerful tool used by<br />
              engineers and architects for analyzing the behaviour of<br />
              beams under various loading conditions.
            </p>
          </div>
        </div>

        <section className="calculator-definition-section text-center py-5">
          <div className="container information-section-of-BeamProperties">
            <hr className="mb-4" />
            <br />
            <h2
              className={`display-4 mb-4 ${expanded ? "expanded" : ""}`}
              style={{ fontWeight: '600' }}
            >
              Beam Deflection Calculator
            </h2>

            <div className={`content ${expanded ? "expanded" : ""}`}>
              <p className="first-important lead mb-4" style={{ fontWeight: "500" }}>
                With the help of the Beam Deflection Calculator, you can easily<br />
                examine how different kinds of beams deflect under various loading scenarios.<br />
                Beam deflection is an important aspect in structural engineering and construction.
              </p>
              <p className="second-important lead mb-4" style={{ fontWeight: "500" }}>
                It ensures the structural integrity of the beam and helps to prevent any potential  deformation or<br /> damage. You can easily find the deflection at any point along the length of the beam<br />  by entering parameters like the beam's material, dimensions, and applied loads.
              </p>
              <p className=" lead" style={{ fontWeight: "500" }}>
                By using this calculator you can easily find the beam's reactions,<br />  maximum deflection, bending moment & shear stress.
              </p>
            </div>

            <hr className="mt-4" />
          </div>
        </section>
        <section className="container-fluid py-4 justify-content-center align-items-center d-flex">
          <div className="row structure-analysis-calculator-calculator">
            {/* Left Section */}
            <div className="col-12 flex-grow-1 col-lg-3 col-md-12 col-sm-12 col-xs-12 text-center py-5 structure-analysis-calculator-calculator-left ps-0 pe-0">
              <div className="d-flex flex-column gap-0 w-100 text-center" style={{ justifyContent: 'center', alignItems: 'center' }}>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option1")}>Length of Beam</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option2")}>Young's Modulus</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option3")}>Cross Section</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option4")}>Support</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option5")}>Load</button>
              </div>
            </div>

            {/* Center Section */}
            <div className="col-12 flex-grow-2  col-lg-5 col-md-12 col-sm-12 col-xs-12 py-0 text-center structure-analysis-calculator-calculator-center bg-white justify-content-center align-items-center d-flex BeamDeflection" style={{ padding: '0' }} >
              {selectedOption === 'option1' && (
                <div className='CenterofDeflectionCalculator' style={{ width: '90%' }}>
                  <div className="number-line-container">
                    <div className="line" />
                    <div className="numbers">
                      {values && values.map((value, index) => (
                        <span
                          key={index}
                          className="number"
                          style={{
                            left: `${(value / length) * 100}%`, // Position each number dynamically
                          }}
                        >
                          {value.toLocaleString()}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {selectedOption === 'option2' && <img src={YoungsModulusImg} alt="Rectangle" className="img-fluid" style={{ height: '100%' }} />}
              {selectedOption === 'option3' && <img src={CrossSectionImg} alt="Hollow Rectangle" className="img-fluid custom-img" />}
              {selectedOption === 'option4' && (
                <div className='CenterofDeflectionCalculator' style={{ width: '90%' }}>
                  <div className="number-line-container">
                    {/* {/* Line  */}
                    <div className="line">
                      {supports.map((support, index) => (
                        <img
                          key={index}
                          src={getSupportImage(support.type)}
                          alt={support.type}
                          className={`support-image ${support.type}-support`} // Dynamically add class based on type
                          style={{
                            left: `${Math.min((support.position / length) * 100, 100)}%`, // Ensure max left is 100%
                            width: 'auto',
                            height: '15px',
                          }}
                        />

                      ))}
                    </div>

                    {/* {/* Render Numbers Dynamically  */}
                    <div className="numbers">
                      {values && values.map((value, index) => (
                        <span
                          key={index}
                          className="number"
                          style={{
                            left: `${(value / length) * 100}%`, // Position each number dynamically
                          }}
                        >
                          {value.toLocaleString()}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {selectedOption === 'option5' && (
                <div className='CenterofDeflectionCalculator' style={{ width: '90%' }}>
                  <div className="number-line-container load">
                    <div className="line">
                      {distributedLoads && distributedLoads.map((load, index) => (
                        <div
                          key={index}
                          className="distributed-load-container"
                          style={{
                            left: `${(load.start_position / length) * 100}%`, // Starting position of the distributed load
                            width: `${((load.end_position - load.start_position) / length) * 100}%`, // Length of the distributed load
                          }}
                        >
                          {((load.start_magnitude > load.end_magnitude && load.end_position > 0) || (load.end_magnitude < load.start_magnitude && load.start_position > 0)) && (
                            <>
                              <div className="distributed-load-magnitude">
                                <div>{load.start_magnitude}</div>
                                <div>{load.end_magnitude}</div>
                              </div>
                              <img
                                src={Non_DistributedLoadSymbolsImg2}
                                alt="Distributed Load"
                                className="distributed-load-image"
                              />
                            </>
                          )}
                          {((load.end_magnitude > load.start_magnitude && load.end_position > 0) || (load.start_magnitude < load.end_magnitude && load.start_position > 0)) && (
                            <>
                              <div className="distributed-load-magnitude">
                                <div>{load.start_magnitude}</div>
                                <div>{load.end_magnitude}</div>
                              </div>
                              <img
                                src={Non_DistributedLoadSymbolsImg1}
                                alt="Distributed Load"
                                className="distributed-load-image"
                              />
                            </>
                          )}
                          {((load.start_magnitude !== 0 || load.end_magnitude !== 0 || load.start_position !== 0 || load.end_position !== 0) &&
                            ((load.end_magnitude === load.start_magnitude) || (load.end_position > 0 && load.start_position > 0))) && (
                              <>
                                <div className="distributed-load-magnitude">
                                  <div>{load.start_magnitude}</div>
                                  <div>{load.end_magnitude}</div>
                                </div>
                                <img
                                  src={DistributedLoadSymbolsImg}
                                  alt="Distributed Load"
                                  className="distributed-load-image"
                                />
                              </>
                            )}

                        </div>
                      ))}
                    </div>

                    {pointLoads && pointLoads.map((pointLoad, index) => (
                      <div
                        key={index}
                        className="point-load-container"
                        style={{
                          left: `${(pointLoad.position / length) * 100}%`, // Calculate position dynamically
                        }}
                      >

                        {(pointLoad.position > 0 || pointLoad.magnitude > 0) && (
                          <>
                            <div className="point-load-magnitude">{pointLoad.magnitude}</div>
                            <img
                              src={PointLoadSymbolsImg}
                              alt="PL"
                              className="point-load-image"
                            />
                          </>
                        )}
                      </div>
                    ))}

                    <div className="numbers">
                      {values && values.map((value, index) => (
                        <span
                          key={index}
                          className="number"
                          style={{
                            left: `${(value / length) * 100}%`, // Position each number dynamically
                          }}
                        >
                          {value.toLocaleString()}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

              )}
            </div>

            {/* Right Section */}
            <div className="col-12 flex-grow-1  col-lg-3 col-md-12 col-sm-12 col-xs-12 text-center py-3 bemProperties structure-analysis-calculator-calculator-right BeamDeflection" >
              <h3 className="text-white mt-3">Input</h3>
              <div className="mt-3">

                {selectedOption === 'option1' && (
                  <>
                    <p className='text-white ' style={{ textAlign: 'center', fontWeight: '600', }}>Beam-Length</p>
                    <br />
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Length of Beam :</p>
                      <div className='Calculator-Side-A' style={{ width: '80%' }}>
                        <h6 className='sigma-symbol' >(L)</h6>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={length}
                            onChange={((e) => handleLengthInputChange(e.target.value))}
                            aria-label="Length of Beam"
                          />
                          <select
                            className='Calculator-select-option'
                            value={lengthSelectedUnit}
                            onChange={((e) => handleLengthUnitChange(e.target.value))}
                          >
                            {LengthUnits && LengthUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </>
                )}
                {selectedOption === 'option2' && (
                  <>
                    <p className='text-white ' style={{ textAlign: 'center', fontWeight: '600', }}>Young's Modulus</p>
                    <br />
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Young's Modulus:</p>
                      <div className='Calculator-Side-A' style={{ width: '80%' }}>
                        <h6 className='sigma-symbol'>(E)</h6>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={youngModules}
                            onChange={((e) => handleYoungModulesChange(e.target.value))}
                            aria-label="Young's Modulus"
                          />
                          <select
                            className='Calculator-select-option'
                            value={YoungModulesselectedUnit}
                            onChange={((e) => handleYoungModuleschange(e.target.value))}
                          >
                            {YoungModulessunits && YoungModulessunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </>
                )}
                {selectedOption === 'option3' && (
                  <>
                    <p className='text-white' style={{ textAlign: 'center', fontWeight: '600', }}>Cross Section</p>
                    <br />

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Area :</p>
                      <div className='Calculator-Side-A' style={{ width: '80%' }}>
                        <h6 className='sigma-symbol'>(A)</h6>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={area}
                            onChange={((e) => handleareaInputChange(e.target.value))}
                            aria-label="Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={areaSelectedUnit}
                            onChange={((e) => handleareaUnitChange(e.target.value))}
                          >
                            {areaUnits && areaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Moment of inertia:</p>
                      <div className='Calculator-Side-A' style={{ width: '80%' }}>
                        <h6 className='sigma-symbol' >(I)</h6>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={inertia}
                            onChange={((e) => handleInertiaInputChange(e.target.value))}
                            aria-label="Moment of inertia"
                          />
                          <select
                            className='Calculator-select-option'
                            value={InertiaSelectedUnit}
                            onChange={((e) => handleInertiaUnitChange(e.target.value))}
                          >
                            {InertiaUnits && InertiaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </>
                )}
                {selectedOption === 'option4' && (
                  <>
                    <p className='text-white' style={{ textAlign: 'center', fontWeight: '600', }}>Support</p>

                    {supports && supports.map((support, index) => (
                      <div
                        key={index}
                        className='SupportOptionsMainDiv'
                        style={{
                          width: '100%',
                          display: 'flex',
                          justifyContent: 'space-evenly',
                          alignItems: 'center',
                        }}
                      >
                        <label style={{ color: '#fff' }}>Type:</label>
                        <div
                          className="Calculator-Side-A Supports"
                          style={{ width: '25%', justifyContent: 'center' }}
                        >
                          <div className="input-and-select-div" style={{ justifyContent: 'center' }}>
                            <select
                              value={support.type}
                              onChange={(e) => handleSupportChange(index, 'type', e.target.value)}
                              style={{
                                outline: 'none',
                                border: 'none',
                              }}
                            >
                              <option value="pinned">Pinned</option>
                              <option value="roller">Roller</option>
                              <option value="fixed">Fixed</option>
                            </select>
                          </div>
                        </div>
                        <label style={{ color: '#fff' }}>Position:</label>
                        <div
                          className="Calculator-Side-A Supports"
                          style={{ width: '30%', justifyContent: 'center' }}
                        >
                          <div className="input-and-select-div" style={{ justifyContent: 'center' }}>
                            <input
                              style={{
                                border: 'none',
                                float: 'right',
                                textAlign: 'center',
                                width: '70%',
                                outline: 'none',
                              }}
                              type="number"
                              value={support.position}
                              onChange={(e) => {
                                const value = Math.min(Number(e.target.value), length); // Ensure value doesn't exceed length
                                handleSupportChange(index, 'position', value); // Update position
                              }}
                              max={length} // This will restrict the input in browsers that support the max attribute
                              aria-label="Position of Support"
                            />

                          </div>
                        </div>
                        <MdDelete
                          className='DeleteButtonOfSupport-Deflection'
                          onClick={() => deleteSupport(index)}
                          style={{
                            color: '#fff',
                          }} />
                      </div>
                    ))}

                    <br />
                    <button type="button" className='BeamLinkBtntoBeamProperies-AddLoad BeamLinkBtntoBeamProperies' onClick={addSupport}>Add Support</button>
                  </>
                )}
                {selectedOption === 'option5' && (
                  <>
                    {distributedLoads &&
                      distributedLoads.map((load, index) => (
                        <div key={index}>
                          <div style={{ display: 'flex', justifyContent: 'space-evenly', alignItems: 'center' }}>
                            <p className="text-white" style={{ textAlign: "center", fontWeight: "600" }}>
                              Distributed Load {index + 1}
                            </p>
                            <MdDelete
                              className='DeleteButtonOfSupport-Deflection'
                              onClick={() => deleteDistributedLoad(index)}
                              style={{
                                color: '#fff',
                              }}
                            />
                          </div>
                          <br />
                          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                            <p className='claculator-conversation-title'>Start Position :</p>
                            <div className='Calculator-Side-A Supports' style={{ width: '50%', justifyContent: 'center' }}>
                              <div className='input-and-select-div' style={{ width: '100%' }}>
                                <input
                                  style={{
                                    border: 'none',
                                    float: 'right',
                                    textAlign: 'center',
                                    width: '100%',
                                    outline: 'none',
                                  }}
                                  type="number"
                                  value={load.start_position}
                                  max={length}
                                  onChange={(e) => {
                                    const value = Math.min(length, parseFloat(e.target.value) || 0);
                                    handleDistributedLoadChange(index, 'start_position', value);
                                  }}
                                  aria-label="Start Position"
                                />
                              </div>
                            </div>
                          </div>
                          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                            <p className='claculator-conversation-title'>End Position :</p>
                            <div className='Calculator-Side-A Supports' style={{ width: '50%', justifyContent: 'center' }}>
                              <div className='input-and-select-div' style={{ width: '100%' }}>
                                <input
                                  style={{
                                    border: 'none',
                                    float: 'right',
                                    textAlign: 'center',
                                    width: '100%',
                                    outline: 'none',
                                  }}
                                  type="number"
                                  value={load.end_position}
                                  max={length}
                                  onChange={(e) => {
                                    const value = Math.min(length, parseFloat(e.target.value) || 0);
                                    handleDistributedLoadChange(index, 'end_position', value);
                                  }}
                                  aria-label="End Position"
                                />
                              </div>
                            </div>
                          </div>
                          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                            <p className='claculator-conversation-title'>Start Magnitude :</p>
                            <div className='Calculator-Side-A Supports' style={{ width: '50%', justifyContent: 'center' }}>
                              <div className='input-and-select-div' style={{ width: '100%' }}>
                                <input
                                  style={{
                                    border: 'none',
                                    float: 'right',
                                    textAlign: 'center',
                                    width: '100%',
                                    outline: 'none',
                                  }}
                                  type="number"
                                  value={load.start_magnitude}
                                  onChange={(e) => handleDistributedLoadChange(index, 'start_magnitude', e.target.value)}
                                  aria-label="Start Magnitude"
                                />
                                <select
                                  value={distributedLoadSelectedUnit}
                                  style={{
                                    outline: 'none',
                                    border: 'none'
                                  }}
                                >
                                  <option key={distributedLoadSelectedUnit} value={distributedLoadSelectedUnit}>
                                    {distributedLoadSelectedUnit}
                                  </option>
                                </select>
                              </div>
                            </div>
                          </div>
                          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                            <p className='claculator-conversation-title'>End Magnitude :</p>
                            <div className='Calculator-Side-A Supports' style={{ width: '50%', justifyContent: 'center' }}>
                              <div className='input-and-select-div' style={{ width: '100%' }}>
                                <input
                                  style={{
                                    border: 'none',
                                    float: 'right',
                                    textAlign: 'center',
                                    width: '100%',
                                    outline: 'none',
                                  }}
                                  type="number"
                                  value={load.end_magnitude}
                                  onChange={(e) => handleDistributedLoadChange(index, 'end_magnitude', e.target.value)}
                                  aria-label="End Magnitude"
                                />
                                <select
                                  value={distributedLoadSelectedUnit}
                                  style={{
                                    outline: 'none',
                                    border: 'none'
                                  }}
                                >
                                  <option key={distributedLoadSelectedUnit} value={distributedLoadSelectedUnit}>
                                    {distributedLoadSelectedUnit}
                                  </option>
                                </select>
                              </div>
                            </div>
                          </div>
                          <br />
                        </div>
                      ))}




                    <br />
                    <button type="button" className='BeamLinkBtntoBeamProperies-AddLoad BeamLinkBtntoBeamProperies' onClick={addDistributedLoad}>Add Distributed Load</button>
                    <br />
                    <br />
                    <hr className='DeflectionHr' style={{ width: '70%', margin: 'auto !important', color: '#fff' }} />
                    <br />

                    {pointLoads &&
                      pointLoads.map((load, index) => (
                        <div key={index}>
                          <div style={{ display: 'flex', justifyContent: 'space-evenly', alignItems: 'center' }}>
                            <p className="text-white" style={{ textAlign: "center", fontWeight: "600" }}>
                              Point Load {index + 1}
                            </p>
                            <MdDelete
                              className='DeleteButtonOfSupport-Deflection'
                              onClick={() => deletePointLoad(index)}
                              style={{
                                color: '#fff',
                              }}
                            />
                          </div>
                          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                            <p className="claculator-conversation-title">Position :</p>
                            <div className="Calculator-Side-A Supports" style={{ width: '50%', justifyContent: 'center' }}>
                              <div className="input-and-select-div" style={{ justifyContent: 'center' }}>
                                <input
                                  style={{
                                    border: 'none',
                                    textAlign: 'center',
                                    width: '100%',
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    outline: 'none',
                                  }}
                                  aria-label="Position of Point Load"
                                  type="number"
                                  value={load.position}
                                  onChange={(e) => {
                                    const value = Math.min(length, parseFloat(e.target.value) || 0);
                                    handlePointLoadChange(index, 'position', value);
                                  }}
                                  step="1"
                                  min="0"
                                  max={length}
                                  placeholder="Position"
                                />
                              </div>
                            </div>
                          </div>
                          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                            <p className="claculator-conversation-title">Magnitude :</p>
                            <div className="Calculator-Side-A Supports" style={{ width: '50%', justifyContent: 'center' }}>
                              <div className="input-and-select-div" style={{ justifyContent: 'center', width: '100%' }}>
                                <input
                                  style={{
                                    border: 'none',
                                    textAlign: 'center',
                                    width: '100%',
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    outline: 'none',
                                  }}
                                  type="number"
                                  value={load.magnitude}
                                  onChange={(e) => handlePointLoadChange(index, 'magnitude', e.target.value)}
                                  placeholder="Magnitude"
                                  aria-label="Magnitude of Point Load"
                                />
                                <select
                                  value={pointLoadSelectedUnit}
                                  style={{
                                    outline: 'none',
                                    border: 'none',
                                  }}
                                >
                                  <option key={pointLoadSelectedUnit} value={pointLoadSelectedUnit}>
                                    {pointLoadSelectedUnit}
                                  </option>
                                </select>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}


                    <br />
                    <button type="button" className='BeamLinkBtntoBeamProperies-AddLoad BeamLinkBtntoBeamProperies' onClick={addPointLoad}>Add Point Load</button>

                    <br />
                    <br />
                    <hr className='DeflectionHr' style={{ width: '70%', margin: 'auto !important', color: '#fff' }} />
                    <br />
                    <br />
                    <button type="submit" className='BeamLinkBtntoBeamProperies-AddLoad BeamLinkBtntoBeamProperies' onClick={handleCombinedClick} >Analyze Beam</button>
                  </>
                )}
              </div>
            </div>
          </div>
        </section>
        <div className={`Grid-of-BeamDeflection-solutions mt-5 ${showfirstDiv ? 'ScrollTransactionone' : ''}  `}>
          <div>
            <div className={`${DropDowmOneMain} `} style={{
              background: '#fff',
              height: '30vw',
              left: '0',
              transform: 'translate(0%, 0%)'
            }}>
              <br />
              <h2 className='DeflectionCalculatorTopic' style={{ fontSize: '3vw', color: 'black', textAlign: 'center' }}>Reaction</h2>
              <div
                className='Rections'
                style={{ position: 'relative', width: '90%', top: '45%', left: '50%', transform: 'translate(-50%, -50%)' }}
              >
                {<div className="number-line-container">
                  <div className="line">

                    {reactionData && reactionData.length > 0 && (
                      <div>
                        <div>
                          {reactionData && reactionData.map((reaction, index) => (
                            <>
                              <img className="reaction-img" key={index} src={PointLoadSymbolsForReactionsImg} alt='reaction' style={{ position: 'absolute', top: '100%', width: '15px', height: '40px', transform: 'translate(-50%,0%)', left: `${(reaction.position / length) * 100}%`, }} />
                              <span
                                key={`position-${index}`}
                                className="reaction-indicator"
                                style={{ left: `${(reaction.position / length) * 100}%`, top: '0%' }}
                              >
                                Position: {reaction.position} mm{index < reactionData.length - 1 ? ', ' : ''}<br />
                                Force: {(reaction.force)} N{index < reactionData.length - 1 ? ', ' : ''}<br />
                                Moment: {(reaction.momentum)} N.m{index < reactionData.length - 1 ? ', ' : ''}
                              </span>
                            </>
                          ))}
                        </div>
                      </div>
                    )}

                  </div>

                  <div className="numbers">
                    {values && values.map((value, index) => (
                      <span
                        key={index}
                        className="number"
                        style={{
                          left: `${(value / length) * 100}%`, // Position each number dynamically
                        }}
                      >
                        {value.toLocaleString()}
                      </span>
                    ))}
                  </div>
                </div>}
                <br />
                <br />
              </div>
            </div>
          </div>

          <div>
            <div className={`${DropDowmOnerightMain} `} style={{
              background: '#fff',
              height: '30vw',
              left: '55%',
              transform: 'translate(0%, 0%)'
            }}>
              <br />
              <h2 className='DeflectionCalculatorTopic' style={{ fontSize: '3vw', color: 'black', textAlign: 'center' }}>Shear Force</h2>
              <Line data={shearForceChartData} options={shearForceChartDataoptions} width="200px" height="100px" />
            </div>
          </div>

        </div>
        <div className={`Grid-of-BeamDeflection-solutions mt-5 BeandingAndDeflection ${showfirstDiv ? 'ScrollTransactionone' : ''}  ${showSecondDiv ? 'ScrollTransactionTwo' : ''}`}>
          <div>
            <div className={`${DropDowmTwoMain}`} style={{
              background: '#fff',
              height: '30vw',
              left: '0%',
              transform: 'translate(0%, 0%)',
              top: '255%'
            }}>
              <br />
              <h2 className='DeflectionCalculatorTopic' style={{ fontSize: '3vw', color: 'black', textAlign: 'center' }}>Bending Moment</h2>
              <Line data={bendingMomentChartData} options={bendingMomentChartDataoptions} width="200px" height="100px" />
            </div>
          </div>
          <div>
            <div className={`${DropDowmTworightMain}`} style={{
              background: '#fff',
              height: '30vw',
              left: '55%',
              transform: 'translate(0%, 0%)',
              top: '255%'
            }}>
              <br />
              <h2 className='DeflectionCalculatorTopic' style={{ fontSize: '3vw', color: 'black', textAlign: 'center' }}>Deflection</h2>
              <Line data={deflectionChartData} options={deflectionChartDataoptions} width="200px" height="100px" />
            </div>
          </div >
        </div>
        <br />
        <br />
        <br />
        <br />
        <hr className='Beam-properties-calculator-hr' />
        <br />
        <VideoPlayerSection />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <InputsParametersFile />
        <br />
        <br />
        <br />
        <hr className='Beam-properties-calculator-hr' />
        <br />
        <br />
        <br />
        <br />
        <br />
        <OutputParameterFile />
        <hr className='Beam-properties-calculator-hr' />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
      </section>
      <section className='cse-header-top'>
        <Link smooth="true" duration={500} offset={-70} onClick={scrollToTop} aria-label="Scroll to top">
          <GrLinkTop className='' />
        </Link>
      </section>
      {/* </section> */}

    </>
  )
}
