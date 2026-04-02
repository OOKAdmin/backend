import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet';
import { GrLinkTop } from "react-icons/gr";

// (Replace with your actual images or keep the same imports if present)
import MaterialpropertyImg from '../../images/Padeye-Material-property.jpg'
import PadeyegeometryImg from '../../images/Padeye-Padeyegeometry.png'
import shackleImg from '../../images/padeye/shackle.png'
import Slingimg from '../../images/Padeye-Sling-Dia.png'
import PadeyeLoadimg from '../../images/padeye load (2)/padeye load (2).png'
import WeldSizeimg from '../../images/PadEye-weld-size.png'
import backgroundJPG from '../../images/Padeye-Background.jpg';
import backgroundWebP from '../../images/Padeye-Background.webp';
import './FourPointRiggingwithSpreaderBarCalculator.css';

export default function FourPointRiggingwithSpreaderBarCalculator() {
    // --- UI State ---
    const [selectedOption, setSelectedOption] = useState("option1");
    const handleOptionChange = (option) => setSelectedOption(option);

    const [isActive3, setIsActive3] = useState(false);
    const toggleClass3 = () => setIsActive3(prev => !prev);

    // Input states (default values taken from the Excel file sample in the workbook)
    
        const Wunits = ['ton', 'kg'];
    
        const WConversionFactors = {
            ton: [1, 1000],  // 1 ton = 1000 kg
            kg: [0.001, 1],  // 1 kg = 0.001 ton
        };
    
        const [WValue, setWValue] = useState(0);
        const [WselectedUnit, setWSelectedUnit] = useState('ton');
        const [WInTon, setWInTon] = useState(0);
    
        const handleWChange = (value) => {
            setWValue(value);
            const factor = WConversionFactors[WselectedUnit][0];
            setWInTon(parseFloat(value) * factor);
        };
    
        const handleWchange = (unit) => {
            let newValue = parseFloat(WValue);
            if (unit === 'kg' && WselectedUnit === 'ton') {
                newValue *= 1000; // convert ton → kg
                if (newValue >= 1000) {
                    newValue = newValue.toExponential(3);
                }
            } else if (unit === 'ton' && WselectedUnit === 'kg') {
                newValue /= 1000; // convert kg → ton
            }
            setWSelectedUnit(unit);
            setWValue(isNaN(newValue) ? 0 : newValue);
        };
        
            const D1units = ['km', 'm'];
        
            const D1ConversionFactors = {
                km: [1, 1000],  // 1 km = 1000 m
                m: [0.001, 1],  // 1 m = 0.001 km
            };
        
            const [D1Value, setD1Value] = useState(0);
            const [D1selectedUnit, setD1SelectedUnit] = useState('km');
            const [D1InKM, setD1InKM] = useState(0);
        
            const handleD1Change = (value) => {
                setD1Value(value);
                const factor = D1ConversionFactors[D1selectedUnit][0];
                setD1InKM(parseFloat(value) * factor);
            };
        
            const handleD1change = (unit) => {
                let newValue = parseFloat(D1Value);
                if (unit === 'm' && D1selectedUnit === 'km') {
                    newValue *= 1000; // km → m
                    if (newValue >= 1000) {
                        newValue = newValue.toExponential(3);
                    }
                } else if (unit === 'km' && D1selectedUnit === 'm') {
                    newValue /= 1000; // m → km
                }
                setD1SelectedUnit(unit);
                setD1Value(isNaN(newValue) ? 0 : newValue);
            };
        
        
            const D2units = ['km', 'm'];
        
            const D2ConversionFactors = {
                km: [1, 1000],  // 1 km = 1000 m
                m: [0.001, 1],  // 1 m = 0.001 km
            };
        
            const [D2Value, setD2Value] = useState(0);
            const [D2selectedUnit, setD2SelectedUnit] = useState('km');
            const [D2InKM, setD2InKM] = useState(0);
        
            const handleD2Change = (value) => {
                setD2Value(value);
                const factor = D2ConversionFactors[D2selectedUnit][0];
                setD2InKM(parseFloat(value) * factor);
            };
        
            const handleD2change = (unit) => {
                let newValue = parseFloat(D2Value);
                if (unit === 'm' && D2selectedUnit === 'km') {
                    newValue *= 1000; // km → m
                    if (newValue >= 1000) {
                        newValue = newValue.toExponential(3);
                    }
                } else if (unit === 'km' && D2selectedUnit === 'm') {
                    newValue /= 1000; // m → km
                }
                setD2SelectedUnit(unit);
                setD2Value(isNaN(newValue) ? 0 : newValue);
            };
        

    const D3units = ['km', 'm'];
        
        const D3ConversionFactors = {
            km: [1, 1000],  // 1 km = 1000 m
            m: [0.001, 1],  // 1 m = 0.001 km
        };
    
        const [D3Value, setD3Value] = useState(0);
        const [D3selectedUnit, setD3SelectedUnit] = useState('km');
        const [D3InKM, setD3InKM] = useState(0);
    
        const handleD3Change = (value) => {
            setD3Value(value);
            const factor = D3ConversionFactors[D3selectedUnit][0];
            setD3InKM(parseFloat(value) * factor);
        };
    
        const handleD3change = (unit) => {
            let newValue = parseFloat(D3Value);
            if (unit === 'm' && D3selectedUnit === 'km') {
                newValue *= 1000; // km → m
                if (newValue >= 1000) {
                    newValue = newValue.toExponential(3);
                }
            } else if (unit === 'km' && D3selectedUnit === 'm') {
                newValue /= 1000; // m → km
            }
            setD3SelectedUnit(unit);
            setD3Value(isNaN(newValue) ? 0 : newValue);
        };
    
    
        const D4units = ['km', 'm'];
    
        const D4ConversionFactors = {
            km: [1, 1000],  // 1 km = 1000 m
            m: [0.001, 1],  // 1 m = 0.001 km
        };
    
        const [D4Value, setD4Value] = useState(0);
        const [D4selectedUnit, setD4SelectedUnit] = useState('km');
        const [D4InKM, setD4InKM] = useState(0);
    
        const handleD4Change = (value) => {
            setD4Value(value);
            const factor = D4ConversionFactors[D4selectedUnit][0];
            setD4InKM(parseFloat(value) * factor);
        };
    
        const handleD4change = (unit) => {
            let newValue = parseFloat(D4Value);
            if (unit === 'm' && D4selectedUnit === 'km') {
                newValue *= 1000; // km → m
                if (newValue >= 1000) {
                    newValue = newValue.toExponential(3);
                }
            } else if (unit === 'km' && D4selectedUnit === 'm') {
                newValue /= 1000; // m → km
            }
            setD4SelectedUnit(unit);
            setD4Value(isNaN(newValue) ? 0 : newValue);
        };
    
    
        const DT1units = ['km', 'm'];
    
        const DT1ConversionFactors = {
            km: [1, 1000],  // 1 km = 1000 m
            m: [0.001, 1],  // 1 m = 0.001 km
        };
    
        const [DT1Value, setDT1Value] = useState(0);
        const [DT1selectedUnit, setDT1SelectedUnit] = useState('km');
        const [DT1InKM, setDT1InKM] = useState(0);
    
        const handleDT1Change = (value) => {
            setDT1Value(value);
            const factor = DT1ConversionFactors[DT1selectedUnit][0];
            setDT1InKM(parseFloat(value) * factor);
        };
    
        const handleDT1change = (unit) => {
            let newValue = parseFloat(DT1Value);
            if (unit === 'm' && DT1selectedUnit === 'km') {
                newValue *= 1000; // km → m
                if (newValue >= 1000) {
                    newValue = newValue.toExponential(3);
                }
            } else if (unit === 'km' && DT1selectedUnit === 'm') {
                newValue /= 1000; // m → km
            }
            setDT1SelectedUnit(unit);
            setDT1Value(isNaN(newValue) ? 0 : newValue);
        };
    
    
        const DT2units = ['km', 'm'];
    
        const DT2ConversionFactors = {
            km: [1, 1000],  // 1 km = 1000 m
            m: [0.001, 1],  // 1 m = 0.001 km
        };
    
        const [DT2Value, setDT2Value] = useState(0);
        const [DT2selectedUnit, setDT2SelectedUnit] = useState('km');
        const [DT2InKM, setDT2InKM] = useState(0);
    
        const handleDT2Change = (value) => {
            setDT2Value(value);
            const factor = DT2ConversionFactors[DT2selectedUnit][0];
            setDT2InKM(parseFloat(value) * factor);
        };
    
        const handleDT2change = (unit) => {
            let newValue = parseFloat(DT2Value);
            if (unit === 'm' && DT2selectedUnit === 'km') {
                newValue *= 1000; // km → m
                if (newValue >= 1000) {
                    newValue = newValue.toExponential(3);
                }
            } else if (unit === 'km' && DT2selectedUnit === 'm') {
                newValue /= 1000; // m → km
            }
            setDT2SelectedUnit(unit);
            setDT2Value(isNaN(newValue) ? 0 : newValue);
        };
    
    
        // --- S1 ---
        const S1units = ['km', 'm'];
    
        const S1ConversionFactors = {
            km: [1, 1000],  // 1 km = 1000 m
            m: [0.001, 1],  // 1 m = 0.001 km
        };
    
        const [S1Value, setS1Value] = useState(0);
        const [S1selectedUnit, setS1SelectedUnit] = useState('km');
        const [S1InKM, setS1InKM] = useState(0);
    
        const handleS1Change = (value) => {
            setS1Value(value);
            const factor = S1ConversionFactors[S1selectedUnit][0];
            setS1InKM(parseFloat(value) * factor);
        };
    
        const handleS1change = (unit) => {
            let newValue = parseFloat(S1Value);
            if (unit === 'm' && S1selectedUnit === 'km') {
                newValue *= 1000;
                if (newValue >= 1000) newValue = newValue.toExponential(3);
            } else if (unit === 'km' && S1selectedUnit === 'm') {
                newValue /= 1000;
            }
            setS1SelectedUnit(unit);
            setS1Value(isNaN(newValue) ? 0 : newValue);
        };
    
    
        // --- S2 ---
        const S2units = ['km', 'm'];
    
        const S2ConversionFactors = {
            km: [1, 1000],
            m: [0.001, 1],
        };
    
        const [S2Value, setS2Value] = useState(0);
        const [S2selectedUnit, setS2SelectedUnit] = useState('km');
        const [S2InKM, setS2InKM] = useState(0);
    
        const handleS2Change = (value) => {
            setS2Value(value);
            const factor = S2ConversionFactors[S2selectedUnit][0];
            setS2InKM(parseFloat(value) * factor);
        };
    
        const handleS2change = (unit) => {
            let newValue = parseFloat(S2Value);
            if (unit === 'm' && S2selectedUnit === 'km') {
                newValue *= 1000;
                if (newValue >= 1000) newValue = newValue.toExponential(3);
            } else if (unit === 'km' && S2selectedUnit === 'm') {
                newValue /= 1000;
            }
            setS2SelectedUnit(unit);
            setS2Value(isNaN(newValue) ? 0 : newValue);
        };
    
    
        // --- S3 ---
        const S3units = ['km', 'm'];
    
        const S3ConversionFactors = {
            km: [1, 1000],
            m: [0.001, 1],
        };
    
        const [S3Value, setS3Value] = useState(0);
        const [S3selectedUnit, setS3SelectedUnit] = useState('km');
        const [S3InKM, setS3InKM] = useState(0);
    
        const handleS3Change = (value) => {
            setS3Value(value);
            const factor = S3ConversionFactors[S3selectedUnit][0];
            setS3InKM(parseFloat(value) * factor);
        };
    
        const handleS3change = (unit) => {
            let newValue = parseFloat(S3Value);
            if (unit === 'm' && S3selectedUnit === 'km') {
                newValue *= 1000;
                if (newValue >= 1000) newValue = newValue.toExponential(3);
            } else if (unit === 'km' && S3selectedUnit === 'm') {
                newValue /= 1000;
            }
            setS3SelectedUnit(unit);
            setS3Value(isNaN(newValue) ? 0 : newValue);
        };
    
    
    const [SW, setSW] = useState(2);          // B18 (spreader bar weight)
    const [RW1, setRW1] = useState(0.2);      // B19
    const [RW2, setRW2] = useState(0.2);      // B20
    const [RW3, setRW3] = useState(0.165);    // B21
    const [RW4, setRW4] = useState(0.165);    // B22

    // Toggle reveal results with small animation imitation
    const [showfirstDiv, setShowFirstDiv] = useState(false);
    const [showSecondDiv, setShowSecondDiv] = useState(false);
    const [showThirdDiv, setShowThirdDiv] = useState(false);

    const handleToggleWithDelay = () => {
        if (!showfirstDiv && !showSecondDiv && !showThirdDiv) {
            setShowFirstDiv(true);
            setTimeout(() => {
                setShowSecondDiv(true);
                setTimeout(() => setShowThirdDiv(true), 400);
            }, 300);
        } else {
            setShowThirdDiv(false);
            setTimeout(() => {
                setShowSecondDiv(false);
                setTimeout(() => setShowFirstDiv(false), 300);
            }, 300);
        }
    };

    const handleCombinedClick = () => {
        handleToggleWithDelay();
        toggleClass3();
    };

    // Helper numeric parse (turn empty to 0)
    const toNum = (v) => {
        const n = parseFloat(v);
        return Number.isFinite(n) ? n : 0;
    };

    // --- Excel-derived Formulas (from uploaded workbook) ---
    // L1..L4 : loads at lifting points (B25..B28)
    // B25 = =B8 * (B10 / B13) * (B11 / B14)
    // B26 = =B8 * (B10 / B13) * (B12 / B14)
    // B27 = =B8 * (B9 / B13) * (B12 / B14)
    // B28 = =B8 * (B9 / B13) * (B11 / B14)

    const numW = toNum(WValue);
    const numD1 = toNum(D1Value);
    const numD2 = toNum(D2Value);
    const numD3 = toNum(D3Value);
    const numD4 = toNum(D4Value);
    const numDT1 = toNum(DT1Value);
    const numDT2 = toNum(DT2Value);
    const numS1 = toNum(S1Value);
    const numS2 = toNum(S2Value);
    const numS3 = toNum(S3Value);
    const numSW = toNum(SW);
    const numRW1 = toNum(RW1);
    const numRW2 = toNum(RW2);
    const numRW3 = toNum(RW3);
    const numRW4 = toNum(RW4);

    // Avoid division by zero by returning 0 if denom is 0
    const safeDiv = (a, b) => b === 0 ? 0 : (a / b);

    const L1 = numW * safeDiv(numD2, numDT1) * safeDiv(numD3, numDT2);
    const L2 = numW * safeDiv(numD2, numDT1) * safeDiv(numD4, numDT2);
    const L3 = numW * safeDiv(numD1, numDT1) * safeDiv(numD4, numDT2);
    const L4 = numW * safeDiv(numD1, numDT1) * safeDiv(numD3, numDT2);

    // T1..T4 on sheet used  = LOAD * (1 / sin(90°)) -> sin(90)=1 so T == L
    const T1 = L1;
    const T2 = L2;
    const T3 = L3;
    const T4 = L4;

    // Heights: B35 = SQRT(B16^2 - B9^2), B36 = SQRT(B17^2 - B10^2)
    const H1 = (numS2 * numS2 - numD1 * numD1) >= 0 ? Math.sqrt(numS2 * numS2 - numD1 * numD1) : NaN;
    const H2 = (numS3 * numS3 - numD2 * numD2) >= 0 ? Math.sqrt(numS3 * numS3 - numD2 * numD2) : NaN;

    // Spreader weights distribution: B39 = B18*(B10/B13), B40 = B18*(B9/B13)
    const spreader_L12 = numSW * safeDiv(numD2, numDT1); // weight of spreader at L1 & L2
    const spreader_L34 = numSW * safeDiv(numD1, numDT1); // weight of spreader at L3 & L4

    // Total loads at spreader ends (B42/B43)
    // B42 = B25+B26+B39+B19  => L1 + L2 + spreader_L12 + RW1
    // B43 = B27+B28+B40+B20  => L3 + L4 + spreader_L34 + RW2
    const TW1 = L1 + L2 + spreader_L12 + numRW1;
    const TW2 = L3 + L4 + spreader_L34 + numRW2;

    // T5/T6 (tensions at slings 5 & 6): B45 = B42*(B16/B35), B46 = B43*(B17/B36)
    // guard H1/H2 not zero/NaN
    const T5 = (H1 && !Number.isNaN(H1) && H1 !== 0) ? TW1 * safeDiv(numS2, H1) : NaN;
    const T6 = (H2 && !Number.isNaN(H2) && H2 !== 0) ? TW2 * safeDiv(numS3, H2) : NaN;

    // Safety factors (simple examples — you can adjust for slings/shackles if desired)
    // We will not compute SF here automatically — keep original approach if needed

    // Scroll to top helper
    const scrollToTop = () => window.scrollTo({ top: 0, behavior: 'smooth' });

    // Render
    return (
        <>
            <Helmet>
                <title>Four Point Rigging Calculator – Excel-matched | OOK</title>
                <meta name="description" content="Four point rigging calculator matched to the provided Excel file (spreader bar logic)." />
            </Helmet>

            {/* Header */}
            <section className='background-white PadEye'>
                <div className="position-relative">
                    <div className="position-relative overflow-hidden" style={{ height: '60vh' }}>
                        <picture>
                            <source type="image/webp" srcSet={backgroundWebP} />
                            <img loading="lazy" src={backgroundJPG} alt="Background" className="h-100"
                                width="600" height="400"
                                style={{ objectFit: 'cover', objectPosition: 'center', width: '100%', transform: 'translateX(0%)' }} />
                        </picture>
                        <div className="Overlay-Black-header deflection"></div>
                    </div>
                    <div className="container text-left text-white position-absolute top-50 translate-middle Padeye" style={{ left: '48%' }}>
                        <h1 className="display-4" style={{ fontWeight: '600' }}>Four Point Rigging Calculator (Excel-matched)</h1>
                        <p className="fs-6">
                            Logic implemented exactly as in your Excel workbook: loads L1–L4, tensions T1–T6, spreader distribution and heights.
                        </p>
                    </div>
                </div>
            </section>

            {/* Input Section */}
            <section className="container-fluid py-4 justify-content-center align-items-center d-flex">
                <div className="row structure-analysis-calculator-calculator">

                    {/* Left Buttons */}
                    <div className="col-12 flex-grow-1 col-lg-3 col-md-12 text-center py-5 structure-analysis-calculator-calculator-left ps-0 pe-0">
                        <div className="d-flex flex-column gap-0 w-100 text-center" style={{ justifyContent: 'center', alignItems: 'center' }}>
                            <button className="btn mb-2" onClick={() => handleOptionChange("option1")}>Material Properties</button>
                            <button className="btn mb-2" onClick={() => handleOptionChange("option2")}>Pad-eye Geometry</button>
                            <button className="btn mb-2" onClick={() => handleOptionChange("option3")}>Shackle Geometry</button>
                            <button className="btn mb-2" onClick={() => handleOptionChange("option4")}>Sling Geometry</button>
                            <button className="btn mb-2" onClick={() => handleOptionChange("option5")}>Pad-eye Load</button>
                            <button className="btn mb-2" onClick={() => handleOptionChange("option6")}>Weld Size</button>
                        </div>
                    </div>

                    {/* Center Image */}
                    <div className="col-12 flex-grow-2 col-lg-5 text-center py-3 structure-analysis-calculator-calculator-center bg-white d-flex justify-content-center align-items-center">
                        {selectedOption === 'option1' && <img src={MaterialpropertyImg} alt="Material Properties" className="img-fluid" />}
                        {selectedOption === 'option2' && <img src={PadeyegeometryImg} alt="Pad-eye Geometry" className="img-fluid" />}
                        {selectedOption === 'option3' && <img src={shackleImg} alt="Shackle Geometry" className="img-fluid" />}
                        {selectedOption === 'option4' && <img src={Slingimg} alt="Sling Geometry" className="img-fluid" />}
                        {selectedOption === 'option5' && <img src={PadeyeLoadimg} alt="Pad-eye Load" className="img-fluid" />}
                        {selectedOption === 'option6' && <img src={WeldSizeimg} alt="Weld Size" className="img-fluid" />}
                    </div>

                    {/* Right Input Fields */}
                    <div className="col-12 flex-grow-1 col-lg-3 text-center py-3 bemProperties structure-analysis-calculator-calculator-right PadeyeInputs">
                        <h2 className="text-white mt-3">Inputs (from Excel)</h2>

                        {/* W */}
                        <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">W (Weight of Cargo)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">W</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={WValue}
                                                onChange={(e) => handleWChange(e.target.value)}
                                            />
                                            <select
                                                className='Calculator-select-option'
                                                value={WselectedUnit}
                                                onChange={((e) => handleWchange(e.target.value))}
                                                label="Select Weight Unit"
                                            >
                                                {Wunits.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                        {/* D1..D4 */}
                        
                                {/* D1 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">D1 (COG from LP1)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">D₁</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={D1Value}
                                                onChange={(e) => handleD1Change(e.target.value)}
                                            />
                                            <select
                                                className='Calculator-select-option'
                                                value={D1selectedUnit}
                                                onChange={((e) => handleD1change(e.target.value))}
                                                label="Select D1 Unit"
                                            >
                                                {D1units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* D2 */}
                                <div className="w-100 d-flex justify-content-cent3er align-items-center">
                                    <p className="claculator-conversation-title">D2 (COG from LP3)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">D₂</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={D2Value}
                                                onChange={(e) => handleD2Change(e.target.value)}
                                            />
                                            <select
                                                className='Calculator-select-option'
                                                value={D2selectedUnit}
                                                onChange={((e) => handleD2change(e.target.value))}
                                                label="Select D2 Unit"
                                            >
                                                {D2units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* D3 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">D3 (COG from LP1 & 3)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">D₃</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={D3Value}
                                                onChange={(e) => handleD3Change(e.target.value)}
                                            />
                                            <select
                                                className='Calculator-select-option'
                                                value={D3selectedUnit}
                                                onChange={((e) => handleD3change(e.target.value))}
                                                label="Select D3 Unit"
                                            >
                                                {D3units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* D4 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">D4 (COG from LP2 & 4)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">D₄</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={D4Value}
                                                onChange={(e) => handleD4Change(e.target.value)}
                                            />
                                            <select
                                                className='Calculator-select-option'
                                                value={D4selectedUnit}
                                                onChange={((e) => handleD4change(e.target.value))}
                                                label="Select D4 Unit"
                                            >
                                                {D4units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                        {/* DT1 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">DT1 (Total Distance D1 + D3)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">DT₁</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={DT1Value}
                                                onChange={(e) => handleDT1Change(e.target.value)}
                                            />

                                            <select
                                                className='Calculator-select-option'
                                                value={D1selectedUnit}
                                                onChange={((e) => handleDT1change(e.target.value))}
                                                label="Select D1 Unit"
                                            >
                                                {D1units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* DT2 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">DT2 (Total Distance D3 + D4)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">DT₂</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={DT2Value}
                                                onChange={(e) => handleDT2Change(e.target.value)}
                                            />
                                            <select
                                                className='Calculator-select-option'
                                                value={D2selectedUnit}
                                                onChange={((e) => handleDT2change(e.target.value))}
                                                label="Select D2 Unit"
                                            >
                                                {D2units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* S1 - S4 */}
                                {/* S1 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">S1 (Sling Length L1)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">S₁</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={S1Value}
                                                onChange={(e) => handleS1Change(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={S1selectedUnit}
                                                onChange={(e) => handleS1change(e.target.value)}
                                                label="Select S1 Unit"
                                            >
                                                {S1units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* S2 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">S2 (Sling Length L2)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">S₂</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={S2Value}
                                                onChange={(e) => handleS2Change(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={S2selectedUnit}
                                                onChange={(e) => handleS2change(e.target.value)}
                                                label="Select S2 Unit"
                                            >
                                                {S2units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* S3 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">S3 (Sling Length L3)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">S₃</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={S3Value}
                                                onChange={(e) => handleS3Change(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={S3selectedUnit}
                                                onChange={(e) => handleS3change(e.target.value)}
                                                label="Select S3 Unit"
                                            >
                                                {S3units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>


                        {/* Spreader weight SW */}
                        <div className="w-100 d-flex justify-content-center align-items-center">
                            <p className="claculator-conversation-title">SW (Spreader Bar Weight)</p>
                            <div className="Calculator-Side-A">
                                <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">SW</p>
                                <div className="input-and-select-div">
                                    <input className="calculator-input" type="number" value={SW} onChange={(e) => setSW(e.target.value)} />
                                </div>
                            </div>
                        </div>

                        {/* RW1..RW4 */}
                        {[{l:'RW1', s:RW1, set:setRW1},{l:'RW2', s:RW2, set:setRW2},{l:'RW3', s:RW3, set:setRW3},{l:'RW4', s:RW4, set:setRW4}].map(item=>(
                            <div key={item.l} className="w-100 d-flex justify-content-center align-items-center">
                                <p className="claculator-conversation-title">{item.l} (Rigging weight)</p>
                                <div className="Calculator-Side-A">
                                    <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">{item.l}</p>
                                    <div className="input-and-select-div">
                                        <input className="calculator-input" type="number" value={item.s} onChange={(e)=>item.set(e.target.value)} />
                                    </div>
                                </div>
                            </div>
                        ))}

                        <div className="FourPointRiggingCalculator structure-analysis-calculator-calculator-right-show-hidden-btn-Padeye-MaterialProperty">
                            <button className="structure-analysis-calculator-calculator-right-show-hidden-btn" onClick={handleCombinedClick}>
                                {isActive3 ? 'Hide' : 'Solve'}
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            {/* Results panel (mirrors the sheet outputs) */}
            <div className={isActive3 ? 'show Sectionmodules' : 'hidden Sectionmodules'} style={{ height: 'auto' }}>
                <br />
                <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '2.2rem' }}>Results (Excel logic)</h2>
                <div className='Section-properties-Solutions' style={{ borderRadius: '10px', width: '90%', margin: 'auto', padding: '1rem' }}>

                    {/* L1..L4 */}
                    {[
                        {label:'LOAD AT LIFTING POINT 1 (L1)', value:L1},
                        {label:'LOAD AT LIFTING POINT 2 (L2)', value:L2},
                        {label:'LOAD AT LIFTING POINT 3 (L3)', value:L3},
                        {label:'LOAD AT LIFTING POINT 4 (L4)', value:L4},
                    ].map(item=>(
                        <div key={item.label} style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                            <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>{item.label}</p>
                            <div className='Calculator-Side-A' style={{ width:'40%' }}>
                                <div className='input-and-select-div'>
                                    <input className='calculator-input' type="number" value={Number.isFinite(item.value) ? item.value.toFixed(6) : ""} readOnly />
                                </div>
                            </div>
                        </div>
                    ))}

                    {/* T1..T4 */}
                    {[
                        {label:'TENSION AT SLING1 (T1)', value:T1},
                        {label:'TENSION AT SLING2 (T2)', value:T2},
                        {label:'TENSION AT SLING3 (T3)', value:T3},
                        {label:'TENSION AT SLING4 (T4)', value:T4},
                    ].map(item=>(
                        <div key={item.label} style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                            <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>{item.label}</p>
                            <div className='Calculator-Side-A' style={{ width:'40%' }}>
                                <div className='input-and-select-div'>
                                    <input className='calculator-input' type="number" value={Number.isFinite(item.value) ? item.value.toFixed(6) : ""} readOnly />
                                </div>
                            </div>
                        </div>
                    ))}

                    {/* H1, H2 */}
                    <div style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>HEIGHT OF SLING H1</p>
                        <div className='Calculator-Side-A' style={{ width:'40%' }}>
                            <div className='input-and-select-div'>
                                <input className='calculator-input' type="number" value={Number.isFinite(H1) ? H1.toFixed(12) : ""} readOnly />
                            </div>
                        </div>
                    </div>
                    <div style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>HEIGHT OF SLING H2</p>
                        <div className='Calculator-Side-A' style={{ width:'40%' }}>
                            <div className='input-and-select-div'>
                                <input className='calculator-input' type="number" value={Number.isFinite(H2) ? H2.toFixed(12) : ""} readOnly />
                            </div>
                        </div>
                    </div>

                    {/* Spreader weights */}
                    <div style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>WEIGHT OF SPREADER BAR AT L1 & L2</p>
                        <div className='Calculator-Side-A' style={{ width:'40%' }}>
                            <div className='input-and-select-div'>
                                <input className='calculator-input' type="number" value={Number.isFinite(spreader_L12) ? spreader_L12.toFixed(12) : ""} readOnly />
                            </div>
                        </div>
                    </div>
                    <div style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>WEIGHT OF SPREADER BAR AT L3 & L4</p>
                        <div className='Calculator-Side-A' style={{ width:'40%' }}>
                            <div className='input-and-select-div'>
                                <input className='calculator-input' type="number" value={Number.isFinite(spreader_L34) ? spreader_L34.toFixed(12) : ""} readOnly />
                            </div>
                        </div>
                    </div>

                    {/* TW1/TW2 */}
                    <div style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>TW1 (TOTAL LOAD AT L1 & L2)</p>
                        <div className='Calculator-Side-A' style={{ width:'40%' }}>
                            <div className='input-and-select-div'>
                                <input className='calculator-input' type="number" value={Number.isFinite(TW1) ? TW1.toFixed(12) : ""} readOnly />
                            </div>
                        </div>
                    </div>
                    <div style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>TW2 (TOTAL LOAD AT L3 & L4)</p>
                        <div className='Calculator-Side-A' style={{ width:'40%' }}>
                            <div className='input-and-select-div'>
                                <input className='calculator-input' type="number" value={Number.isFinite(TW2) ? TW2.toFixed(12) : ""} readOnly />
                            </div>
                        </div>
                    </div>

                    {/* T5 / T6 */}
                    <div style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>TENSION AT SLING 5 (T5)</p>
                        <div className='Calculator-Side-A' style={{ width:'40%' }}>
                            <div className='input-and-select-div'>
                                <input className='calculator-input' type="number" value={Number.isFinite(T5) ? T5.toFixed(12) : ""} readOnly />
                            </div>
                        </div>
                    </div>
                    <div style={{ display:'flex', justifyContent:'center', alignItems:'center', margin:'0.4rem 0' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ width:'45%', fontWeight:600 }}>TENSION AT SLING 6 (T6)</p>
                        <div className='Calculator-Side-A' style={{ width:'40%' }}>
                            <div className='input-and-select-div'>
                                <input className='calculator-input' type="number" value={Number.isFinite(T6) ? T6.toFixed(12) : ""} readOnly />
                            </div>
                        </div>
                    </div>

                </div>
            </div>

            {/* Scroll to Top */}
            <section className='cse-header-top'>
                <Link onClick={scrollToTop}>
                    <GrLinkTop />
                </Link>
            </section>
        </>
    );
}
