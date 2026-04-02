import React, { useState, useEffect } from 'react'

// images
import MaterialpropertyImg from '../../images/Padeye-Material-property.jpg'
import PadeyegeometryImg from '../../images/Padeye-Padeyegeometry.png'
import shackleImg from '../../images/padeye/shackle.png'
import Slingimg from '../../images/Padeye-Sling-Dia.png'
import PadeyeLoadimg from '../../images/padeye load (2)/padeye load (2).png'
import WeldSizeimg from '../../images/PadEye-weld-size.png'

// modules
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet';

// icons
import { GrLinkTop } from "react-icons/gr";

// Background Image
import backgroundJPG from '../../images/Padeye-Background.jpg';
import backgroundWebP from '../../images/Padeye-Background.webp';

//css
import './FourPointRiggingCalculator.css';

export default function FourPointRiggingCalculator() {

    // Scroll and section effects
    const scrollToTop = () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const [selectedOption, setSelectedOption] = useState("option1");
    const handleOptionChange = (option) => setSelectedOption(option);

    const [isActive3, setIsActive3] = useState(false);
    const toggleClass3 = () => setIsActive3(prev => !prev);

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
                setTimeout(() => setShowThirdDiv(true), 2000);
            }, 2000);
        } else {
            setShowThirdDiv(false);
            setTimeout(() => {
                setShowSecondDiv(false);
                setTimeout(() => setShowFirstDiv(false), 2000);
            }, 2000);
        }
    };

    const handleCombinedClick = () => {
        handleToggleWithDelay();
        toggleClass3();
    };

    // --- Individual Input States & Handlers ---

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



    const COGunits = ['km', 'm'];

    const COGConversionFactors = {
        km: [1, 1000],  // 1 km = 1000 m
        m: [0.001, 1],  // 1 m = 0.001 km
    };

    const [COGValue, setCOGValue] = useState(0);
    const [COGselectedUnit, setCOGSelectedUnit] = useState('km');
    const [COGInKM, setCOGInKM] = useState(0);

    const handleCOGChange = (value) => {
        setCOGValue(value);
        const factor = COGConversionFactors[COGselectedUnit][0];
        setCOGInKM(parseFloat(value) * factor);
    };

    const handleCOGchange = (unit) => {
        let newValue = parseFloat(COGValue);
        if (unit === 'm' && COGselectedUnit === 'km') {
            newValue *= 1000; // convert km → m
            if (newValue >= 1000) {
                newValue = newValue.toExponential(3);
            }
        } else if (unit === 'km' && COGselectedUnit === 'm') {
            newValue /= 1000; // convert m → km
        }
        setCOGSelectedUnit(unit);
        setCOGValue(isNaN(newValue) ? 0 : newValue);
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


    // --- S4 ---
    const S4units = ['km', 'm'];

    const S4ConversionFactors = {
        km: [1, 1000],
        m: [0.001, 1],
    };

    const [S4Value, setS4Value] = useState(0);
    const [S4selectedUnit, setS4SelectedUnit] = useState('km');
    const [S4InKM, setS4InKM] = useState(0);

    const handleS4Change = (value) => {
        setS4Value(value);
        const factor = S4ConversionFactors[S4selectedUnit][0];
        setS4InKM(parseFloat(value) * factor);
    };

    const handleS4change = (unit) => {
        let newValue = parseFloat(S4Value);
        if (unit === 'm' && S4selectedUnit === 'km') {
            newValue *= 1000;
            if (newValue >= 1000) newValue = newValue.toExponential(3);
        } else if (unit === 'km' && S4selectedUnit === 'm') {
            newValue /= 1000;
        }
        setS4SelectedUnit(unit);
        setS4Value(isNaN(newValue) ? 0 : newValue);
    };

    // --- RW1 ---
    const RW1units = ['t', 'kg'];

    const RW1ConversionFactors = {
        t: [1, 1000],   // 1 ton = 1000 kg
        kg: [0.001, 1], // 1 kg = 0.001 ton
    };

    const [RW1Value, setRW1Value] = useState(0);
    const [RW1selectedUnit, setRW1SelectedUnit] = useState('t');
    const [RW1InTon, setRW1InTon] = useState(0);

    const handleRW1Change = (value) => {
        setRW1Value(value);
        const factor = RW1ConversionFactors[RW1selectedUnit][0];
        setRW1InTon(parseFloat(value) * factor);
    };

    const handleRW1change = (unit) => {
        let newValue = parseFloat(RW1Value);
        if (unit === 'kg' && RW1selectedUnit === 't') {
            newValue *= 1000; // t → kg
            if (newValue >= 1000) newValue = newValue.toExponential(3);
        } else if (unit === 't' && RW1selectedUnit === 'kg') {
            newValue /= 1000; // kg → t
        }
        setRW1SelectedUnit(unit);
        setRW1Value(isNaN(newValue) ? 0 : newValue);
    };


    // --- RW2 ---
    const RW2units = ['t', 'kg'];

    const RW2ConversionFactors = {
        t: [1, 1000],
        kg: [0.001, 1],
    };

    const [RW2Value, setRW2Value] = useState(0);
    const [RW2selectedUnit, setRW2SelectedUnit] = useState('t');
    const [RW2InTon, setRW2InTon] = useState(0);

    const handleRW2Change = (value) => {
        setRW2Value(value);
        const factor = RW2ConversionFactors[RW2selectedUnit][0];
        setRW2InTon(parseFloat(value) * factor);
    };

    const handleRW2change = (unit) => {
        let newValue = parseFloat(RW2Value);
        if (unit === 'kg' && RW2selectedUnit === 't') {
            newValue *= 1000;
            if (newValue >= 1000) newValue = newValue.toExponential(3);
        } else if (unit === 't' && RW2selectedUnit === 'kg') {
            newValue /= 1000;
        }
        setRW2SelectedUnit(unit);
        setRW2Value(isNaN(newValue) ? 0 : newValue);
    };


    // --- RW3 ---
    const RW3units = ['t', 'kg'];

    const RW3ConversionFactors = {
        t: [1, 1000],
        kg: [0.001, 1],
    };

    const [RW3Value, setRW3Value] = useState(0);
    const [RW3selectedUnit, setRW3SelectedUnit] = useState('t');
    const [RW3InTon, setRW3InTon] = useState(0);

    const handleRW3Change = (value) => {
        setRW3Value(value);
        const factor = RW3ConversionFactors[RW3selectedUnit][0];
        setRW3InTon(parseFloat(value) * factor);
    };

    const handleRW3change = (unit) => {
        let newValue = parseFloat(RW3Value);
        if (unit === 'kg' && RW3selectedUnit === 't') {
            newValue *= 1000;
            if (newValue >= 1000) newValue = newValue.toExponential(3);
        } else if (unit === 't' && RW3selectedUnit === 'kg') {
            newValue /= 1000;
        }
        setRW3SelectedUnit(unit);
        setRW3Value(isNaN(newValue) ? 0 : newValue);
    };


    // --- RW4 ---
    const RW4units = ['t', 'kg'];

    const RW4ConversionFactors = {
        t: [1, 1000],
        kg: [0.001, 1],
    };

    const [RW4Value, setRW4Value] = useState(0);
    const [RW4selectedUnit, setRW4SelectedUnit] = useState('t');
    const [RW4InTon, setRW4InTon] = useState(0);

    const handleRW4Change = (value) => {
        setRW4Value(value);
        const factor = RW4ConversionFactors[RW4selectedUnit][0];
        setRW4InTon(parseFloat(value) * factor);
    };

    const handleRW4change = (unit) => {
        let newValue = parseFloat(RW4Value);
        if (unit === 'kg' && RW4selectedUnit === 't') {
            newValue *= 1000;
            if (newValue >= 1000) newValue = newValue.toExponential(3);
        } else if (unit === 't' && RW4selectedUnit === 'kg') {
            newValue /= 1000;
        }
        setRW4SelectedUnit(unit);
        setRW4Value(isNaN(newValue) ? 0 : newValue);
    };


    // --- WLL Sling ---
    const WLLSlingunits = ['t', 'kg'];

    const WLLSlingConversionFactors = {
        t: [1, 1000],
        kg: [0.001, 1],
    };

    const [WLLSlingValue, setWLLSlingValue] = useState(0);
    const [WLLSlingselectedUnit, setWLLSlingSelectedUnit] = useState('t');
    const [WLLSlingInTon, setWLLSlingInTon] = useState(0);

    const handleWLLSlingChange = (value) => {
        setWLLSlingValue(value);
        const factor = WLLSlingConversionFactors[WLLSlingselectedUnit][0];
        setWLLSlingInTon(parseFloat(value) * factor);
    };

    const handleWLLSlingchange = (unit) => {
        let newValue = parseFloat(WLLSlingValue);
        if (unit === 'kg' && WLLSlingselectedUnit === 't') {
            newValue *= 1000;
            if (newValue >= 1000) newValue = newValue.toExponential(3);
        } else if (unit === 't' && WLLSlingselectedUnit === 'kg') {
            newValue /= 1000;
        }
        setWLLSlingSelectedUnit(unit);
        setWLLSlingValue(isNaN(newValue) ? 0 : newValue);
    };


    // --- WLL Shackle ---
    const WLLShackleunits = ['t', 'kg'];

    const WLLShackleConversionFactors = {
        t: [1, 1000],
        kg: [0.001, 1],
    };

    const [WLLShackleValue, setWLLShackleValue] = useState(0);
    const [WLLShackleselectedUnit, setWLLShackleSelectedUnit] = useState('t');
    const [WLLShackleInTon, setWLLShackleInTon] = useState(0);

    const handleWLLShackleChange = (value) => {
        setWLLShackleValue(value);
        const factor = WLLShackleConversionFactors[WLLShackleselectedUnit][0];
        setWLLShackleInTon(parseFloat(value) * factor);
    };

    const handleWLLShacklechange = (unit) => {
        let newValue = parseFloat(WLLShackleValue);
        if (unit === 'kg' && WLLShackleselectedUnit === 't') {
            newValue *= 1000;
            if (newValue >= 1000) newValue = newValue.toExponential(3);
        } else if (unit === 't' && WLLShackleselectedUnit === 'kg') {
            newValue /= 1000;
        }
        setWLLShackleSelectedUnit(unit);
        setWLLShackleValue(isNaN(newValue) ? 0 : newValue);
    };


    // --- Calculations ---
    // --- Tension Calculations (same as Excel) ---
    // const T1 = ((W + RW1) * (D2 / DT1) * (D4 / DT2)) 2* (S1 / Math.sqrt(S1 ** 2 - (D1 ** 2 + D3 ** 2)));
    const T1 = ((WInTon - (-RW1InTon)) * (D2InKM / DT1InKM) * (D4InKM / DT2InKM)) * (S1InKM / Math.sqrt(S1InKM ** 2 - (D1InKM ** 2 + D3InKM ** 2)));
    const T2 = ((WInTon - (-RW2InTon)) * (D2InKM / DT1InKM) * (D3InKM / DT2InKM)) * (S2InKM / Math.sqrt(S2InKM ** 2 - (D1InKM ** 2 + D4InKM ** 2)));
    const T3 = ((WInTon - (-RW3InTon)) * (D1InKM / DT1InKM) * (D4InKM / DT2InKM)) * (S3InKM / Math.sqrt(S3InKM ** 2 - (D2InKM ** 2 + D3InKM ** 2)));
    const T4 = ((WInTon - (-RW4InTon)) * (D1InKM / DT1InKM) * (D3InKM / DT2InKM)) * (S4InKM / Math.sqrt(S4InKM ** 2 - (D2InKM ** 2 + D4InKM ** 2)));

    const L1 = (WInTon - (-RW1InTon)) * ((D1InKM / DT1InKM) * (D3InKM / DT2InKM));
    const L2 = (WInTon - (-RW2InTon)) * (D2InKM / DT1InKM) * (D4InKM / DT2InKM);
    const L3 = (WInTon - (-RW3InTon)) * (D2InKM / DT1InKM) * (D3InKM / DT2InKM);
    const L4 = (WInTon - (-RW4InTon)) * (D1InKM / DT1InKM) * (D4InKM / DT2InKM);

    const safetyFactorSling = WLLSlingInTon / Math.max(T1, T2, T3, T4);

    const safetyFactorShackle = WLLShackleInTon / Math.max(L1, L2, L3, L4);

    return (
        <>
            <Helmet>
                <title>Four Point Rigging Calculator – OOK | Lifting Point Analysis</title>
                <meta name="description" content="Calculate four-point rigging loads and dimensions using our OOK Calculator – the go-to tool for structural engineers." />
                <link rel="canonical" href="https://www.ookcalculator.com/FourPointRiggingCalculator" />
            </Helmet>

            {/* Header */}
            <section className='background-white PadEye'>
                <div className="position-relative">
                    <div className="position-relative overflow-hidden" style={{ height: '85vh' }}>
                        <picture>
                            <source type="image/webp" srcSet={backgroundWebP} />
                            <img loading="lazy" src={backgroundJPG} alt="Background" className="h-100"
                                width="600" height="400"
                                style={{ objectFit: 'cover', objectPosition: 'center', width: '100%', transform: 'translateX(0%)' }} />
                        </picture>
                        <div className="Overlay-Black-header deflection"></div>
                    </div>
                    <div className="container text-left text-white position-absolute top-50 translate-middle Padeye" style={{ left: '48%' }}>
                        <h1 className="display-4" style={{ fontWeight: '600' }}>Four Point Rigging Calculator</h1>
                        <br />
                        <p className="fs-5">
                            Four Point Rigging Calculator is a tool used in engineering and construction to <br />determine the required dimensions and specifications for padeyes,<br /> which are integral for lifting and rigging systems.
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
                        <h2 className="text-white mt-3">Input</h2>

                        {selectedOption === 'option1' && (
                            <>
                                {/* Excel Inputs */}

                                {/* W (Weight of Cargo) */}
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

                                {/* COG */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">COG (Center of Gravity)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">COG</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={COGValue}
                                                onChange={(e) => handleCOGChange(e.target.value)}
                                            />
                                            <select
                                                className='Calculator-select-option'
                                                value={COGselectedUnit}
                                                onChange={((e) => handleCOGchange(e.target.value))}
                                                label="Select COG Unit"
                                            >
                                                {COGunits.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

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

                                {/* S4 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">S4 (Sling Length L4)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">S₄</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={S4Value}
                                                onChange={(e) => handleS4Change(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={S4selectedUnit}
                                                onChange={(e) => handleS4change(e.target.value)}
                                                label="Select S4 Unit"
                                            >
                                                {S4units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>


                                {/* RW1 - RW4 */}
                                {/* RW1 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">RW1 (Rigging Weight 1)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">RW₁</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={RW1Value}
                                                onChange={(e) => handleRW1Change(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={RW1selectedUnit}
                                                onChange={(e) => handleRW1change(e.target.value)}
                                                label="Select RW1 Unit"
                                            >
                                                {RW1units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* RW2 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">RW2 (Rigging Weight 2)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">RW₂</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={RW2Value}
                                                onChange={(e) => handleRW2Change(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={RW2selectedUnit}
                                                onChange={(e) => handleRW2change(e.target.value)}
                                                label="Select RW2 Unit"
                                            >
                                                {RW2units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* RW3 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">RW3 (Rigging Weight 3)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">RW₃</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={RW3Value}
                                                onChange={(e) => handleRW3Change(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={RW3selectedUnit}
                                                onChange={(e) => handleRW3change(e.target.value)}
                                                label="Select RW3 Unit"
                                            >
                                                {RW3units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* RW4 */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">RW4 (Rigging Weight 4)</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">RW₄</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={RW4Value}
                                                onChange={(e) => handleRW4Change(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={RW4selectedUnit}
                                                onChange={(e) => handleRW4change(e.target.value)}
                                                label="Select RW4 Unit"
                                            >
                                                {RW4units.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>



                                {/* WLL of Sling */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">WLL (Work Load Limit) of Sling</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">WLLₛ</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={WLLSlingValue}
                                                onChange={(e) => handleWLLSlingChange(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={WLLSlingselectedUnit}
                                                onChange={(e) => handleWLLSlingchange(e.target.value)}
                                                label="Select WLL Sling Unit"
                                            >
                                                {WLLSlingunits.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* WLL of Shackle */}
                                <div className="w-100 d-flex justify-content-center align-items-center">
                                    <p className="claculator-conversation-title">WLL (Work Load Limit) of Shackle</p>
                                    <div className="Calculator-Side-A">
                                        <br />
                                        <p className="sigma-symbol d-flex justify-content-center align-items-center m-0">WLLₛₕ</p>
                                        <div className="input-and-select-div">
                                            <input
                                                style={{ transform: 'translate(5px, 0px)' }}
                                                className="calculator-input"
                                                type="number"
                                                value={WLLShackleValue}
                                                onChange={(e) => handleWLLShackleChange(e.target.value)}
                                            />
                                            <select
                                                className="Calculator-select-option"
                                                value={WLLShackleselectedUnit}
                                                onChange={(e) => handleWLLShacklechange(e.target.value)}
                                                label="Select RW4 Unit"
                                            >
                                                {WLLShackleunits.map((unit) => (
                                                    <option key={unit} value={unit}>
                                                        {unit}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                <div className="FourPointRiggingCalculator structure-analysis-calculator-calculator-right-show-hidden-btn-Padeye-MaterialProperty">
                                    <button className="structure-analysis-calculator-calculator-right-show-hidden-btn" onClick={handleCombinedClick}>
                                        {isActive3 ? 'Hide' : 'Solve'}
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </section>
            <div className={isActive3 ? 'show  Sectionmodules  ' : 'hidden  Sectionmodules  '} style={{
                height: '45vw',
            }}>
                <br />
                <br />
                <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Four Point Rigging Calculator</h2>

                <br />
                <br />
                <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            TENSION AT SLING1:
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isNaN(T1) ? "" : T1.toFixed(6)}
                                    readOnly
                                    id="TENSIONATSLING1"
                                    name="value"
                                />
                            </div>
                        </div>
                    </div>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            TENSION AT SLING2:
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isNaN(T2) ? "" : T2.toFixed(6)}
                                    readOnly
                                    id="TENSIONATSLING2"
                                    name="value"
                                />
                            </div>
                        </div>
                    </div>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            TENSION AT SLING3:
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isNaN(T3) ? "" : T3.toFixed(6)}
                                    readOnly
                                    id="TENSIONATSLING3"
                                    name="value"
                                />
                            </div>
                        </div>
                    </div>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            TENSION AT SLING4:
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isNaN(T4) ? "" : T4.toFixed(6)}
                                    readOnly
                                    id="TENSIONATSLING4"
                                    name="value"
                                />
                            </div>
                        </div>
                    </div>

                    {/* L1 */}
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            LOAD AT LIFTING POINT 1 (L1):
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isNaN(L1) ? "" : L1}
                                    readOnly
                                    id="LOADATL1" name="value"
                                />
                            </div>
                        </div>
                    </div>

                    {/* L2 */}
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            LOAD AT LIFTING POINT 2 (L2):
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isNaN(L2) ? "" : L2}
                                    readOnly
                                    id="LOADATL2" name="value"
                                />
                            </div>
                        </div>
                    </div>

                    {/* L3 */}
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            LOAD AT LIFTING POINT 3 (L3):
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isNaN(L3) ? "" : L3}
                                    readOnly
                                    id="LOADATL3" name="value"
                                />
                            </div>
                        </div>
                    </div>

                    {/* L4 */}
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            LOAD AT LIFTING POINT 4 (L4):
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isNaN(L4) ? "" : L4}
                                    readOnly
                                    id="LOADATL4" name="value"
                                />
                            </div>
                        </div>
                    </div>
                    {/* Safety Factor - Sling */}
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            SAFETY FACTOR OF SLING:
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isFinite(safetyFactorSling) ? safetyFactorSling.toFixed(6) : ""}
                                    readOnly
                                    id="SAFETYFACTORSLING" name="value"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Safety Factor - Shackle */}
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.2vw', fontWeight: '600' }}>
                            SAFETY FACTOR OF SHACKLE:
                        </p>
                        <div className='Calculator-Side-A'>
                            <div className='input-and-select-div'>
                                <input
                                    className='calculator-input'
                                    type="number"
                                    value={isFinite(safetyFactorShackle) ? safetyFactorShackle.toFixed(6) : ""}
                                    readOnly
                                    id="SAFETYFACTORSHACKLE" name="value"
                                />
                            </div>
                        </div>
                    </div>

                </div >
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
