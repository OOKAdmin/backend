import React, { useState, useEffect } from 'react'
import { Helmet } from 'react-helmet';

// CSS
import '../../Css/BeamProperties.css'
import '../../Css/BeamDeflection.css'
import '../../Css/NumberLine.css'
import '../../Css/AboutUS.css'
import '../../Css/Navbar.css'
import '../../Css/Padeye.css'

// Background Image
// import Background from '../../images/Padeye-Background.jpg'
import backgroundJPG from '../../images/Padeye-Background.jpg'; // Fallback JPEG/PNG
import backgroundWebP from '../../images/Padeye-Background.webp'; // WebP version
// Images
import MaterialpropertyImg from '../../images/Padeye-Material-property.jpg'
import PadeyegeometryImg from '../../images/Padeye-Padeyegeometry.png'
import shackleImg from '../../images/padeye/shackle.png'
import Slingimg from '../../images/Padeye-Sling-Dia.png'
import PadeyeLoadimg from '../../images/padeye load (2)/padeye load (2).png'
import WeldSizeimg from '../../images/PadEye-weld-size.png'
import Img from '../../images/Padeye-PadeyeFileOutputparameters-tensile-stress-area.png'
import puiSign from '../../images/phi.svg'
// import Img from '../../../images/Padeye-PadeyeFileOutputparameters-tensile-stress-area.png'

// Files
import Materialproperty from './Topics/Materialproperty';
import PadeyeGeometry from './Topics/PadeyeGeometry';
import PadEyeLoad from './Topics/PadEyeLoad';
import ShackleGeometry from './Topics/ShackleGeometry';
import SlingGeometry from './Topics/SlingGeometry';
import WeldSize from './Topics/WeldSize';

// modules
import { Link } from 'react-router-dom';

// icons
import { GrLinkTop } from "react-icons/gr";
import PadeyeFile from './Formula/PadeyeFile'
import ShackleFile from './Formula/ShackleFile'
import SlingsFile from './Formula/SlingsFile'
import Outputs from './Topics/Outputs'
import { LuDot } from "react-icons/lu";
import VideoPlayerSection from './VideoPlayer/VideoPlayerSection'

export default function Padeye() {
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
    setIsActive3(previsActive3 => !previsActive3);

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
    } else {
      // Close in reverse order: Third, Second, then First
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

  // Dynamic class names with conditional hidden class
  const DropDowmOneMain = `
    rightPadeyeDropDown
    AllowableStressedAndDesignLoads
    PadeyeSolutions
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hidden padeyeHidden' : 'showPadEyeOutputs'}
    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  `;

  const DropDowmOnerightMain = `
    rightPadeyeDropDown
    GeometryCheck
    PadeyeSolutions
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hidden padeyeHidden' : 'showPadEyeOutputs'}
    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  `;

  const DropDowmOnerightsecondMain = `
    STRESSCHECKSATPINHOLE
    secondrightPadeyeDropDown
    PadeyeSolutions
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hidden padeyeHidden' : 'showPadEyeOutputs'}
    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  `;

  const DropDowmTwoMain = `
    WELDCHECKATCHEEKPLATEWELD
    ScrollTransactionTwoPadeyeDropDown
    PadeyeSolutions
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hidden padeyeHidden' : 'showPadEyeOutputs'}
    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
    ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
  `;

  const DropDowmTworightMain = `
    STRESSCHECKSATBASEPLATE
    ScrollTransactionTworightPadeyeDropDown
    PadeyeSolutions
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hidden padeyeHidden' : 'showPadEyeOutputs'}
    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
    ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
  `;

  const DropDowmThirdMain = `
    ScrollTransactionThirdPadeyeDropDown
    WELDSTRESSCHECKOFBASEWELD
    PadeyeSolutions
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hidden padeyeHidden' : 'showPadEyeOutputs'}
    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
    ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
    ${showThirdDiv ? 'ScrollTransactionThree' : ''}
  `;

  const DropDowmThirdrightMain = `
    FINALLCHECKS
    ScrollTransactionThirdrightPadeyeDropDown
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hidden padeyeHidden' : 'showPadEyeOutputs'}
    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
    ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
    ${showThirdDiv ? 'ScrollTransactionThree' : ''}
  `;


  // Inputs
  // Material Properties
  // Material Yield Stress
  const MaterialYieldStressunits = ['Mpa', 'Pa'];
  const MaterialYieldStressConversionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };
  const [MaterialYieldStresValue, setMaterialYieldStressValue] = useState(0);
  const [MaterialYieldStresselectedUnit, setMaterialYieldStressSelectedUnit] = useState('Mpa');
  const [MaterialYieldStressInMPa, setMaterialYieldStressInMPa] = useState(0);

  const handleMaterialYieldStresChange = (value) => {
    setMaterialYieldStressValue(value);
    const factor = MaterialYieldStressConversionFactors[MaterialYieldStresselectedUnit][0];
    setMaterialYieldStressInMPa(parseFloat(value) * factor);
  };

  const handleMaterialYieldStreschange = (unit) => {
    let newValue = parseFloat(MaterialYieldStresValue);
    if (unit === 'Pa' && MaterialYieldStresselectedUnit === 'Mpa') {
      newValue *= 1e6;
      if (newValue >= 1000) {
        newValue = newValue.toExponential(3);
      }
    } else if (unit === 'Mpa' && MaterialYieldStresselectedUnit === 'Pa') {
      newValue /= 1e6;
    }
    setMaterialYieldStressSelectedUnit(unit);
    setMaterialYieldStressValue(isNaN(newValue) ? 0 : newValue);
  };
  // Electrode Tensile Strength
  const ElectrodeTensileStrengthunits = ['Mpa', 'Pa'];

  const ElectrodeTensileStrengthFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [ElectrodeTensileStrength, setElectrodeTensileStrength] = useState(0);
  const [ElectrodeTensileStrengthSelectedUnit, setElectrodeTensileStrengthSelectedUnit] = useState('Mpa');
  const [ElectrodeTensileStrengthInMpa, setElectrodeTensileStrengthInMpa] = useState(0);

  const handleElectrodeTensileStrength = (value) => {
    setElectrodeTensileStrength(value);
    const factor = ElectrodeTensileStrengthFactors[ElectrodeTensileStrengthSelectedUnit][0];
    setElectrodeTensileStrengthInMpa(parseFloat(value) * factor);
  };

  const handleElectrodeTensileStrengthSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(ElectrodeTensileStrength) * ElectrodeTensileStrengthFactors[ElectrodeTensileStrengthSelectedUnit][0];
    let convertedValue = newMetricValueInMM / ElectrodeTensileStrengthFactors[unit][0];
    if (unit === 'Pa' && convertedValue >= 1000) {
      convertedValue = convertedValue.toExponential(3);
    }
    setElectrodeTensileStrengthSelectedUnit(unit);
    setElectrodeTensileStrength(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Modulus of Elasticity
  const ModulusofElasticityunits = ['Mpa', 'Pa'];

  const ModulusofElasticityFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [ModulusofElasticityValue, setModulusofElasticityValue] = useState(0);
  const [ModulusofElasticitySelectedUnit, setModulusofElasticitySelectedUnit] = useState('Mpa');
  const [ModulusofElasticityValueInMpa, setModulusofElasticityValueInMpa] = useState(0);

  const handleModulusofElasticChange = (value) => {
    setModulusofElasticityValue(value);
    const factor = ModulusofElasticityFactors[ModulusofElasticitySelectedUnit][0];
    setModulusofElasticityValueInMpa(parseFloat(value) * factor);
  };

  const handleModulusofElasticUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(ModulusofElasticityValue) * ModulusofElasticityFactors[ModulusofElasticitySelectedUnit][0];
    let convertedValue = newMetricValueInMM / ModulusofElasticityFactors[unit][0];
    if (unit === 'Pa' && convertedValue >= 1000) {
      convertedValue = convertedValue.toExponential(3);
    }
    setModulusofElasticitySelectedUnit(unit);
    setModulusofElasticityValue(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Poissons Ratio
  const [PoissonsRatio, setPoissonsRatio] = useState(0)
  const handlePoissonsRatioChange = (value) => {
    setPoissonsRatio(value);
  };


  // Pad eye Geometry
  // RadiusofMainPlate
  const RadiusofMainPlateunits = ['mm', 'cm'];
  const RadiusofMainPlateconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [RadiusofMainPlateValue, setRadiusofMainPlateValue] = useState(0);
  const [RadiusofMainPlateselectedUnit, setRadiusofMainPlateSelectedUnit] = useState('mm');
  const [internalRadiusofMainPlateValue, setInternalRadiusofMainPlateValue] = useState(0); // Always in mm
  const handleRadiusofMainPlateInputChange = (value) => {
    setRadiusofMainPlateValue(value);
    const factor = RadiusofMainPlateconversionFactors[RadiusofMainPlateselectedUnit][0];
    setInternalRadiusofMainPlateValue(parseFloat(value) * factor);
  };

  const handleRadiusofMainPlateUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(RadiusofMainPlateValue) * RadiusofMainPlateconversionFactors[RadiusofMainPlateselectedUnit][0];
    const convertedValue = newMetricValueInMM / RadiusofMainPlateconversionFactors[unit][0];
    setRadiusofMainPlateSelectedUnit(unit);
    setRadiusofMainPlateValue(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Thickness of Main Plate
  const ThicknessofMainPlateunits = ['mm', 'cm'];

  const ThicknessofMainPlateConversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };
  const [ThicknessofMainPlateValue, setThicknessofMainPlateValue] = useState(0);
  const [ThicknessofMainPlateSelectedUnit, setThicknessofMainPlateSelectedUnit] = useState('mm');
  const [internalThicknessofMainPlateValue, setInternalThicknessofMainPlateValue] = useState(0); // Always in mm
  const handleThicknessofMainPlateValue = (value) => {
    setThicknessofMainPlateValue(value);
    const factor = ThicknessofMainPlateConversionFactors[ThicknessofMainPlateSelectedUnit][0];
    setInternalThicknessofMainPlateValue(parseFloat(value) * factor);
  };

  const handleThicknessofMainPlateSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(ThicknessofMainPlateValue) * ThicknessofMainPlateConversionFactors[ThicknessofMainPlateSelectedUnit][0];
    const convertedValue = newMetricValueInMM / ThicknessofMainPlateConversionFactors[unit][0];
    setThicknessofMainPlateSelectedUnit(unit);
    setThicknessofMainPlateValue(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Diameterofeyepinhole
  const Diameterofeyepinholeunits = ['mm', 'cm'];

  const DiameterofeyepinholeconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };
  const [DiameterofeyepinholeValue, setDiameterofeyepinholeValue] = useState(0);
  const [DiameterofeyepinholeselectedUnit, setDiameterofeyepinholeSelectedUnit] = useState('mm');

  const [internalDiameterofeyepinholeValue, setInternalDiameterofeyepinholeValue] = useState(0); // Always in mm
  const handleDiameterofeyepinholeChange = (value) => {
    setDiameterofeyepinholeValue(value);
    const factor = DiameterofeyepinholeconversionFactors[DiameterofeyepinholeselectedUnit][0];
    setInternalDiameterofeyepinholeValue(parseFloat(value) * factor);
  };

  const handleDiameterofeyepinholeUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(DiameterofeyepinholeValue) * DiameterofeyepinholeconversionFactors[DiameterofeyepinholeselectedUnit][0];
    const convertedValue = newMetricValueInMM / DiameterofeyepinholeconversionFactors[unit][0];
    setDiameterofeyepinholeSelectedUnit(unit);
    setDiameterofeyepinholeValue(isNaN(convertedValue) ? 0 : convertedValue);
  };


  // Diameter of Cheek Plate
  const DiameterofCheekPlateunits = ['mm', 'cm'];

  const DiameterofCheekPlateConversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [DiameterofCheekPlateInputValue, setDiameterofCheekPlateInputValue] = useState(0);
  const [DiameterofCheekPlateSelectedUnit, setDiameterofCheekPlateSelectedUnit] = useState('mm');

  const [internalDiameterofCheekPlateInputValue, setInternalDiameterofCheekPlateInputValue] = useState(0); // Always in mm

  const handleDiameterofCheekPlateInputValue = (value) => {
    setDiameterofCheekPlateInputValue(value);
    const factor = DiameterofCheekPlateConversionFactors[DiameterofCheekPlateSelectedUnit][0];
    setInternalDiameterofCheekPlateInputValue(parseFloat(value) * factor);
  };

  const handleDiameterofCheekPlateSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(DiameterofCheekPlateInputValue) * DiameterofCheekPlateConversionFactors[DiameterofCheekPlateSelectedUnit][0];
    const convertedValue = newMetricValueInMM / DiameterofCheekPlateConversionFactors[unit][0];
    setDiameterofCheekPlateSelectedUnit(unit);
    setDiameterofCheekPlateInputValue(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Thickness of Cheek Plate
  const ThicknessofCheekPlateunits = ['mm', 'cm'];

  const ThicknessofCheekPlateConversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [ThicknessofCheekPlateInputValue, setThicknessofCheekPlateInputValue] = useState(0);
  const [ThicknessofCheekPlateSelectedUnit, setThicknessofCheekPlateSelectedUnit] = useState('mm');

  const [internalThicknessofCheekPlateInputValue, setInternalThicknessofCheekPlateInputValue] = useState(0); // Always in mm

  const handleThicknessofCheekPlateInputValue = (value) => {
    setThicknessofCheekPlateInputValue(value);
    const factor = ThicknessofCheekPlateConversionFactors[ThicknessofCheekPlateSelectedUnit][0];
    setInternalThicknessofCheekPlateInputValue(parseFloat(value) * factor);
  };

  const handleThicknessofCheekPlateSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(ThicknessofCheekPlateInputValue) * ThicknessofCheekPlateConversionFactors[ThicknessofCheekPlateSelectedUnit][0];
    const convertedValue = newMetricValueInMM / ThicknessofCheekPlateConversionFactors[unit][0];
    setThicknessofCheekPlateSelectedUnit(unit);
    setThicknessofCheekPlateInputValue(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Total Height of Pad-eye
  const TotalHeightofPadeyeunits = ['mm', 'cm'];

  const TotalHeightofPadeyeconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [inputTotalHeightofPadeyeValue, setInputTotalHeightofPadeyeValue] = useState(0);
  const [selectedTotalHeightofPadeyeUnit, setSelectedTotalHeightofPadeyeUnit] = useState('mm');
  const [internalinputTotalHeightofPadeyeValue, setInternalinputTotalHeightofPadeyeValue] = useState(0); // Always in mm

  const handleInputTotalHeightofPadeyeChange = (value) => {
    setInputTotalHeightofPadeyeValue(value);
    const factor = TotalHeightofPadeyeconversionFactors[selectedTotalHeightofPadeyeUnit][0];
    setInternalinputTotalHeightofPadeyeValue(parseFloat(value) * factor);
  };

  const handleUnitTotalHeightofPadeyeChange = (unit) => {
    const newMetricValueInMM = parseFloat(inputTotalHeightofPadeyeValue) * TotalHeightofPadeyeconversionFactors[selectedTotalHeightofPadeyeUnit][0];
    const convertedValue = newMetricValueInMM / TotalHeightofPadeyeconversionFactors[unit][0];
    setSelectedTotalHeightofPadeyeUnit(unit);
    setInputTotalHeightofPadeyeValue(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Length of Base Plate

  const LengthofBasePlateunits = ['mm', 'cm'];

  const LengthofBasePlateConversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [LengthofBasePlateInputValue, setLengthofBasePlateInputValue] = useState(0);
  const [LengthofBasePlateSelectedUnit, setLengthofBasePlateSelectedUnit] = useState('mm');
  const [internalLengthofBasePlateInputValue, setInternalLengthofBasePlateInputValue] = useState(0); // Always in mm

  const handleLengthofBasePlateInputValue = (value) => {
    setLengthofBasePlateInputValue(value);
    const factor = LengthofBasePlateConversionFactors[LengthofBasePlateSelectedUnit][0];
    setInternalLengthofBasePlateInputValue(parseFloat(value) * factor);
  };

  const handleLengthofBasePlateSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(LengthofBasePlateInputValue) * LengthofBasePlateConversionFactors[LengthofBasePlateSelectedUnit][0];
    const convertedValue = newMetricValueInMM / LengthofBasePlateConversionFactors[unit][0];
    setLengthofBasePlateSelectedUnit(unit);
    setLengthofBasePlateInputValue(isNaN(convertedValue) ? 0 : convertedValue);
  };


  // shackel Geometry
  // Shackle SWL

  const ShackleSWLUnits = ['MT', 'N'];
  const ShackleSWLconversionFactors = {
    MT: [1, 0.0001019716],
    N: [9806.65, 1],
  };

  const [ShackleSWL, setShackleSWL] = useState(0);
  const [ShackleSWLUnit, setShackleSWLUnit] = useState('MT');
  const [ShackleSWLInMT, setShackleSWLInMT] = useState(0);

  const handleShackleSWLvalue = (value) => {
    setShackleSWL(value);
    const factor = ShackleSWLconversionFactors[ShackleSWLUnit][0];
    setShackleSWLInMT(parseFloat(value) * factor);
  };

  const handleShackleSWLUnit = (unit) => {
    let convertedValue;
    if (ShackleSWLUnit === 'MT') {
      convertedValue = parseFloat(ShackleSWL) / ShackleSWLconversionFactors['MT'][0] * ShackleSWLconversionFactors[unit][0];
    } else {
      convertedValue = parseFloat(ShackleSWL) / ShackleSWLconversionFactors['N'][0] * ShackleSWLconversionFactors[unit][0];
    }
    if (unit === 'MT' && (convertedValue.toFixed(3).length > 6 || convertedValue.toString().length > 5)) {
      convertedValue = convertedValue.toFixed(5);
    }
    setShackleSWLUnit(unit);
    setShackleSWL(isNaN(convertedValue) ? 0 : convertedValue);
  };
  // Shackle Inside Length
  const ShackleInsideLengthUnits = ['mm', 'cm'];

  const ShackleInsideLengthconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };
  const [ShackleInsideLength, setShackleInsideLength] = useState(0);
  const [ShackleInsideLengthUnit, setShackleInsideLengthUnit] = useState('mm');

  const [internalShackleInsideLength, setInternalShackleInsideLength] = useState(0); // Always in mm

  const handleShackleInsideLengthvalue = (value) => {
    setShackleInsideLength(value);
    const factor = ShackleInsideLengthconversionFactors[ShackleInsideLengthUnit][0];
    setInternalShackleInsideLength(parseFloat(value) * factor);
  };

  const handleShackleInsideLengthUnit = (unit) => {
    const newMetricValueInMM = parseFloat(ShackleInsideLength) * ShackleInsideLengthconversionFactors[ShackleInsideLengthUnit][0];
    const convertedValue = newMetricValueInMM / ShackleInsideLengthconversionFactors[unit][0];
    setShackleInsideLengthUnit(unit);
    setShackleInsideLength(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Shackle Jaw Width
  const ShackleJawWidthUnits = ['mm', 'cm'];

  const ShackleJawWidthconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };
  const [ShackleJawWidth, setShackleJawWidth] = useState(0);
  const [ShackleJawWidthUnit, setShackleJawWidthUnit] = useState('mm');

  const [internalShackleJawWidth, setInternalShackleJawWidth] = useState(0); // Always in mm

  const handleShackleJawWidthvalue = (value) => {
    setShackleJawWidth(value);
    const factor = ShackleJawWidthconversionFactors[ShackleJawWidthUnit][0];
    setInternalShackleJawWidth(parseFloat(value) * factor);
  };

  const handleShackleJawWidthUnit = (unit) => {
    const newMetricValueInMM = parseFloat(ShackleJawWidth) * ShackleJawWidthconversionFactors[ShackleJawWidthUnit][0];
    const convertedValue = newMetricValueInMM / ShackleJawWidthconversionFactors[unit][0];
    setShackleJawWidthUnit(unit);
    setShackleJawWidth(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Shackle Pin Diameter
  const ShacklePinDiameterUnits = ['mm', 'cm'];

  const ShacklePinDiameterconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };
  const [ShacklePinDiameter, setShacklePinDiameter] = useState(0);
  const [ShacklePinDiameterUnit, setShacklePinDiameterUnit] = useState('mm');

  const [internalShacklePinDiameter, setInternalShacklePinDiameter] = useState(0); // Always in mm

  const handleShacklePinDiametervalue = (value) => {
    setShacklePinDiameter(value);
    const factor = ShacklePinDiameterconversionFactors[ShacklePinDiameterUnit][0];
    setInternalShacklePinDiameter(parseFloat(value) * factor);
  };

  const handleShacklePinDiameterUnit = (unit) => {
    const newMetricValueInMM = parseFloat(ShacklePinDiameter) * ShacklePinDiameterconversionFactors[ShacklePinDiameterUnit][0];
    const convertedValue = newMetricValueInMM / ShacklePinDiameterconversionFactors[unit][0];
    setShacklePinDiameterUnit(unit);
    setShacklePinDiameter(isNaN(convertedValue) ? 0 : convertedValue);
  };


  // SlingDiameter
  const SlingDiameterUnits = ['mm', 'cm'];

  const SlingDiameterconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };
  const [SlingDiameter, setSlingDiameter] = useState(0);
  const [SlingDiameterUnit, setSlingDiameterUnit] = useState('mm');
  const [internalSlingDiameter, setInternalSlingDiameter] = useState(0); // Always in mm

  const handleSlingDiametervalue = (value) => {
    setSlingDiameter(value);
    const factor = SlingDiameterconversionFactors[SlingDiameterUnit][0];
    setInternalSlingDiameter(parseFloat(value) * factor);
  };

  const handleSlingDiameterUnit = (unit) => {
    const newMetricValueInMM = parseFloat(SlingDiameter) * SlingDiameterconversionFactors[SlingDiameterUnit][0];
    const convertedValue = newMetricValueInMM / SlingDiameterconversionFactors[unit][0];
    setSlingDiameterUnit(unit);
    setSlingDiameter(isNaN(convertedValue) ? 0 : convertedValue);
  };


  // pad-eye load
  // Load on Pad eye
  const LoadonPadeyeunits = ['MT', 'N'];
  const LoadonPadeyeFactors = {
    MT: [1, 0.0001019716],
    N: [9806.65, 1],
  };

  const [LoadonPadeyeValue, setLoadonPadeyeValue] = useState(0);
  const [LoadonPadeyeSelectedUnit, setLoadonPadeyeSelectedUnit] = useState('MT');
  const [LoadonPadeyeValueInMT, setLoadonPadeyeValueInMT] = useState(0);

  const handleLoadonPadeyeValueChange = (value) => {
    setLoadonPadeyeValue(value);
    const factor = LoadonPadeyeFactors[LoadonPadeyeSelectedUnit][0];
    setLoadonPadeyeValueInMT(parseFloat(value) * factor);
  };

  const handleLoadonPadeyeUnitChange = (unit) => {
    let convertedValue;
    if (LoadonPadeyeSelectedUnit === 'MT') {
      convertedValue = parseFloat(LoadonPadeyeValue) / LoadonPadeyeFactors['MT'][0] * LoadonPadeyeFactors[unit][0];
    } else {
      convertedValue = parseFloat(LoadonPadeyeValue) / LoadonPadeyeFactors['N'][0] * LoadonPadeyeFactors[unit][0];
    }
    if (unit === 'MT' && (convertedValue.toFixed(3).length > 6 || convertedValue.toString().length > 5)) {
      convertedValue = convertedValue.toFixed(5);
    }
    setLoadonPadeyeSelectedUnit(unit);
    setLoadonPadeyeValue(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Angle of Load with Vertical

  const AngleofLoadwithVerticalunit = ['deg'];
  const AngleofLoadwithVerticalFactors = {
    deg: [1],
  };
  const [AngleofLoadwithVerticalvalue, setAngleofLoadwithVerticalvalue] = useState(0);
  const [AngleofLoadwithVerticalSelectedUnit, setAngleofLoadwithVerticalSelectedUnit] = useState('deg');
  const handleAngleofLoadwithVerticalvalue = (values) => {
    setAngleofLoadwithVerticalvalue(values);
  };
  const handleAngleofLoadwithVerticalSelectedUnit = (units) => {
    setAngleofLoadwithVerticalSelectedUnit(units);
    const DLFFactors = AngleofLoadwithVerticalFactors[units][AngleofLoadwithVerticalunit.indexOf(AngleofLoadwithVerticalSelectedUnit)];
    setAngleofLoadwithVerticalvalue(isNaN(DLFFactors) ? 0 : (parseFloat(AngleofLoadwithVerticalvalue) / DLFFactors));
  };



  // Out of Plane Angle
  const OutofPlaneAngleUnits = ['deg'];
  const OutofPlaneAngleFactors = {
    deg: [1],
  };
  const [OutofPlaneAngleValue, setOutofPlaneAngleValue] = useState(0);
  const [OutofPlaneAngleselectedUnit, setOutofPlaneAngleselectedUnit] = useState('deg');
  const handleOutofPlaneAngleValueStresChange = (value) => {
    setOutofPlaneAngleValue(value);
  };
  const handleOutofPlaneAngleunitchange = (unit) => {
    setOutofPlaneAngleselectedUnit(unit);
    const OutofPlaneAngleFactor = OutofPlaneAngleFactors[unit][OutofPlaneAngleUnits.indexOf(OutofPlaneAngleselectedUnit)];
    setOutofPlaneAngleValue(isNaN(OutofPlaneAngleFactor) ? 0 : (parseFloat(OutofPlaneAngleValue) / OutofPlaneAngleFactor));
  };

  // DLF Value
  const [DLFValue, setDLFValue] = useState(0);
  const handleDLFValue = (values) => {
    setDLFValue(values);
  };



  // weld size
  // Base Weld Leg Size
  const BaseWeldLegSizeUnits = ['mm', 'cm'];

  const BaseWeldLegSizeconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };
  const [BaseWeldLegSize, setBaseWeldLegSize] = useState(0);
  const [BaseWeldLegSizeUnit, setBaseWeldLegSizeUnit] = useState('mm');
  const [internalBaseWeldLegSize, setInternalBaseWeldLegSize] = useState(0); // Always in mm

  const handleBaseWeldLegSizevalue = (value) => {
    setBaseWeldLegSize(value);
    const factor = BaseWeldLegSizeconversionFactors[BaseWeldLegSizeUnit][0];
    setInternalBaseWeldLegSize(parseFloat(value) * factor);
  };

  const handleBaseWeldLegSizeUnit = (unit) => {
    const newMetricValueInMM = parseFloat(BaseWeldLegSize) * BaseWeldLegSizeconversionFactors[BaseWeldLegSizeUnit][0];
    const convertedValue = newMetricValueInMM / BaseWeldLegSizeconversionFactors[unit][0];
    setBaseWeldLegSizeUnit(unit);
    setBaseWeldLegSize(isNaN(convertedValue) ? 0 : convertedValue);
  };

  // Cheek Plate Weld Leg Size
  const CheekPlateWeldLegSizeUnits = ['mm', 'cm'];

  const CheekPlateWeldLegSizeconversionFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };
  const [CheekPlateWeldLegSize, setCheekPlateWeldLegSize] = useState(0);
  const [CheekPlateWeldLegSizeUnit, setCheekPlateWeldLegSizeUnit] = useState('mm');
  const [internalCheekPlateWeldLegSize, setInternalCheekPlateWeldLegSize] = useState(0); // Always in mm

  const handleCheekPlateWeldLegSizevalue = (value) => {
    setCheekPlateWeldLegSize(value);
    const factor = CheekPlateWeldLegSizeconversionFactors[CheekPlateWeldLegSizeUnit][0];
    setInternalCheekPlateWeldLegSize(parseFloat(value) * factor);
  };

  const handleCheekPlateWeldLegSizeUnit = (unit) => {
    const newMetricValueInMM = parseFloat(CheekPlateWeldLegSize) * CheekPlateWeldLegSizeconversionFactors[CheekPlateWeldLegSizeUnit][0];
    const convertedValue = newMetricValueInMM / CheekPlateWeldLegSizeconversionFactors[unit][0];
    setCheekPlateWeldLegSizeUnit(unit);
    setCheekPlateWeldLegSize(isNaN(convertedValue) ? 0 : convertedValue);
  };


  const AllowableBearingStressunits = ['Mpa', 'Pa'];
  const AllowableBearingStressConversionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [AllowableBearingStressValue, setAllowableBearingStressValue] = useState(0);
  const [AllowableBearingStressselectedUnit, setAllowableBearingStressSelectedUnit] = useState('Mpa');

  const handleAllowableBearingStressUnitChange = (unit) => {
    const AllowableBearingStressFactor = AllowableBearingStressConversionFactors[unit][AllowableBearingStressunits.indexOf(AllowableBearingStressselectedUnit)];
    let newValue = parseFloat(AllowableBearingStressValue) / AllowableBearingStressFactor;
    newValue = formatValue(newValue, unit);
    setAllowableBearingStressSelectedUnit(unit);
    setAllowableBearingStressValue(newValue);
  };

  const calculateAllowableBearingStressValue = () => {
    const MaterialYieldStress = parseFloat(MaterialYieldStressInMPa);
    if (!isNaN(MaterialYieldStress)) {
      let newValue = MaterialYieldStress * 0.9;
      newValue = formatValue(newValue, AllowableBearingStressselectedUnit);
      setAllowableBearingStressValue(isNaN(newValue) ? 0 : newValue);
    }
  };
  const formatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateAllowableBearingStressValue();
  }, [MaterialYieldStressInMPa]);



  const AllowableBendingStressInPlaneunits = ['Mpa', 'Pa'];
  const AllowableBendingStressInPlaneconversionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [AllowableBendingStressInPlaneValue, setAllowableBendingStressInPlaneValue] = useState(0);
  const [AllowableBendingStressInPlaneselectedUnit, setAllowableBendingStressInPlaneselectedUnit] = useState('Mpa');

  const handleAllowableBendingStressInPlaneUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = AllowableBendingStressInPlaneconversionFactors[unit][AllowableBendingStressInPlaneunits.indexOf(AllowableBendingStressInPlaneselectedUnit)];
    let newValue = parseFloat(AllowableBendingStressInPlaneValue) / AllowableBendingStressInPlaneFactor;
    newValue = AllowableBendingStressInPlaneFormatValue(newValue, unit);
    setAllowableBendingStressInPlaneselectedUnit(unit);
    setAllowableBendingStressInPlaneValue(newValue);
  };

  const calculateAllowableBendingStressInPlaneValue = () => {
    const MaterialYieldStress = parseFloat(MaterialYieldStressInMPa);
    if (!isNaN(MaterialYieldStress)) {
      let newValue = MaterialYieldStress * 0.6;
      newValue = AllowableBendingStressInPlaneFormatValue(newValue, AllowableBendingStressInPlaneselectedUnit);
      setAllowableBendingStressInPlaneValue(isNaN(newValue) ? 0 : newValue);
    }
  };

  const AllowableBendingStressInPlaneFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateAllowableBendingStressInPlaneValue();
  }, [MaterialYieldStressInMPa]);







  const AllowableBendingStressOutofPlaneunits = ['Mpa', 'Pa'];
  const AllowableBendingStressOutofPlaneconversionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [AllowableBendingStressOutofPlaneValue, setAllowableBendingStressOutofPlaneValue] = useState(0);
  const [AllowableBendingStressOutofPlaneselectedUnit, setAllowableBendingStressOutofPlaneselectedUnit] = useState('Mpa');

  const handleAllowableBendingStressOutofPlaneselectedUnitChange = (unit) => {
    const AllowableBendingStressOutofPlaneFactor = AllowableBendingStressOutofPlaneconversionFactors[unit][AllowableBendingStressOutofPlaneunits.indexOf(AllowableBendingStressOutofPlaneselectedUnit)];
    let newValue = parseFloat(AllowableBendingStressOutofPlaneValue) / AllowableBendingStressOutofPlaneFactor;
    newValue = AllowableBendingStressOutofPlaneFormatValue(newValue, unit);
    setAllowableBendingStressOutofPlaneselectedUnit(unit);
    setAllowableBendingStressOutofPlaneValue(newValue);
  };

  const calculateAllowableBendingStressOutofPlaneValue = () => {
    const MaterialYieldStress = parseFloat(MaterialYieldStressInMPa);
    if (!isNaN(MaterialYieldStress)) {
      let newValue = MaterialYieldStress * 0.75;
      newValue = AllowableBendingStressOutofPlaneFormatValue(newValue, AllowableBendingStressOutofPlaneselectedUnit);
      setAllowableBendingStressOutofPlaneValue(isNaN(newValue) ? 0 : newValue);
    }
  };

  const AllowableBendingStressOutofPlaneFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateAllowableBendingStressOutofPlaneValue();
  }, [MaterialYieldStressInMPa]);


  const AllowableTensileStressUnits = ['Mpa', 'Pa'];
  const [AllowableTensileStressValue, setAllowableTensileStressValue] = useState(0);
  const [AllowableTensileStressSelectedUnit, setAllowableTensileStressSelectedUnit] = useState('Mpa');

  const handleAllowableTensileStressselectedUnitChange = (unit) => {
    let newValue = parseFloat(AllowableTensileStressValue);
    if (unit === 'Pa' && AllowableTensileStressSelectedUnit === 'Mpa') {
      newValue *= 1e6;
    } else if (unit === 'Mpa' && AllowableTensileStressSelectedUnit === 'Pa') {
      newValue /= 1e6;
    }
    newValue = AllowableTensileStressFormatValue(newValue, unit);
    setAllowableTensileStressSelectedUnit(unit);
    setAllowableTensileStressValue(newValue);
  };

  const calculateAllowableTensileStressValue = () => {
    const MaterialYieldStressFloat = parseFloat(MaterialYieldStressInMPa);
    if (!isNaN(MaterialYieldStressFloat)) {
      let newValue = MaterialYieldStressFloat * 0.6;
      newValue = AllowableTensileStressFormatValue(newValue, AllowableTensileStressSelectedUnit);
      setAllowableTensileStressValue(isNaN(newValue) ? 0 : newValue);
    }
  };

  const AllowableTensileStressFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      } else {
        return floatValue.toFixed(2);
      }
    }
  };

  useEffect(() => {
    calculateAllowableTensileStressValue();
  }, [MaterialYieldStressInMPa]);

  const AllowableTensileStressatpinholeUnits = ['MPa', 'Pa'];
  const [AllowableTensileStressatpinholeValue, setAllowableTensileStressatpinholeValue] = useState(0);
  const [AllowableTensileStressatpinholeSelectedUnit, setAllowableTensileStressatpinholeSelectedUnit] = useState('MPa');

  const handleAllowableTensileStressatpinholeselectedUnitChange = (unit) => {
    let newValue = parseFloat(AllowableTensileStressatpinholeValue);
    if (unit === 'Pa' && AllowableTensileStressatpinholeSelectedUnit === 'MPa') {
      newValue *= 1e6;
    } else if (unit === 'MPa' && AllowableTensileStressatpinholeSelectedUnit === 'Pa') {
      newValue /= 1e6;
    }
    newValue = AllowableTensileStressatpinholeFormatValue(newValue, unit);
    setAllowableTensileStressatpinholeSelectedUnit(unit);
    setAllowableTensileStressatpinholeValue(newValue);
  };

  const calculateAllowableTensileStressatpinholeValue = () => {
    const MaterialYieldStressFloat = parseFloat(MaterialYieldStressInMPa);
    if (!isNaN(MaterialYieldStressFloat)) {
      let newValue = MaterialYieldStressFloat * 0.45;
      newValue = AllowableTensileStressatpinholeFormatValue(newValue, AllowableTensileStressatpinholeSelectedUnit);
      setAllowableTensileStressatpinholeValue(isNaN(newValue) ? 0 : newValue);
    }
  };

  const AllowableTensileStressatpinholeFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'MPa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateAllowableTensileStressatpinholeValue();
  }, [MaterialYieldStressInMPa]);


  const AllowableShearStressUnits = ['Mpa', 'Pa'];
  const AllowableShearStressFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [AllowableShearStressValue, setAllowableShearStressValue] = useState(0);
  const [AllowableShearStressSelectedUnit, setAllowableShearStressSelectedUnit] = useState('Mpa');

  const handleAllowableShearStressselectedUnitChange = (unit) => {
    const AllowableShearStressfactor = AllowableShearStressFactors[unit][AllowableShearStressUnits.indexOf(AllowableShearStressSelectedUnit)];
    let newValue = parseFloat(AllowableShearStressValue) / AllowableShearStressfactor;
    newValue = AllowableShearStressFormatValue(newValue, unit);
    setAllowableShearStressSelectedUnit(unit);
    setAllowableShearStressValue(newValue);
  };

  const calculateAllowableShearStressValue = () => {
    const MaterialYieldStressFloat = parseFloat(MaterialYieldStressInMPa);
    if (!isNaN(MaterialYieldStressFloat)) {
      let newValue = MaterialYieldStressFloat * 0.4;
      newValue = AllowableShearStressFormatValue(newValue, AllowableShearStressSelectedUnit);
      setAllowableShearStressValue(isNaN(newValue) ? 0 : newValue);
    }
  };

  const AllowableShearStressFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateAllowableShearStressValue();
  }, [MaterialYieldStressInMPa]);

  const AllowableHertzStressatPinHoleUnits = ['Mpa', 'Pa'];
  const AllowableHertzStressatPinHoleFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [AllowableHertzStressatPinHoleValue, setAllowableHertzStressatPinHoleValue] = useState(0);
  const [AllowableHertzStressatPinHoleSelectedUnit, setAllowableHertzStressatPinHoleSelectedUnit] = useState('Mpa');

  const handleAllowableHertzStressatPinHoleselectedUnitChange = (unit) => {
    const AllowableHertzStressfactor = AllowableHertzStressatPinHoleFactors[unit][AllowableHertzStressatPinHoleUnits.indexOf(AllowableHertzStressatPinHoleSelectedUnit)];
    let newValue = parseFloat(AllowableHertzStressatPinHoleValue) / AllowableHertzStressfactor;
    newValue = AllowableHertzStressatPinHoleFormatValue(newValue, unit);
    setAllowableHertzStressatPinHoleSelectedUnit(unit);
    setAllowableHertzStressatPinHoleValue(newValue);
  };

  const calculateAllowableHertzStressatPinHoleValue = () => {
    const MaterialYieldStressFloat = parseFloat(MaterialYieldStressInMPa);
    if (!isNaN(MaterialYieldStressFloat)) {
      let newValue = MaterialYieldStressFloat * 2.5;
      newValue = AllowableHertzStressatPinHoleFormatValue(newValue, AllowableHertzStressatPinHoleSelectedUnit);
      setAllowableHertzStressatPinHoleValue(isNaN(newValue) ? 0 : newValue);
    }
  };

  const AllowableHertzStressatPinHoleFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateAllowableHertzStressatPinHoleValue();
  }, [MaterialYieldStressInMPa]);

  const AllowableWeldStressUnits = ['Mpa', 'Pa'];
  const AllowableWeldStressFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [AllowableWeldStressValue, setAllowableWeldStressValue] = useState(0);
  const [AllowableWeldStressSelectedUnit, setAllowableWeldStressSelectedUnit] = useState('Mpa');

  const handleAllowableWeldStressselectedUnitChange = (unit) => {
    const AllowableWeldStressfactor = AllowableWeldStressFactors[unit][AllowableWeldStressUnits.indexOf(AllowableWeldStressSelectedUnit)];
    let newValue = parseFloat(AllowableWeldStressValue) / AllowableWeldStressfactor;
    newValue = AllowableWeldStressFormatValue(newValue, unit);
    setAllowableWeldStressSelectedUnit(unit);
    setAllowableWeldStressValue(newValue);
  };

  const calculateAllowableWeldStressValue = () => {
    const ElectrodeTensileStrengthFloat = parseFloat(ElectrodeTensileStrengthInMpa);
    if (!isNaN(ElectrodeTensileStrengthFloat)) {
      let newValue = ElectrodeTensileStrengthFloat * 0.3;
      newValue = AllowableWeldStressFormatValue(newValue, AllowableWeldStressSelectedUnit);
      setAllowableWeldStressValue(isNaN(newValue) ? 0 : newValue);
    }
  };

  const AllowableWeldStressFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateAllowableWeldStressValue();
  }, [ElectrodeTensileStrengthInMpa]);




  const cosRadians = (degrees) => {
    const radians = degrees * Math.PI / 180;
    return Math.cos(radians);
  };

  const sinRadians = (degrees) => {
    const radians = degrees * Math.PI / 180;
    return Math.sin(radians);
  };



  const LoadinVerticalDirectionZdirectionUnits = ['MT', 'N'];
  const LoadinVerticalDirectionZdirectionFactors = {
    MT: [1, 0.0001019716],
    N: [9806.65, 1],
  };

  const [LoadinVerticalDirectionZdirectionValue, setLoadinVerticalDirectionZdirectionValue] = useState(0);
  const [LoadinVerticalDirectionZdirectionSelectedUnit, setLoadinVerticalDirectionZdirectionSelectedUnit] = useState('MT');

  const handleLoadinVerticalDirectionZdirectionselectedUnitChange = (unit) => {
    setLoadinVerticalDirectionZdirectionSelectedUnit(unit);
    const factor = LoadinVerticalDirectionZdirectionFactors[unit][LoadinVerticalDirectionZdirectionUnits.indexOf(LoadinVerticalDirectionZdirectionSelectedUnit)];
    setLoadinVerticalDirectionZdirectionValue((parseFloat(LoadinVerticalDirectionZdirectionValue) * factor).toFixed(2));
  };

  const calculateLoadinVerticalDirectionZdirectionValue = () => {
    const LoadonPadeye = LoadonPadeyeValueInMT
    const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
    const LoadinVerticalDirectionZdirectionValue = (LoadonPadeye * multiplyValue).toFixed(2);
    setLoadinVerticalDirectionZdirectionValue(isNaN(LoadinVerticalDirectionZdirectionValue) ? 0 : LoadinVerticalDirectionZdirectionValue);
  };
  useEffect(() => {
    calculateLoadinVerticalDirectionZdirectionValue();
  }, [LoadonPadeyeValueInMT, AngleofLoadwithVerticalvalue]);



  const LoadinHorizontalDirectionYdirectionUnits = ['MT', 'N'];
  const LoadinHorizontalDirectionYdirectionFactors = {
    MT: [1, 0.0001019716],
    N: [9806.65, 1],
  };

  const [LoadinHorizontalDirectionYdirection, setLoadinHorizontalDirectionYdirection] = useState(0);
  const [LoadinHorizontalDirectionYdirectionSelectedUnit, setLoadinHorizontalDirectionYdirectionSelectedUnit] = useState('MT');

  const handleLoadinHorizontalDirectionYdirectionselectedUnitChange = (unit) => {
    setLoadinHorizontalDirectionYdirectionSelectedUnit(unit);
    const factor = LoadinHorizontalDirectionYdirectionFactors[unit][LoadinHorizontalDirectionYdirectionUnits.indexOf(LoadinHorizontalDirectionYdirectionSelectedUnit)];
    setLoadinHorizontalDirectionYdirection((parseFloat(LoadinHorizontalDirectionYdirection) * factor).toFixed(2));
  };

  const calculateLoadinHorizontalDirectionYdirectionValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const AngleofLoadwithVertical = parseFloat(AngleofLoadwithVerticalvalue);
    const pi = Math.PI;
    const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVertical * 3.14159265358979 / 180) * Math.cos(OutofPlaneAngleValue * 3.14159265358979 / 180)
    const LoadinHorizontalDirectionYdirection = (LoadonPadeye * LoadinHorizontalDirectionYdirectionMultipleValue).toFixed(2);
    setLoadinHorizontalDirectionYdirection(isNaN(LoadinHorizontalDirectionYdirection) ? 0 : LoadinHorizontalDirectionYdirection); // Adjust to your precision needs
  };
  useEffect(() => {
    calculateLoadinHorizontalDirectionYdirectionValue();
  }, [LoadonPadeyeValueInMT, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue]);



  const OutofplaneLoadLateralLoadXdirectionUnits = ['MT', 'N'];
  const OutofplaneLoadLateralLoadXdirectionFactors = {
    MT: [1, 0.0001019716],
    N: [9806.65, 1],
  };

  const [OutofplaneLoadLateralLoadXdirection, setOutofplaneLoadLateralLoadXdirection] = useState(0);
  const [OutofplaneLoadLateralLoadXdirectionSelectedUnit, setOutofplaneLoadLateralLoadXdirectionSelectedUnit] = useState('MT');

  const handleOutofplaneLoadLateralLoadXdirectionselectedUnitChange = (unit) => {
    setOutofplaneLoadLateralLoadXdirectionSelectedUnit(unit);
    const factor = OutofplaneLoadLateralLoadXdirectionFactors[unit][OutofplaneLoadLateralLoadXdirectionUnits.indexOf(OutofplaneLoadLateralLoadXdirectionSelectedUnit)];
    setOutofplaneLoadLateralLoadXdirection((parseFloat(OutofplaneLoadLateralLoadXdirection) * factor).toFixed(2));
  };

  const calculateOutofplaneLoadLateralLoadXdirectionValue = () => {
    const D13 = parseFloat(LoadonPadeyeValueInMT);
    const D30 = parseFloat(AngleofLoadwithVerticalvalue);
    const D31 = parseFloat(OutofPlaneAngleValue);
    const radiansD30 = D30 * Math.PI / 180;
    const radiansD31 = D31 * Math.PI / 180;
    const sineD30 = Math.sin(radiansD30);
    const sineD31 = Math.sin(radiansD31);
    const calculationResult = (D13 * sineD30 * sineD31).toFixed(2);
    setOutofplaneLoadLateralLoadXdirection(isNaN(calculationResult) ? 0 : calculationResult); // Adjust to your precision needs
  };
  useEffect(() => {
    calculateOutofplaneLoadLateralLoadXdirectionValue();
  }, [LoadonPadeyeValueInMT, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue]);




  const PadEyeDesignLoadUnits = ['MT', 'N'];
  const PadEyeDesignLoadFactors = {
    MT: [1, 0.0001019716],
    N: [9806.65, 1],
  };

  const [PadEyeDesignLoad, setPadEyeDesignLoad] = useState(0);
  const [PadEyeDesignLoadSelectedUnit, setPadEyeDesignLoadSelectedUnit] = useState('MT');

  const handlePadEyeDesignLoadselectedUnitChange = (unit) => {
    setPadEyeDesignLoadSelectedUnit(unit);
    const factor = PadEyeDesignLoadFactors[unit][PadEyeDesignLoadUnits.indexOf(PadEyeDesignLoadSelectedUnit)];
    setPadEyeDesignLoad((parseFloat(PadEyeDesignLoad) * factor).toFixed(2));
  };

  const calculatePadEyeDesignLoadValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const DLF = parseFloat(DLFValue)
    const PadEyeDesignLoad = (LoadonPadeye * DLF).toFixed(2);
    setPadEyeDesignLoad(isNaN(PadEyeDesignLoad) ? 0 : PadEyeDesignLoad);
  };
  useEffect(() => {
    calculatePadEyeDesignLoadValue();
  }, [LoadonPadeyeValueInMT, DLFValue]);




  // DesignLoadinVerticalDirectionZdirection
  const DesignLoadinVerticalDirectionZdirectionUnits = ['MT', 'N'];
  const DesignLoadinVerticalDirectionZdirectionFactors = {
    MT: [1, 0.00010],
    N: [9806.65, 1],
  };

  const [DesignLoadinVerticalDirectionZdirection, setDesignLoadinVerticalDirectionZdirection] = useState(0);
  const [DesignLoadinVerticalDirectionZdirectionSelectedUnit, setDesignLoadinVerticalDirectionZdirectionSelectedUnit] = useState('MT');

  const handleDesignLoadinVerticalDirectionZdirectionselectedUnitChange = (unit) => {
    setDesignLoadinVerticalDirectionZdirectionSelectedUnit(unit);
    const factor = DesignLoadinVerticalDirectionZdirectionFactors[unit][DesignLoadinVerticalDirectionZdirectionUnits.indexOf(DesignLoadinVerticalDirectionZdirectionSelectedUnit)];
    setDesignLoadinVerticalDirectionZdirection((parseFloat(DesignLoadinVerticalDirectionZdirection) * factor).toFixed(2));
  };

  const calculateDesignLoadinVerticalDirectionZdirectionValue = () => {
    const LoadonPadeye = LoadonPadeyeValueInMT
    const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
    const LoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
    const DLF = parseFloat(DLFValue);
    const DesignLoadinVerticalDirectionZdirection = (LoadinVerticalDirectionZdirectionValue * DLF).toFixed(2);
    setDesignLoadinVerticalDirectionZdirection(isNaN(DesignLoadinVerticalDirectionZdirection) ? 0 : DesignLoadinVerticalDirectionZdirection); // Adjust to your precision needs
  };

  useEffect(() => {
    calculateDesignLoadinVerticalDirectionZdirectionValue();
  }, [DLFValue, LoadonPadeyeValueInMT]);




  // DesignLoadinHorizontalDirectionYdirection

  const DesignLoadinHorizontalDirectionYdirectionUnits = ['MT', 'N'];
  const DesignLoadinHorizontalDirectionYdirectionFactors = {
    MT: [1, 0.000101971621],
    N: [9806.65, 1],
  };

  const [DesignLoadinHorizontalDirectionYdirection, setDesignLoadinHorizontalDirectionYdirection] = useState(0);
  const [DesignLoadinHorizontalDirectionYdirectionSelectedUnit, setDesignLoadinHorizontalDirectionYdirectionSelectedUnit] = useState('MT');

  const handleDesignLoadinHorizontalDirectionYdirectionselectedUnitChange = (unit) => {
    setDesignLoadinHorizontalDirectionYdirectionSelectedUnit(unit);
    const factor = DesignLoadinHorizontalDirectionYdirectionFactors[unit][DesignLoadinHorizontalDirectionYdirectionUnits.indexOf(DesignLoadinHorizontalDirectionYdirectionSelectedUnit)];
    setDesignLoadinHorizontalDirectionYdirection((parseFloat(DesignLoadinHorizontalDirectionYdirection) * factor).toFixed(2)); // Adjust to your precision needs
  };

  const calculateDesignLoadinHorizontalDirectionYdirectionValue = () => {
    const pi = Math.PI;
    const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
    const LoadinHorizontalDirectionYdirection = (LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue).toFixed(1);
    const DesignLoadinHorizontalDirectionYdirection = (LoadinHorizontalDirectionYdirection * DLFValue).toFixed(3);
    setDesignLoadinHorizontalDirectionYdirection(isNaN(DesignLoadinHorizontalDirectionYdirection) ? 0 : DesignLoadinHorizontalDirectionYdirection); // Adjust to your precision needs
  };

  useEffect(() => {
    calculateDesignLoadinHorizontalDirectionYdirectionValue();
  }, [LoadonPadeyeValueInMT, AngleofLoadwithVerticalvalue, DLFValue, OutofPlaneAngleValue]); // Dependencies to ensure calculation runs when these values change

  // DesignLoadOutofPlaneLateralXdirection2




  const DesignLoadOutofPlaneLateralXdirectionUnits = ['MT', 'N'];
  const DesignLoadOutofPlaneLateralXdirectionFactors = {
    MT: [1, 0.000101971621],
    N: [9806.65, 1],
  };

  const [DesignLoadOutofPlaneLateralXdirection, setDesignLoadOutofPlaneLateralXdirection] = useState(0);
  const [DesignLoadOutofPlaneLateralXdirectionSelectedUnit, setDesignLoadOutofPlaneLateralXdirectionSelectedUnit] = useState('MT');

  const handleDesignLoadOutofPlaneLateralXdirectionselectedUnitChange = (unit) => {
    const currentFactor = DesignLoadOutofPlaneLateralXdirectionFactors[unit][DesignLoadOutofPlaneLateralXdirectionUnits.indexOf(DesignLoadOutofPlaneLateralXdirectionSelectedUnit)];
    const newValue = parseFloat((DesignLoadOutofPlaneLateralXdirection) * currentFactor);
    setDesignLoadOutofPlaneLateralXdirection(newValue.toFixed(7)); // Adjust to your precision needs
    setDesignLoadOutofPlaneLateralXdirectionSelectedUnit(unit);
  };

  const calculateDesignLoadOutofPlaneLateralXdirectionValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT)
    const DLF = parseFloat(DLFValue);
    const pi = Math.PI;
    const PadEyeDesignLoadvalue = LoadonPadeye * DLF;
    const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
    const OutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
    const DesignLoadOutofPlaneLateralXdirection = ((OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)).toFixed(2);
    setDesignLoadOutofPlaneLateralXdirection(isNaN(DesignLoadOutofPlaneLateralXdirection) ? 0 : DesignLoadOutofPlaneLateralXdirection); // Adjust to your precision needs
  };

  useEffect(() => {
    calculateDesignLoadOutofPlaneLateralXdirectionValue();
  }, [DLFValue, LoadonPadeyeValueInMT]);





  // Geometry check
  const ClearancebetweenpinholediaandpindiaUnits = ['mm', 'cm'];
  const ClearancebetweenpinholediaandpindiaFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [Clearancebetweenpinholediaandpindia, setClearancebetweenpinholediaandpindia] = useState(0);
  const [ClearancebetweenpinholediaandpindiaSelectedunit, setClearancebetweenpinholediaandpindiaSelectedUnit] = useState('mm');

  const handleClearancebetweenpinholediaandpindiaselectedUnitChange = (unit) => {
    setClearancebetweenpinholediaandpindiaSelectedUnit(unit);
    const ClearancebetweenpinholediaandpindiaFactor = ClearancebetweenpinholediaandpindiaFactors[unit][ClearancebetweenpinholediaandpindiaUnits.indexOf(ClearancebetweenpinholediaandpindiaSelectedunit)];
    setClearancebetweenpinholediaandpindia((parseFloat(Clearancebetweenpinholediaandpindia) / ClearancebetweenpinholediaandpindiaFactor).toFixed(3));
  };

  const calculateClearancebetweenpinholediaandpindiaValue = () => {
    const Diameterofeyepinhole = parseFloat(internalDiameterofeyepinholeValue);
    const Clearancebetweenpinholediaandpindia = Diameterofeyepinhole - internalShacklePinDiameter;
    setClearancebetweenpinholediaandpindia(isNaN(Clearancebetweenpinholediaandpindia) ? 0 : Clearancebetweenpinholediaandpindia.toFixed(3));
  };
  useEffect(() => {
    calculateClearancebetweenpinholediaandpindiaValue();
  }, [internalDiameterofeyepinholeValue, internalShacklePinDiameter]);



  const LengthClearanceofShackleUnits = ['mm', 'cm'];
  const LengthClearanceofShackleFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [LengthClearanceofShackle, setLengthClearanceofShackle] = useState(0);
  const [LengthClearanceofShackleselectedunit, setLengthClearanceofShackleSelectedUnit] = useState('mm');

  const handleLengthClearanceofShackleselectedUnitChange = (unit) => {
    setLengthClearanceofShackleSelectedUnit(unit);
    const ClearancebetweenpinholediaandpindiaFactor = LengthClearanceofShackleFactors[unit][LengthClearanceofShackleUnits.indexOf(LengthClearanceofShackleselectedunit)];
    setLengthClearanceofShackle((parseFloat(LengthClearanceofShackle) / ClearancebetweenpinholediaandpindiaFactor).toFixed(3));
  };

  const calculateLengthClearanceofShackleValue = () => {
    const ShackleInsideLengthvalue = parseFloat(internalShackleInsideLength);
    const Plusvalue = ShackleInsideLengthvalue + (internalShacklePinDiameter / 2)
    const minusValue = internalRadiusofMainPlateValue - (- internalSlingDiameter);
    const LengthClearanceofShackle = (Plusvalue - minusValue).toFixed(3);
    setLengthClearanceofShackle(isNaN(LengthClearanceofShackle) ? 0 : LengthClearanceofShackle);
  };
  useEffect(() => {
    calculateLengthClearanceofShackleValue();
  }, [internalShackleInsideLength, internalShacklePinDiameter, internalSlingDiameter, internalRadiusofMainPlateValue]);





  const WidthClearanceofShackleUnits = ['mm', 'cm'];
  const WidthClearanceofShackleFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [WidthClearanceofShackle, setWidthClearanceofShackle] = useState(0);
  const [WidthClearanceofShackleSelectedUnit, setWidthClearanceofShackleSelectedunit] = useState('mm');

  const handleWidthClearanceofShackleUnitChange = (unit) => {
    setWidthClearanceofShackleSelectedunit(unit);
    const WidthClearanceofShackleFactor = WidthClearanceofShackleFactors[unit][WidthClearanceofShackleUnits.indexOf(WidthClearanceofShackleSelectedUnit)];
    setWidthClearanceofShackle((parseFloat(WidthClearanceofShackle) / WidthClearanceofShackleFactor));
  };

  const calculateWidthClearanceofShackleValue = () => {
    const ShackleJawWidthvalue = parseFloat(internalShackleJawWidth);
    const WidthClearanceofShackleMinusValue = ((ShackleJawWidthvalue - internalThicknessofMainPlateValue));
    const WidthClearanceofShackle = (WidthClearanceofShackleMinusValue - (2 * internalThicknessofCheekPlateInputValue));
    setWidthClearanceofShackle(isNaN(WidthClearanceofShackle) ? 0 : WidthClearanceofShackle);
  };
  useEffect(() => {
    calculateWidthClearanceofShackleValue();
  }, [internalShackleJawWidth, internalThicknessofMainPlateValue, internalThicknessofCheekPlateInputValue]);



  // Stress Checks at Pin Hole
  // Bearing Area


  const pi = Math.PI;
  const PI = 22 / 7
  const piby4 = pi / 4

  const BearingAreaUnits = ['mm²', 'cm²'];
  const BearingAreaFactors = {
    'mm²': [1, 0.01],
    'cm²': [100, 1],
  };

  const [BearingArea, setBearingArea] = useState(0);
  const [BearingAreaSelectedUnit, setBearingAreaSelectedUnit] = useState('mm²');

  const handleBearingAreaUnitChange = (unit) => {
    setBearingAreaSelectedUnit(unit);
    const BearingAreaFactor = BearingAreaFactors[unit][BearingAreaUnits.indexOf(BearingAreaSelectedUnit)];
    setBearingArea((parseFloat(BearingArea) / BearingAreaFactor).toFixed(4));
  };

  const calculateBearingAreaValue = () => {
    const Dp = parseFloat(internalShacklePinDiameter);
    const t = parseFloat(internalThicknessofMainPlateValue);
    const tc = parseFloat(internalThicknessofCheekPlateInputValue);
    const BearingArea = piby4 * Dp * (t + 2 * tc)
    setBearingArea(isNaN(BearingArea) ? 0 : BearingArea.toFixed(3));
  };
  useEffect(() => {
    calculateBearingAreaValue();
  }, [internalShacklePinDiameter, internalThicknessofMainPlateValue, internalThicknessofCheekPlateInputValue]);

  const BearingStressActualUnits = ['Mpa', 'Pa'];
  const BearingStressActualFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [BearingStressActual, setBearingStressActual] = useState(0);
  const [BearingStressActualSelectedUnit, setBearingStressActualSelectedUnit] = useState('Mpa');

  const handleBearingStressActualUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = BearingStressActualFactors[unit][BearingStressActualUnits.indexOf(BearingStressActualSelectedUnit)];
    let newValue = parseFloat(BearingStressActual) / AllowableBendingStressInPlaneFactor;
    newValue = BearingStressActualFormatValue(newValue, unit);
    setBearingStressActualSelectedUnit(unit);
    setBearingStressActual(newValue);
  };

  const calculateBearingStressActualValue = () => {
    const LoadonPadeyeValueInMTValue = parseFloat(LoadonPadeyeValueInMT);
    const DLF = parseFloat(DLFValue);
    const ShacklePinDiameter = parseFloat(internalShacklePinDiameter);
    const ThicknessofMainPlateValue = parseFloat(internalThicknessofMainPlateValue);
    const ThicknessofCheekPlateInputValue = parseFloat(internalThicknessofCheekPlateInputValue);
    if (!isNaN(LoadonPadeyeValueInMTValue) && !isNaN(DLF) && !isNaN(ShacklePinDiameter) && !isNaN(ThicknessofMainPlateValue) && !isNaN(ThicknessofCheekPlateInputValue)) {
      const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
      const DLF = parseFloat(DLFValue)
      const PadEyeDesignLoad = LoadonPadeye * DLF;
      const Dp = parseFloat(internalShacklePinDiameter);
      const t = parseFloat(internalThicknessofMainPlateValue);
      const tc = parseFloat(internalThicknessofCheekPlateInputValue);
      const BearingArea = piby4 * Dp * (t + 2 * tc)
      const BearingStressActual = (PadEyeDesignLoad / BearingArea * 9810);
      setBearingStressActual(isNaN(BearingStressActual) ? 0 : BearingStressActual.toFixed(2));
    }
  };

  const BearingStressActualFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateBearingStressActualValue();
  }, [LoadonPadeyeValueInMT, DLFValue, internalShacklePinDiameter, internalThicknessofMainPlateValue, internalThicknessofCheekPlateInputValue]);



  //Shear Stress Check at Pin Hole     

  const RadiusofCheekPlateUnits = ['mm', 'cm'];
  const RadiusofCheekPlateFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [RadiusofCheekPlate, setRadiusofCheekPlate] = useState(0);
  const [RadiusofCheekPlateSelectedUnit, setRadiusofCheekPlateSelectedUnit] = useState('mm');

  const handleRadiusofCheekPlateUnitChange = (unit) => {
    setRadiusofCheekPlateSelectedUnit(unit);
    const RadiusofCheekPlateFactor = RadiusofCheekPlateFactors[unit][RadiusofCheekPlateUnits.indexOf(RadiusofCheekPlateSelectedUnit)];
    setRadiusofCheekPlate((parseFloat(RadiusofCheekPlate) / RadiusofCheekPlateFactor));
  };

  const calculateRadiusofCheekPlateValue = () => {
    const RadiusofCheekPlate = (internalDiameterofCheekPlateInputValue / 2).toFixed(2);
    setRadiusofCheekPlate(isNaN(RadiusofCheekPlate) ? 0 : RadiusofCheekPlate);
  };
  useEffect(() => {
    calculateRadiusofCheekPlateValue();
  }, [internalDiameterofCheekPlateInputValue]);


  // Shear Area of pin hole - See Fig 4



  const ShearAreaofpinholeUnits = ['mm²', 'cm²'];
  const ShearAreaofpinholeFactors = {
    'mm²': [1, 0.01],
    'cm²': [100, 1],
  };

  const [ShearAreaofpinhole, setShearAreaofpinhole] = useState(0);
  const [ShearAreaofpinholeSelectedUnit, setShearAreaofpinholeSelectedUnit] = useState('mm²');

  const handleShearAreaofpinholeUnitChange = (unit) => {
    setShearAreaofpinholeSelectedUnit(unit);
    const ShearAreaofpinholeFactor = ShearAreaofpinholeFactors[unit][ShearAreaofpinholeUnits.indexOf(ShearAreaofpinholeSelectedUnit)];
    setShearAreaofpinhole((parseFloat(ShearAreaofpinhole) / ShearAreaofpinholeFactor));
  };

  const calculateShearAreaofpinholeValue = () => {
    const R = parseFloat(internalRadiusofMainPlateValue);
    const De = parseFloat(internalDiameterofeyepinholeValue);
    const t = parseFloat(internalThicknessofMainPlateValue);
    const RadiusofCheekPlateValue = internalDiameterofCheekPlateInputValue / 2;
    const tc = parseFloat(internalThicknessofCheekPlateInputValue);
    const ShearAreaofpinholePluesValue1 = (R - De / 2) * t;
    const ShearAreaofpinholePluesValue2 = 2 * (RadiusofCheekPlateValue - De / 2) * tc;
    const ShearAreaofpinhole = 2 * (ShearAreaofpinholePluesValue1 + ShearAreaofpinholePluesValue2);
    setShearAreaofpinhole(isNaN(ShearAreaofpinhole) ? 0 : ShearAreaofpinhole);
  };
  useEffect(() => {
    calculateShearAreaofpinholeValue();
  }, [internalRadiusofMainPlateValue, internalDiameterofeyepinholeValue, internalThicknessofCheekPlateInputValue, internalThicknessofMainPlateValue, internalDiameterofCheekPlateInputValue]);


  // Shear Stress at pin hole

  const ShearStressatpinholeUnits = ['Mpa', 'Pa'];
  const ShearStressatpinholeFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [ShearStressatpinhole, setShearStressatpinhole] = useState(0);
  const [ShearStressatpinholeSelectedUnit, setShearStressatpinholeSelectedUnit] = useState('Mpa');

  const handleShearStressatpinholeUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = ShearStressatpinholeFactors[unit][ShearStressatpinholeUnits.indexOf(ShearStressatpinholeSelectedUnit)];
    let newValue = parseFloat(ShearStressatpinhole) / AllowableBendingStressInPlaneFactor;
    newValue = calculateShearStressatpinholeValue(newValue, unit);
    setShearStressatpinholeSelectedUnit(unit);
    setShearStressatpinhole(newValue);
  };

  const calculateShearStressatpinhole = () => {
    const DiameterofCheekPlate = parseFloat(internalDiameterofCheekPlateInputValue);
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const t = parseFloat(internalThicknessofMainPlateValue);
    const R = parseFloat(internalRadiusofMainPlateValue);
    const Rc = parseFloat(internalDiameterofCheekPlateInputValue / 2);
    const De = parseFloat(internalDiameterofeyepinholeValue);
    const tc = parseFloat(internalThicknessofCheekPlateInputValue);
    const DLF = parseFloat(DLFValue)
    if (!isNaN(DiameterofCheekPlate) && !isNaN(LoadonPadeye) && !isNaN(t) && !isNaN(R) && !isNaN(Rc) && !isNaN(De) && !isNaN(tc) && !isNaN(DLF)) {
      const ShearAreaofpinhole = 2 * ((R - De / 2) * t + 2 * (Rc - De / 2) * tc);
      const pd = LoadonPadeye * DLF;
      const ShearStressatpinhole = (pd * 9810 / ShearAreaofpinhole).toFixed(2)
      setShearStressatpinhole(isNaN(ShearStressatpinhole) ? 0 : ShearStressatpinhole);
    }
  };

  const calculateShearStressatpinholeValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateShearStressatpinhole();
  }, [internalDiameterofCheekPlateInputValue, LoadonPadeyeValueInMT, internalThicknessofMainPlateValue, internalRadiusofMainPlateValue, internalDiameterofeyepinholeValue, internalThicknessofCheekPlateInputValue, DLFValue,]);


  // TensileAreaforsectionAA
  const TensileAreaforsectionAAUnits = ['mm²', 'cm²'];
  const TensileAreaforsectionAAFactors = {
    'mm²': [1, 0.01],
    'cm²': [100, 1],
  };

  const [TensileAreaforsectionAA, setTensileAreaforsectionAA] = useState(0);
  const [TensileAreaforsectionAASelectedUnit, setTensileAreaforsectionAASelectedUnit] = useState('mm²');

  const handleTensileAreaforsectionAAUnitChange = (unit) => {
    setTensileAreaforsectionAASelectedUnit(unit);
    const TensileAreaforsectionAAFactor = TensileAreaforsectionAAFactors[unit][TensileAreaforsectionAAUnits.indexOf(TensileAreaforsectionAASelectedUnit)];
    setTensileAreaforsectionAA((parseFloat(TensileAreaforsectionAA) / TensileAreaforsectionAAFactor).toFixed(3));
  };

  const calculateTensileAreaforsectionAAValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const t = parseFloat(internalThicknessofMainPlateValue);
    const R = parseFloat(internalRadiusofMainPlateValue);
    const DiameterofCheekPlate = parseFloat(internalDiameterofCheekPlateInputValue);
    const RadiusofCheekPlate = DiameterofCheekPlate / 2;
    const Rc = parseFloat(DiameterofCheekPlate / 2);
    const De = parseFloat(internalDiameterofeyepinholeValue);
    const tc = parseFloat(internalThicknessofCheekPlateInputValue);
    const DLF = parseFloat(DLFValue)
    const ShearAreaofpinhole = 2 * ((R - De / 2) * t + 2 * (Rc - De / 2) * tc);
    const PadEyeDesignLoad = LoadonPadeye * DLF;
    const TensileAreaforsectionAA = 2 * ((R - De / 2) * t + 2 * (Rc - De / 2) * tc);
    setTensileAreaforsectionAA(isNaN(TensileAreaforsectionAA) ? 0 : TensileAreaforsectionAA.toFixed(3));
  };
  useEffect(() => {
    calculateTensileAreaforsectionAAValue();
  }, [internalRadiusofMainPlateValue, internalDiameterofeyepinholeValue, RadiusofCheekPlate, internalThicknessofCheekPlateInputValue, internalThicknessofMainPlateValue, LoadonPadeyeValueInMT, internalDiameterofCheekPlateInputValue]);

  // const TensileStressatpinholeActualatsectionAA = pd * 9810 / TensileAreaforsectionAA


  const TensileStressatpinholeActualatsectionAAUnits = ['Mpa', 'Pa'];
  const TensileStressatpinholeActualatsectionAAFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [TensileStressatpinholeActualatsectionAA, setTensileStressatpinholeActualatsectionAA] = useState(0);
  const [TensileStressatpinholeActualatsectionAASelectedUnit, setTensileStressatpinholeActualatsectionAASelectedUnit] = useState('Mpa');

  const handleTensileStressatpinholeActualatsectionAAUnitChange = (unit) => {
    const AllowableBearingStressFactor = TensileStressatpinholeActualatsectionAAFactors[unit][TensileStressatpinholeActualatsectionAAUnits.indexOf(TensileStressatpinholeActualatsectionAASelectedUnit)];
    let newValue = parseFloat(TensileStressatpinholeActualatsectionAA) / AllowableBearingStressFactor;
    newValue = TensileStressatpinholeActualatsectionAAFormatValue(newValue, unit);
    setTensileStressatpinholeActualatsectionAASelectedUnit(unit);
    setTensileStressatpinholeActualatsectionAA(newValue);
  };

  const calculateTensileStressatpinholeActualatsectionAAValue = () => {
    const DiameterofCheekPlate = parseFloat(internalDiameterofCheekPlateInputValue);
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const t = parseFloat(internalThicknessofMainPlateValue);
    const R = parseFloat(internalRadiusofMainPlateValue);
    const Rc = DiameterofCheekPlate / 2;
    const De = parseFloat(internalDiameterofeyepinholeValue);
    const tc = parseFloat(internalThicknessofCheekPlateInputValue);
    const DLF = parseFloat(DLFValue)
    const pd = LoadonPadeye * DLF;
    const TensileAreaforsectionAA = 2 * ((R - De / 2) * t + 2 * (Rc - De / 2) * tc);
    const TensileStressatpinholeActualatsectionAA = pd * 9810 / TensileAreaforsectionAA;
    setTensileStressatpinholeActualatsectionAA(isNaN(TensileStressatpinholeActualatsectionAA) ? 0 : TensileStressatpinholeActualatsectionAA.toFixed(3));
  };
  const TensileStressatpinholeActualatsectionAAFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateTensileStressatpinholeActualatsectionAAValue();
  }, [internalDiameterofCheekPlateInputValue, LoadonPadeyeValueInMT, internalThicknessofMainPlateValue, internalRadiusofMainPlateValue, internalDiameterofeyepinholeValue, internalThicknessofCheekPlateInputValue, DLFValue,]);



  // TensileAreaforsectionBB
  const TensileAreaforsectionBBUnits = ['mm²', 'cm²'];
  const TensileAreaforsectionBBFactors = {
    'mm²': [1, 0.01],
    'cm²': [100, 1],
  };

  const [TensileAreaforsectionBB, setTensileAreaforsectionBB] = useState(0);
  const [TensileAreaforsectionBBSelectedUnit, setTensileAreaforsectionBBSelectedUnit] = useState('mm²');

  const handleTensileAreaforsectionBBUnitChange = (unit) => {
    setTensileAreaforsectionBBSelectedUnit(unit);
    const TensileAreaforsectionAAFactor = TensileAreaforsectionBBFactors[unit][TensileAreaforsectionBBUnits.indexOf(TensileAreaforsectionBBSelectedUnit)];
    setTensileAreaforsectionBB((parseFloat(TensileAreaforsectionBB) / TensileAreaforsectionAAFactor).toFixed(3));
  };

  const calculateTensileAreaforsectionBBValue = () => {
    const DiameterofCheekPlate = parseFloat(internalDiameterofCheekPlateInputValue);
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const t = parseFloat(internalThicknessofMainPlateValue);
    const R = parseFloat(internalRadiusofMainPlateValue);
    const RadiusofCheekPlate = DiameterofCheekPlate / 2;
    const Rc = parseFloat(RadiusofCheekPlate);
    const De = parseFloat(internalDiameterofeyepinholeValue);
    const tc = parseFloat(internalThicknessofCheekPlateInputValue);
    const DLF = parseFloat(DLFValue)
    const PadEyeDesignLoad = LoadonPadeye * DLF;
    const TensileAreaforsectionBB = ((2 * R) + (pi * Rc / 2)) * t
    setTensileAreaforsectionBB(isNaN(TensileAreaforsectionBB) ? 0 : TensileAreaforsectionBB.toFixed(3));
  };
  useEffect(() => {
    calculateTensileAreaforsectionBBValue();
  }, [internalRadiusofMainPlateValue, RadiusofCheekPlate, internalThicknessofMainPlateValue, LoadonPadeyeValueInMT, internalDiameterofeyepinholeValue, internalDiameterofCheekPlateInputValue, internalThicknessofCheekPlateInputValue]);

  // TensileStressatpinholeActualatsectionBB

  const TensileStressatpinholeActualatsectionBBUnits = ['Mpa', 'Pa'];
  const TensileStressatpinholeActualatsectionBBFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [TensileStressatpinholeActualatsectionBB, setTensileStressatpinholeActualatsectionBB] = useState(0);
  const [TensileStressatpinholeActualatsectionBBSelectedUnit, setTensileStressatpinholeActualatsectionBBSelectedUnit] = useState('Mpa');

  const handleTensileStressatpinholeActualatsectionBBUnitChange = (unit) => {
    const AllowableBearingStressFactor = TensileStressatpinholeActualatsectionBBFactors[unit][TensileStressatpinholeActualatsectionBBUnits.indexOf(TensileStressatpinholeActualatsectionBBSelectedUnit)];
    let newValue = parseFloat(TensileStressatpinholeActualatsectionBB) / AllowableBearingStressFactor;
    newValue = TensileStressatpinholeActualatsectionBBFormatValue(newValue, unit);
    setTensileStressatpinholeActualatsectionBBSelectedUnit(unit);
    setTensileStressatpinholeActualatsectionBB(newValue);
  };

  const calculateTensileStressatpinholeActualatsectionBBValue = () => {
    const DiameterofCheekPlate = parseFloat(internalDiameterofCheekPlateInputValue);
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const t = parseFloat(internalThicknessofMainPlateValue);
    const R = parseFloat(internalRadiusofMainPlateValue);
    const Rc = DiameterofCheekPlate / 2;
    const DLF = parseFloat(DLFValue)
    const PadEyeDesignLoad = LoadonPadeye * DLF;
    const TensileAreaforsectionBB = ((2 * R) + (pi * Rc / 2)) * t
    const TensileStressatpinholeActualatsectionBB = PadEyeDesignLoad * 9810 / TensileAreaforsectionBB
    setTensileStressatpinholeActualatsectionBB(isNaN(TensileStressatpinholeActualatsectionBB) ? 0 : TensileStressatpinholeActualatsectionBB.toFixed(3));
  };
  const TensileStressatpinholeActualatsectionBBFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateTensileStressatpinholeActualatsectionBBValue();
  }, [internalDiameterofCheekPlateInputValue, LoadonPadeyeValueInMT, internalThicknessofMainPlateValue, internalRadiusofMainPlateValue, DLFValue,]);

  // Hertz/Contact Stress Check at Pin Hole

  // DesignLoadperunitLength
  const DesignLoadperunitLengthUnits = ['MT/mm'];
  const DesignLoadperunitLengthFactors = {
    'MT/mm': [1],
  };

  const [DesignLoadperunitLength, setDesignLoadperunitLength] = useState(0);
  const [DesignLoadperunitLengthSelectedUnit, setDesignLoadperunitLengthSelectedUnit] = useState('MT/mm');

  const handleDesignLoadperunitLengthUnitChange = (unit) => {
    setDesignLoadperunitLengthSelectedUnit(unit);
    const DesignLoadperunitLengthFactor = DesignLoadperunitLengthFactors[unit][DesignLoadperunitLengthUnits.indexOf(DesignLoadperunitLengthSelectedUnit)];
    setDesignLoadperunitLength((parseFloat(DesignLoadperunitLength) / DesignLoadperunitLengthFactor).toFixed(3));
  };

  const calculateDesignLoadperunitLengthValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const DLF = parseFloat(DLFValue)
    const PadEyeDesignLoad = (LoadonPadeye * DLF).toFixed(2);
    const pd = parseFloat(PadEyeDesignLoad);
    const t = parseFloat(internalThicknessofMainPlateValue);
    const tc = parseFloat(internalThicknessofCheekPlateInputValue);
    const DesignLoadperunitLength = pd / (t + 2 * tc);
    setDesignLoadperunitLength(isNaN(DesignLoadperunitLength) ? 0 : DesignLoadperunitLength.toFixed(3));
  };
  useEffect(() => {
    calculateDesignLoadperunitLengthValue();
  }, [PadEyeDesignLoad, internalThicknessofMainPlateValue, internalThicknessofCheekPlateInputValue]);


  const HertzStressatPinHoleUnits = ['Mpa', 'Pa'];
  const HertzStressatPinHoleFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [HertzStressatPinHole, setHertzStressatPinHole] = useState(0);
  const [HertzStressatPinHoleSelectedUnit, setHertzStressatPinHoleSelectedUnit] = useState('Mpa');

  const handleHertzStressatPinHoleUnitChange = (unit) => {
    const AllowableBearingStressFactor = HertzStressatPinHoleFactors[unit][HertzStressatPinHoleUnits.indexOf(HertzStressatPinHoleSelectedUnit)];
    let newValue = parseFloat(HertzStressatPinHole) / AllowableBearingStressFactor;
    newValue = HertzStressatPinHoleFormatValue(newValue, unit);
    setHertzStressatPinHoleSelectedUnit(unit);
    setHertzStressatPinHole(newValue);
  };

  const calculateHertzStressatPinHoleValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const DLF = parseFloat(DLFValue)
    const pd = LoadonPadeye * DLF;
    const t = parseFloat(internalThicknessofMainPlateValue);
    const tc = parseFloat(internalThicknessofCheekPlateInputValue);
    const D38 = pd / (t + 2 * tc);
    const D39 = parseFloat(ModulusofElasticityValueInMpa);
    const D40 = parseFloat(PoissonsRatio);
    const D41 = parseFloat(internalDiameterofeyepinholeValue)
    const D42 = parseFloat(internalShacklePinDiameter)
    const HertzStressatPinHole = (Math.sqrt(D38 * 9810 * D39 * (D41 - D42) / (PI * (1 - Math.pow(D40, 2)) * D41 * D42))).toFixed(3);
    // SQRT(D38*9810*D39*(D41-D42)/(PI()*(1-D40^2)*D41*D42))
    setHertzStressatPinHole(isNaN(HertzStressatPinHole) ? 0 : HertzStressatPinHole);
  };
  const HertzStressatPinHoleFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(5);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateHertzStressatPinHoleValue();
  }, [LoadonPadeyeValueInMT, DLFValue, internalThicknessofMainPlateValue, internalThicknessofCheekPlateInputValue, ModulusofElasticityValueInMpa, PoissonsRatio, internalDiameterofeyepinholeValue, internalShacklePinDiameter]);

  // Stress Checks at Base Plate
  // Tensile Stress Check


  const TensileAreaUnits = ['mm²', 'cm²'];
  const TensileAreaFactors = {
    'mm²': [1, 0.01],
    'cm²': [100, 1],
  };

  const [TensileArea, setTensileArea] = useState(0);
  const [TensileAreaSelectedUnit, setTensileAreaSelected] = useState('mm²');

  const handleTensileAreaUnitChange = (unit) => {
    setTensileAreaSelected(unit);
    const TensileAreaFactor = TensileAreaFactors[unit][TensileAreaUnits.indexOf(TensileAreaSelectedUnit)];
    setTensileArea((parseFloat(TensileArea) / TensileAreaFactor));
  };

  const calculateTensileAreaValue = () => {
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const TensileArea = L * T;
    setTensileArea(isNaN(TensileArea) ? 0 : TensileArea);
  };
  useEffect(() => {
    calculateTensileAreaValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue]);


  // TensileStressatBaseActual


  const TensileStressatBaseActualUnits = ['Mpa', 'Pa'];
  const TensileStressatBaseActualFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [TensileStressatBaseActual, setTensileStressatBaseActual] = useState(0);
  const [TensileStressatBaseActualSelectedUnit, setTensileStressatBaseActualSelectedUnit] = useState('Mpa');

  const handleTensileStressatBaseActualUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = TensileStressatBaseActualFactors[unit][TensileStressatBaseActualUnits.indexOf(TensileStressatBaseActualSelectedUnit)];
    let newValue = parseFloat(TensileStressatBaseActual) / AllowableBendingStressInPlaneFactor;
    newValue = TensileStressatBaseActualValue(newValue, unit);
    setTensileStressatBaseActualSelectedUnit(unit);
    setTensileStressatBaseActual(newValue);
  };

  const calculateTensileStressatBaseActualValue = () => {
    const LoadonPadeye = LoadonPadeyeValueInMT
    const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
    const LoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
    const DLF = parseFloat(DLFValue);
    const ValueofDesignLoadinVerticalDirectionZdirection = LoadinVerticalDirectionZdirectionValue * DLF;
    const pt = parseFloat(ValueofDesignLoadinVerticalDirectionZdirection)
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const TensileAreavalue = L * T;
    const atb = parseFloat(TensileAreavalue)
    const TensileStressatBaseActual = (pt * 9810 / atb).toFixed(2);
    setTensileStressatBaseActual(isNaN(TensileStressatBaseActual) ? 0 : TensileStressatBaseActual);
  };

  const TensileStressatBaseActualValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateTensileStressatBaseActualValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue, LoadonPadeyeValueInMT, DLFValue]);


  // Moment lever for Horizontal/Lateral Force

  const MomentleverforHorizontalLateralForceUnits = ['mm', 'cm'];
  const MomentleverforHorizontalLateralForceFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [MomentleverforHorizontalLateralForce, setMomentleverforHorizontalLateralForce] = useState(0);
  const [MomentleverforHorizontalLateralForceSelectedUnit, setMomentleverforHorizontalLateralForceSelectedUnit] = useState('mm');

  const handleMomentleverforHorizontalLateralForceUnitChange = (unit) => {
    setMomentleverforHorizontalLateralForceSelectedUnit(unit);
    const MomentleverforHorizontalLateralForceFactor = MomentleverforHorizontalLateralForceFactors[unit][MomentleverforHorizontalLateralForceUnits.indexOf(MomentleverforHorizontalLateralForceSelectedUnit)];
    setMomentleverforHorizontalLateralForce((parseFloat(MomentleverforHorizontalLateralForce) / MomentleverforHorizontalLateralForceFactor).toFixed(2));
  };

  const calculateMomentleverforHorizontalLateralForceValue = () => {
    const H = parseFloat(internalinputTotalHeightofPadeyeValue)
    const R = parseFloat(internalRadiusofMainPlateValue)
    const MomentleverforHorizontalLateralForce = (H - R).toFixed(2);
    setMomentleverforHorizontalLateralForce(isNaN(MomentleverforHorizontalLateralForce) ? 0 : MomentleverforHorizontalLateralForce);
  };
  useEffect(() => {
    calculateMomentleverforHorizontalLateralForceValue();
  }, [internalinputTotalHeightofPadeyeValue, internalRadiusofMainPlateValue]);



  const MomentleverforVerticalForceUnits = ['mm', 'cm'];
  const MomentleverforVerticalForceFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [MomentleverforVerticalForce, setMomentleverforVerticalForce] = useState(0);
  const [MomentleverforVerticalForceSelectedUnit, setMomentleverforVerticalForceSelectedUnit] = useState('mm');

  const handleMomentleverforVerticalForceUnitChange = (unit) => {
    setMomentleverforVerticalForceSelectedUnit(unit);
    const MomentleverforVerticalForceFactor = MomentleverforVerticalForceFactors[unit][MomentleverforVerticalForceUnits.indexOf(MomentleverforVerticalForceSelectedUnit)];
    setMomentleverforVerticalForce((parseFloat(MomentleverforVerticalForce) / MomentleverforVerticalForceFactor).toFixed(2));
  };

  const calculateMomentleverforVerticalForceValue = () => {
    const L = internalLengthofBasePlateInputValue
    const R = internalRadiusofMainPlateValue
    const MomentleverforVerticalForce = (R - (L / 2)).toFixed(2);
    setMomentleverforVerticalForce(isNaN(MomentleverforVerticalForce) ? 0 : MomentleverforVerticalForce);
  };
  useEffect(() => {
    calculateMomentleverforVerticalForceValue();
  }, [internalLengthofBasePlateInputValue, internalRadiusofMainPlateValue]);



  // TotalDesignMomentaboutXaxisInplane
  const TotalDesignMomentaboutXaxisInplaneUnits = ['MT-Mm'];
  const TotalDesignMomentaboutXaxisInplaneFactors = {
    'MT-mm': [1]
  };

  const [TotalDesignMomentaboutXaxisInplane, setTotalDesignMomentaboutXaxisInplane] = useState(0);
  const [TotalDesignMomentaboutXaxisInplaneSelectedUnit, setTotalDesignMomentaboutXaxisInplaneSelectedUnit] = useState('MT-mm');

  const handleTotalDesignMomentaboutXaxisInplaneUnitChange = (unit) => {
    setTotalDesignMomentaboutXaxisInplaneSelectedUnit(unit);
    const MomentleverforVerticalForceFactor = TotalDesignMomentaboutXaxisInplaneFactors[unit][TotalDesignMomentaboutXaxisInplaneUnits.indexOf(TotalDesignMomentaboutXaxisInplaneSelectedUnit)];
    setTotalDesignMomentaboutXaxisInplane((parseFloat(TotalDesignMomentaboutXaxisInplane) / MomentleverforVerticalForceFactor));
  };

  const calculateTotalDesignMomentaboutXaxisInplaneValue = () => {
    const DLF = parseFloat(DLFValue);
    const H = parseFloat(internalinputTotalHeightofPadeyeValue)
    const R = parseFloat(internalRadiusofMainPlateValue)
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const pi = Math.PI;

    const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
    const LoadinHorizontalDirectionYdirection = (LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue);
    const D6 = (LoadinHorizontalDirectionYdirection * DLFValue)

    const LoadonPadeye = LoadonPadeyeValueInMT
    const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * (Math.PI / 180));
    const LoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
    const D7 = (LoadinVerticalDirectionZdirectionValue * DLF).toFixed(2)

    const D9 = (H - R).toFixed(2)

    const D10 = (R - (L / 2)).toFixed(2)
    // const TotalDesignMomentaboutXaxisInplane = D6
    const TotalDesignMomentaboutXaxisInplane = (D6 * D9 + D7 * D10).toFixed(2)
    setTotalDesignMomentaboutXaxisInplane(isNaN(TotalDesignMomentaboutXaxisInplane) ? 0 : TotalDesignMomentaboutXaxisInplane);
  };
  useEffect(() => {
    calculateTotalDesignMomentaboutXaxisInplaneValue();
  }, [DLFValue, internalinputTotalHeightofPadeyeValue, internalRadiusofMainPlateValue, internalLengthofBasePlateInputValue, LoadonPadeyeValueInMT, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue]);


  // Section Modulus of Base about X-axis


  const SectionModulusofBaseaboutXaxisUnits = ['mm³', 'cm³'];
  const SectionModulusofBaseaboutXaxisFactors = {
    'mm³': [1, 0.001],
    'cm³': [1000, 1],
  };

  const [SectionModulusofBaseaboutXaxis, setSectionModulusofBaseaboutXaxis] = useState(0);
  const [SectionModulusofBaseaboutXaxisSelectedUnit, setSectionModulusofBaseaboutXaxisSelectedUnit] = useState('mm³');

  const handleSectionModulusofBaseaboutXaxisUnitChange = (unit) => {
    setSectionModulusofBaseaboutXaxisSelectedUnit(unit);
    const SectionModulusofBaseaboutXaxisFactor = SectionModulusofBaseaboutXaxisFactors[unit][SectionModulusofBaseaboutXaxisUnits.indexOf(SectionModulusofBaseaboutXaxisSelectedUnit)];
    setSectionModulusofBaseaboutXaxis((parseFloat(SectionModulusofBaseaboutXaxis) / SectionModulusofBaseaboutXaxisFactor).toFixed(2));
  };

  const calculateSectionModulusofBaseaboutXaxisValue = () => {
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const SectionModulusofBaseaboutXaxis = T * Math.pow(L, 2) / 6;
    setSectionModulusofBaseaboutXaxis(isNaN(SectionModulusofBaseaboutXaxis) ? 0 : SectionModulusofBaseaboutXaxis.toFixed(2));
  };
  useEffect(() => {
    calculateSectionModulusofBaseaboutXaxisValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue]);



  const BendingStressActualaboutXaxisInplaneUnits = ['Mpa', 'Pa'];
  const BendingStressActualaboutXaxisInplaneFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [BendingStressActualaboutXaxisInplane, setBendingStressActualaboutXaxisInplane] = useState(0);
  const [BendingStressActualaboutXaxisInplaneSelectedUnit, setBendingStressActualaboutXaxisInplaneSelectedUnit] = useState('Mpa');

  const handleBendingStressActualaboutXaxisInplaneUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = BendingStressActualaboutXaxisInplaneFactors[unit][BendingStressActualaboutXaxisInplaneUnits.indexOf(BendingStressActualaboutXaxisInplaneSelectedUnit)];
    let newValue = parseFloat(BendingStressActualaboutXaxisInplane) / AllowableBendingStressInPlaneFactor;
    newValue = BendingStressActualaboutXaxisInplaneValue(newValue, unit);
    setBendingStressActualaboutXaxisInplaneSelectedUnit(unit);
    setBendingStressActualaboutXaxisInplane(newValue);
  };

  const calculateBendingStressActualaboutXaxisInplaneValue = () => {
    const H = internalinputTotalHeightofPadeyeValue
    const R = internalRadiusofMainPlateValue
    const pi = Math.PI;
    const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
    const LoadinHorizontalDirectionYdirection = (LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue).toFixed(1);
    const pyd = (LoadinHorizontalDirectionYdirection * DLFValue).toFixed(2)
    const LoadonPadeye = LoadonPadeyeValueInMT
    const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
    const LoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
    const DLF = DLFValue;
    const h = H - R
    const pzd = LoadinVerticalDirectionZdirectionValue * DLF;
    const L = internalLengthofBasePlateInputValue
    const e = R - (L / 2);
    const Mdxx = pyd * h + pzd * e;
    // const L = internalLengthofBasePlateInputValue
    const T = internalThicknessofMainPlateValue
    const Zbxx = T * Math.pow(L, 2) / 6;
    const BendingStressActualaboutXaxisInplane = Mdxx * 9810 / Zbxx;
    setBendingStressActualaboutXaxisInplane(isNaN(BendingStressActualaboutXaxisInplane) ? 0 : BendingStressActualaboutXaxisInplane.toFixed(2));
  };

  const BendingStressActualaboutXaxisInplaneValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(2);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateBendingStressActualaboutXaxisInplaneValue();
  }, [internalinputTotalHeightofPadeyeValue, internalRadiusofMainPlateValue, DLFValue, internalLengthofBasePlateInputValue, internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue]);

  // Total Design Moment about Y-axis (Out-of Plane)

  const TotalDesignMomentaboutYaxisOutofPlaneUnits = ['MT-Mm'];
  const TotalDesignMomentaboutYaxisOutofPlaneFactors = {
    'MT-mm': [1]
  };

  const [TotalDesignMomentaboutYaxisOutofPlane, setTotalDesignMomentaboutYaxisOutofPlane] = useState(0);
  const [TotalDesignMomentaboutYaxisOutofPlaneSelectedUnit, setTotalDesignMomentaboutYaxisOutofPlaneSelectedUnit] = useState('MT-Mm');

  const handleTotalDesignMomentaboutYaxisOutofPlaneUnitChange = (unit) => {
    setTotalDesignMomentaboutYaxisOutofPlaneSelectedUnit(unit);
    const TotalDesignMomentaboutYaxisOutofPlaneFactor = TotalDesignMomentaboutYaxisOutofPlaneFactors[unit][TotalDesignMomentaboutYaxisOutofPlaneUnits.indexOf(TotalDesignMomentaboutYaxisOutofPlaneSelectedUnit)];
    setTotalDesignMomentaboutYaxisOutofPlane((parseFloat(TotalDesignMomentaboutYaxisOutofPlane) / TotalDesignMomentaboutYaxisOutofPlaneFactor));
  };

  const calculateTotalDesignMomentaboutYaxisOutofPlaneValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT)
    const DLF = parseFloat(DLFValue);
    const pi = Math.PI;
    const H = parseFloat(internalinputTotalHeightofPadeyeValue)
    const R = parseFloat(internalRadiusofMainPlateValue)
    const Hvalue = H - R;
    const PadEyeDesignLoadvalue = LoadonPadeye * DLF;
    const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
    const OutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
    const pxd = (OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)
    const h = parseFloat(Hvalue)
    const TotalDesignMomentaboutYaxisOutofPlane = (pxd * h).toFixed(3);
    setTotalDesignMomentaboutYaxisOutofPlane(isNaN(TotalDesignMomentaboutYaxisOutofPlane) ? 0 : TotalDesignMomentaboutYaxisOutofPlane);
  };
  useEffect(() => {
    calculateTotalDesignMomentaboutYaxisOutofPlaneValue();
  }, [MomentleverforHorizontalLateralForce, LoadonPadeyeValueInMT, DLFValue, internalinputTotalHeightofPadeyeValue, internalRadiusofMainPlateValue]);

  // Section Modulus of Base about Y-axis 

  const SectionModulusofBaseaboutYaxisUnits = ['mm³', 'cm³'];
  const SectionModulusofBaseaboutYaxisFactors = {
    'mm³': [1, 0.001],
    'cm³': [1000, 1],
  };

  const [SectionModulusofBaseaboutYaxis, setSectionModulusofBaseaboutYaxis] = useState(0);
  const [SectionModulusofBaseaboutYaxisSelectedUnit, setSectionModulusofBaseaboutYaxisSelectedUnit] = useState('mm³');

  const handleSectionModulusofBaseaboutYaxisUnitChange = (unit) => {
    setSectionModulusofBaseaboutYaxisSelectedUnit(unit);
    const SectionModulusofBaseaboutYaxisFactor = SectionModulusofBaseaboutYaxisFactors[unit][SectionModulusofBaseaboutYaxisUnits.indexOf(SectionModulusofBaseaboutYaxisSelectedUnit)];
    setSectionModulusofBaseaboutYaxis((parseFloat(SectionModulusofBaseaboutYaxis) / SectionModulusofBaseaboutYaxisFactor));
  };

  const calculateSectionModulusofBaseaboutYaxisValue = () => {
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const SectionModulusofBaseaboutYaxis = L * Math.pow(T, 2) / 6;
    setSectionModulusofBaseaboutYaxis(isNaN(SectionModulusofBaseaboutYaxis) ? 0 : SectionModulusofBaseaboutYaxis);
  };
  useEffect(() => {
    calculateSectionModulusofBaseaboutYaxisValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue]);


  // Bending Stress (Actual) - about Y-axis (Out-of-Plane)
  const BendingStressActualaboutYaxisOutofPlaneUnit = ['Mpa', 'Pa'];
  const BendingStressActualaboutYaxisOutofPlaneFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [BendingStressActualaboutYaxisOutofPlane, setBendingStressActualaboutYaxisOutofPlane] = useState(0);
  const [BendingStressActualaboutYaxisOutofPlaneSelectedUnit, setBendingStressActualaboutYaxisOutofPlaneSelectedUnit] = useState('Mpa');

  const handleBendingStressActualaboutYaxisOutofPlaneUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = BendingStressActualaboutYaxisOutofPlaneFactors[unit][BendingStressActualaboutYaxisOutofPlaneUnit.indexOf(BendingStressActualaboutYaxisOutofPlaneSelectedUnit)];
    let newValue = parseFloat(BendingStressActualaboutYaxisOutofPlane) / AllowableBendingStressInPlaneFactor;
    newValue = BendingStressActualaboutYaxisOutofPlaneValue(newValue, unit);
    setBendingStressActualaboutYaxisOutofPlaneSelectedUnit(unit);
    setBendingStressActualaboutYaxisOutofPlane(newValue);
  };

  const calculateBendingStressActualaboutYaxisOutofPlaneValue = () => {
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT)
    const DLF = parseFloat(DLFValue);
    if (!isNaN(L) && !isNaN(T) && !isNaN(LoadonPadeye) && !isNaN(DLF)) {
      const h = parseFloat(MomentleverforHorizontalLateralForce)
      const pi = Math.PI;
      const PadEyeDesignLoadvalue = LoadonPadeye * DLF;
      const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
      const OutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
      const pxd = (OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)
      const Mdyy = pxd * h
      const Zbyy = L * Math.pow(T, 2) / 6
      const BendingStressActualaboutYaxisOutofPlane = Mdyy * 9810 / Zbyy;
      setBendingStressActualaboutYaxisOutofPlane(isNaN(BendingStressActualaboutYaxisOutofPlane) ? 0 : BendingStressActualaboutYaxisOutofPlane.toFixed(2));
    }
  };

  const BendingStressActualaboutYaxisOutofPlaneValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateBendingStressActualaboutYaxisOutofPlaneValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue, LoadonPadeyeValueInMT, DLFValue,]);

  // Shear Stress Check at Base

  // Shear Area at Base (Horizontal)
  // ShearAreaatBaseHorizontal
  const ShearAreaatBaseHorizontalUnit = ['mm²', 'cm²'];
  const ShearAreaatBaseHorizontalFactors = {
    'mm²': [1, 0.01],
    'cm²': [100, 1],
  };

  const [ShearAreaatBaseHorizontal, setShearAreaatBaseHorizontal] = useState(0);
  const [ShearAreaatBaseHorizontalSelectedUnit, setShearAreaatBaseHorizontalSelectedUnit] = useState('mm²');

  const handleShearAreaatBaseHorizontalUnitChange = (unit) => {
    setShearAreaatBaseHorizontalSelectedUnit(unit);
    const ShearAreaatBaseHorizontalFactor = ShearAreaatBaseHorizontalFactors[unit][ShearAreaatBaseHorizontalUnit.indexOf(ShearAreaatBaseHorizontalSelectedUnit)];
    setShearAreaatBaseHorizontal((parseFloat(ShearAreaatBaseHorizontal) / ShearAreaatBaseHorizontalFactor));
  };

  const calculateShearAreaatBaseHorizontalValue = () => {
    const L = internalLengthofBasePlateInputValue
    const T = internalThicknessofMainPlateValue
    const ShearAreaatBaseHorizontal = L * T;
    setShearAreaatBaseHorizontal(isNaN(ShearAreaatBaseHorizontal) ? 0 : ShearAreaatBaseHorizontal);
  };
  useEffect(() => {
    calculateShearAreaatBaseHorizontalValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue]);


  // Shear Stress at Base (Actual) - Horizontal (Y direction)
  // ShearStressatBaseActualHorizontalYdirection
  const ShearStressatBaseActualHorizontalYdirectionUnit = ['Mpa', 'Pa'];
  const ShearStressatBaseActualHorizontalYdirectionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [ShearStressatBaseActualHorizontalYdirection, setShearStressatBaseActualHorizontalYdirection] = useState(0);
  const [ShearStressatBaseActualHorizontalYdirectionSelectedUnit, setShearStressatBaseActualHorizontalYdirectionSelectedUnit] = useState('Mpa');

  const handleShearStressatBaseActualHorizontalYdirectionUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = ShearStressatBaseActualHorizontalYdirectionFactors[unit][ShearStressatBaseActualHorizontalYdirectionUnit.indexOf(ShearStressatBaseActualHorizontalYdirectionSelectedUnit)];
    let newValue = parseFloat(ShearStressatBaseActualHorizontalYdirection) / AllowableBendingStressInPlaneFactor;
    newValue = ShearStressatBaseActualHorizontalYdirectionValue(newValue, unit);
    setShearStressatBaseActualHorizontalYdirectionSelectedUnit(unit);
    setShearStressatBaseActualHorizontalYdirection(newValue);
  };

  const calculateShearStressatBaseActualHorizontalYdirectionValue = () => {
    const pi = Math.PI;
    const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
    const LoadinHorizontalDirectionYdirection = (LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue).toFixed(1);
    const pyd = (LoadinHorizontalDirectionYdirection * DLFValue).toFixed(2)
    const L = internalLengthofBasePlateInputValue
    const T = internalThicknessofMainPlateValue
    const asb = L * T
    const ShearStressatBaseActualHorizontalYdirection = (pyd * 9810 / asb).toFixed(3);
    setShearStressatBaseActualHorizontalYdirection(isNaN(ShearStressatBaseActualHorizontalYdirection) ? 0 : ShearStressatBaseActualHorizontalYdirection, OutofPlaneAngleValue);
  };

  const ShearStressatBaseActualHorizontalYdirectionValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3)
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateShearStressatBaseActualHorizontalYdirectionValue();
  }, [internalLengthofBasePlateInputValue, DLFValue, LoadonPadeyeValueInMT, internalThicknessofMainPlateValue]);

  // Shear Stress at Base (Actual) - Lateral (X direction)
  // ShearStressatBaseActualLateralXdirection
  const ShearStressatBaseActualLateralXdirectionUnits = ['Mpa', 'Pa'];
  const ShearStressatBaseActualLateralXdirectionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [ShearStressatBaseActualLateralXdirection, setShearStressatBaseActualLateralXdirection] = useState(0);
  const [ShearStressatBaseActualLateralXdirectionSelectedUnit, setShearStressatBaseActualLateralXdirectionSelectedUnit] = useState('Mpa');

  const handleShearStressatBaseActualLateralXdirectionUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = ShearStressatBaseActualLateralXdirectionFactors[unit][ShearStressatBaseActualLateralXdirectionUnits.indexOf(ShearStressatBaseActualLateralXdirectionSelectedUnit)];
    let newValue = parseFloat(ShearStressatBaseActualLateralXdirection) / AllowableBendingStressInPlaneFactor;
    newValue = ShearStressatBaseActualLateralXdirectionValue(newValue, unit);
    setShearStressatBaseActualLateralXdirectionSelectedUnit(unit);
    setShearStressatBaseActualLateralXdirection(newValue);
  };

  const calculateShearStressatBaseActualLateralXdirectionValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT)
    const DLF = parseFloat(DLFValue);
    const pi = Math.PI;
    const PadEyeDesignLoadvalue = LoadonPadeyeValueInMT * DLF;
    const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
    const OutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
    const pxd = (OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue);
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const asb = L * T
    const ShearStressatBaseActualLateralXdirection = (pxd * 9810 / asb).toFixed(2);
    setShearStressatBaseActualLateralXdirection(isNaN(ShearStressatBaseActualLateralXdirection) ? 0 : ShearStressatBaseActualLateralXdirection);
  };

  const ShearStressatBaseActualLateralXdirectionValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateShearStressatBaseActualLateralXdirectionValue();
  }, [
    LoadonPadeyeValueInMT, DLFValue, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue, internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue,]);

  // DesignLoadOutofPlaneLateralXdirection

  const VonMisesStressatBaseActualUnits = ['Mpa', 'Pa'];
  const VonMisesStressatBaseActualFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [VonMisesStressatBaseActual, setVonMisesStressatBaseActual] = useState(0);
  const [VonMisesStressatBaseActualSelectedUnit, setVonMisesStressatBaseActualSelectedUnit] = useState('Mpa');

  const handleVonMisesStressatBaseActualUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = VonMisesStressatBaseActualFactors[unit][VonMisesStressatBaseActualUnits.indexOf(VonMisesStressatBaseActualSelectedUnit)];
    let newValue = parseFloat(VonMisesStressatBaseActual) / AllowableBendingStressInPlaneFactor;
    newValue = VonMisesStressatBaseActualValue(newValue, unit);
    setVonMisesStressatBaseActualSelectedUnit(unit);
    setVonMisesStressatBaseActual(newValue);
  };

  const calculateVonMisesStressatBaseActualValue = () => {
    const LoadonPadeye = LoadonPadeyeValueInMT
    const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
    const LoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
    const DLF = parseFloat(DLFValue);
    const ValueofDesignLoadinVerticalDirectionZdirection = LoadinVerticalDirectionZdirectionValue * DLF;
    const pt = parseFloat(ValueofDesignLoadinVerticalDirectionZdirection)
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const TensileAreavalue = L * T;
    const atb = parseFloat(TensileAreavalue)
    const TensileStressatBaseActualValue = pt * 9810 / atb;
    const sigmaTb = parseFloat(TensileStressatBaseActualValue)

    const H = parseFloat(internalinputTotalHeightofPadeyeValue)
    const R = parseFloat(internalRadiusofMainPlateValue)
    const pi = Math.PI;
    const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
    const LoadinHorizontalDirectionYdirection = (LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue).toFixed(1);
    const pyd = (LoadinHorizontalDirectionYdirection * DLFValue).toFixed(2)
    const h = H - R
    const pzd = LoadinVerticalDirectionZdirectionValue * DLF;
    const MomentleverforVerticalForce = R - (L / 2);
    const e = parseFloat(MomentleverforVerticalForce)
    const Mdxx = pyd * h + pzd * e;
    const Zbxx = T * Math.pow(L, 2) / 6;
    const BendingStressActualaboutXaxisInplane = Mdxx * 9810 / Zbxx;
    const sigmaBbx = parseFloat(BendingStressActualaboutXaxisInplane)

    const PadEyeDesignLoadvalue = LoadonPadeye * DLF;
    const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
    const OutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
    const pxd = (OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)
    const Mdyy = pxd * h
    const Zbyy = L * Math.pow(T, 2) / 6
    const BendingStressActualaboutYaxisOutofPlane = Mdyy * 9810 / Zbyy;
    const sigmaBby = parseFloat(BendingStressActualaboutYaxisOutofPlane)

    const asb = L * T
    const ShearStressatBaseActualHorizontalYdirection = (pyd * 9810 / asb).toFixed(3);
    const tauSbx = parseFloat(ShearStressatBaseActualHorizontalYdirection)


    const ShearStressatBaseActualLateralXdirection = (pxd * 9810 / asb).toFixed(2);
    const tauSby = parseFloat(ShearStressatBaseActualLateralXdirection)

    const sigmaTerm = Math.pow(sigmaTb + sigmaBbx + sigmaBby, 2);
    const tauTerm = 3 * (Math.pow(tauSbx, 2) + Math.pow(tauSby, 2));
    const VonMisesStressatBaseActual = Math.sqrt(sigmaTerm + tauTerm);
    setVonMisesStressatBaseActual(isNaN(VonMisesStressatBaseActual) ? 0 : VonMisesStressatBaseActual.toFixed(3));
  };

  const VonMisesStressatBaseActualValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateVonMisesStressatBaseActualValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue, LoadonPadeyeValueInMT, DLFValue,]);

  // Weld Stress Check of Base Weld
  // Weld Dimensions and Design Loads
  // Throatthickness
  const ThroatthicknessUnits = ['mm', 'cm'];
  const ThroatthicknessFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [Throatthickness, setThroatthickness] = useState(0);
  const [ThroatthicknessSelectedUnit, setThroatthicknessSelectedUnit] = useState('mm');

  const handleThroatthicknessUnitChange = (unit) => {
    setThroatthicknessSelectedUnit(unit);
    const ThroatthicknessFactor = ThroatthicknessFactors[unit][ThroatthicknessUnits.indexOf(ThroatthicknessSelectedUnit)];
    setThroatthickness((parseFloat(Throatthickness) / ThroatthicknessFactor).toFixed(2));
  };

  const calculateThroatthicknessValue = () => {
    const Throatthickness = (internalBaseWeldLegSize / Math.sqrt(2)).toFixed(2);
    setThroatthickness(isNaN(Throatthickness) ? 0 : Throatthickness);
  };
  useEffect(() => {
    calculateThroatthicknessValue();
  }, [internalDiameterofeyepinholeValue, internalShacklePinDiameter, internalBaseWeldLegSize]);


  // WeldCheckatCheekPlateWeld
  const WeldCheckatCheekPlateWeldThroatthicknessUnits = ['mm', 'cm'];
  const WeldCheckatCheekPlateWeldThroatthicknessFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [WeldCheckatCheekPlateWeldThroatthickness, setWeldCheckatCheekPlateWeldThroatthickness] = useState(0);
  const [WeldCheckatCheekPlateWeldThroatthicknessSelectedUnit, setWeldCheckatCheekPlateWeldThroatthicknessSelectedUnit] = useState('mm');

  const handleWeldCheckatCheekPlateWeldThroatthicknessUnitChange = (unit) => {
    setWeldCheckatCheekPlateWeldThroatthicknessSelectedUnit(unit);
    const WeldCheckatCheekPlateWeldThroatthicknessFactor = WeldCheckatCheekPlateWeldThroatthicknessFactors[unit][WeldCheckatCheekPlateWeldThroatthicknessUnits.indexOf(WeldCheckatCheekPlateWeldThroatthicknessSelectedUnit)];
    setWeldCheckatCheekPlateWeldThroatthickness((parseFloat(WeldCheckatCheekPlateWeldThroatthickness) / WeldCheckatCheekPlateWeldThroatthicknessFactor));
  };

  const calculateWeldCheckatCheekPlateWeldThroatthicknessValue = () => {
    const WeldCheckatCheekPlateWeldThroatthickness = (internalCheekPlateWeldLegSize / Math.sqrt(2)).toFixed(1);
    setWeldCheckatCheekPlateWeldThroatthickness(isNaN(WeldCheckatCheekPlateWeldThroatthickness) ? 0 : WeldCheckatCheekPlateWeldThroatthickness);
  };
  useEffect(() => {
    calculateWeldCheckatCheekPlateWeldThroatthicknessValue();
  }, [internalDiameterofeyepinholeValue, internalShacklePinDiameter, internalCheekPlateWeldLegSize]);



  // Total Weld Length
  const TotalWeldLengthUnits = ['mm', 'cm'];
  const TotalWeldLengthFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [TotalWeldLength, setTotalWeldLength] = useState(0);
  const [TotalWeldLengthSelectedUnit, setTotalWeldLengthSelectedUnit] = useState('mm');

  const handleTotalWeldLengthUnitChange = (unit) => {
    setTotalWeldLengthSelectedUnit(unit);
    const TotalWeldLengthFactor = TotalWeldLengthFactors[unit][TotalWeldLengthUnits.indexOf(TotalWeldLengthSelectedUnit)];
    setTotalWeldLength((parseFloat(TotalWeldLength) / TotalWeldLengthFactor));
  };

  const calculateTotalWeldLengthValue = () => {
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const TotalWeldLength = (2 * L) + (2 * T)
    setTotalWeldLength(isNaN(TotalWeldLength) ? 0 : TotalWeldLength);
  };
  useEffect(() => {
    calculateTotalWeldLengthValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue]);


  // Shear Stress at Base Weld

  const ShearStressatBaseWeldUnits = ['mm²', 'cm²'];
  const ShearStressatBaseWeldFactors = {
    'mm²': [1, 0.01],
    'cm²': [100, 1],
  };

  const [ShearStressatBaseWeld, setShearStressatBaseWeld] = useState(0);
  const [ShearStressatBaseWeldSelectedUnit, setShearStressatBaseWeldSelectedUnit] = useState('mm²');

  const handleShearStressatBaseWeldUnitChange = (unit) => {
    setShearStressatBaseWeldSelectedUnit(unit);
    const ShearStressatBaseWeldFactor = ShearStressatBaseWeldFactors[unit][ShearStressatBaseWeldUnits.indexOf(ShearStressatBaseWeldSelectedUnit)];
    setShearStressatBaseWeld((parseFloat(ShearStressatBaseWeld) / ShearStressatBaseWeldFactor).toFixed(3));
  };

  const calculateShearStressatBaseWeldValue = () => {
    const squarerootof2 = Math.sqrt(2)
    const twt = internalBaseWeldLegSize / squarerootof2;
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const Lw = (2 * L) + (2 * T)
    const ShearStressatBaseWeld = twt * Lw
    setShearStressatBaseWeld(isNaN(ShearStressatBaseWeld) ? 0 : ShearStressatBaseWeld.toFixed(3));
  };
  useEffect(() => {
    calculateShearStressatBaseWeldValue();
  }, [Throatthickness, TotalWeldLength, internalThicknessofMainPlateValue, internalLengthofBasePlateInputValue, internalBaseWeldLegSize]);


  // Shear Stress at Base Weld(Actual) - Horizontal (Y -direction)

  const ShearStressatBaseWeldActualHorizontalYdirectionUnits = ['Mpa', 'Pa'];
  const ShearStressatBaseWeldActualHorizontalYdirectionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [ShearStressatBaseWeldActualHorizontalYdirection, setShearStressatBaseWeldActualHorizontalYdirection] = useState(0);
  const [ShearStressatBaseWeldActualHorizontalYdirectionSelectedUnit, setShearStressatBaseWeldActualHorizontalYdirectionSelectedUnit] = useState('Mpa');

  const handleShearStressatBaseWeldActualHorizontalYdirectionUnitChange = (unit) => {
    const AllowableBearingStressFactor = ShearStressatBaseWeldActualHorizontalYdirectionFactors[unit][ShearStressatBaseWeldActualHorizontalYdirectionUnits.indexOf(ShearStressatBaseWeldActualHorizontalYdirectionSelectedUnit)];
    let newValue = parseFloat(ShearStressatBaseWeldActualHorizontalYdirection) / AllowableBearingStressFactor;
    newValue = ShearStressatBaseWeldActualHorizontalYdirectionFormatValue(newValue, unit);
    setShearStressatBaseWeldActualHorizontalYdirectionSelectedUnit(unit);
    setShearStressatBaseWeldActualHorizontalYdirection(newValue);
  };

  const calculateShearStressatBaseWeldActualHorizontalYdirectionValue = () => {
    const pi = Math.PI
    const squarerootof2 = Math.sqrt(2)
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
    const LoadinHorizontalDirectionYdirection = LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue;
    const twt = internalBaseWeldLegSize / squarerootof2;
    const Lw = (2 * L) + (2 * T)
    const Asbw = twt * Lw
    const Pyd = LoadinHorizontalDirectionYdirection * DLFValue
    const ShearStressatBaseWeldActualHorizontalYdirection = ((Pyd * 9810) / Asbw).toFixed(2)
    setShearStressatBaseWeldActualHorizontalYdirection(isNaN(ShearStressatBaseWeldActualHorizontalYdirection) ? 0 : ShearStressatBaseWeldActualHorizontalYdirection);
  };
  const ShearStressatBaseWeldActualHorizontalYdirectionFormatValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateShearStressatBaseWeldActualHorizontalYdirectionValue();
  }, [
    internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue, internalBaseWeldLegSize, DLFValue,]);

  // Shear Stress at Base Weld(Actual) - Lateral (X - direction)


  const ShearStressatBaseWeldActualLateralXdirectionUnit = ['Mpa', 'Pa'];
  const ShearStressatBaseWeldActualLateralXdirectionFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [ShearStressatBaseWeldActualLateralXdirection, setShearStressatBaseWeldActualLateralXdirection] = useState(0);
  const [ShearStressatBaseWeldActualLateralXdirectionSelectedUnit, setShearStressatBaseWeldActualLateralXdirectionSelectedUnit] = useState('Mpa');

  const handleShearStressatBaseWeldActualLateralXdirectionUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = ShearStressatBaseWeldActualLateralXdirectionFactors[unit][ShearStressatBaseWeldActualLateralXdirectionUnit.indexOf(ShearStressatBaseWeldActualLateralXdirectionSelectedUnit)];
    let newValue = parseFloat(ShearStressatBaseWeldActualLateralXdirection) / AllowableBendingStressInPlaneFactor;
    newValue = ShearStressatBaseWeldActualLateralXdirectionValue(newValue, unit);
    setShearStressatBaseWeldActualLateralXdirectionSelectedUnit(unit);
    setShearStressatBaseWeldActualLateralXdirection(newValue);
  };

  const calculateShearStressatBaseWeldActualLateralXdirectionValue = () => {
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT)
    const DLF = parseFloat(DLFValue);
    if (!isNaN(L) && !isNaN(T) && !isNaN(LoadonPadeye) && !isNaN(DLF)) {
      const pi = Math.PI
      const squarerootof2 = Math.sqrt(2)
      const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
      const LoadinHorizontalDirectionYdirection = LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue;
      const twt = internalBaseWeldLegSize / squarerootof2;
      const Lw = (2 * L) + (2 * T)
      const Asbw = twt * Lw
      const Pyd = LoadinHorizontalDirectionYdirection * DLFValue
      const PadEyeDesignLoadvalue = LoadonPadeye * DLF;
      const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
      const OutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
      const Pxd = (OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)
      const ShearStressatBaseWeldActualLateralXdirection = (Pxd * 9810) / Asbw
      setShearStressatBaseWeldActualLateralXdirection(isNaN(ShearStressatBaseWeldActualLateralXdirection) ? 0 : ShearStressatBaseWeldActualLateralXdirection.toFixed(3));
    }
  };

  const ShearStressatBaseWeldActualLateralXdirectionValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateShearStressatBaseWeldActualLateralXdirectionValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue, LoadonPadeyeValueInMT, DLFValue,]);



  // Tensile Area at Base
  const TensileAreaatBaseUnit = ['mm²', 'cm²'];
  const TensileAreaatBaseFactors = {
    'mm²': [1, 0.01,],
    'cm²': [100, 1,],
  };

  const [TensileAreaatBase, setTensileAreaatBase] = useState(0);
  const [TensileAreaatBaseSelectedUnit, setTensileAreaatBaseSelectedUnit] = useState('mm²');

  const handleTensileAreaatBaseUnitChange = (unit) => {
    setTensileAreaatBaseSelectedUnit(unit);
    const TensileAreaatBaseFactor = TensileAreaatBaseFactors[unit][TensileAreaatBaseUnit.indexOf(TensileAreaatBaseSelectedUnit)];
    setTensileAreaatBase((parseFloat(TensileAreaatBase) / TensileAreaatBaseFactor).toFixed(3));
  };

  const calculateTensileAreaatBaseValue = () => {
    const squarerootof2 = Math.sqrt(2)
    const twt = internalBaseWeldLegSize / squarerootof2;
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const Lw = (2 * L) + (2 * T)
    const TensileAreaatBase = twt * Lw
    setTensileAreaatBase(isNaN(TensileAreaatBase) ? 0 : TensileAreaatBase.toFixed(3));
  };
  useEffect(() => {
    calculateTensileAreaatBaseValue();
  }, [Throatthickness, TotalWeldLength]);



  // TensileStressatBaseWeldActual

  const TensileStressatBaseWeldActualUnit = ['Mpa', 'Pa'];
  const TensileStressatBaseWeldActualFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [TensileStressatBaseWeldActual, setTensileStressatBaseWeldActual] = useState(0);
  const [TensileStressatBaseWeldActualSelectedUnit, setTensileStressatBaseWeldActualSelectedUnit] = useState('Mpa');

  const handleTensileStressatBaseWeldActualUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = TensileStressatBaseWeldActualFactors[unit][TensileStressatBaseWeldActualUnit.indexOf(TensileStressatBaseWeldActualSelectedUnit)];
    let newValue = parseFloat(TensileStressatBaseWeldActual) / AllowableBendingStressInPlaneFactor;
    newValue = TensileStressatBaseWeldActualValue(newValue, unit);
    setTensileStressatBaseWeldActualSelectedUnit(unit);
    setTensileStressatBaseWeldActual(newValue);
  };

  const calculateTensileStressatBaseWeldActualValue = () => {
    const L = internalLengthofBasePlateInputValue
    const T = internalThicknessofMainPlateValue
    const LoadonPadeye = LoadonPadeyeValueInMT
    const DLF = DLFValue;
    const squarerootof2 = Math.sqrt(2)
    const twt = internalBaseWeldLegSize / squarerootof2;
    const Lw = (2 * L) + (2 * T)
    const Asbw = twt * Lw
    const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
    const LoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
    const Pyd = LoadinVerticalDirectionZdirectionValue * DLF;
    const TensileStressatBaseWeldActual = (Pyd * 9810) / Asbw
    setTensileStressatBaseWeldActual(isNaN(TensileStressatBaseWeldActual) ? 0 : TensileStressatBaseWeldActual.toFixed(3));

  };

  const TensileStressatBaseWeldActualValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateTensileStressatBaseWeldActualValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue, LoadonPadeyeValueInMT, DLFValue, internalBaseWeldLegSize, AngleofLoadwithVerticalvalue]);

  // Section Modulus of Base Weld 
  // Section Modulus of Weld about X-axis
  const SectionModulusofWeldaboutXaxisUnit = ['mm³', 'cm³'];
  const SectionModulusofWeldaboutXaxisFactors = {
    'mm³': [1, 0.001],
    'cm³': [1000, 1],
  };

  const [SectionModulusofWeldaboutXaxis, setSectionModulusofWeldaboutXaxis] = useState(0);
  const [SectionModulusofWeldaboutXaxisSelectedUnit, setSectionModulusofWeldaboutXaxisSelectedUnit] = useState('mm³');

  const handleSectionModulusofWeldaboutXaxisUnitChange = (unit) => {
    setSectionModulusofWeldaboutXaxisSelectedUnit(unit);
    const SectionModulusofWeldaboutXaxisFactor = SectionModulusofWeldaboutXaxisFactors[unit][SectionModulusofWeldaboutXaxisUnit.indexOf(SectionModulusofWeldaboutXaxisSelectedUnit)];
    setSectionModulusofWeldaboutXaxis((parseFloat(SectionModulusofWeldaboutXaxis) / SectionModulusofWeldaboutXaxisFactor).toFixed(3));
  };

  const calculateSectionModulusofWeldaboutXaxisValue = () => {
    const b = parseFloat(internalThicknessofMainPlateValue)
    const d = parseFloat(internalLengthofBasePlateInputValue)
    const tw = parseFloat(internalBaseWeldLegSize)
    const SectionModulusofWeldaboutXaxis = (b * d + Math.pow(d, 2) / 3) * tw;
    setSectionModulusofWeldaboutXaxis(isNaN(SectionModulusofWeldaboutXaxis) ? 0 : SectionModulusofWeldaboutXaxis.toFixed(3));
  };
  useEffect(() => {
    calculateSectionModulusofWeldaboutXaxisValue();
  }, [internalThicknessofMainPlateValue, internalLengthofBasePlateInputValue, internalBaseWeldLegSize]);


  // Section Modulus of Weld about Y-axis
  const SectionModulusofWeldaboutYaxisUnit = ['mm³', 'cm³'];
  const SectionModulusofWeldaboutYaxisFactors = {
    'mm³': [1, 0.001],
    'cm³': [1000, 1],
  };

  const [SectionModulusofWeldaboutYaxis, setSectionModulusofWeldaboutYaxis] = useState(0);
  const [SectionModulusofWeldaboutYaxisSelectedUnit, setSectionModulusofWeldaboutYaxisSelectedUnit] = useState('mm³');

  const handleSectionModulusofWeldaboutYaxisUnitChange = (unit) => {
    setSectionModulusofWeldaboutYaxisSelectedUnit(unit);
    const SectionModulusofWeldaboutYaxisFactor = SectionModulusofWeldaboutYaxisFactors[unit][SectionModulusofWeldaboutYaxisUnit.indexOf(SectionModulusofWeldaboutYaxisSelectedUnit)];
    setSectionModulusofWeldaboutYaxis((parseFloat(SectionModulusofWeldaboutYaxis) / SectionModulusofWeldaboutYaxisFactor).toFixed(3));
  };

  const calculateSectionModulusofWeldaboutYaxisValue = () => {
    const b = parseFloat(internalThicknessofMainPlateValue)
    const d = parseFloat(internalLengthofBasePlateInputValue)
    const tw = parseFloat(internalBaseWeldLegSize)
    const SectionModulusofWeldaboutYaxis = (b * d + Math.pow(b, 2) / 3) * tw;
    setSectionModulusofWeldaboutYaxis(isNaN(SectionModulusofWeldaboutYaxis) ? 0 : SectionModulusofWeldaboutYaxis.toFixed(3));
  };
  useEffect(() => {
    calculateSectionModulusofWeldaboutYaxisValue();
  }, [internalThicknessofMainPlateValue, internalLengthofBasePlateInputValue, internalBaseWeldLegSize]);

  // Bending Stress Check at Base Weld
  // Total Design Moment about X-axis (In-plane)
  // TotalDesignMomentaboutXaxisInplane
  // BendingStressatBaseWeldActualaboutXAxisInPlane
  const BendingStressatBaseWeldActualaboutXAxisInPlaneUnit = ['Mpa', 'Pa'];
  const BendingStressatBaseWeldActualaboutXAxisInPlaneFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [BendingStressatBaseWeldActualaboutXAxisInPlane, setBendingStressatBaseWeldActualaboutXAxisInPlane] = useState(0);
  const [BendingStressatBaseWeldActualaboutXAxisInPlaneSelectedUnit, setBendingStressatBaseWeldActualaboutXAxisInPlaneSelectedUnit] = useState('Mpa');

  const handleBendingStressatBaseWeldActualaboutXAxisInPlaneUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = BendingStressatBaseWeldActualaboutXAxisInPlaneFactors[unit][BendingStressatBaseWeldActualaboutXAxisInPlaneUnit.indexOf(BendingStressatBaseWeldActualaboutXAxisInPlaneSelectedUnit)];
    let newValue = parseFloat(BendingStressatBaseWeldActualaboutXAxisInPlane) / AllowableBendingStressInPlaneFactor;
    newValue = BendingStressatBaseWeldActualaboutXAxisInPlaneValue(newValue, unit);
    setBendingStressatBaseWeldActualaboutXAxisInPlaneSelectedUnit(unit);
    setBendingStressatBaseWeldActualaboutXAxisInPlane(newValue);
  };

  const calculateBendingStressatBaseWeldActualaboutXAxisInPlaneValue = () => {
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT)
    const DLF = parseFloat(DLFValue);
    if (!isNaN(L) && !isNaN(T) && !isNaN(LoadonPadeye) && !isNaN(DLF)) {
      const H = parseFloat(internalinputTotalHeightofPadeyeValue)
      const R = parseFloat(internalRadiusofMainPlateValue)
      const pi = Math.PI;

      const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
      const LoadinHorizontalDirectionYdirection = (LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue);
      const D6 = (LoadinHorizontalDirectionYdirection * DLFValue)

      const LoadonPadeye = LoadonPadeyeValueInMT
      const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * (Math.PI / 180));
      const LoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
      const D7 = (LoadinVerticalDirectionZdirectionValue * DLF).toFixed(2)

      const D9 = (H - R).toFixed(2)

      const D10 = (R - (L / 2)).toFixed(2)
      const TotalDesignMomentaboutXaxisInplane = (D6 * D9 + D7 * D10).toFixed(2)
      const Mdxx = parseFloat(TotalDesignMomentaboutXaxisInplane)

      const b = parseFloat(internalThicknessofMainPlateValue)
      const d = parseFloat(internalLengthofBasePlateInputValue)
      const tw = parseFloat(internalBaseWeldLegSize)
      const SectionModulusofWeldaboutXaxis = (b * d + Math.pow(d, 2) / 3) * tw;
      const Zwbxx = parseFloat(SectionModulusofWeldaboutXaxis)

      const BendingStressatBaseWeldActualaboutXAxisInPlane = Mdxx * 9810 / Zwbxx
      setBendingStressatBaseWeldActualaboutXAxisInPlane(isNaN(BendingStressatBaseWeldActualaboutXAxisInPlane) ? 0 : BendingStressatBaseWeldActualaboutXAxisInPlane.toFixed(3));
    }
  };

  const BendingStressatBaseWeldActualaboutXAxisInPlaneValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateBendingStressatBaseWeldActualaboutXAxisInPlaneValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue, LoadonPadeyeValueInMT, DLFValue,]);


  // BendingStressActualaboutYaxisOutofPlane
  // BendingStressaboutYaxisOutofPlane

  const BendingStressaboutYaxisOutofPlaneUnit = ['Mpa', 'Pa'];
  const BendingStressaboutYaxisOutofPlaneFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [BendingStressaboutYaxisOutofPlane, setBendingStressaboutYaxisOutofPlane] = useState(0);
  const [BendingStressaboutYaxisOutofPlaneSelectedUnit, setBendingStressaboutYaxisOutofPlaneSelectedUnit] = useState('Mpa');

  const handleBendingStressaboutYaxisOutofPlaneUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = BendingStressaboutYaxisOutofPlaneFactors[unit][BendingStressaboutYaxisOutofPlaneUnit.indexOf(BendingStressaboutYaxisOutofPlaneSelectedUnit)];
    let newValue = parseFloat(BendingStressaboutYaxisOutofPlane) / AllowableBendingStressInPlaneFactor;
    newValue = BendingStressaboutYaxisOutofPlaneValue(newValue, unit);
    setBendingStressaboutYaxisOutofPlaneSelectedUnit(unit);
    setBendingStressaboutYaxisOutofPlane(newValue);
  };

  const calculateBendingStressaboutYaxisOutofPlaneValue = () => {
    const b = parseFloat(internalThicknessofMainPlateValue)
    const d = parseFloat(internalLengthofBasePlateInputValue)
    const tw = parseFloat(internalBaseWeldLegSize)
    const ZwbyyValue = (b * d + Math.pow(b, 2) / 3) * tw;

    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT)
    const DLF = parseFloat(DLFValue);
    const pi = Math.PI;
    const PadEyeDesignLoadvalue = LoadonPadeye * DLF;
    const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
    const OutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
    const pxd = (OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)
    const H = parseFloat(internalinputTotalHeightofPadeyeValue)
    const R = parseFloat(internalRadiusofMainPlateValue)
    const MomentleverforHorizontalLateralForce = (H - R).toFixed(2);
    const h = parseFloat(MomentleverforHorizontalLateralForce)
    const MdxxValue = pxd * h;

    const Mdxx = parseFloat(MdxxValue)
    const Zwbyy = parseFloat(ZwbyyValue)
    const BendingStressaboutYaxisOutofPlane = (Mdxx * 9810 / Zwbyy).toFixed(3)
    setBendingStressaboutYaxisOutofPlane(isNaN(BendingStressaboutYaxisOutofPlane) ? 0 : BendingStressaboutYaxisOutofPlane);
  };

  const BendingStressaboutYaxisOutofPlaneValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateBendingStressaboutYaxisOutofPlaneValue();
  }, [internalThicknessofMainPlateValue, internalLengthofBasePlateInputValue, internalBaseWeldLegSize, LoadonPadeyeValueInMT, DLFValue, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue, internalinputTotalHeightofPadeyeValue, internalRadiusofMainPlateValue,]);

  const TotalStressatBaseWeldActualUnit = ['Mpa', 'Pa'];
  const TotalStressatBaseWeldActualFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [TotalStressatBaseWeldActual, setTotalStressatBaseWeldActual] = useState(0);
  const [TotalStressatBaseWeldActualSelectedUnit, setTotalStressatBaseWeldActualSelectedUnit] = useState('Mpa');

  const handleTotalStressatBaseWeldActualUnitChange = (unit) => {
    const AllowableBendingStressInPlaneFactor = TotalStressatBaseWeldActualFactors[unit][TotalStressatBaseWeldActualUnit.indexOf(TotalStressatBaseWeldActualSelectedUnit)];
    let newValue = parseFloat(TotalStressatBaseWeldActual) / AllowableBendingStressInPlaneFactor;
    newValue = TotalStressatBaseWeldActualValue(newValue, unit);
    setTotalStressatBaseWeldActualSelectedUnit(unit);
    setTotalStressatBaseWeldActual(newValue);
  };

  const calculateTotalStressatBaseWeldActualValue = () => {
    const L = parseFloat(internalLengthofBasePlateInputValue)
    const T = parseFloat(internalThicknessofMainPlateValue)
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT)
    const DLF = parseFloat(DLFValue);
    const squarerootof2 = Math.sqrt(2)
    const twt = internalBaseWeldLegSize / squarerootof2;
    const Lw = (2 * L) + (2 * T)
    const Asbw = twt * Lw
    const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
    const LoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
    const Pyd = LoadinVerticalDirectionZdirectionValue * DLF;
    const TensileStressatBaseWeldActual = ((Pyd * 9810) / Asbw).toFixed(3)
    const σtbwAct = parseFloat(TensileStressatBaseWeldActual)

    const H = parseFloat(internalinputTotalHeightofPadeyeValue)
    const R = parseFloat(internalRadiusofMainPlateValue)
    const pi = Math.PI;

    const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
    const LoadinHorizontalDirectionYdirection = (LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue);
    const D6 = (LoadinHorizontalDirectionYdirection * DLFValue)

    const D7 = (LoadinVerticalDirectionZdirectionValue * DLF).toFixed(2)

    const D9 = (H - R).toFixed(2)

    const D10 = (R - (L / 2)).toFixed(2)
    const TotalDesignMomentaboutXaxisInplane = (D6 * D9 + D7 * D10).toFixed(2)
    const Mdxx = parseFloat(TotalDesignMomentaboutXaxisInplane)

    const b = parseFloat(internalThicknessofMainPlateValue)
    const d = parseFloat(internalLengthofBasePlateInputValue)
    const tw = parseFloat(internalBaseWeldLegSize)
    const SectionModulusofWeldaboutXaxis = (b * d + Math.pow(d, 2) / 3) * tw;
    const Zwbxx = parseFloat(SectionModulusofWeldaboutXaxis)

    const BendingStressatBaseWeldActualaboutXAxisInPlane = Mdxx * 9810 / Zwbxx
    setBendingStressatBaseWeldActualaboutXAxisInPlane(BendingStressatBaseWeldActualaboutXAxisInPlane.toFixed(3));
    const σbbwxAct = parseFloat(isNaN(BendingStressatBaseWeldActualaboutXAxisInPlane) ? 0 : BendingStressatBaseWeldActualaboutXAxisInPlane)


    const ZwbyyValue = (b * d + Math.pow(b, 2) / 3) * tw;
    const PadEyeDesignLoadvalue = LoadonPadeye * DLF;
    const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
    const OutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
    const pxd = (OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)
    const h = parseFloat(MomentleverforHorizontalLateralForce)
    const MdxxValue = pxd * h;
    const Zwbyy = parseFloat(ZwbyyValue)
    const BendingStressaboutYaxisOutofPlane = (MdxxValue * 9810 / Zwbyy).toFixed(3)
    const σbbyAct = parseFloat(isNaN(BendingStressaboutYaxisOutofPlane) ? 0 : BendingStressaboutYaxisOutofPlane)

    const ShearStressatBaseWeldActualHorizontalYdirectionsquarerootof2 = Math.sqrt(2)
    const ShearStressatBaseWeldActualHorizontalYdirectionL = parseFloat(internalLengthofBasePlateInputValue)
    const ShearStressatBaseWeldActualHorizontalYdirectionT = parseFloat(internalThicknessofMainPlateValue)
    const ShearStressatBaseWeldActualHorizontalYdirectionLoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
    const ShearStressatBaseWeldActualHorizontalYdirectionLoadinHorizontalDirectionYdirection = LoadonPadeyeValueInMT * ShearStressatBaseWeldActualHorizontalYdirectionLoadinHorizontalDirectionYdirectionMultipleValue;
    const ShearStressatBaseWeldActualHorizontalYdirectiontwt = internalBaseWeldLegSize / ShearStressatBaseWeldActualHorizontalYdirectionsquarerootof2;
    const ShearStressatBaseWeldActualHorizontalYdirectionLw = (2 * ShearStressatBaseWeldActualHorizontalYdirectionL) + (2 * ShearStressatBaseWeldActualHorizontalYdirectionT)
    const ShearStressatBaseWeldActualHorizontalYdirectionAsbw = ShearStressatBaseWeldActualHorizontalYdirectiontwt * ShearStressatBaseWeldActualHorizontalYdirectionLw
    const ShearStressatBaseWeldActualHorizontalYdirectionPyd = ShearStressatBaseWeldActualHorizontalYdirectionLoadinHorizontalDirectionYdirection * DLFValue
    const ShearStressatBaseWeldActualHorizontalYdirection = ((ShearStressatBaseWeldActualHorizontalYdirectionPyd * 9810) / ShearStressatBaseWeldActualHorizontalYdirectionAsbw).toFixed(2)
    const τsbwyAct = parseFloat(isNaN(ShearStressatBaseWeldActualHorizontalYdirection) ? 0 : ShearStressatBaseWeldActualHorizontalYdirection)


    const Pxd = (OutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)
    const ShearStressatBaseWeldActualLateralXdirection = (Pxd * 9810) / Asbw
    setShearStressatBaseWeldActualLateralXdirection(ShearStressatBaseWeldActualLateralXdirection.toFixed(3));
    const τsbwxAct = parseFloat(ShearStressatBaseWeldActualLateralXdirection)
    const TotalStressatBaseWeldActual = (Math.sqrt(Math.pow(σtbwAct + σbbwxAct + σbbyAct, 2) + Math.pow(τsbwyAct, 2) + Math.pow(τsbwxAct, 2))).toFixed(3);
    setTotalStressatBaseWeldActual(isNaN(TotalStressatBaseWeldActual) ? 0 : TotalStressatBaseWeldActual);
  };

  const TotalStressatBaseWeldActualValue = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(3);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(4);
      }
      return floatValue.toFixed(3);
    }
  };

  useEffect(() => {
    calculateTotalStressatBaseWeldActualValue();
  }, [internalLengthofBasePlateInputValue, internalThicknessofMainPlateValue, LoadonPadeyeValueInMT, DLFValue, internalBaseWeldLegSize, AngleofLoadwithVerticalvalue, internalinputTotalHeightofPadeyeValue, internalRadiusofMainPlateValue, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue, LoadonPadeyeValueInMT, internalThicknessofMainPlateValue, internalLengthofBasePlateInputValue, internalBaseWeldLegSize, AngleofLoadwithVerticalvalue, OutofPlaneAngleValue]);

  // Weld Length
  const WeldLengthUnit = ['mm', 'cm'];
  const WeldLengthFactors = {
    mm: [1, 0.1],
    cm: [10, 1],
  };

  const [WeldLength, setWeldLength] = useState(0);
  const [WeldLengthSelectedUnit, setWeldLengthSelectedUnit] = useState('mm');

  const handleWeldLengthUnitChange = (unit) => {
    setWeldLengthSelectedUnit(unit);
    const TotalStressatBaseWeldActualFactor = WeldLengthFactors[unit][WeldLengthUnit.indexOf(WeldLengthSelectedUnit)];
    setWeldLength((parseFloat(WeldLength) / TotalStressatBaseWeldActualFactor));
  };

  const calculateWeldLengthValue = () => {
    const WeldLength = ((Math.PI * internalDiameterofCheekPlateInputValue) / 2).toFixed(4);
    setWeldLength(isNaN(WeldLength) ? 0 : WeldLength);
  };
  useEffect(() => {
    calculateWeldLengthValue();
  }, [internalDiameterofCheekPlateInputValue]);



  // Weld Area at Cheek Plate
  const WeldAreaatCheekPlateUnit = ['mm²', 'cm²'];
  const WeldAreaatCheekPlateFactors = {
    'mm²': [1, 0.01],
    'cm²': [100, 1],
  };

  const [WeldAreaatCheekPlate, setWeldAreaatCheekPlate] = useState(0);
  const [WeldAreaatCheekPlateSelectedUnit, setWeldAreaatCheekPlateSelectedUnit] = useState('mm²');

  const handleWeldAreaatCheekPlateUnitChange = (unit) => {
    setWeldAreaatCheekPlateSelectedUnit(unit);
    const TotalStressatBaseWeldActualFactor = WeldAreaatCheekPlateFactors[unit][WeldAreaatCheekPlateUnit.indexOf(WeldAreaatCheekPlateSelectedUnit)];
    setWeldAreaatCheekPlate((parseFloat(WeldAreaatCheekPlate) / TotalStressatBaseWeldActualFactor));
  };

  const calculateWeldAreaatCheekPlateValue = () => {
    const DiameterofCheekPlate = parseFloat(internalDiameterofCheekPlateInputValue)
    const WeldLengthValue = (Math.PI * internalDiameterofCheekPlateInputValue) / 2
    const Throatthicknessvalue = internalCheekPlateWeldLegSize / 1.41421356
    const WeldAreaatCheekPlate = (Throatthicknessvalue * WeldLengthValue).toFixed(3);
    setWeldAreaatCheekPlate(isNaN(WeldAreaatCheekPlate) ? 0 : WeldAreaatCheekPlate);
  };
  useEffect(() => {
    calculateWeldAreaatCheekPlateValue();
  }, [internalDiameterofCheekPlateInputValue, internalCheekPlateWeldLegSize, internalCheekPlateWeldLegSize,
  ]);


  // Design Load on one cheek plate

  const DesignLoadononecheekplateUnit = ['MT', 'N'];
  const DesignLoadononecheekplateFactors = {
    MT: [1, 0.0001019716],
    N: [9806.65, 1],
  };

  const [DesignLoadononecheekplate, setDesignLoadononecheekplate] = useState(0);
  const [DesignLoadononecheekplateSelectedUnit, setDesignLoadononecheekplateSelectedUnit] = useState('MT');

  const handleDesignLoadononecheekplateUnitChange = (unit) => {
    setDesignLoadononecheekplateSelectedUnit(unit);
    const factor = DesignLoadononecheekplateFactors[unit][LoadinVerticalDirectionZdirectionUnits.indexOf(DesignLoadononecheekplateSelectedUnit)];
    setDesignLoadononecheekplate((parseFloat(DesignLoadononecheekplate) * factor).toFixed(2));
  };

  const calculateDesignLoadononecheekplateValue = () => {
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const DLF = parseFloat(DLFValue)
    const PadEyeDesignLoadValue = (LoadonPadeye * DLF).toFixed(2);
    const pd = parseFloat(PadEyeDesignLoadValue)
    const tc = parseFloat(internalThicknessofCheekPlateInputValue)
    const t = parseFloat(internalThicknessofMainPlateValue)
    const DesignLoadononecheekplate = (pd * tc / (t + 2 * tc)).toFixed(2);
    setDesignLoadononecheekplate(isNaN(DesignLoadononecheekplate) ? 0 : DesignLoadononecheekplate);
  };
  useEffect(() => {
    calculateDesignLoadononecheekplateValue();
  }, [LoadonPadeyeValueInMT, internalThicknessofCheekPlateInputValue, internalThicknessofMainPlateValue]);


  const WeldStressatCheekWeldActualUnit = ['Mpa', 'Pa'];
  const WeldStressatCheekWeldActualFactors = {
    Mpa: [1, 1e6],
    Pa: [1e-6, 1],
  };

  const [WeldStressatCheekWeldActual, setWeldStressatCheekWeldActual] = useState(0);
  const [WeldStressatCheekWeldActualSelectedUnit, setWeldStressatCheekWeldActualSelectedUnit] = useState('Mpa');

  const handleWeldStressatCheekWeldActualUnitChange = (unit) => {
    const AllowableBearingStressFactor = WeldStressatCheekWeldActualFactors[unit][WeldStressatCheekWeldActualUnit.indexOf(WeldStressatCheekWeldActualSelectedUnit)];
    let newValue = parseFloat(WeldStressatCheekWeldActual) / AllowableBearingStressFactor;
    newValue = FormatValueforWeldStressAtCheekWeldActual(newValue, unit);
    setWeldStressatCheekWeldActualSelectedUnit(unit);
    setWeldStressatCheekWeldActual(newValue);
  };

  const calculateWeldStressatCheekWeldActualValue = () => {
    const internalThicknessofMainPlateValueValue = parseFloat(internalThicknessofMainPlateValue);
    const internalThicknessofCheekPlateInputValueValue = parseFloat(internalThicknessofCheekPlateInputValue);
    const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
    const DLF = parseFloat(DLFValue)
    const PadEyeDesignLoadValue = (LoadonPadeye * DLF).toFixed(2);
    if (!isNaN(internalThicknessofMainPlateValueValue) && !isNaN(internalThicknessofCheekPlateInputValueValue) && !isNaN(PadEyeDesignLoadValue)) {

      const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
      const DLF = parseFloat(DLFValue)
      const pd = (LoadonPadeye * DLF).toFixed(2);
      const tc = parseFloat(internalThicknessofCheekPlateInputValue)
      const t = parseFloat(internalThicknessofMainPlateValue)
      const DesignLoadononecheekplateValue = pd * tc / (t + 2 * tc);
      const DiameterofCheekPlate = parseFloat(internalDiameterofCheekPlateInputValue)
      const WeldLengthValue = (Math.PI * DiameterofCheekPlate) / 2
      const Throatthicknessvalue = internalCheekPlateWeldLegSize / 1.41421356
      const WeldAreaatCheekPlate = Throatthicknessvalue * WeldLengthValue;
      const WeldStressatCheekWeldActual = (DesignLoadononecheekplateValue * 9810 / WeldAreaatCheekPlate).toFixed(2);
      setWeldStressatCheekWeldActual(isNaN(WeldStressatCheekWeldActual) ? 0 : WeldStressatCheekWeldActual);
    }
  };
  const FormatValueforWeldStressAtCheekWeldActual = (value, unit) => {
    const floatValue = parseFloat(value);
    if (unit === 'Mpa') {
      return floatValue.toFixed(2);
    } else {
      if (floatValue >= 1000) {
        return floatValue.toExponential(3);
      }
      return floatValue.toFixed(2);
    }
  };

  useEffect(() => {
    calculateWeldStressatCheekWeldActualValue();
  }, [internalThicknessofMainPlateValue, internalThicknessofCheekPlateInputValue, LoadonPadeyeValueInMT, DLFValue, internalThicknessofCheekPlateInputValue, internalThicknessofMainPlateValue, internalDiameterofCheekPlateInputValue, internalCheekPlateWeldLegSize,]);



  // Final Checks Conditions 

  // Geometry Check 
  // Main Plate Radius Condition
  if (internalRadiusofMainPlateValue > 1.25 * internalDiameterofeyepinholeValue) {
    var MainPlateRadius = <p>OK</p>;
  } else {
    var MainPlateRadius = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Shackle Clearance checks
  // ShackleClearanceChecks

  const ShackleInsideLengthvalue = parseFloat(internalShackleInsideLength);
  // const LengthClearanceofShackle = Plusvalue - minusValue;
  const LengthClearanceofShackleValue = (internalDiameterofeyepinholeValue - internalShacklePinDiameter).toFixed(3);

  if (LengthClearanceofShackleValue > 2 && LengthClearanceofShackleValue < 6) {
    var ShackleClearanceChecks = <p>OK</p>;
  } else {
    var ShackleClearanceChecks = <p style={{ color: '#bd2323' }}>NOT OK</p>
  }

  // Stress Check At Pin Hole Conditions
  // Bearing Area
  const Dp = parseFloat(internalShacklePinDiameter);
  const t = parseFloat(internalThicknessofMainPlateValue);
  const tc = parseFloat(internalThicknessofCheekPlateInputValue);
  const BearingAreaValue = piby4 * Dp * (t + 2 * tc)
  const LoadonPadeye = parseFloat(LoadonPadeyeValueInMT);
  const DLF = parseFloat(DLFValue)
  const ValueofDesignBearingLoad = (LoadonPadeye * DLF).toFixed(2);
  const ValueofBearingStressActual = ValueofDesignBearingLoad / BearingAreaValue * 9810
  const ValueofBearingStressAllowed = MaterialYieldStressInMPa * 0.9
  if (ValueofBearingStressActual < ValueofBearingStressAllowed) {
    var BearingStressCheckatPinHole = <p>OK</p>;
  } else {
    var BearingStressCheckatPinHole = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Shear Area of pin hole Condition
  const Rc = internalDiameterofCheekPlateInputValue / 2
  const ValueofShearAreaofpinhole = 2 * ((internalRadiusofMainPlateValue - internalDiameterofeyepinholeValue / 2) * internalThicknessofMainPlateValue + 2 * (Rc - internalDiameterofeyepinholeValue / 2) * internalThicknessofCheekPlateInputValue)
  const ValueofShearStressatpinhole = ValueofDesignBearingLoad * 9810 / ValueofShearAreaofpinhole
  const AllowableShearStressatpinhole = MaterialYieldStressInMPa * 0.4
  if (ValueofShearStressatpinhole < AllowableShearStressatpinhole) {
    var ShearAreaofPinHole = <p>OK</p>;
  } else {
    var ShearAreaofPinHole = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Tensile Stress Check at Pin Hole

  // A-A
  const R = parseFloat(internalRadiusofMainPlateValue);
  const De = parseFloat(internalDiameterofeyepinholeValue);
  const ValueofDLF = parseFloat(DLFValue)
  const ValueofTensileAreaforsectionAA = 2 * ((R - De / 2) * t + 2 * (Rc - De / 2) * tc);
  const pd = LoadonPadeye * ValueofDLF;
  const ValueofTensileStressatpinholeActualatsectionAA = pd * 9810 / ValueofTensileAreaforsectionAA;

  // B-B
  const valueofPadEyeDesignLoad = LoadonPadeye * ValueofDLF;
  const ValueofTensileAreaforsectionBB = ((2 * R) + (pi * Rc / 2)) * t
  const ValueofTensileStressatpinholeActualatsectionBB = valueofPadEyeDesignLoad * 9810 / ValueofTensileAreaforsectionBB

  const TensileStressatpinholeAllowed = MaterialYieldStressInMPa * 0.45

  const maxD31D33 = Math.max(ValueofTensileStressatpinholeActualatsectionAA, ValueofTensileStressatpinholeActualatsectionBB);

  if (maxD31D33 < TensileStressatpinholeAllowed) {
    var TensileStressCheckatPinHole = <p>OK</p>;
  } else {
    var TensileStressCheckatPinHole = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Hertz/Contact Stress Check at Pin Hole

  const D38 = pd / (t + 2 * tc);
  const D39 = parseFloat(ModulusofElasticityValueInMpa);
  const D40 = parseFloat(PoissonsRatio);
  const D41 = parseFloat(internalDiameterofeyepinholeValue)
  const D42 = parseFloat(internalShacklePinDiameter)
  const ValueofHertzStressatPinHole = (Math.sqrt(D38 * 9810 * D39 * (D41 - D42) / (PI * (1 - Math.pow(D40, 2)) * D41 * D42))).toFixed(3);

  const AllowableHertzStressatPinHole = MaterialYieldStressInMPa * 2.5

  if (ValueofHertzStressatPinHole < AllowableHertzStressatPinHole) {
    var HertzContactStressCheckatPinHole = <p>OK</p>;
  } else {
    var HertzContactStressCheckatPinHole = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Stress Checks at Base Plate
  // Tensile Stress Check at Base
  const multiplyValue = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
  const ValueofLoadinVerticalDirectionZdirectionValue = LoadonPadeye * multiplyValue;
  const ValueofDesignLoadinVerticalDirectionZdirection = ValueofLoadinVerticalDirectionZdirectionValue * ValueofDLF;
  const pt = parseFloat(ValueofDesignLoadinVerticalDirectionZdirection)
  const L = parseFloat(internalLengthofBasePlateInputValue)
  const T = parseFloat(internalThicknessofMainPlateValue)
  const TensileAreavalue = L * T;
  const atb = parseFloat(TensileAreavalue)
  const ValueofTensileStressatBaseActual = (pt * 9810 / atb).toFixed(2);

  const AllowableTensileStress = MaterialYieldStressInMPa * 0.6

  if (ValueofTensileStressatBaseActual < AllowableTensileStress) {
    var TensileStressCheckatBase = <p>OK</p>;
  } else {
    var TensileStressCheckatBase = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }


  // Weld Check at Cheek Plate Weld
  // Stress Check at Cheek Plate Weld
  const DesignLoadononecheekplateValue = pd * tc / (t + 2 * tc);
  const DiameterofCheekPlate = parseFloat(internalDiameterofCheekPlateInputValue)
  const WeldLengthValue = (Math.PI * DiameterofCheekPlate) / 2
  const Throatthicknessvalue = internalCheekPlateWeldLegSize / 1.41421356
  const ValueofWeldAreaatCheekPlate = Throatthicknessvalue * WeldLengthValue;
  const ValueofWeldStressatCheekWeldActual = (DesignLoadononecheekplateValue * 9810 / ValueofWeldAreaatCheekPlate).toFixed(2);

  const AllowableWeldStress = ElectrodeTensileStrengthInMpa * 0.3
  if (ValueofWeldStressatCheekWeldActual < AllowableWeldStress) {
    var StressCheckatCheekPlateWeld = <p>OK</p>;
  } else {
    var StressCheckatCheekPlateWeld = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Bending Stress Check at Base (X-axis)
  const H = parseFloat(internalinputTotalHeightofPadeyeValue)
  const LoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
  const ValueofLoadinHorizontalDirectionYdirection = (LoadonPadeyeValueInMT * LoadinHorizontalDirectionYdirectionMultipleValue);
  const D6 = (ValueofLoadinHorizontalDirectionYdirection * DLFValue)
  const ValueofmultiplyValue = Math.cos(AngleofLoadwithVerticalvalue * (Math.PI / 180));
  const valueofLoadinVerticalDirectionZdirectionValue = LoadonPadeye * ValueofmultiplyValue;
  const D7 = (valueofLoadinVerticalDirectionZdirectionValue * DLF).toFixed(2)
  const D9 = (H - R).toFixed(2)
  const D10 = (R - (L / 2)).toFixed(2)
  const ValueofTotalDesignMomentaboutXaxisInplane = (D6 * D9 + D7 * D10).toFixed(2)
  const ValueofMdxx = parseFloat(ValueofTotalDesignMomentaboutXaxisInplane)
  const b = parseFloat(internalThicknessofMainPlateValue)
  const d = parseFloat(internalLengthofBasePlateInputValue)
  const tw = parseFloat(internalBaseWeldLegSize)
  const ValueofSectionModulusofWeldaboutXaxis = (b * d + Math.pow(d, 2) / 3) * tw;
  const Zwbxx = parseFloat(ValueofSectionModulusofWeldaboutXaxis)
  const ValueofBendingStressatBaseWeldActualaboutXAxisInPlane = ValueofMdxx * 9810 / Zwbxx

  const AllowableBendingStressInPlane = MaterialYieldStressInMPa * 0.6
  if (ValueofBendingStressatBaseWeldActualaboutXAxisInPlane < AllowableBendingStressInPlane) {
    var BendingStressCheckatBaseXaxis = <p>OK</p>;
  } else {
    var BendingStressCheckatBaseXaxis = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Bending Stress Check at Base (Y-axis)
  const h = parseFloat(MomentleverforHorizontalLateralForce)
  const Mathpi = Math.PI;
  const PadEyeDesignLoadvalue = LoadonPadeye * DLF;
  const OutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * Mathpi / 180) * Math.sin(OutofPlaneAngleValue * Mathpi / 180)
  const ValueofOutofplaneLoadLateralLoadXdirection = LoadonPadeye * OutofplaneLoadLateralLoadXdirectionMultiplyValue;
  const pxd = (ValueofOutofplaneLoadLateralLoadXdirection * DLF) + (0.05 * PadEyeDesignLoadvalue)
  const Mdyy = pxd * h
  const Zbyy = L * Math.pow(T, 2) / 6
  const ValueofBendingStressActualaboutYaxisOutofPlane = Mdyy * 9810 / Zbyy;

  const AllowableBendingStressOutofPlane = MaterialYieldStressInMPa * 0.75

  if (ValueofBendingStressActualaboutYaxisOutofPlane < AllowableBendingStressOutofPlane) {
    var BendingStressCheckatBaseYaxis = <p>OK</p>;
  } else {
    var BendingStressCheckatBaseYaxis = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Shear Stress Check at Base (Y direction)
  const ValueofLoadinHorizontalDirectionYdirectionMultipleValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
  const LoadinHorizontalDirectionYdirectionValue = (LoadonPadeyeValueInMT * ValueofLoadinHorizontalDirectionYdirectionMultipleValue).toFixed(1);
  const pyd = (LoadinHorizontalDirectionYdirectionValue * DLFValue).toFixed(2)
  const asb = L * T
  const ValueofShearStressatBaseActualHorizontalYdirection = (pyd * 9810 / asb).toFixed(3);

  const AllowableShearStress = MaterialYieldStressInMPa * 0.4

  if (ValueofShearStressatBaseActualHorizontalYdirection < AllowableShearStress) {
    var ShearStressCheckatBaseYdirection = <p>OK</p>;
  } else {
    var ShearStressCheckatBaseYdirection = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Shear Stress Check at Base (X direction)

  const ValueofOutofplaneLoadLateralLoadXdirectionMultiplyValue = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
  const OutofplaneLoadLateralLoadXdirectionValue = LoadonPadeye * ValueofOutofplaneLoadLateralLoadXdirectionMultiplyValue;
  const Valueofpxd = (OutofplaneLoadLateralLoadXdirectionValue * DLF) + (0.05 * PadEyeDesignLoadvalue);
  const ValueofShearStressatBaseActualLateralXdirection = (Valueofpxd * 9810 / asb).toFixed(2);

  if (ValueofShearStressatBaseActualLateralXdirection < AllowableShearStress) {
    var ShearStressCheckatBaseXdirection = <p>OK</p>;
  } else {
    var ShearStressCheckatBaseXdirection = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }

  // Combined Stress Check as per AISC H2-1
  const combinedStressCheckasperAISCH21 = (ValueofTensileStressatBaseActual / AllowableTensileStress) + (ValueofBendingStressatBaseWeldActualaboutXAxisInPlane / AllowableBendingStressInPlane) + (ValueofBendingStressActualaboutYaxisOutofPlane / AllowableBendingStressOutofPlane)
  if (combinedStressCheckasperAISCH21 < 1) {
    var combinedStressCheckasperAISCH21Condition = <p>OK</p>;
  } else {
    var combinedStressCheckasperAISCH21Condition = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }


  //Combined Stress Check at Base
  const squarerootof2 = Math.sqrt(2)
  const twt = internalBaseWeldLegSize / squarerootof2;
  const Lw = (2 * L) + (2 * T)
  const Asbw = twt * Lw
  const Valueofmultiply = Math.cos(AngleofLoadwithVerticalvalue * Math.PI / 180);
  const ValueLoadinVerticalDirectionZdirectionValue = LoadonPadeye * Valueofmultiply;
  const Pyd = ValueLoadinVerticalDirectionZdirectionValue * DLF;
  const ValueofTensileStressatBaseWeldActual = (Pyd * 9810) / Asbw

  const ValueofLoadinHorizontalDirectionYdirectionMultiple = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.cos(OutofPlaneAngleValue * pi / 180);
  const ValueofLoadinHorizontalDirectionYdirectionValue = LoadonPadeyeValueInMT * ValueofLoadinHorizontalDirectionYdirectionMultiple;
  const ValueoofPyd = ValueofLoadinHorizontalDirectionYdirectionValue * DLFValue
  const ValueofShearStressatBaseWeldActualHorizontalYdirection = ((ValueoofPyd * 9810) / Asbw).toFixed(2)


  const ValueofOutofplaneLoadLateralLoadXdirectionMultiply = Math.sin(AngleofLoadwithVerticalvalue * pi / 180) * Math.sin(OutofPlaneAngleValue * pi / 180)
  const ValueofOutofplaneLoadLateralLoadXdirectionValue = LoadonPadeye * ValueofOutofplaneLoadLateralLoadXdirectionMultiply;
  const Pxd = (ValueofOutofplaneLoadLateralLoadXdirectionValue * DLF) + (0.05 * PadEyeDesignLoadvalue)
  const ValueofShearStressatBaseWeldActualLateralXdirection = (Pxd * 9810) / Asbw

  const sigmaSumSquared = Math.pow(ValueofTensileStressatBaseWeldActual - (- ValueofBendingStressatBaseWeldActualaboutXAxisInPlane) - (-ValueofBendingStressActualaboutYaxisOutofPlane), 2);
  const tauSbwxSquared = Math.pow(ValueofShearStressatBaseWeldActualHorizontalYdirection, 2);
  const tauSbwySquared = Math.pow(ValueofShearStressatBaseWeldActualLateralXdirection, 2);
  const CombinedStressCheckatBase = Math.sqrt(sigmaSumSquared + tauSbwxSquared + tauSbwySquared);

  if (CombinedStressCheckatBase < AllowableWeldStress) {
    var CombinedStressCheckatBaseCondition = <p>OK</p>;
  } else {
    var CombinedStressCheckatBaseCondition = <p style={{ color: '#bd2323' }}>NOT OK</p>;
  }
  return (
    <>
      <Helmet>
        <title>Pad Eye Design Calculator – OOK | Lifting Point Analysis</title>
        <meta
          name="description"
          content="Design and analyze pad eyes with confidence. Calculate design loads, allowable stresses, and geometry constraints for lifting points according to industry standards. Essential for offshore and structural rigging engineering professionals."
        />
        <link rel="canonical" href="https://www.ook-calculator.com/PadEye" />
      </Helmet>

      <div className='Background-Black'></div>
      <section className='background-white PadEye'>
        <div className="position-relative">
          {/* Image section */}
          <div className="position-relative overflow-hidden" style={{ height: '85vh' }}>
            <picture>
              <source type="image/webp" srcSet={backgroundWebP} />
              <img
                loading="lazy"
                src={backgroundJPG}
                alt="Offshore Rigging Padeye Design and Load Analysis Diagram"
                className="h-100"
                width="600"
                height="400"
                fetchpriority="high"
                decoding="async"
                style={{
                  objectFit: 'cover',
                  objectPosition: 'center',
                  width: '100%',
                  transform: 'translateX(0%)'
                }}
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
            <h1 className="display-4" style={{ fontWeight: '600' }}>Pad-Eye Calculator</h1>
            <p className="fs-5">
              Padeye calculator is a tool used in engineering and<br /> construction to determine the required dimensions and<br /> specifications for padeyes, which are integral for lifting<br /> and rigging systems.
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
              OOK Pad-Eye Calculator
            </h2>

            <div className={`content ${expanded ? "expanded" : ""}`}>
              <p className="first-important lead mb-4" style={{ fontWeight: "500" }}>
                Calculator help us to design a pad-eye to be used for<br /> lifting during transport.
              </p>
              <p className="second-important lead mb-4" style={{ fontWeight: "500" }}>
                Calculator considers a number of factors, to make sure the Padeye <br />can safely support the intended load without failing, <br />including the weight of the load, the angle of lift,<br /> the material and thickness of the structure,<br /> and safety considerations.
              </p>
              <p className=" lead" style={{ fontWeight: "500" }}>
                By using this calculator you can easily find the beam's reactions,<br />  maximum deflection, bending moment & shear stress.
              </p>
            </div>

            <hr className="Beam-properties-calculator-hr" />
            <br />

            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column' }}>
              <h3 className={`calculator-defination-section   Beam-properties-calculator-heading ${Sectionthird ? 'expanded' : ''}`} style={{ color: '#000', fontSize: '1.8vw' }}>
                <span style={{ fontFamily: 'none' }}>I</span>t can be used for the design of a <br />standard Pad-eye with:
              </h3>
              <br />
              <ul style={{ listStylePosition: "outside", paddingLeft: "20px" }}>


                <li>
                  <p className='calculator-defination-section lead ' style={{ textAlign: 'left', width: '100%', fontWeight: '500', color: '#1d1d1dbf', lineHeight: 'inherit', }}>
                    No brackets
                  </p>
                </li>
                <li>

                  <p className='calculator-defination-section lead ' style={{ textAlign: 'left', width: '100%', fontWeight: '500', color: '#1d1d1dbf', lineHeight: 'inherit', }}>
                    Single cheek plate on each side
                  </p>
                </li>
              </ul>
              <br />
              <br />
              <p className='calculator-defination-section  ' style={{ color: '#000', fontSize: '1.8vw' }}>
                <span style={{ fontFamily: 'none' }}>I</span>t makes the following checks:
              </p>
              <br />

              <ul>


                <li>
                  <p className='calculator-defination-section lead ' style={{ textAlign: 'left', width: '100%', fontWeight: '500', color: '#1d1d1dbf', lineHeight: 'inherit', }}>

                    <span> Geometry check: Main plate radius, Shackle clearances</span>
                  </p>
                </li>

                <li>
                  <p className='calculator-defination-section lead ' style={{ textAlign: 'left', width: '100%', fontWeight: '500', color: '#1d1d1dbf', lineHeight: 'inherit', }}>

                    <span> Stress Check for Pin Hole (Tensile, Bearing, Shear, Hertz stress)</span>
                  </p>
                </li>

                <li>
                  <p className='calculator-defination-section lead ' style={{ textAlign: 'left', width: '100%', fontWeight: '500', color: '#1d1d1dbf', lineHeight: 'inherit', }}>

                    <span> Stress check for Base Plate (Tensile, Bending, Shear, Von Mises, and Combined)</span>
                  </p>
                </li>

                <li>
                  <p className='calculator-defination-section lead ' style={{ textAlign: 'left', width: '100%', fontWeight: '500', color: '#1d1d1dbf', lineHeight: 'inherit', }}>

                    <span> Stress Check for Base Weld (Tensile, Bending, Shear, Total Stress)</span>
                  </p>
                </li>

                <li>
                  <p className='calculator-defination-section lead ' style={{ textAlign: 'left', width: '100%', fontWeight: '500', color: '#1d1d1dbf', lineHeight: 'inherit', }}>

                    <span> Shear Stress Check for Cheek Plate Weld</span>
                  </p>
                </li>
              </ul>
            </div>
            <hr className="Beam-properties-calculator-hr" />
            <br />

          </div>
        </section>

        <section className="container-fluid py-4 justify-content-center align-items-center d-flex">
          <div className="row structure-analysis-calculator-calculator">
            <div className="col-12 flex-grow-1 col-lg-3 col-md-12 col-sm-12 col-xs-12 text-center py-5 structure-analysis-calculator-calculator-left ps-0 pe-0">
              <div className="d-flex flex-column gap-0 w-100 text-center" style={{ justifyContent: 'center', alignItems: 'center' }}>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option1")}>Material Properties</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option2")}>Pad-eye Geometry</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option3")}>Shackle Geometry</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option4")}>Sling Geometry</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option5")}>Pad-eye Load </button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option6")}>Weld Size</button>
              </div>
            </div>


            <div className="col-12 flex-grow-2  col-lg-5 col-md-12 col-sm-12 col-xs-12 text-center py-3 structure-analysis-calculator-calculator-center bg-white justify-content-center align-items-center d-flex" >
              {selectedOption === 'option1' && <img src={MaterialpropertyImg} alt="Material Properties" className="img-fluid" />}
              {selectedOption === 'option2' && <img src={PadeyegeometryImg} alt="Pad-eye Geometry" className="img-fluid" />}
              {selectedOption === 'option3' && <img src={shackleImg} alt="Shackle Geometry" className="img-fluid" />}
              {selectedOption === 'option4' && <img src={Slingimg} alt="Sling Geometry" className="img-fluid" />}
              {selectedOption === 'option5' && <img src={PadeyeLoadimg} alt="Pad-eye Load " className="img-fluid" />}
              {selectedOption === 'option6' && <img src={WeldSizeimg} alt="Weld Size" className="img-fluid" />}
            </div>

            <div className="col-12 flex-grow-1  col-lg-3 col-md-12 col-sm-12 col-xs-12 text-center py-3 bemProperties structure-analysis-calculator-calculator-right PadeyeInputs" >
              <h2 className="text-white mt-3">Input</h2>
              <div className="mt-3">
                {selectedOption === 'option1' && (
                  <>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Material Yield Stress</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>
                          <span className='power' style={{ top: '-3px' }}>σ</span>
                          <span className='LowerPower'>y</span>
                        </p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={MaterialYieldStresValue}
                            onChange={((e) => handleMaterialYieldStresChange(e.target.value))}
                            aria-label="Material Yield Stress"
                          />
                          <select
                            className='Calculator-select-option'
                            value={MaterialYieldStresselectedUnit}
                            onChange={((e) => handleMaterialYieldStreschange(e.target.value))}
                          >
                            {MaterialYieldStressunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Electrode Tensile Strength
                        <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (for Weld)</span>
                      </p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>
                          <span className='power' style={{ top: '-3px' }}>σ</span>
                          <span className='LowerPower'>u</span>
                        </p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={ElectrodeTensileStrength}
                            onChange={((e) => handleElectrodeTensileStrength(e.target.value))}
                            aria-label="Electrode Tensile Strength"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ElectrodeTensileStrengthSelectedUnit}
                            onChange={((e) => handleElectrodeTensileStrengthSelectedUnit(e.target.value))}
                          >
                            {ElectrodeTensileStrengthunits.map((Unit) => (
                              <option key={Unit} value={Unit}>
                                {Unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Modulus of Elasticity</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>E</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={ModulusofElasticityValue}
                            onChange={((e) => handleModulusofElasticChange(e.target.value))}
                            aria-label="Modulus of Elasticity"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ModulusofElasticitySelectedUnit}
                            onChange={((e) => handleModulusofElasticUnitChange(e.target.value))}
                          >
                            {ModulusofElasticityunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Poisson's Ratio</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>V</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={PoissonsRatio}
                            onChange={((e) => handlePoissonsRatioChange(e.target.value))}
                            aria-label="Poisson's Ratio"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ModulusofElasticitySelectedUnit}
                            onChange={((e) => handleModulusofElasticUnitChange(e.target.value))}
                            style={{ opacity: '0' }}
                          >
                            {ModulusofElasticityunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-Padeye-MaterialProperty'>
                      <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={handleCombinedClick}>{isActive3 ? 'Hide' : 'Solve'}</button>
                    </div>
                  </>
                )}
                {selectedOption === 'option2' && (
                  <>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Radius of Main Plate</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>R</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={RadiusofMainPlateValue}
                            onChange={((e) => handleRadiusofMainPlateInputChange(e.target.value))}
                            aria-label="Radius of Main Plate"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RadiusofMainPlateselectedUnit}
                            onChange={((e) => handleRadiusofMainPlateUnitChange(e.target.value))}
                          >
                            {RadiusofMainPlateunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Thickness of Main Plate</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>t</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={ThicknessofMainPlateValue}
                            onChange={((e) => handleThicknessofMainPlateValue(e.target.value))}
                            aria-label="Thickness of Main Plate"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ThicknessofMainPlateSelectedUnit}
                            onChange={((e) => handleThicknessofMainPlateSelectedUnit(e.target.value))}
                          >
                            {ThicknessofMainPlateunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Diameter of eye pin hole </p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>De</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={DiameterofeyepinholeValue}
                            onChange={((e) => handleDiameterofeyepinholeChange(e.target.value))}
                            aria-label="Diameter of eye pin hole "
                          />
                          <select
                            className='Calculator-select-option'
                            value={DiameterofeyepinholeselectedUnit}
                            onChange={((e) => handleDiameterofeyepinholeUnitChange(e.target.value))}
                          >
                            {Diameterofeyepinholeunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Diameter of Cheek Plate</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>Dc</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={DiameterofCheekPlateInputValue}
                            onChange={((e) => handleDiameterofCheekPlateInputValue(e.target.value))}
                            aria-label="Diameter of Cheek Plate"
                          />
                          <select
                            className='Calculator-select-option'
                            value={DiameterofCheekPlateSelectedUnit}
                            onChange={((e) => handleDiameterofCheekPlateSelectedUnit(e.target.value))}
                          >
                            {DiameterofCheekPlateunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Thickness of Cheek Plate</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>tc</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={ThicknessofCheekPlateInputValue}
                            onChange={((e) => handleThicknessofCheekPlateInputValue(e.target.value))}
                            aria-label="Thickness of Cheek Plate"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ThicknessofCheekPlateSelectedUnit}
                            onChange={((e) => handleThicknessofCheekPlateSelectedUnit(e.target.value))}
                          >
                            {ThicknessofCheekPlateunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Total Height of Pad-eye</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>H</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={inputTotalHeightofPadeyeValue}
                            onChange={((e) => handleInputTotalHeightofPadeyeChange(e.target.value))}
                            aria-label="Total Height of Pad-eye"
                          />
                          <select
                            className='Calculator-select-option'
                            value={selectedTotalHeightofPadeyeUnit}
                            onChange={((e) => handleUnitTotalHeightofPadeyeChange(e.target.value))}
                          >
                            {TotalHeightofPadeyeunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Length of Base Plate</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>L</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={LengthofBasePlateInputValue}
                            onChange={((e) => handleLengthofBasePlateInputValue(e.target.value))}
                            aria-label="Length of Base Plate"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LengthofBasePlateSelectedUnit}
                            onChange={((e) => handleLengthofBasePlateSelectedUnit(e.target.value))}
                          >
                            {LengthofBasePlateunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-Padeye-Geomtry'>
                      <br />
                      <button className='structure-analysis-calculator-calculator-right-show-hidden-btn' onClick={handleCombinedClick}>{isActive3 ? 'Hide' : 'Solve'}</button>
                    </div>
                  </>
                )}
                {selectedOption === 'option3' && (
                  <>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Shackle SWL</p>
                      <p className='sigma-symbol' style={{ color: '#fff', marginRight: '10px' }}>(
                        <span className='power' style={{ top: '0px' }}>SWL</span>
                        <span className='LowerPower' style={{ fontSize: '0.5vw', top: '3px', position: 'relative' }} >sh</span>
                        )
                      </p>
                      <div className='Calculator-Side-A' style={{ width: '52%' }}>
                        <br />
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={ShackleSWL}
                            onChange={(e) => handleShackleSWLvalue(e.target.value)}
                            aria-label="Shackle SWL"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ShackleSWLUnit}
                            onChange={(e) => handleShackleSWLUnit(e.target.value)}
                          >
                            {ShackleSWLUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Shackle Inside Length</p>
                      <p className='sigma-symbol' style={{ color: '#fff', marginRight: '10px' }}>(
                        <span className='power' style={{ top: '0px' }}>L</span>
                        <span className='LowerPower' style={{ fontSize: '0.5vw', top: '3px', position: 'relative' }}>sh</span>
                        )
                      </p>
                      <div className='Calculator-Side-A' style={{ width: '52%' }}>
                        <br />
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={ShackleInsideLength}
                            onChange={(e) => handleShackleInsideLengthvalue(e.target.value)}
                            aria-label="Shackle Inside Length"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ShackleInsideLengthUnit}
                            onChange={(e) => handleShackleInsideLengthUnit(e.target.value)}
                          >
                            {ShackleInsideLengthUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Shackle Jaw Width</p>
                      <p className='sigma-symbol' style={{ color: '#fff', marginRight: '10px' }}>(
                        <span className='power' style={{ top: '0px' }}>W</span>
                        <span className='LowerPower' style={{ fontSize: '0.5vw', top: '3px', position: 'relative' }}>sh</span>
                        )
                      </p>
                      <div className='Calculator-Side-A' style={{ width: '52%' }}>
                        <br />
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={ShackleJawWidth}
                            onChange={(e) => handleShackleJawWidthvalue(e.target.value)}
                            aria-label="Shackle Jaw Width"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ShackleJawWidthUnit}
                            onChange={(e) => handleShackleJawWidthUnit(e.target.value)}
                          >
                            {ShackleJawWidthUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Shackle Pin Diameter</p>
                      <p className='sigma-symbol' style={{ color: '#fff', marginRight: '10px', }}>(
                        <span className='power' style={{ top: '0px' }}>D</span>
                        <span className='LowerPower' style={{ fontSize: '0.5vw', top: '3px', position: 'relative' }}>p</span>
                        )
                      </p>
                      <div className='Calculator-Side-A' style={{ width: '52%' }}>
                        <br />
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={ShacklePinDiameter}
                            onChange={(e) => handleShacklePinDiametervalue(e.target.value)}
                            aria-label="Shackle Pin Diameter"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ShacklePinDiameterUnit}
                            onChange={(e) => handleShacklePinDiameterUnit(e.target.value)}
                          >
                            {ShacklePinDiameterUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-Padeye-Shackle-Geometry'>
                      <button className='structure-analysis-calculator-calculator-right-show-hidden-btn' onClick={handleCombinedClick}>{isActive3 ? 'Hide' : 'Solve'}</button>
                    </div>
                  </>
                )}
                {selectedOption === 'option4' && (
                  <>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Sling Diameter</p>
                      <div className='Calculator-Side-A' style={{ width: '65%' }}>
                        <br />
                        <p className='sigma-symbol'>
                          <span className='power' style={{ top: '-3px' }}>D</span>
                          <span className='LowerPower' style={{ fontSize: '0.55vw' }}>sling</span>
                        </p>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={SlingDiameter}
                            onChange={(e) => handleSlingDiametervalue(e.target.value)}
                            aria-label="Sling Diameter"
                          />
                          <select
                            className='Calculator-select-option'
                            value={SlingDiameterUnit}
                            onChange={(e) => handleSlingDiameterUnit(e.target.value)}
                          >
                            {SlingDiameterUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-Padeye-Sling'>
                      <button className='structure-analysis-calculator-calculator-right-show-hidden-btn' onClick={handleCombinedClick}>{isActive3 ? 'Hide' : 'Solve'}</button>
                    </div>
                  </>
                )}
                {selectedOption === 'option5' && (
                  <>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Load on Pad-eye</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>P</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={LoadonPadeyeValue}
                            onChange={(e) => handleLoadonPadeyeValueChange(e.target.value)}
                            aria-label="Load on Pad-eye"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LoadonPadeyeSelectedUnit}
                            onChange={(e) => handleLoadonPadeyeUnitChange(e.target.value)}
                          >
                            {LoadonPadeyeunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Angle of Load with Vertical</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>θ</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={AngleofLoadwithVerticalvalue}
                            onChange={(e) => handleAngleofLoadwithVerticalvalue(e.target.value)}
                            aria-label="Angle of Load with Vertical"
                          />
                          <select
                            className='Calculator-select-option'
                            value={AngleofLoadwithVerticalSelectedUnit}
                            onChange={(e) => handleAngleofLoadwithVerticalSelectedUnit(e.target.value)}
                          >
                            {AngleofLoadwithVerticalunit.map((Unit) => (
                              <option key={Unit} value={Unit}>
                                {Unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Out-of Plane Angle</p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>
                          <span className='puiSignUpperI' style={{}}>I</span>
                          <span className='puiSignO' style={{}}>O</span>
                          <span className='puiSignlowerI' style={{}}>I</span></p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={OutofPlaneAngleValue}
                            onChange={(e) => handleOutofPlaneAngleValueStresChange(e.target.value)}
                            aria-label="Out-of Plane Angle"
                          />
                          <select
                            className='Calculator-select-option'
                            value={OutofPlaneAngleselectedUnit}
                            onChange={(e) => handleOutofPlaneAngleunitchange(e.target.value)}
                          >
                            {OutofPlaneAngleUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Dynamic Load Factor </p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>DLF</p>
                        <div className='input-and-select-div'>
                          <input
                            style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={DLFValue}
                            onChange={(e) => handleDLFValue(e.target.value)}
                            aria-label="Dynamic Load Factor "
                          />
                          <select
                            className='Calculator-select-option'
                            value={OutofPlaneAngleselectedUnit}
                            onChange={(e) => handleOutofPlaneAngleunitchange(e.target.value)}
                            style={{ opacity: '0' }}
                          >
                            {OutofPlaneAngleUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-Padeye-Load'>
                      <button className='structure-analysis-calculator-calculator-right-show-hidden-btn' onClick={handleCombinedClick}>{isActive3 ? 'Hide' : 'Solve'}</button>
                    </div>
                  </>
                )}
                {selectedOption === 'option6' && (
                  <>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Base Weld Leg Size
                      </p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>
                          <span className='power' style={{ top: '-3px' }}>t</span>
                          <span className='LowerPower'>w</span>
                        </p>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={BaseWeldLegSize}
                            onChange={((e) => handleBaseWeldLegSizevalue(e.target.value))}
                            aria-label="Base Weld Leg Size"
                          />
                          <select
                            className='Calculator-select-option'
                            value={BaseWeldLegSizeUnit}
                            onChange={((e) => handleBaseWeldLegSizeUnit(e.target.value))}
                          >
                            {BaseWeldLegSizeUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>Cheek Plate Weld Leg Size
                      </p>
                      <div className='Calculator-Side-A'>
                        <br />
                        <p className='sigma-symbol'>
                          <span className='power' style={{ top: '-3px' }}>t</span>
                          <span className='LowerPower'>wc</span>
                        </p>
                        <div className='input-and-select-div'>
                          <input style={{ transform: 'translate(5px, 0px)' }}
                            className='calculator-input'
                            type="number"
                            value={CheekPlateWeldLegSize}
                            onChange={(e) => handleCheekPlateWeldLegSizevalue(e.target.value)}
                            aria-label="Cheek Plate Weld Leg Size"
                          />
                          <select
                            className='Calculator-select-option'
                            value={CheekPlateWeldLegSizeUnit}
                            onChange={(e) => handleCheekPlateWeldLegSizeUnit(e.target.value)}
                          >
                            {CheekPlateWeldLegSizeUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div >
                    <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-Padeye-WeldSize'>
                      <button className='structure-analysis-calculator-calculator-right-show-hidden-btn' onClick={handleCombinedClick}>{isActive3 ? 'Hide' : 'Solve'}</button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </section>
        <div className={`Grid-of-padeye-solutions mt-5 ${showfirstDiv ? 'ScrollTransactionone' : ''}  `}>
          <div className={DropDowmOneMain} style={{
            height: '45vw',
            left: '0'
          }}>
            <h2 className='text-center  text-white'>ALLOWABLE STRESSES & DESIGN LOADS</h2>
            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Allowable Stresses:</p>
            <br />
            <div className='' style={{
              borderRadius: '10px',
            }}>      <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Allowable Bearing Stress</p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>be(allow)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={AllowableBearingStressValue}
                      readOnly
                      aria-label="Allowable Bearing Stress"
                    />
                    <select
                      className='Calculator-select-option'
                      value={AllowableBearingStressselectedUnit}
                      onChange={(e) => handleAllowableBearingStressUnitChange(e.target.value)}
                    >
                      {AllowableBearingStressunits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>


              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Allowable Bending Stress
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (In-Plane)</span>
                </p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>bd(allow)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={AllowableBendingStressInPlaneValue}
                      readOnly
                      aria-label="Allowable Bending Stress In-Plane"
                    />
                    <select
                      className='Calculator-select-option'
                      value={AllowableBendingStressInPlaneselectedUnit}
                      onChange={(e) => handleAllowableBendingStressInPlaneUnitChange(e.target.value)}
                    >
                      {AllowableBendingStressInPlaneunits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Allowable Bending Stress
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out of-plane)</span>

                </p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>bdo(allow)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={AllowableBendingStressOutofPlaneValue}
                      readOnly
                      aria-label="Allowable Bending Stress Out-of-Plane"
                    />
                    <select
                      className='Calculator-select-option'
                      value={AllowableBendingStressOutofPlaneselectedUnit}
                      onChange={(e) => handleAllowableBendingStressOutofPlaneselectedUnitChange(e.target.value)}
                    >
                      {AllowableBendingStressOutofPlaneunits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Allowable Tensile Stress</p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>t(allow) </span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={AllowableTensileStressValue}
                      readOnly
                      aria-label="Allowable Tensile Stress"
                    />
                    <select
                      className='Calculator-select-option'
                      value={AllowableTensileStressSelectedUnit}
                      onChange={(e) => handleAllowableTensileStressselectedUnitChange(e.target.value)}
                    >
                      {AllowableTensileStressUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Allowable Tensile Stress at pin hole</p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>tp(allow)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={AllowableTensileStressatpinholeValue}
                      readOnly
                      aria-label="Allowable Tensile Stress at pin hole"
                    />
                    <select
                      className='Calculator-select-option'
                      value={AllowableTensileStressatpinholeSelectedUnit}
                      onChange={(e) => handleAllowableTensileStressatpinholeselectedUnitChange(e.target.value)}
                    >
                      {AllowableTensileStressatpinholeUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Allowable Shear Stress</p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>τ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>s(allow)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={AllowableShearStressValue}
                      readOnly
                      aria-label="Allowable Shear Stress"
                    />
                    <select
                      className='Calculator-select-option'
                      value={AllowableShearStressSelectedUnit}
                      onChange={(e) => handleAllowableShearStressselectedUnitChange(e.target.value)}
                    >
                      {AllowableShearStressUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Allowable Hertz Stress
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Pin Hole)</span>
                </p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>H(allow)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={AllowableHertzStressatPinHoleValue}
                      readOnly
                      aria-label="Allowable Hertz Stress at Pin Hole"
                    />
                    <select
                      className='Calculator-select-option'
                      value={AllowableHertzStressatPinHoleSelectedUnit}
                      onChange={(e) => handleAllowableHertzStressatPinHoleselectedUnitChange(e.target.value)}
                    >
                      {AllowableHertzStressatPinHoleUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>

                <p className='claculator-conversation-title'>Allowable Weld Stress</p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>w(allow)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={AllowableWeldStressValue}
                      readOnly
                      aria-label="Allowable Weld Stress"
                    />
                    <select
                      className='Calculator-select-option'
                      value={AllowableWeldStressSelectedUnit}
                      onChange={(e) => handleAllowableWeldStressselectedUnitChange(e.target.value)}
                    >
                      {AllowableWeldStressUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <br />
              <br />
              <p className='text-white' style={{ fontSize: '1.8vw' }}>Design Loads:</p>
              <br />

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>

                <p className='claculator-conversation-title'>Load in Vertical Direction
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Z direction)</span>
                </p>
                <div className='Calculator-Side-A '>
                  <p className='sigma-symbol'>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>P</span>

                    <span className='LowerPower'>z</span></p>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={LoadinVerticalDirectionZdirectionValue}
                      readOnly
                      aria-label="Load in Vertical Direction (Z direction)"
                    />
                    <select
                      className='Calculator-select-option'
                      value={LoadinVerticalDirectionZdirectionSelectedUnit}
                      onChange={(e) => handleLoadinVerticalDirectionZdirectionselectedUnitChange(e.target.value)}
                    >
                      {LoadinVerticalDirectionZdirectionUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>



              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Load in Horizontal Direction
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y direction)</span>
                </p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>P</span>
                    <span className='LowerPower'>y</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={LoadinHorizontalDirectionYdirection}
                      readOnly
                      aria-label="Load in Horizontal Direction (Y direction)"
                    />
                    <select
                      className='Calculator-select-option'
                      value={LoadinHorizontalDirectionYdirectionSelectedUnit}
                      onChange={(e) => handleLoadinHorizontalDirectionYdirectionselectedUnitChange(e.target.value)}
                    >
                      {LoadinHorizontalDirectionYdirectionUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>



              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Out-of-plane Load
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X direction)</span>
                </p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>P</span>
                    <span className='LowerPower'>x</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={OutofplaneLoadLateralLoadXdirection}
                      readOnly
                      aria-label="Out-of-plane Load (X direction)"
                    />
                    <select
                      className='Calculator-select-option'
                      value={OutofplaneLoadLateralLoadXdirectionSelectedUnit}
                      onChange={(e) => handleOutofplaneLoadLateralLoadXdirectionselectedUnitChange(e.target.value)}
                    >
                      {OutofplaneLoadLateralLoadXdirectionUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>


              {/* PadEyeDesignLoad   */}
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>
                  Pad-eye Design Load
                </p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>P</span>
                    <span className='LowerPower'>d</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={PadEyeDesignLoad}
                      readOnly
                      aria-label="Pad-eye Design Load"
                    />
                    <select
                      className='Calculator-select-option'
                      value={PadEyeDesignLoadSelectedUnit}
                      onChange={(e) => handlePadEyeDesignLoadselectedUnitChange(e.target.value)}
                    >
                      {PadEyeDesignLoadUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>
                  Design Load in Vertical Direction
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Z-direction)</span>
                </p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>P</span>
                    <span className='LowerPower'>zd</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={DesignLoadinVerticalDirectionZdirection}
                      readOnly
                      aria-label="Design Load in Vertical Direction (Z-direction)"
                    />
                    <select
                      className='Calculator-select-option'
                      value={DesignLoadinVerticalDirectionZdirectionSelectedUnit}
                      onChange={(e) => handleDesignLoadinVerticalDirectionZdirectionselectedUnitChange(e.target.value)}
                    >
                      {DesignLoadinVerticalDirectionZdirectionUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>
                  Design Load in Horizontal Direction
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y-direction)</span>
                </p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>P</span>
                    <span className='LowerPower'>yd</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={DesignLoadinHorizontalDirectionYdirection}
                      onChange={(e) => calculateLoadinHorizontalDirectionYdirectionValue(e.target.value)}
                      readOnly
                      aria-label="Design Load in Horizontal Direction (Y-direction)"
                    />
                    <select
                      className='Calculator-select-option'
                      value={DesignLoadinHorizontalDirectionYdirectionSelectedUnit}
                      onChange={(e) => handleDesignLoadinHorizontalDirectionYdirectionselectedUnitChange(e.target.value)}
                    >
                      {DesignLoadinHorizontalDirectionYdirectionUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>
                  Design Load Out-of-Plane
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Lateral, X-direction)</span>

                </p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>P</span>
                    <span className='LowerPower'>xd</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={DesignLoadOutofPlaneLateralXdirection}
                      readOnly
                      aria-label="Design Load Out-of-Plane (Lateral, X-direction)"
                    />
                    <select
                      className='Calculator-select-option'
                      value={DesignLoadOutofPlaneLateralXdirectionSelectedUnit}
                      onChange={(e) => handleDesignLoadOutofPlaneLateralXdirectionselectedUnitChange(e.target.value)}
                    >
                      {DesignLoadOutofPlaneLateralXdirectionUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className='Grid-of-2-smaller-padeye-solutions'>
            <div className={DropDowmOnerightMain} style={{
              height: '20vw',
              left: '50%',
              top: '115%',
              width: '38vw',
            }}>
              <h2 className='text-center text-white'>GEOMETRY CHECK</h2>
              <br />
              <div className='' style={{
                borderRadius: '10px',
              }}>
                <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <p className='claculator-conversation-title'>Clearance between pin hole dia and pin dia</p>
                  <div className='Calculator-Side-A'>
                    <br />
                    <p className='sigma-symbol'>
                      <span className='power' style={{ fontSize: '1vw', top: '-1px' }}>δD</span>
                    </p>
                    <div className='input-and-select-div'>
                      <input
                        className='calculator-input'
                        type="number"
                        value={Clearancebetweenpinholediaandpindia}
                        readOnly
                        aria-label="Clearance between pin hole dia and pin dia"
                      />
                      <select
                        className='Calculator-select-option'
                        value={ClearancebetweenpinholediaandpindiaSelectedunit}
                        onChange={(e) => handleClearancebetweenpinholediaandpindiaselectedUnitChange(e.target.value)}
                      >
                        {ClearancebetweenpinholediaandpindiaUnits.map((unit) => (
                          <option key={unit} value={unit}>
                            {unit}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
                <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <p className='claculator-conversation-title'>Length Clearance of Shackle</p>
                  <div className='Calculator-Side-A'>
                    <br />
                    <p className='sigma-symbol sigmawithshorterlowerPower'>
                      <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>δL</span>
                      <span className='LowerPower'>shackle</span>
                    </p>
                    <div className='input-and-select-div'>
                      <input
                        className='calculator-input'
                        type="number"
                        value={LengthClearanceofShackle}
                        readOnly
                        aria-label="Length Clearance of Shackle"
                      />
                      <select
                        className='Calculator-select-option'
                        value={LengthClearanceofShackleselectedunit}
                        onChange={(e) => handleLengthClearanceofShackleselectedUnitChange(e.target.value)}
                      >
                        {LengthClearanceofShackleUnits.map((unit) => (
                          <option key={unit} value={unit}>
                            {unit}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
                <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <p className='claculator-conversation-title'>Width Clearance of Shackle</p>
                  <div className='Calculator-Side-A'>
                    <br />
                    <p className='sigma-symbol sigmawithshorterlowerPower'>
                      <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>δW</span>
                      <span className='LowerPower'>shackle</span>
                    </p>
                    <div className='input-and-select-div'>
                      <input
                        className='calculator-input'
                        type="number"
                        value={WidthClearanceofShackle}
                        readOnly
                        aria-label="Width Clearance of Shackle"
                      />
                      <select
                        className='Calculator-select-option'
                        value={WidthClearanceofShackleSelectedUnit}
                        onChange={(e) => handleWidthClearanceofShackleUnitChange(e.target.value)}
                      >
                        {WidthClearanceofShackleUnits.map((unit) => (
                          <option key={unit} value={unit}>
                            {unit}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className={DropDowmOnerightsecondMain} style={{
              height: '22vw',
              width: '38vw',
            }}>
              <h2 className='text-center text-white'>WELD CHECK AT CHEEK PLATE WELD</h2>
              <br />
              <br />

              <div className=''>

                <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <p className='claculator-conversation-title'>Throat Thickness</p>
                  <div className='Calculator-Side-A'>
                    <br />
                    <p className='sigma-symbol'>

                      <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>t</span>
                      <span className='LowerPower'>wtc</span>
                    </p>
                    <div className='input-and-select-div'>
                      <input
                        className='calculator-input'
                        type="number"
                        value={WeldCheckatCheekPlateWeldThroatthickness}
                        readOnly
                        aria-label="Throat Thickness"
                      />
                      <select
                        className='Calculator-select-option'
                        value={WeldCheckatCheekPlateWeldThroatthicknessSelectedUnit}
                        onChange={(e) => handleWeldCheckatCheekPlateWeldThroatthicknessUnitChange(e.target.value)}
                      >
                        {WeldCheckatCheekPlateWeldThroatthicknessUnits.map((unit) => (
                          <option key={unit} value={unit}>
                            {unit}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>

                <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <p className='claculator-conversation-title'>Weld Length</p>
                  <div className='Calculator-Side-A'>
                    <br />
                    <p className='sigma-symbol'>

                      <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>L</span>
                      <span className='LowerPower'>wc</span>
                    </p>
                    <div className='input-and-select-div'>
                      <input
                        className='calculator-input'
                        type="number"
                        value={WeldLength}
                        readOnly
                        aria-label="Weld Length"
                      />
                      <select
                        className='Calculator-select-option'
                        value={WeldLengthSelectedUnit}
                        onChange={(e) => handleWeldLengthUnitChange(e.target.value)}
                      >
                        {WeldLengthUnit.map((unit) => (
                          <option key={unit} value={unit}>
                            {unit}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>

                <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <p className='claculator-conversation-title'>Weld Area at Cheek Plate</p>
                  <div className='Calculator-Side-A'>
                    <br />
                    <p className='sigma-symbol'>

                      <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>A</span>
                      <span className='LowerPower'>wc</span>
                    </p>
                    <div className='input-and-select-div'>
                      <input
                        className='calculator-input'
                        type="number"
                        value={WeldAreaatCheekPlate}
                        readOnly
                        aria-label="Weld Area at Cheek Plate"
                      />
                      <select
                        className='Calculator-select-option'
                        value={WeldAreaatCheekPlateSelectedUnit}
                        onChange={(e) => handleWeldAreaatCheekPlateUnitChange(e.target.value)}
                      >
                        {WeldAreaatCheekPlateUnit.map((unit) => (
                          <option key={unit} value={unit}>
                            {unit}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>

                <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <p className='claculator-conversation-title'>Design Load on one cheek plate</p>
                  <div className='Calculator-Side-A'>
                    <br />
                    <p className='sigma-symbol'>

                      <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>P</span>
                      <span className='LowerPower'>dc</span>
                    </p>
                    <div className='input-and-select-div'>
                      <input
                        className='calculator-input'
                        type="number"
                        value={DesignLoadononecheekplate}
                        readOnly
                        aria-label="Design Load on one cheek plate"
                      />
                      <select
                        className='Calculator-select-option'
                        value={DesignLoadononecheekplateSelectedUnit}
                        onChange={(e) => handleDesignLoadononecheekplateUnitChange(e.target.value)}
                      >
                        {DesignLoadononecheekplateUnit.map((unit) => (
                          <option key={unit} value={unit}>
                            {unit}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>

                <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <p className='claculator-conversation-title'>Weld Stress at Cheek Weld
                    <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
                  </p>
                  <div className='Calculator-Side-A '>
                    <p className='sigma-symbol '>
                      <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>σ</span>
                      <span className='LowerPower'>wc</span>
                    </p>
                    <br />
                    <div className='input-and-select-div'>
                      <input
                        className='calculator-input'
                        type="number"
                        value={WeldStressatCheekWeldActual}
                        readOnly
                        aria-label="Weld Stress at Cheek Weld (Act)"
                      />
                      <select
                        className='Calculator-select-option'
                        value={WeldStressatCheekWeldActualSelectedUnit}
                        onChange={(e) => handleWeldStressatCheekWeldActualUnitChange(e.target.value)}
                      >
                        {WeldStressatCheekWeldActualUnit.map((unit) => (
                          <option key={unit} value={unit}>
                            {unit}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className={`Grid-of-padeye-solutions mt-5 BothStressCheck ${showfirstDiv ? 'ScrollTransactionone' : ''} ${showSecondDiv ? 'ScrollTransactionTwo' : ''} `}>
          <div className={DropDowmTwoMain} style={{
            height: '45vw',
            left: '0%',
            width: '38vw',
            top: '290%',
            zIndex: '1'
          }}>
            <h2 className='text-center text-white'>STRESS CHECKS AT PIN HOLE</h2>
            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Bearing Stress Check at Pin Hole:</p>
            <br />
            <div className=''>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Bearing Area</p>
                <div className='Calculator-Side-A'>
                  <p className='sigma-symbol'>

                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>A</span>
                    <span className='LowerPower'>be</span>
                  </p>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={BearingArea}
                      readOnly
                      aria-label="Bearing Area"
                    />
                    <select
                      className='Calculator-select-option'
                      value={BearingAreaSelectedUnit}
                      onChange={(e) => handleBearingAreaUnitChange(e.target.value)}
                    >
                      {BearingAreaUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Bearing Stress
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                </p>
                <div className='Calculator-Side-A ' style={{ width: '40%' }}>
                  <p className='sigma-symbol '>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>σ</span>
                    <span className='LowerPower'>be</span>
                  </p>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={BearingStressActual}
                      readOnly
                      aria-label="Bearing Stress (Act)"
                    />
                    <select
                      className='Calculator-select-option'
                      value={BearingStressActualSelectedUnit}
                      onChange={(e) => handleBearingStressActualUnitChange(e.target.value)}
                    >
                      {BearingStressActualUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <br />
              <br />
              <p className='text-white' style={{ fontSize: '1.8vw' }}>Shear Stress Check at Pin Hole:</p>
              <br />

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Radius of Cheek Plate</p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>

                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>R</span>
                    <span className='LowerPower'>c</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={RadiusofCheekPlate}
                      readOnly
                      aria-label="Radius of Cheek Plate"
                    />
                    <select
                      className='Calculator-select-option'
                      value={RadiusofCheekPlateSelectedUnit}
                      onChange={(e) => handleRadiusofCheekPlateUnitChange(e.target.value)}
                    >
                      {RadiusofCheekPlateUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Shear Area of pin hole</p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>

                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>A</span>
                    <span className='LowerPower'>sp</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={ShearAreaofpinhole}
                      readOnly
                      aria-label="Shear Area of pin hole"
                    />
                    <select
                      className='Calculator-select-option'
                      value={ShearAreaofpinholeSelectedUnit}
                      onChange={(e) => handleShearAreaofpinholeUnitChange(e.target.value)}
                    >
                      {ShearAreaofpinholeUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Shear Stress at pin hole</p>
                <p className='sigma-symbol outer-sigma-symbol'>

                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>sp(act)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={ShearStressatpinhole}
                      readOnly
                      aria-label="Shear Stress at pin hole"
                    />
                    <select
                      className='Calculator-select-option'
                      value={ShearStressatpinholeSelectedUnit}
                      onChange={(e) => handleShearStressatpinholeUnitChange(e.target.value)}
                    >
                      {ShearStressatpinholeUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <br />
              <br />
              <p className='text-white' style={{ fontSize: '1.8vw' }}>Tensile Stress Check at Pin Hole:</p>
              <br />

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title' >Tensile Area for section
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> A-A </span>
                </p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>

                    <span className='power' style={{ fontSize: '1vw !important', top: '-3px' }}>A</span>
                    <span className='LowerPower'>t1p</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={TensileAreaforsectionAA}
                      readOnly
                      aria-label="Tensile Area for section A-A"
                    />
                    <select
                      className='Calculator-select-option'
                      value={TensileAreaforsectionAASelectedUnit}
                      onChange={(e) => handleTensileAreaforsectionAAUnitChange(e.target.value)}
                    >
                      {TensileAreaforsectionAAUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title' style={{ fontSize: '1.1vw' }}>Tensile Stress at pin hole
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                  at section
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> A-A </span>
                </p>
                <div className='Calculator-Side-A '>
                  <p className='sigma-symbol '>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>σ</span>
                    <span className='LowerPower' style={{ position: 'relative', top: '0px', }}>tp</span>
                  </p>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={TensileStressatpinholeActualatsectionAA}
                      readOnly
                      aria-label="Tensile Stress at pin hole (Act) at section A-A"
                    />
                    <select
                      className='Calculator-select-option'
                      value={TensileStressatpinholeActualatsectionAASelectedUnit}
                      onChange={(e) => handleTensileStressatpinholeActualatsectionAAUnitChange(e.target.value)}
                    >
                      {TensileStressatpinholeActualatsectionAAUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>



              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Tensile Area for section
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> B-B </span>
                </p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>

                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>A</span>
                    <span className='LowerPower'>t2p</span>
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={TensileAreaforsectionBB}
                      readOnly
                      aria-label="Tensile Area for section B-B"
                    />
                    <select
                      className='Calculator-select-option'
                      value={TensileAreaforsectionBBSelectedUnit}
                      onChange={(e) => handleTensileAreaforsectionBBUnitChange(e.target.value)}
                    >
                      {TensileAreaforsectionBBUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>



              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title' style={{ fontSize: '1.1vw' }}>Tensile Stress at pin hole
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                  at section
                  <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> B-B </span>
                </p>
                <div className='Calculator-Side-A'>
                  <p className='sigma-symbol '>
                    <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>σ</span>
                    <span className='LowerPower' style={{ position: 'relative', top: '0px', }}>tp</span>
                  </p>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={TensileStressatpinholeActualatsectionBB}
                      readOnly
                      aria-label="Tensile Stress at pin hole (Act) at section B-B"
                    />
                    <select
                      className='Calculator-select-option'
                      value={TensileStressatpinholeActualatsectionBBSelectedUnit}
                      onChange={(e) => handleTensileStressatpinholeActualatsectionBBUnitChange(e.target.value)}
                    >
                      {TensileStressatpinholeActualatsectionBBUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <br />
              <br />

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <img style={{ width: '20vw', height: '18vw' }} src={Img} alt='' />
              </div>
              <br />
              <br />
              <p className='text-white' style={{ fontSize: '1.8vw' }}>Hertz/Contact Stress Check at Pin Hole:</p>
              <br />

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Design Load per unit Length</p>
                <div className='Calculator-Side-A'>
                  <br />
                  <p className='sigma-symbol'>
                    P
                  </p>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={DesignLoadperunitLength}
                      readOnly
                      aria-label="Design Load per unit Length"
                    />
                    <select
                      className='Calculator-select-option'
                      value={DesignLoadperunitLengthSelectedUnit}
                      onChange={(e) => handleDesignLoadperunitLengthUnitChange(e.target.value)}
                    >
                      {DesignLoadperunitLengthUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>


              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <p className='claculator-conversation-title'>Hertz Stress at Pin Hole</p>
                <p className='sigma-symbol outer-sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px', color: 'white' }}>σ</span>
                  <span className='LowerPower' style={{ color: 'white' }}>H(act)</span>
                </p>
                <div className='Calculator-Side-A outer-sigma-symbol-input'>
                  <br />
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={HertzStressatPinHole}
                      readOnly
                      aria-label="Hertz Stress at Pin Hole"
                    />
                    <select
                      className='Calculator-select-option'
                      value={HertzStressatPinHoleSelectedUnit}
                      onChange={(e) => handleHertzStressatPinHoleUnitChange(e.target.value)}
                    >
                      {HertzStressatPinHoleUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className={DropDowmTworightMain} style={{
            height: '45vw',
            width: '38vw',
            zIndex: ' 0',
          }}>
            <h2 className='text-center text-white'>STRESS CHECKS AT BASE PLATE</h2>
            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Tensile Stress Check:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Tensile Area</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>A</span>
                  <span className='LowerPower'>tb</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TensileArea}
                    readOnly
                    aria-label="Tensile Area"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TensileAreaSelectedUnit}
                    onChange={(e) => handleTensileAreaUnitChange(e.target.value)}
                  >
                    {TensileAreaUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Tensile Stress at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}>(Act)</span>
              </p>
              <div className='Calculator-Side-A '>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>σ</span>
                  <span className='LowerPower'>tb</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TensileStressatBaseActual}
                    readOnly
                    aria-label="Tensile Stress at Base (Act)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TensileStressatBaseActualSelectedUnit}
                    onChange={(e) => handleTensileStressatBaseActualUnitChange(e.target.value)}
                  >
                    {TensileStressatBaseActualUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Bending Stress Check:</p>
            <br />
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Moment lever for Horizontal/Lateral Force</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol' style={{ fontSize: '1vw' }}>
                  h
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={MomentleverforHorizontalLateralForce}
                    readOnly
                    aria-label="Moment lever for Horizontal/Lateral Force"
                  />
                  <select
                    className='Calculator-select-option'
                    value={MomentleverforHorizontalLateralForceSelectedUnit}
                    onChange={(e) => handleMomentleverforHorizontalLateralForceUnitChange(e.target.value)}
                  >
                    {MomentleverforHorizontalLateralForceUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Moment lever for Vertical Force</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol' style={{ fontSize: '1vw' }}>
                  e
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={MomentleverforVerticalForce}
                    readOnly
                    aria-label="Moment lever for Vertical Force"
                  />
                  <select
                    className='Calculator-select-option'
                    value={MomentleverforVerticalForceSelectedUnit}
                    onChange={(e) => handleMomentleverforVerticalForceUnitChange(e.target.value)}
                  >
                    {MomentleverforVerticalForceUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Total Design Moment about X-axis
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (In-plane)</span>
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>M</span>
                  <span className='LowerPower'>dxx</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TotalDesignMomentaboutXaxisInplane}
                    readOnly
                    aria-label="Total Design Moment about X-axis (In-plane)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TotalDesignMomentaboutXaxisInplaneSelectedUnit}
                    onChange={(e) => handleTotalDesignMomentaboutXaxisInplaneUnitChange(e.target.value)}
                  >
                    {TotalDesignMomentaboutXaxisInplaneUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>



            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Section Modulus of Base about X-axis</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>Z</span>
                  <span className='LowerPower'>dxx</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={SectionModulusofBaseaboutXaxis}
                    readOnly
                    aria-label="Section Modulus of Base about X-axis"
                  />
                  <select
                    className='Calculator-select-option'
                    value={SectionModulusofBaseaboutXaxisSelectedUnit}
                    onChange={(e) => handleSectionModulusofBaseaboutXaxisUnitChange(e.target.value)}
                  >
                    {SectionModulusofBaseaboutXaxisUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Bending Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
                - about X-axis
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (In-plane)</span>
              </p>
              <div className='Calculator-Side-A '>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>σ</span>
                  <span className='LowerPower'>bbx</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={BendingStressActualaboutXaxisInplane}
                    readOnly
                    aria-label="Bending Stress about X-axis (In-plane) (Act)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={BendingStressActualaboutXaxisInplaneSelectedUnit}
                    onChange={(e) => handleBendingStressActualaboutXaxisInplaneUnitChange(e.target.value)}
                  >
                    {BendingStressActualaboutXaxisInplaneUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Total Design Moment about Y-axis
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out-of Plane)</span>
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>M</span>
                  <span className='LowerPower'>dyy</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TotalDesignMomentaboutYaxisOutofPlane}
                    readOnly
                    aria-label="Total Design Moment about Y-axis (Out-of Plane)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TotalDesignMomentaboutYaxisOutofPlaneSelectedUnit}
                    onChange={(e) => handleTotalDesignMomentaboutYaxisOutofPlaneUnitChange(e.target.value)}
                  >
                    {TotalDesignMomentaboutYaxisOutofPlaneUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>



            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Section Modulus of Base about Y-axis</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>Z</span>
                  <span className='LowerPower'>byy</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={SectionModulusofBaseaboutYaxis}
                    readOnly
                    aria-label="Section Modulus of Base about Y-axis"
                  />
                  <select
                    className='Calculator-select-option'
                    value={SectionModulusofBaseaboutYaxisSelectedUnit}
                    onChange={(e) => handleSectionModulusofBaseaboutYaxisUnitChange(e.target.value)}
                  >
                    {SectionModulusofBaseaboutYaxisUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>




            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Bending Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
                - about Y-axis
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out-of-Plane)</span>
              </p>
              <div className='Calculator-Side-A '>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw !important', top: '-3px' }}>σ</span>
                  <span className='LowerPower'>bby</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={BendingStressActualaboutYaxisOutofPlane}
                    readOnly
                    aria-label="Bending Stress about Y-axis (Out-of-Plane) (Act)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={BendingStressActualaboutYaxisOutofPlaneSelectedUnit}
                    onChange={(e) => handleBendingStressActualaboutYaxisOutofPlaneUnitChange(e.target.value)}
                  >
                    {BendingStressActualaboutYaxisOutofPlaneUnit.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Shear Stress Check at Base:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Shear Area at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Horizontal)</span>
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>A</span>
                  <span className='LowerPower'>sb</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={ShearAreaatBaseHorizontal}
                    readOnly
                    aria-label="Shear Area at Base (Horizontal)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={ShearAreaatBaseHorizontalSelectedUnit}
                    onChange={(e) => handleShearAreaatBaseHorizontalUnitChange(e.target.value)}
                  >
                    {ShearAreaatBaseHorizontalUnit.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Shear Stress at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                - Horizontal
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y direction)</span>
              </p>
              <div className='Calculator-Side-A '>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>τ</span>
                  <span className='LowerPower'>sby</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={ShearStressatBaseActualHorizontalYdirection}
                    readOnly
                    aria-label="Shear Stress at Base (Act) - Horizontal (Y direction)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={ShearStressatBaseActualHorizontalYdirectionSelectedUnit}
                    onChange={(e) => handleShearStressatBaseActualHorizontalYdirectionUnitChange(e.target.value)}
                  >
                    {ShearStressatBaseActualHorizontalYdirectionUnit.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Shear Stress at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}>(Act)</span>
                - Lateral
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}>(X direction)</span>
              </p>
              <div className='Calculator-Side-A'>
                <p className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>τ</span>
                  <span className='LowerPower'>sbx(act)</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={ShearStressatBaseActualLateralXdirection}
                    readOnly
                    aria-label="Shear Stress at Base (Act) - Lateral (X direction)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={ShearStressatBaseActualLateralXdirectionSelectedUnit}
                    onChange={(e) => handleShearStressatBaseActualLateralXdirectionUnitChange(e.target.value)}
                  >
                    {ShearStressatBaseActualLateralXdirectionUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Von Mises Stress Check at Base:</p>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Von Mises Stress at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
              </p>
              <div className='Calculator-Side-A '>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>τ</span>
                  <span className='LowerPower'>sbx</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={VonMisesStressatBaseActual}
                    readOnly
                    aria-label="Von Mises Stress at Base (Act)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={VonMisesStressatBaseActualSelectedUnit}
                    onChange={(e) => handleVonMisesStressatBaseActualUnitChange(e.target.value)}
                  >
                    {VonMisesStressatBaseActualUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

        </div>
        <div className={`Grid-of-padeye-solutions mt-5 weldPlusFinalCheck ${showfirstDiv ? 'ScrollTransactionone' : ''} ${showSecondDiv ? 'ScrollTransactionTwo' : ''} ${showThirdDiv ? 'ScrollTransactionThree' : ''}`}>
          <div className={DropDowmThirdMain} style={{
            height: '45vw',
            left: '0%',
            width: '38vw',
            top: '465%',
            zIndex: ' 1',
          }}>
            <h2 className='text-center text-white'>WELD STRESS CHECK OF BASE WELD</h2>
            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Tensile Stress Check:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Throat Thickness</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>t</span>
                  <span className='LowerPower'>wt</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={Throatthickness}
                    readOnly
                    aria-label="Throat thickness"
                  />
                  <select
                    className='Calculator-select-option'
                    value={ThroatthicknessSelectedUnit}
                    onChange={(e) => handleThroatthicknessUnitChange(e.target.value)}
                  >
                    {ThroatthicknessUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Total Weld Length</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol'>

                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>L</span>
                  <span className='LowerPower'>w</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TotalWeldLength}
                    readOnly
                    aria-label="Total Weld Length"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TotalWeldLengthSelectedUnit}
                    onChange={(e) => handleTotalWeldLengthUnitChange(e.target.value)}
                  >
                    {TotalWeldLengthUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Moment lever for Horizontal/Lateral Force</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol' style={{ fontSize: '1vw' }}>
                  h
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={MomentleverforHorizontalLateralForce}
                    readOnly
                    aria-label="Moment lever for Horizontal/Lateral Force"
                  />
                  <select
                    className='Calculator-select-option'
                    value={MomentleverforHorizontalLateralForceSelectedUnit}
                    onChange={(e) => handleMomentleverforHorizontalLateralForceUnitChange(e.target.value)}
                  >
                    {MomentleverforHorizontalLateralForceUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Moment lever for Vertical Force</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol' style={{ fontSize: '1vw' }}>
                  e
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={MomentleverforVerticalForce}
                    readOnly
                    aria-label="Moment lever for Vertical Force"
                  />
                  <select
                    className='Calculator-select-option'
                    value={MomentleverforVerticalForceSelectedUnit}
                    onChange={(e) => handleMomentleverforVerticalForceUnitChange(e.target.value)}
                  >
                    {MomentleverforVerticalForceUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Shear Stress at Base Weld:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>Shear Area of Base Weld</p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>A</span>
                  <span className='LowerPower'>sbw</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={ShearStressatBaseWeld}
                    readOnly
                    aria-label="Shear Area of Base Weld"
                  />
                  <select
                    className='Calculator-select-option'
                    value={ShearStressatBaseWeldSelectedUnit}
                    onChange={(e) => handleShearStressatBaseWeldUnitChange(e.target.value)}
                  >
                    {ShearStressatBaseWeldUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Shear Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                - Horizontal
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y -direction) </span>
              </p>
              <div className='Calculator-Side-A '>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>τ</span>
                  <span className='LowerPower'>sbwy</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={ShearStressatBaseWeldActualHorizontalYdirection}
                    readOnly
                    aria-label="Shear Stress at Base Weld (Act) - Horizontal (Y -direction)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={ShearStressatBaseWeldActualHorizontalYdirectionSelectedUnit}
                    onChange={(e) => handleShearStressatBaseWeldActualHorizontalYdirectionUnitChange(e.target.value)}
                  >
                    {ShearStressatBaseWeldActualHorizontalYdirectionUnits.map((unit) => (<option key={unit} value={unit}>{unit}</option>))}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Shear Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                - Lateral
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X - direction) </span>
              </p>
              <div className='Calculator-Side-A '>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>τ</span>
                  <span className='LowerPower'>sbwx</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={ShearStressatBaseWeldActualLateralXdirection}
                    readOnly
                    aria-label="Shear Stress at Base Weld (Act) - Lateral (X - direction)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={ShearStressatBaseWeldActualLateralXdirectionSelectedUnit}
                    onChange={(e) => handleShearStressatBaseWeldActualLateralXdirectionUnitChange(e.target.value)}
                  >
                    {ShearStressatBaseWeldActualLateralXdirectionUnit.map((unit) => (<option key={unit} value={unit}>{unit}</option>))}
                  </select>
                </div>
              </div>
            </div>
            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Tensile Stress at Base Weld:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Tensile Area at Base
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>A</span>
                  <span className='LowerPower'>tbw</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={ShearStressatBaseWeld}
                    readOnly
                    aria-label="Tensile Area at Base"
                  />
                  <select
                    className='Calculator-select-option'
                    value={ShearStressatBaseWeldSelectedUnit}
                    onChange={(e) => handleShearStressatBaseWeldUnitChange(e.target.value)}
                  >
                    {ShearStressatBaseWeldUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Tensile Stress at Base Weld
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act)</span>
              </p>
              <div className='Calculator-Side-A ' >
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw !important', top: '-3px' }}>σ</span>
                  <span className='LowerPower'>tbw</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TensileStressatBaseWeldActual}
                    readOnly
                    aria-label="Tensile Stress at Base Weld (Act)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TensileStressatBaseWeldActualSelectedUnit}
                    onChange={(e) => handleTensileStressatBaseWeldActualUnitChange(e.target.value)}
                  >
                    {TensileStressatBaseWeldActualUnit.map((unit) => (<option key={unit} value={unit}>{unit}</option>))}
                  </select>
                </div>
              </div>
            </div>


            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Section Modulus of Base Weld:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Section Modulus of Weld about X-axis
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>Z</span>
                  <span className='LowerPower'>wbxx</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={SectionModulusofWeldaboutXaxis}
                    readOnly
                    aria-label="Section Modulus of Weld about X-axis"
                  />
                  <select
                    className='Calculator-select-option'
                    value={SectionModulusofWeldaboutXaxisSelectedUnit}
                    onChange={(e) => handleSectionModulusofWeldaboutXaxisUnitChange(e.target.value)}
                  >
                    {SectionModulusofWeldaboutXaxisUnit.map((unit) => (<option key={unit} value={unit}>{unit}</option>))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Section Modulus of Weld about Y-axis
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>Z</span>
                  <span className='LowerPower'>wbyy</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={SectionModulusofWeldaboutYaxis}
                    readOnly
                    aria-label="Section Modulus of Weld about Y-axis"
                  />
                  <select
                    className='Calculator-select-option'
                    value={SectionModulusofWeldaboutYaxisSelectedUnit}
                    onChange={(e) => handleSectionModulusofWeldaboutYaxisUnitChange(e.target.value)}
                  >
                    {SectionModulusofWeldaboutYaxisUnit.map((unit) => (<option key={unit} value={unit}>{unit}</option>))}
                  </select>
                </div>
              </div>
            </div>

            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Bending Stress Check at Base Weld:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Total Design Moment about X-axis
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (In-plane)</span>
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>M</span>
                  <span className='LowerPower'>dxx</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TotalDesignMomentaboutXaxisInplane}
                    readOnly
                    aria-label="Total Design Moment about X-axis (In-plane)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TotalDesignMomentaboutXaxisInplaneSelectedUnit}
                    onChange={(e) => handleTotalDesignMomentaboutXaxisInplaneUnitChange(e.target.value)}
                  >
                    {TotalDesignMomentaboutXaxisInplaneUnits.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Bending Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                - about X Axis
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}>  (In Plane) </span>
              </p>
              <div className='Calculator-Side-A '>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>σ</span>
                  <span className='LowerPower'>bbwx</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={BendingStressatBaseWeldActualaboutXAxisInPlane}
                    readOnly
                    aria-label="Bending Stress at Base Weld about X Axis (In Plane) (Act)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={BendingStressatBaseWeldActualaboutXAxisInPlaneSelectedUnit}
                    onChange={(e) => handleBendingStressatBaseWeldActualaboutXAxisInPlaneUnitChange(e.target.value)}
                  >
                    {BendingStressatBaseWeldActualaboutXAxisInPlaneUnit.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Total Design Moment about Y-axis
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out-of Plane) </span>
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>M</span>
                  <span className='LowerPower'>dyy</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TotalDesignMomentaboutYaxisOutofPlane}
                    readOnly
                    aria-label="Total Design Moment about Y-axis (Out-of Plane)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TotalDesignMomentaboutYaxisOutofPlaneSelectedUnit}
                    onChange={(e) => handleTotalDesignMomentaboutYaxisOutofPlaneUnitChange(e.target.value)}
                  >
                    {BendingStressActualaboutYaxisOutofPlaneUnit.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>




            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Bending Stress
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
                - about Y-axis
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Out-of Plane) </span>
              </p>
              <div className='Calculator-Side-A'>
                <p className='sigma-symbol '>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>σ</span>
                  <span className='LowerPower'>bby</span>
                </p>
                <br />
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={BendingStressaboutYaxisOutofPlane}
                    readOnly
                    aria-label="Bending Stress about Y-axis (Out-of Plane)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={BendingStressaboutYaxisOutofPlaneSelectedUnit}
                    onChange={(e) => handleBendingStressaboutYaxisOutofPlaneUnitChange(e.target.value)}
                  >
                    {BendingStressaboutYaxisOutofPlaneUnit.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <br />
            <br />
            <p className='text-white' style={{ fontSize: '1.8vw' }}>Total Stress Check at Base Weld:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title'>
                Total Stress at Base Weld
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Act) </span>
              </p>
              <div className='Calculator-Side-A'>
                <br />
                <p className='sigma-symbol sigmawithshorterlowerPower'>
                  <span className='power' style={{ fontSize: '1vw', top: '-3px' }}>M</span>
                  <span className='LowerPower'>dyy</span>
                </p>
                <div className='input-and-select-div'>
                  <input
                    className='calculator-input'
                    type="number"
                    value={TotalStressatBaseWeldActual}
                    readOnly
                    aria-label="Total Stress at Base Weld (Act)"
                  />
                  <select
                    className='Calculator-select-option'
                    value={TotalStressatBaseWeldActualSelectedUnit}
                    onChange={(e) => handleTotalStressatBaseWeldActualUnitChange(e.target.value)}
                  >
                    {TotalStressatBaseWeldActualUnit.map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>
          <div className={DropDowmThirdrightMain} style={{
            height: '45vw',
            left: '52%',
            width: '38vw',
            top: '465%',
          }}>
            <h2 className='text-center text-white'>FINAL CHECKS</h2>
            <br />
            <br />
            <p className='text-white Finalchecks' style={{ fontSize: '1.8vw' }}>Geometry check:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Main Plate Radius
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {MainPlateRadius}
              </div>
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Shackle Clearance Checks
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {ShackleClearanceChecks}
              </div>
            </div>
            <br />
            <p className='text-white Finalchecks' style={{ fontSize: '1.8vw' }}>Stress Checks at Pin Hole:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Bearing Stress Check at Pin Hole
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {BearingStressCheckatPinHole}
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Shear Stress Check at Pin Hole
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {ShearAreaofPinHole}
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Tensile Stress Check at Pin Hole
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {TensileStressCheckatPinHole}
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Hertz/Contact Stress Check at Pin Hole
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {HertzContactStressCheckatPinHole}
              </div>
            </div>
            <br />
            <p className='text-white Finalchecks' style={{ fontSize: '1.8vw' }}>Stress Check Base Plate:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Tensile Stress Check at Base
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {TensileStressCheckatBase}
              </div>
            </div>


            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Bending Stress Check at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X-axis) </span>
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {BendingStressCheckatBaseXaxis}
              </div>
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Bending Stress Check at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y-axis) </span>
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {BendingStressCheckatBaseYaxis}
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Shear Stress Check at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (Y direction) </span>
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {ShearStressCheckatBaseYdirection}
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Shear Stress Check at Base
                <span className='calculator-coversation-title-span' style={{ fontSize: '0.7vw', position: 'relative', top: '3px' }}> (X direction) </span>
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {ShearStressCheckatBaseXdirection}
              </div>
            </div>

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Combined Stress Check as per AISC H2-1
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {combinedStressCheckasperAISCH21Condition}
              </div>
            </div>

            <br />
            <p className='text-white Finalchecks' style={{ fontSize: '1.8vw' }}>Weld Stress Check of Base Weld:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Combined Stress Check at Base
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {CombinedStressCheckatBaseCondition}
              </div>
            </div>

            <br />
            <p className='text-white Finalchecks' style={{ fontSize: '1.8vw' }}>Weld Check at Cheek Plate Weld:</p>
            <br />

            <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <p className='claculator-conversation-title Finalcheck'>
                Stress Check at Cheek Plate Weld
              </p>
              <div className='Calculator-Side-A Finalcheck' style={{ width: '25%', justifyContent: 'center' }}>
                {StressCheckatCheekPlateWeld}
              </div>
            </div>

          </div>
        </div>
        <div className={showfirstDiv ? ' padeye height110 ' : ' padeye  height0 '} ></div>
        <br />
        <br />
        <br />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <VideoPlayerSection />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <PadeyeFile />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <ShackleFile />
        <br />
        <br />
        <br />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <SlingsFile />
        <br />
        <br />
        <br />
        <section className='cse-header-top'>
          <Link smooth='true' duration={500} offset={-70} onClick={scrollToTop} aria-label="Scroll to top">
            <GrLinkTop className='' />
          </Link>
        </section>
      </section >
    </>
  )
}
