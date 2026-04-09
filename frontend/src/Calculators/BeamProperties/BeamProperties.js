import React, { useState, useEffect } from 'react'
import backgroundWebP from '../../images/Beam-Properties-Background.webp'; // Optimized WebP version

// Caculator Topic Images
import Square from '../../images/Beam-Properties-Square.png'
import Rectangle from '../../images/Beam-Properties-Rectangle.png'
import HollowRectangle from '../../images/Beam-Properties-Hollow-Rect.png'
import TSection from '../../images/Beam-Properties-T.png'
import CChannel from '../../images/Beam-Properties-C-Channel.png'
import IBeam from '../../images/Beam-Properties-I.jpg'
import LSection from '../../images/Beam-Properties-L.png'
import Circle from '../../images/Beam-Properties-Circle.png'
import HollowCircle from '../../images/Beam-Properties-Hollow-Circle.png'

// CSS
import '../../Css/BeamProperties.css'
import '../../Css/BeamDeflection.css'
import '../../Css/NumberLine.css'
import '../../Css/AboutUS.css'
import '../../Css/Navbar.css'
import '../../Css/Padeye.css'

// Shape topic Files
import AreaOfSection from './FormulaSections/AreaOfSection'
import PrincipleAxis from './FormulaSections/PrincipleAxis'
import TorsionalConstant from './FormulaSections/TorsionalConstant'
import SectionModulus from './FormulaSections/SectionModulus'
import AreaMomentsOfInertia from './FormulaSections/AreaMomentsOfInertia'
import Centroid from './FormulaSections/Centroid'

// modules
import { Link } from 'react-router-dom';

// icons
import { GrLinkTop } from "react-icons/gr";
import VideoPlayerSection from './VideoPlayer/VideoPlayerSection'
import { Helmet } from 'react-helmet';

export default function BeamProperties() {


  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };
  const [expanded, setExpanded] = useState(false);
  const [Sectionthird, setSectionthird] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 250 && !expanded) {
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

  const [selectedOption, setSelectedOption] = useState("option1");

  const handleOptionChange = (option) => {
    setSelectedOption(option)
  }

  const [isActive3, setIsActive3] = useState(false);

  const toggleClass3 = () => {
    setIsActive3(previsActive3 => !previsActive3);

  };

  const [MetricOrImperial, setMetricOrImperial] = useState("option1");

  const toggleMetricOrImperial = (option) => {
    setMetricOrImperial(option);

  };

  // square calculation
  const SquareMetricUnits = ['mm', 'cm', 'm'];
  const SquareMetricConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const SquareAreaUnits = ['mm²', 'cm²', 'm²'];
  const SquareAreaConversionFactors = {
    'mm²': [1, 0.01, 0.000001],
    'cm²': [100, 1, 0.0001],
    'm²': [1000000, 10000, 1],
  };

  const SquareSquareCentroidUnits = ['mm', 'cm', 'm'];
  const SquareSquareCentroidConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const SquareSquareMomentOfInertiaUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const SquareMomentOfInertiaConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const SquareSectionModulesUnits = ['mm³', 'cm³', 'm³'];
  const SquareSectionModulesConversionFactors = {
    'mm³': [1, 0.001, 1e-9],
    'cm³': [1000, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };

  const SquareTorsionalConstantUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const SquareTorsionalConstantConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [SquaremetricInputValue, setSquareMetricInputValue] = useState('0');
  const [SquaremetricSelectedUnit, setSquareMetricSelectedUnit] = useState('mm');
  const [SquareinternalMetricValue, setSquareInternalMetricValue] = useState(0); // Always in mm

  const [Squarearea, setSquareArea] = useState(0);
  const [SquareareaUnit, setSquareAreaUnit] = useState('mm²');

  const [Squarecentroid, setSquareCentroid] = useState(0);
  const [SquarecentroidUnit, setSquareCentroidUnit] = useState('mm');

  const [SquaremomentOfInertia, setSquareMomentOfInertia] = useState(0);
  const [SquaremomentOfInertiaUnit, setSquareMomentOfInertiaUnit] = useState('mm⁴');

  const [SquaresectionModules, setSquareSectionModules] = useState(0);
  const [SquaresectionModulesUnit, setSquareSectionModulesUnit] = useState('mm³');

  const [SquaretorsionalConstant, setSquareTorsionalConstant] = useState(0);
  const [SquaretorsionalConstantUnit, setSquareTorsionalConstantUnit] = useState('mm⁴');

  const handleMetricInputChange = (value) => {
    setSquareMetricInputValue(value);
    const factor = SquareMetricConversionFactors[SquaremetricSelectedUnit][0];
    setSquareInternalMetricValue(parseFloat(value) * factor);
  };

  const handleMetricUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(SquaremetricInputValue) * SquareMetricConversionFactors[SquaremetricSelectedUnit][0];
    const convertedValue = newMetricValueInMM / SquareMetricConversionFactors[unit][0];
    setSquareMetricSelectedUnit(unit);
    setSquareMetricInputValue(convertedValue);
  };

  useEffect(() => {
    const value = parseFloat(SquaremetricInputValue) * SquareMetricConversionFactors[SquaremetricSelectedUnit][0];
    if (!isNaN(value)) {
      const areaValue = (value * value).toFixed(5);
      setSquareArea((areaValue / SquareAreaConversionFactors[SquareareaUnit][0]));
    } else {
      setSquareArea('');
    }
  }, [SquaremetricInputValue, SquareareaUnit, SquaremetricSelectedUnit]);

  useEffect(() => {
    const value = SquareinternalMetricValue;
    const SquarecentroidValue = (value / 2).toFixed(2);
    setSquareCentroid((SquarecentroidValue / SquareSquareCentroidConversionFactors[SquarecentroidUnit][0]));
  }, [SquareinternalMetricValue, SquarecentroidUnit]);

  useEffect(() => {
    const value = SquareinternalMetricValue;
    if (!isNaN(value)) {
      const inertia = ((value ** 4) / 12).toFixed(3);
      const convertedValue = (inertia * SquareMomentOfInertiaConversionFactors['mm⁴'][SquareSquareMomentOfInertiaUnits.indexOf(SquaremomentOfInertiaUnit)]);
      setSquareMomentOfInertia(convertedValue);
    }
  }, [SquareinternalMetricValue, SquaremomentOfInertiaUnit]);

  useEffect(() => {
    const value = SquareinternalMetricValue;
    if (!isNaN(value)) {
      const SquarecentroidValue = (value / 2);
      const SquaremomentOfInertiaValue = ((value ** 4) / 12);
      const SquaresectionModulesValue = (SquaremomentOfInertiaValue / SquarecentroidValue).toFixed(4);
      const convertedValue = (SquaresectionModulesValue * SquareSectionModulesConversionFactors['mm³'][SquareSectionModulesUnits.indexOf(SquaresectionModulesUnit)]);
      setSquareSectionModules(convertedValue);
    }
  }, [SquareinternalMetricValue, SquaresectionModulesUnit]);

  useEffect(() => {
    const value = SquareinternalMetricValue;
    if (!isNaN(value)) {
      const torsional = (9 * Math.pow(value, 4)) / 64;
      const convertedValue = (torsional * SquareTorsionalConstantConversionFactors['mm⁴'][SquareTorsionalConstantUnits.indexOf(SquaretorsionalConstantUnit)]);
      setSquareTorsionalConstant(convertedValue);
    }
  }, [SquareinternalMetricValue, SquaretorsionalConstantUnit]);


  const SquareImperialunits = ['in'];
  const ImperialFactors = {
    in: [1],
  };

  const [SquareImperialinputValue, setSquareImperialInputValue] = useState(0);
  const [SquareImperialselectedUnit, setSquareImperialSelectedUnit] = useState('in');

  const SquarehandleImperialInputChange = (value) => {
    setSquareImperialInputValue(value);
  };

  const SquarehandleImperialUnitChange = (unit) => {
    setSquareImperialSelectedUnit(unit);
    const Imperialfactor = ImperialFactors[unit][SquareImperialunits.indexOf(SquareImperialselectedUnit)];
    setSquareImperialInputValue((parseFloat(SquareImperialinputValue) / Imperialfactor).toFixed(3));
  };

  const ImperialSquareAreaUnits = ['in²'];
  const ImperialSquareAreaConversionFactors = {
    'in²': [1],
  };
  const [SquareImperialArea, setSquareImperialArea] = useState(0);
  const [SquareImperialAreaUnit, setSquareImperialAreaUnit] = useState('in²');

  const handleSquareImperialAreaUnitChange = (unit) => {
    const currentAreaInNewUnit = SquareImperialArea * ImperialSquareAreaConversionFactors[SquareImperialAreaUnit][ImperialSquareAreaUnits.indexOf(unit)];
    setSquareImperialArea(currentAreaInNewUnit);
    setSquareImperialAreaUnit(unit);
  };

  const calculateSquareImperialAreaInputChangeValue = (e) => {
    const SquareImperialinputValue = parseFloat(e.target.value);
    if (!isNaN(SquareImperialinputValue)) {
      const SquareImperialArea = (SquareImperialinputValue * SquareImperialinputValue);
      setSquareImperialArea(SquareImperialArea);
    } else {
      setSquareImperialArea('');
    }
    setSquareImperialInputValue(e.target.value);
  };

  useEffect(() => {
    if (SquareImperialinputValue !== '') {
      calculateSquareImperialAreaInputChangeValue({ target: { value: SquareImperialinputValue } });
    }
  }, [SquareImperialinputValue]);



  const [SquareImperialSquareCentroid, setSquareImperialSquareCentroid] = useState(0);
  const [SquareImperialSquareCentroidSelectedunit, setSquareImperialSquareCentroidSelectedUnit] = useState('in');

  const handleSquareImperialSquareCentroidUnitChange = (unit) => {
    setSquareImperialSquareCentroidSelectedUnit(unit);
  };

  const calculateSquareImperialSquareCentroidInputChangeValue = () => {
    const Input = SquareImperialinputValue;
    const SquareImperialSquareCentroid = Input / 2;
    setSquareImperialSquareCentroid(SquareImperialSquareCentroid);
  };

  useEffect(() => {
    calculateSquareImperialSquareCentroidInputChangeValue();
  }, [SquareImperialinputValue]);



  const ImperialSquareSquareMomentOfInertiaUnits = ['in⁴'];
  const SquareImperialSquareMomentOfInertiaConversionFactors = {
    'in⁴': [1],
  };
  const [SquareImperialSquaremomentOfInertia, setSquareImperialSquareMomentOfInertia] = useState(0);
  const [SquareImperialSquaremomentOfInertiaSelectedUnit, setSquareImperialSquareMomentOfInertiaSelectedUnit] = useState('in⁴');

  const handleSquareImperialSquareMomentOfInertiaUnitChange = (newUnit) => {
    const currentFactor = SquareImperialSquareMomentOfInertiaConversionFactors[SquareImperialSquaremomentOfInertiaSelectedUnit][ImperialSquareSquareMomentOfInertiaUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(SquareImperialSquaremomentOfInertia) * currentFactor;
    setSquareImperialSquareMomentOfInertiaSelectedUnit(newUnit);
    setSquareImperialSquareMomentOfInertia(convertedValue.toString());
  };

  const calculateSquareImperialSquareMomentOfInertia = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const inertia = ((parsedValue ** 4) / 12).toFixed(3);
      const currentFactor = SquareImperialSquareMomentOfInertiaConversionFactors['in⁴'][ImperialSquareSquareMomentOfInertiaUnits.indexOf(SquareImperialSquaremomentOfInertiaSelectedUnit)];
      setSquareImperialSquareMomentOfInertia((inertia * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateSquareImperialSquareMomentOfInertia(SquareImperialinputValue);
  }, [SquareImperialinputValue, SquareImperialSquaremomentOfInertiaSelectedUnit]);

  const ImperialSquareSectionModulesUnits = ['in³'];
  const SquareImperialSectionModulesConversionFactors = {
    'in³': [1],
  };
  const [SquareImperialSectionModules, setSquareImperialSectionModules] = useState(0);
  const [SquareImperialSectionModulesSelectedUnit, setSquareImperialSectionModulesSelectedUnit] = useState('in³');

  const handleSquareImperialSectionModulesUnitChange = (newUnit) => {
    const currentFactor = SquareImperialSectionModulesConversionFactors[SquareImperialSectionModulesSelectedUnit][ImperialSquareSectionModulesUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(SquareImperialSectionModules) * currentFactor;
    setSquareImperialSectionModulesSelectedUnit(newUnit);
    setSquareImperialSectionModules(convertedValue.toString());
  };

  const calculateSquareImperialSectionModules = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const SquarecentroidValue = (parsedValue / 2).toFixed(3);
      const SquaremomentOfInertiaValue = ((parsedValue ** 4) / 12).toFixed(3);
      const sectionModules = (SquaremomentOfInertiaValue / SquarecentroidValue).toFixed(3);
      const currentFactor = SquareImperialSectionModulesConversionFactors['in³'][ImperialSquareSectionModulesUnits.indexOf(SquareImperialSectionModulesSelectedUnit)];
      setSquareImperialSectionModules((sectionModules * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateSquareImperialSectionModules(SquareImperialinputValue);
  }, [SquareImperialinputValue, SquareImperialSectionModulesSelectedUnit]);


  const ImperialSquareTorsionalConstantUnits = ['in⁴'];
  const ImperialSquareTorsionalConstantConversionFactors = {
    'in⁴': [1],
  };
  const [SquareImperialTorsionalConstant, setSquareImperialTorsionalConstant] = useState(0);
  const [SquareImperialTorsionalConstantSelectedUnit, setSquareImperialTorsionalConstantSelectedUnit] = useState('in⁴');

  const handleSquareImperialTorsionalConstantUnitChange = (newUnit) => {
    const currentFactor = ImperialSquareTorsionalConstantConversionFactors[SquareImperialTorsionalConstantSelectedUnit][ImperialSquareTorsionalConstantUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(SquareImperialTorsionalConstant) * currentFactor;
    setSquareImperialTorsionalConstantSelectedUnit(newUnit);
    setSquareImperialTorsionalConstant(convertedValue.toString());
  };

  const calculateSquareImperialTorsionalConstant = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const torsional = ((9 * Math.pow(parsedValue, 4)) / 64).toFixed(3);
      const currentFactor = ImperialSquareTorsionalConstantConversionFactors['in⁴'][ImperialSquareTorsionalConstantUnits.indexOf(SquareImperialTorsionalConstantSelectedUnit)];
      setSquareImperialTorsionalConstant((torsional * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateSquareImperialTorsionalConstant(SquareImperialinputValue);
  }, [SquareImperialinputValue, SquareImperialTorsionalConstantSelectedUnit]);



  // rectangle calculation

  const Rectangleunits = ['mm', 'cm', 'm'];
  // const [exponent, setExponent] = useState(4);

  const RectangleconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [RectangleinputValue, setRectangleInputValue] = useState(0);
  const [RectangleselectedUnit, setRectangleSelectedUnit] = useState('mm');
  const [internalRectangleinputValue, setInternalRectangleinputValue] = useState(0); // Always in mm

  const handleInputChange = (value) => {
    setRectangleInputValue(value);
    const factor = RectangleconversionFactors[RectangleselectedUnit][0];
    setInternalRectangleinputValue(parseFloat(value) * factor);
  };

  const handleUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(RectangleinputValue) * RectangleconversionFactors[RectangleselectedUnit][0];
    const convertedValue = newMetricValueInMM / RectangleconversionFactors[unit][0];
    setRectangleSelectedUnit(unit);
    setRectangleInputValue(convertedValue.toFixed(3));
  };


  const RectangleHeightunits = ['mm', 'cm', 'm'];

  const RectangleHeightConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [RectangleHeightInputValue, setRectangleHeightInputValue] = useState(0);
  const [RectangleheightSelectedUnit, setRectangleHeightSelectedUnit] = useState('mm');
  const [internalRectangleHeightInputValue, setInternalRectangleHeightInputValue] = useState(0); // Always in mm

  const handleRectangleHeightInputValue = (value) => {
    setRectangleHeightInputValue(value);
    const factor = RectangleHeightConversionFactors[RectangleheightSelectedUnit][0];
    setInternalRectangleHeightInputValue(parseFloat(value) * factor);
  };

  const handleRectangleHeightSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(RectangleHeightInputValue) * RectangleHeightConversionFactors[RectangleheightSelectedUnit][0];
    const convertedValue = newMetricValueInMM / RectangleHeightConversionFactors[unit][0];
    setRectangleHeightSelectedUnit(unit);
    setRectangleHeightInputValue(convertedValue.toFixed(3));
  };


  const b = internalRectangleinputValue
  const d = internalRectangleHeightInputValue



  const RectangleAreaUnits = ['mm²', 'cm²', 'm²'];
  const RectangleAreaConversionFactors = {
    'mm²': [1, 0.01, 0.000001],
    'cm²': [100, 1, 0.0001],
    'm²': [1000000, 10000, 1],
  };
  const [RectangleArea, setRectangleArea] = useState(0);
  const [RectangleAreaUnit, setRectangleAreaUnit] = useState('mm²');
  useEffect(() => {
    const value = parseFloat(internalRectangleinputValue) * RectangleconversionFactors[RectangleselectedUnit][0];
    if (!isNaN(value)) {
      const RectangleareaValue = (b * d);
      setRectangleArea((RectangleareaValue / RectangleAreaConversionFactors[RectangleAreaUnit][0]));
    } else {
      setRectangleArea('');
    }
  }, [internalRectangleinputValue, internalRectangleHeightInputValue, RectangleAreaUnit, RectangleselectedUnit]);


  // CentroidXcUnits
  const RectangleCentroidXcUnits = ['mm', 'cm', 'm'];

  const [RectangleCentroidXc, setRectangleCentroidXc] = useState(0);
  const [RectangleCentroidXcSelectedUnit, setRectangleCentroidXcSelectedUnit] = useState('mm');

  useEffect(() => {
    const value = internalRectangleinputValue;
    const centroidValue = (value / 2);
    setRectangleCentroidXc((centroidValue / RectangleconversionFactors[RectangleCentroidXcSelectedUnit][0]));
  }, [internalRectangleinputValue, RectangleCentroidXcSelectedUnit]);


  // CentriodYcUnits
  const RectangleCentriodYcUnits = ['mm', 'cm', 'm'];

  const [RectangleCentriodYc, setRectangleCentriodYc] = useState(0);
  const [RectangleCentriodYcSelectedUnit, setRectangleCentriodYcSelectedUnit] = useState('mm');


  useEffect(() => {
    const d = internalRectangleHeightInputValue;
    const centroidValue = (d / 2);
    setRectangleCentriodYc((centroidValue / RectangleconversionFactors[RectangleCentriodYcSelectedUnit][0]));
  }, [internalRectangleHeightInputValue, RectangleCentriodYcSelectedUnit]);




  // MomentOfInertia Ix
  const [RectanglemomentOfInertiaIx, setRectangleMomentOfInertiaIx] = useState(0);
  const [RectanglemomentOfInertiaIxSelectedUnit, setRectangleMomentOfInertiaIxSelectedUnit] = useState('mm⁴');

  const RectangleMomentOfInertiaIxUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const RectangleMomentOfInertiaIxConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  useEffect(() => {
    const value = b;
    const dvalue = d;

    if (!isNaN(value)) {
      const RectanglemomentOfInertiaIx = (value * Math.pow(dvalue, 3)) / 12;

      // Convert RectanglemomentOfInertiaIx based on the selected unit
      const convertedValue = (RectanglemomentOfInertiaIx * RectangleMomentOfInertiaIxConversionFactors['mm⁴'][RectangleMomentOfInertiaIxUnits.indexOf(RectanglemomentOfInertiaIxSelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = RectanglemomentOfInertiaIxSelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setRectangleMomentOfInertiaIx(formattedValue);
    }
  }, [b, d, RectanglemomentOfInertiaIxSelectedUnit]);


  // MomentOfInertia Iy
  const [RectanglemomentOfInertiaIy, setRectangleMomentOfInertiaIy] = useState(0);
  const [RectanglemomentOfInertiaIySelectedUnit, setRectangleMomentOfInertiaIySelectedUnit] = useState('mm⁴');

  const RectangleMomentOfInertiaIyUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const RectangleMomentOfInertiaIyConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  useEffect(() => {
    const value = b;
    const dvalue = d;

    if (!isNaN(value)) {
      const RectanglemomentOfInertiaIy = ((dvalue * Math.pow(value, 3)) / 12);

      // Convert RectanglemomentOfInertiaIy based on the selected unit
      const convertedValue = (RectanglemomentOfInertiaIy * RectangleMomentOfInertiaIyConversionFactors['mm⁴'][RectangleMomentOfInertiaIyUnits.indexOf(RectanglemomentOfInertiaIySelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = RectanglemomentOfInertiaIySelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setRectangleMomentOfInertiaIy(formattedValue);
    }
  }, [b, d, RectanglemomentOfInertiaIySelectedUnit]);


  // SectionModules Sx
  const RectangleSectionModulesSxUnits = ['mm³', 'cm³', 'm³'];
  const RectangleSectionModulesSxConversionUnit = {
    'mm³': [1, 0.001, 1e-9],
    'cm³': [1000, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [RectangleSectionModulesSx, setRectangleSectionModulesSx] = useState(0);
  const [RectangleSectionModulesSxSelectedUnit, setRectangleSectionModulesSxSelectedUnit] = useState('mm³');

  useEffect(() => {
    const value = b;
    const dvalue = d;

    if (!isNaN(value)) {
      const momentOfInertiaValue = ((value * Math.pow(dvalue, 3)) / 12).toFixed(2);
      const RectanglesectionModulesValue = (b * Math.pow(d, 2) / 6);

      // Convert RectanglesectionModulesValue based on the selected unit
      const convertedValue = RectanglesectionModulesValue * RectangleSectionModulesSxConversionUnit['mm³'][RectangleSectionModulesSxUnits.indexOf(RectangleSectionModulesSxSelectedUnit)];

      // Format the output based on the selected unit
      const formattedValue = RectangleSectionModulesSxSelectedUnit === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation
        : convertedValue.toFixed(3); // 3 decimals for other units

      setRectangleSectionModulesSx(formattedValue);
    }
  }, [b, d, RectangleSectionModulesSxSelectedUnit]);


  // const RectangleSectionModulesSx = Ix / Yc

  // Sy
  const RectangleSectionModulesSyUnits = ['mm³', 'cm³', 'm³'];
  const RectangleSectionModulesSyConversionUnit = {
    'mm³': [1, 0.001, 1e-9],
    'cm³': [1000, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };

  const [RectangleSectionModulesSy, setRectangleSectionModulesSy] = useState(0);
  const [RectangleSectionModulesSelectedUnitSy, setRectangleSectionModulesSelectedUnitSy] = useState('mm³');

  useEffect(() => {
    const value = b;

    if (!isNaN(value)) {
      const RectanglesectionModulesValue = (d * Math.pow(b, 2) / 6);

      // Convert RectanglesectionModulesValue based on the selected unit
      const convertedValue = (RectanglesectionModulesValue * RectangleSectionModulesSyConversionUnit['mm³'][RectangleSectionModulesSyUnits.indexOf(RectangleSectionModulesSelectedUnitSy)]);

      // Format the output based on the selected unit
      const formattedValue = RectangleSectionModulesSelectedUnitSy === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m³
        : convertedValue.toFixed(3); // 3 decimals for other units

      setRectangleSectionModulesSy(formattedValue);
    }
  }, [b, d, RectangleSectionModulesSelectedUnitSy]);

  // RectangleTorsionalConstant

  const RectangleTorsionalConstantUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const RectangleTorsionalConstantConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [RectangletorsionalConstant, setRectangleTorsionalConstant] = useState(0);
  const [RectangletorsionalConstantSelectedUnit, setRectangleTorsionalConstantSelectedUnit] = useState('mm⁴');

  useEffect(() => {
    const value = b;
    const dvalue = d;

    if (!isNaN(value)) {
      const torsional = ((value * Math.pow(dvalue, 3)) / 3 - 0.21 * Math.pow(dvalue, 4) + 0.0175 * Math.pow(dvalue, 8) / Math.pow(value, 4)).toFixed(2);

      // Convert torsional value based on the selected unit
      const convertedValue = (torsional * RectangleTorsionalConstantConversionFactors['mm⁴'][RectangleTorsionalConstantUnits.indexOf(RectangletorsionalConstantSelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = RectangletorsionalConstantSelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setRectangleTorsionalConstant(formattedValue);
    }
  }, [b, d, RectangletorsionalConstantSelectedUnit]);

  // const torsionalConstant = (parsedB * Math.pow(parsedD, 3)) / 3 - 0.21 * Math.pow(parsedD, 4) + 0.0175 * Math.pow(parsedD, 8) / Math.pow(parsedB, 4);



  // imperial units


  const RectangleunitsImperial = ['in'];
  const RectangleconversionFactorsImperial = {
    in: [1],
  };

  const [RectangleinputValueImperial, setRectangleInputValueImperial] = useState(0);
  const [RectangleselectedUnitImperial, setRectangleSelectedUnitImperial] = useState('in');

  const handleInputChangeImperial = (value) => {
    setRectangleInputValueImperial(value);
  };

  const handleUnitChangeImperial = (unit) => {
    setRectangleSelectedUnitImperial(unit);
    const factor = RectangleconversionFactorsImperial[unit][RectangleunitsImperial.indexOf(RectangleselectedUnitImperial)];
    setRectangleInputValueImperial((parseFloat(RectangleinputValueImperial) / factor).toFixed(4));
  };

  const RectangleHeightunitsImperial = ['in'];
  const RectangleHeightConversionFactorsImperial = {
    in: [1],
  };

  const [RectangleHeightInputValueImperial, setRectangleHeightInputValueImperial] = useState(0);
  const [RectangleheightSelectedUnitImperial, setRectangleHeightSelectedUnitImperial] = useState('in');

  const handleRectangleHeightInputValueImperial = (values) => {
    setRectangleHeightInputValueImperial(values);
  };

  const handleRectangleHeightSelectedUnitImperial = (units) => {
    setRectangleHeightSelectedUnitImperial(units);
    const Hrightfactor = RectangleHeightConversionFactorsImperial[units][RectangleHeightunitsImperial.indexOf(RectangleheightSelectedUnitImperial)];
    setRectangleHeightInputValueImperial((parseFloat(RectangleHeightInputValueImperial) / Hrightfactor).toFixed(4));
  };

  const RectangleAreaUnitsImperial = ['in²'];
  const RectangleAreaConversionFactorsImperial = {
    'in²': [1],
  };

  const [RectangleAreaImperial, setRectangleAreaImperial] = useState(0);
  const [RectangleAreaUnitImperial, setRectangleAreaUnitImperial] = useState('in²');

  // Handle change in area unit
  const handleAreaUnitChangeImperial = (unit) => {
    const currentAreaInNewUnit = RectangleAreaImperial / RectangleAreaConversionFactorsImperial[unit][RectangleAreaUnitsImperial.indexOf(RectangleAreaUnitImperial)];
    setRectangleAreaImperial(currentAreaInNewUnit);
    setRectangleAreaUnitImperial(unit);
  };

  // Calculate area based on input values
  const calculatehandleAreaInputChangeValueImperial = (value) => {
    const b = parseFloat(RectangleinputValueImperial);
    const d = parseFloat(RectangleHeightInputValueImperial);

    if (!isNaN(b) && !isNaN(d)) {
      const areaValue = b * d;
      setRectangleAreaImperial(areaValue);
    } else {
      setRectangleAreaImperial('');
    }
  };

  useEffect(() => {
    calculatehandleAreaInputChangeValueImperial();
  }, [RectangleinputValueImperial, RectangleHeightInputValueImperial]);

  // CentroidXcUnits
  const ReactangleCentroidXcUnitsImperial = ['in'];
  const ReactangleCentroidXcconversionFactorsImperial = {
    in: [1],
  };

  const [ReactangleCentroidXcImperial, setReactangleCentroidXcImperial] = useState(0);
  const [ReactangleCentroidXcSelectedUnitImperial, setReactangleCentroidXcSelectedUnitImperial] = useState('in');

  const handleReactangleCentroidXcUnitChangeImperial = (unit) => {
    setReactangleCentroidXcSelectedUnitImperial(unit);
    const CentroidFactor = ReactangleCentroidXcconversionFactorsImperial[unit][ReactangleCentroidXcUnitsImperial.indexOf(ReactangleCentroidXcSelectedUnitImperial)];
    setReactangleCentroidXcImperial((parseFloat(ReactangleCentroidXcImperial) / CentroidFactor).toFixed(3));
  };

  const calculateReactangleCentroidXcInputChangeValueImperial = () => {
    const b = parseFloat(RectangleinputValueImperial);
    const ReactangleCentroidXc = b / 2;
    setReactangleCentroidXcImperial(ReactangleCentroidXc);
  };

  useEffect(() => {
    calculateReactangleCentroidXcInputChangeValueImperial();
  }, [RectangleinputValueImperial]);

  // CentriodYcUnits
  const ReactangleCentriodYcUnitsImperial = ['in'];
  const ReactangleCentriodYcconversionFactorsImperial = {
    in: [1],
  };

  const [ReactangleCentriodYcImperial, setReactangleCentriodYcImperial] = useState(0);
  const [ReactangleCentriodYcSelectedUnitImperial, setReactangleCentriodYcSelectedUnitImperial] = useState('in');

  const handleReactangleCentriodYcUnitChangeImperial = (unit) => {
    setReactangleCentriodYcSelectedUnitImperial(unit);
    const ReactangleCentriodYcFactor = ReactangleCentriodYcconversionFactorsImperial[unit][ReactangleCentriodYcUnitsImperial.indexOf(ReactangleCentriodYcSelectedUnitImperial)];
    setReactangleCentriodYcImperial((parseFloat(ReactangleCentriodYcImperial) / ReactangleCentriodYcFactor).toFixed(3));
  };

  const calculateReactangleCentriodYcInputChangeValueImperial = () => {
    const d = parseFloat(RectangleHeightInputValueImperial);
    const ReactangleCentriodYc = d / 2;
    setReactangleCentriodYcImperial(ReactangleCentriodYc);
  };

  useEffect(() => {
    calculateReactangleCentriodYcInputChangeValueImperial();
  }, [RectangleHeightInputValueImperial]);

  // MomentOfInertia Ix
  const [ReactanglemomentOfInertiaIxImperial, setReactangleMomentOfInertiaIxImperial] = useState(0);
  const [ReactanglemomentOfInertiaIxSelectedUnitImperial, setReactangleMomentOfInertiaIxSelectedUnitImperial] = useState('in⁴');

  const ReactangleMomentOfInertiaIxUnitsImperial = ['in⁴'];
  const ReactangleMomentOfInertiaIxConversionFactorsImperial = {
    'in⁴': [1],
  };

  const handleReactangleMomentOfInertiaIxUnitChangeImperial = (newUnit) => {
    const currentFactor = ReactangleMomentOfInertiaIxConversionFactorsImperial[ReactanglemomentOfInertiaIxSelectedUnitImperial][ReactangleMomentOfInertiaIxUnitsImperial.indexOf(newUnit)];
    const convertedValue = parseFloat(ReactanglemomentOfInertiaIxImperial) * currentFactor;
    setReactangleMomentOfInertiaIxSelectedUnitImperial(newUnit);
    setReactangleMomentOfInertiaIxImperial(convertedValue.toString());
  };

  const calculateReactangleMomentOfInertiaIxImperial = (b, d) => {
    const parsedB = parseFloat(b);
    const parsedD = parseFloat(d);
    if (!isNaN(parsedB) && !isNaN(parsedD)) {
      // .toFixed(1)
      // .toFixed(2)
      const ReactanglemomentOfInertiaIx = ((parsedB * Math.pow(parsedD, 3)) / 12).toFixed(1);
      const currentFactor = ReactangleMomentOfInertiaIxConversionFactorsImperial['in⁴'][ReactangleMomentOfInertiaIxUnitsImperial.indexOf(ReactanglemomentOfInertiaIxSelectedUnitImperial)];
      setReactangleMomentOfInertiaIxImperial((ReactanglemomentOfInertiaIx * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateReactangleMomentOfInertiaIxImperial(RectangleinputValueImperial, RectangleHeightInputValueImperial);
  }, [RectangleinputValueImperial, RectangleHeightInputValueImperial, ReactanglemomentOfInertiaIxSelectedUnitImperial]);

  // MomentOfInertia Iy
  const [ReactanglemomentOfInertiaIyImperial, setReactangleMomentOfInertiaIyImperial] = useState(0);
  const [ReactanglemomentOfInertiaIySelectedUnitImperial, setReactangleMomentOfInertiaIySelectedUnitImperial] = useState('in⁴');

  const ReactangleMomentOfInertiaIyUnitsImperial = ['in⁴'];
  const ReactangleMomentOfInertiaIyConversionFactorsImperial = {
    'in⁴': [1],
  };

  const handleReactangleMomentOfInertiaIyUnitChangeImperial = (newUnit) => {
    const currentFactor = ReactangleMomentOfInertiaIyConversionFactorsImperial[ReactanglemomentOfInertiaIySelectedUnitImperial][ReactangleMomentOfInertiaIyUnitsImperial.indexOf(newUnit)];
    const convertedValue = parseFloat(ReactanglemomentOfInertiaIyImperial) * currentFactor;
    setReactangleMomentOfInertiaIySelectedUnitImperial(newUnit);
    setReactangleMomentOfInertiaIyImperial(convertedValue.toString());
  };

  const calculateReactangleMomentOfInertiaIyImperial = (b, d) => {
    const parsedB = parseFloat(b);
    const parsedD = parseFloat(d);
    if (!isNaN(parsedB) && !isNaN(parsedD)) {
      const ReactanglemomentOfInertiaIy = ((parsedD * Math.pow(parsedB, 3)) / 12).toFixed(1);
      const currentFactor = ReactangleMomentOfInertiaIyConversionFactorsImperial['in⁴'][ReactangleMomentOfInertiaIyUnitsImperial.indexOf(ReactanglemomentOfInertiaIySelectedUnitImperial)];
      setReactangleMomentOfInertiaIyImperial((ReactanglemomentOfInertiaIy * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateReactangleMomentOfInertiaIyImperial(RectangleinputValueImperial, RectangleHeightInputValueImperial);
  }, [RectangleinputValueImperial, RectangleHeightInputValueImperial, ReactanglemomentOfInertiaIySelectedUnitImperial]);

  // SectionModules Sx
  const ReactangleSectionModulesSxUnitsImperial = ['in³'];
  const ReactangleSectionModulesSxConversionUnitImperial = {
    'in³': [1],
  };
  const [ReactangleSectionModulesSxImperial, setReactangleSectionModulesSxImperial] = useState(0);
  const [ReactangleSectionModulesSxSelectedUnitImperial, setReactangleSectionModulesSxSelectedUnitImperial] = useState('in³');
  const handleReactangleSectionModulesSxUnitChangeImperial = (units) => {
    setReactangleSectionModulesSxSelectedUnitImperial(units);
    const ReactangleSectionModulesSxfactors = ReactangleSectionModulesSxConversionUnitImperial[units][ReactangleSectionModulesSxUnitsImperial.indexOf(ReactangleSectionModulesSxSelectedUnitImperial)];
    setReactangleSectionModulesSxImperial((parseFloat(ReactangleSectionModulesSxImperial) / ReactangleSectionModulesSxfactors));
  };

  const calculateReactangleSectionModulesSxInputChangeImperial = () => {
    const d = parseFloat(RectangleHeightInputValueImperial);
    const b = parseFloat(RectangleinputValueImperial)
    const ReactangleSectionModulesSxImperial = (b * Math.pow(d, 2) / 6).toFixed(2);
    setReactangleSectionModulesSxImperial(ReactangleSectionModulesSxImperial);

  };

  useEffect(() => {
    calculateReactangleSectionModulesSxInputChangeImperial();
  }, [ReactanglemomentOfInertiaIxImperial, RectangleHeightInputValueImperial]);

  // SectionModules Sy
  const ReactangleSectionModulesSyUnitsImperial = ['in³'];
  const ReactangleSectionModulesSyConversionUnitImperial = {
    'in³': [1],
  };
  const [ReactangleSectionModulesSyImperial, setReactangleSectionModulesSyImperial] = useState(0);
  const [ReactangleSectionModulesSySelectedUnitImperial, setReactangleSectionModulesSySelectedUnitImperial] = useState('in³');
  const handleReactangleSectionModulesSyUnitChangeImperial = (units) => {
    setReactangleSectionModulesSySelectedUnitImperial(units);
    const ReactangleSectionModulesSyfactors = ReactangleSectionModulesSyConversionUnitImperial[units][ReactangleSectionModulesSyUnitsImperial.indexOf(ReactangleSectionModulesSySelectedUnitImperial)];
    setReactangleSectionModulesSyImperial((parseFloat(ReactangleSectionModulesSyImperial) / ReactangleSectionModulesSyfactors));
  };

  const calculateReactangleSectionModulesSyInputChangeImperial = () => {
    const d = parseFloat(RectangleHeightInputValueImperial);
    const b = parseFloat(RectangleinputValueImperial)
    const ReactangleSectionModulesSyImperial = (d * Math.pow(b, 2) / 6).toFixed(2);
    setReactangleSectionModulesSyImperial(ReactangleSectionModulesSyImperial);

  };

  useEffect(() => {
    calculateReactangleSectionModulesSyInputChangeImperial();
  }, [ReactanglemomentOfInertiaIyImperial, RectangleinputValueImperial]);


  const ReactangleTorsionalConstantUnitsImperial = ['in⁴'];
  const ReactangleTorsionalConstantConversionFactorsImperial = {
    'in⁴': [1],
  };

  const [ReactangletorsionalConstantImperial, setReactangleTorsionalConstantImperial] = useState(0);
  const [ReactangletorsionalConstantSelectedUnitImperial, setReactangleTorsionalConstantSelectedUnitImperial] = useState('in⁴');

  const handleReactangleTorsionalConstantUnitChangeImperial = (newUnit) => {
    const currentFactor = ReactangleTorsionalConstantConversionFactorsImperial[ReactangletorsionalConstantSelectedUnitImperial][ReactangleTorsionalConstantUnitsImperial.indexOf(newUnit)];
    const convertedValue = parseFloat(ReactangletorsionalConstantImperial) / currentFactor;
    setReactangleTorsionalConstantSelectedUnitImperial(newUnit);
    setReactangleTorsionalConstantImperial(convertedValue.toString());
  };

  const calculateReactangleTorsionalConstantImperial = (b, d) => {
    const parsedB = parseFloat(b);
    const parsedD = parseFloat(d);

    if (!isNaN(parsedB) && !isNaN(parsedD)) {
      const ReactangletorsionalConstant = ((parsedB * Math.pow(parsedD, 3)) / 3 - 0.21 * Math.pow(parsedD, 4) + 0.0175 * Math.pow(parsedD, 8) / Math.pow(parsedB, 4)).toFixed(2);
      const currentFactor = ReactangleTorsionalConstantConversionFactorsImperial['in⁴'][ReactangleTorsionalConstantUnitsImperial.indexOf(ReactangletorsionalConstantSelectedUnitImperial)];
      setReactangleTorsionalConstantImperial((ReactangletorsionalConstant * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateReactangleTorsionalConstantImperial(RectangleinputValueImperial, RectangleHeightInputValueImperial);
  }, [RectangleinputValueImperial, RectangleHeightInputValueImperial, ReactangletorsionalConstantSelectedUnitImperial]);


  // hollow rectangle calculation

  const HollowReactangleUnits = ['mm', 'cm', 'm'];
  const HollowReactangleConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [HollowReactangleinputValue, setHollowReactangleInputValue] = useState(0);
  const [HollowReactangleselectedUnit, setHollowReactangleSelectedUnit] = useState('mm');
  const [internalHollowReactangleinputValue, setInternalHollowReactangleinputValue] = useState(0); // Always in mm

  const handleHollowReactangleInputChange = (value) => {
    setHollowReactangleInputValue(value);
    const factor = HollowReactangleConversionFactors[HollowReactangleselectedUnit][0];
    setInternalHollowReactangleinputValue(parseFloat(value) * factor);
  };

  const handleHollowReactangleUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(HollowReactangleinputValue) * HollowReactangleConversionFactors[HollowReactangleselectedUnit][0];
    const convertedValue = newMetricValueInMM / HollowReactangleConversionFactors[unit][0];
    setHollowReactangleSelectedUnit(unit);
    setHollowReactangleInputValue(convertedValue.toFixed(5));
  };


  const HollowReactangleHeightunits = ['mm', 'cm', 'm'];
  const HollowReactangleHeightConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [HollowReactangleHeightInputValue, setHollowReactangleHeightInputValue] = useState(0);
  const [HollowReactangleheightSelectedUnit, setHollowReactangleheightSelectedUnit] = useState('mm');
  const [internalHollowReactangleHeightInputValue, setinternalHollowReactangleHeightInputValue] = useState(0); // Always in mm

  const handleHollowReactangleHeightInputValue = (value) => {
    setHollowReactangleHeightInputValue(value);
    const factor = HollowReactangleHeightConversionFactors[HollowReactangleheightSelectedUnit][0];
    setinternalHollowReactangleHeightInputValue(parseFloat(value) * factor);
  };

  const handleHollowReactangleHeightSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(HollowReactangleHeightInputValue) * HollowReactangleHeightConversionFactors[HollowReactangleheightSelectedUnit][0];
    const convertedValue = newMetricValueInMM / HollowReactangleHeightConversionFactors[unit][0];
    setHollowReactangleheightSelectedUnit(unit);
    setHollowReactangleHeightInputValue(convertedValue.toFixed(5));
  };



  const HollowReactangleInnerunits = ['mm', 'cm', 'm'];
  const HollowReactangleInnerconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [inputHollowReactangleInnerValue, setinputHollowReactangleInnerValue] = useState(0);
  const [selectedHollowReactangleInnerUnit, setSelectedHollowReactangleInnerUnit] = useState('mm');
  const [internalinputHollowReactangleInnerValue, setInternalinputHollowReactangleInnerValue] = useState(0); // Always in mm

  const handleInputHollowReactangleinnerChange = (value) => {
    setinputHollowReactangleInnerValue(value);
    const factor = HollowReactangleInnerconversionFactors[selectedHollowReactangleInnerUnit][0];
    setInternalinputHollowReactangleInnerValue(parseFloat(value) * factor);
  };

  const handleUnitHollowReactangleinnerChange = (unit) => {
    const newMetricValueInMM = parseFloat(inputHollowReactangleInnerValue) * HollowReactangleInnerconversionFactors[selectedHollowReactangleInnerUnit][0];
    const convertedValue = newMetricValueInMM / HollowReactangleInnerconversionFactors[unit][0];
    setSelectedHollowReactangleInnerUnit(unit);
    setinputHollowReactangleInnerValue(convertedValue.toFixed(5));
  };

  const HollowReactangleHeightInnerunits = ['mm', 'cm', 'm'];
  const HollowReactangleHeightInnerConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [HollowReactangleHeightInnerInputValue, setHollowReactangleHeightInnerInputValue] = useState(0);
  const [HollowReactangleheightinnerSelectedUnit, setHollowReactangleheightinnerSelectedUnit] = useState('mm');
  const [internalHollowReactangleHeightInnerInputValue, setInternalHollowReactangleHeightInnerInputValue] = useState(0); // Always in mm

  const handleHollowReactangleHeightinnerInputValue = (value) => {
    setHollowReactangleHeightInnerInputValue(value);
    const factor = HollowReactangleHeightInnerConversionFactors[HollowReactangleheightinnerSelectedUnit][0];
    setInternalHollowReactangleHeightInnerInputValue(parseFloat(value) * factor);
  };

  const handleHollowReactangleHeightinnerSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(HollowReactangleHeightInnerInputValue) * HollowReactangleHeightInnerConversionFactors[HollowReactangleheightinnerSelectedUnit][0];
    const convertedValue = newMetricValueInMM / HollowReactangleHeightInnerConversionFactors[unit][0];
    setHollowReactangleheightinnerSelectedUnit(unit);
    setHollowReactangleHeightInnerInputValue(convertedValue.toFixed(5));
  };
  // Area
  const HollowReactangleAreaUnits = ['mm²', 'cm²', 'm²'];
  const HollowReactangleAreaConversionFactors = {
    'mm²': [1, 0.01, 0.000001],
    'cm²': [100, 1, 0.0001],
    'm²': [1000000, 10000, 1],
  };
  const [HollowReactangleArea, setHollowReactangleArea] = useState(0);
  const [HollowReactangleAreaUnit, setHollowReactangleAreaUnit] = useState('mm²');

  useEffect(() => {
    const value = parseFloat(internalHollowReactangleinputValue) * HollowReactangleHeightConversionFactors[HollowReactangleheightSelectedUnit][0];
    if (!isNaN(value)) {
      const b = internalHollowReactangleinputValue
      const d = internalHollowReactangleHeightInputValue
      const bi = internalinputHollowReactangleInnerValue
      const di = internalHollowReactangleHeightInnerInputValue
      const HollowReactangleareaValue = (d * b) - (di * bi);
      setHollowReactangleArea((HollowReactangleareaValue / HollowReactangleAreaConversionFactors[HollowReactangleAreaUnit][0]));
    } else {
      setHollowReactangleArea('');
    }
  }, [internalHollowReactangleinputValue, internalinputHollowReactangleInnerValue, internalHollowReactangleHeightInnerInputValue, internalHollowReactangleHeightInputValue, HollowReactangleAreaUnit]);


  // CentroidXc
  const HollowReactangleCentroidXcUnits = ['mm', 'cm', 'm'];

  const [HollowReactanglecentroidXc, setHollowReactangleCentroidXc] = useState(0);
  const [HollowReactanglecentroidXcSelectedUnit, setHollowReactangleCentroidXcSelectedUnit] = useState('mm');
  useEffect(() => {
    const value = internalHollowReactangleinputValue;
    const centroidValue = (value / 2);
    setHollowReactangleCentroidXc((centroidValue / HollowReactangleConversionFactors[HollowReactanglecentroidXcSelectedUnit][0]));
  }, [internalHollowReactangleinputValue, internalinputHollowReactangleInnerValue, internalHollowReactangleHeightInnerInputValue, internalHollowReactangleHeightInputValue, HollowReactanglecentroidXcSelectedUnit]);

  // CentroidYc
  const HollowReactangleCentroidYcUnits = ['mm', 'cm', 'm'];

  const [HollowReactanglecentroidYc, setHollowReactangleCentroidYc] = useState(0);
  const [HollowReactanglecentroidYcSelectedUnit, setHollowReactangleCentroidYcSelectedUnit] = useState('mm');

  // CentriodYcUnits

  useEffect(() => {
    const d = internalHollowReactangleHeightInputValue;
    const centroidValue = (d / 2);
    setHollowReactangleCentroidYc((centroidValue / HollowReactangleConversionFactors[HollowReactanglecentroidYcSelectedUnit][0]));
  }, [internalHollowReactangleinputValue, internalinputHollowReactangleInnerValue, internalHollowReactangleHeightInnerInputValue, internalHollowReactangleHeightInputValue, HollowReactanglecentroidYcSelectedUnit]);

  // Ix
  const HollowReactangleMomentOfInertiaIxUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const HollowReactangleMomentOfInertiaIxConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [HollowReactanglemomentOfInertiaIx, setHollowReactangleMomentOfInertiaIx] = useState(0);
  const [HollowReactanglemomentOfInertiaIxSelectedUnit, setHollowReactangleMomentOfInertiaIxSelectedUnit] = useState('mm⁴');
  const HollowReactangleb = internalHollowReactangleinputValue;
  const HollowReactangled = internalHollowReactangleHeightInputValue;
  const HollowReactanglebi = internalinputHollowReactangleInnerValue;
  const HollowReactangledi = internalHollowReactangleHeightInnerInputValue;

  useEffect(() => {
    const value = HollowReactangleb;
    if (!isNaN(value)) {
      const HollowReactanglemomentOfInertiaIx = ((HollowReactangleb * Math.pow(HollowReactangled, 3)) - (HollowReactanglebi * Math.pow(HollowReactangledi, 3))) / 12; // Removed .toFixed(1) here

      // Convert HollowReactanglemomentOfInertiaIx based on the selected unit
      const convertedValue = (HollowReactanglemomentOfInertiaIx * HollowReactangleMomentOfInertiaIxConversionFactors['mm⁴'][HollowReactangleMomentOfInertiaIxUnits.indexOf(HollowReactanglemomentOfInertiaIxSelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = HollowReactanglemomentOfInertiaIxSelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setHollowReactangleMomentOfInertiaIx(formattedValue);
    }
  }, [HollowReactangleb, HollowReactangled, HollowReactanglebi, HollowReactangledi, HollowReactanglemomentOfInertiaIxSelectedUnit]);

  // Moment of Inertia Iy
  const HollowReactangleMomentOfInertiaIyUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const HollowReactangleMomentOfInertiaIyConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [HollowReactanglemomentOfInertiaIy, setHollowReactangleMomentOfInertiaIy] = useState(0);
  const [HollowReactanglemomentOfInertiaIySelectedUnit, setHollowReactangleMomentOfInertiaIySelectedUnit] = useState('mm⁴');

  useEffect(() => {
    const value = HollowReactangleb;
    if (!isNaN(value)) {
      const HollowReactanglemomentOfInertiaIy = ((HollowReactangled * Math.pow(HollowReactangleb, 3)) - (HollowReactangledi * Math.pow(HollowReactanglebi, 3))) / 12; // Removed .toFixed(1) here

      // Convert HollowReactanglemomentOfInertiaIy based on the selected unit
      const convertedValue = (HollowReactanglemomentOfInertiaIy * HollowReactangleMomentOfInertiaIyConversionFactors['mm⁴'][HollowReactangleMomentOfInertiaIyUnits.indexOf(HollowReactanglemomentOfInertiaIySelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = HollowReactanglemomentOfInertiaIySelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setHollowReactangleMomentOfInertiaIy(formattedValue);
    }
  }, [HollowReactangleb, HollowReactangled, HollowReactanglebi, HollowReactangledi, HollowReactanglemomentOfInertiaIySelectedUnit]);


  // Section Modulus Sx
  const HollowReactangleSectionModulusSxUnits = ['mm³', 'cm³', 'm³'];
  const HollowReactangleSectionModulusSxConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };

  const [HollowReactanglesectionModulusSx, setHollowReactangleSectionModulusSx] = useState(0);
  const [HollowReactanglesectionModulusSxSelectedUnit, setHollowReactangleSectionModulusSxSelectedUnit] = useState('mm³');

  useEffect(() => {
    const value = HollowReactangleb;
    if (!isNaN(value)) {
      const momentOfInertia = ((HollowReactangleb * Math.pow(HollowReactangled, 3)) - (HollowReactanglebi * Math.pow(HollowReactangledi, 3))) / 12;
      const centroid = HollowReactangled / 2;
      const sectionModulesValue = momentOfInertia / centroid; // Removed .toFixed(1) here

      // Convert sectionModulesValue based on the selected unit
      const convertedValue = (sectionModulesValue * HollowReactangleSectionModulusSxConversionFactors['mm³'][HollowReactangleSectionModulusSxUnits.indexOf(HollowReactanglesectionModulusSxSelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = HollowReactanglesectionModulusSxSelectedUnit === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m³
        : convertedValue.toFixed(3); // 3 decimals for other units

      setHollowReactangleSectionModulusSx(formattedValue);
    }
  }, [HollowReactangleb, HollowReactangled, HollowReactanglebi, HollowReactangledi, HollowReactanglesectionModulusSxSelectedUnit]);

  //  Sy

  const HollowReactangleSectionModulusSyUnits = ['mm³', 'cm³', 'm³'];
  const HollowReactangleSectionModulusSyConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };

  const [HollowReactangleSectionModulusSy, setHollowReactangleSectionModulusSy] = useState(0);
  const [HollowReactangleSectionModulusSySelectedUnit, setHollowReactangleSectionModulusSySelectedUnit] = useState('mm³');

  useEffect(() => {
    const value = HollowReactangleb;
    if (!isNaN(value)) {
      const momentOfInertia = ((HollowReactangled * Math.pow(HollowReactangleb, 3)) - (HollowReactangledi * Math.pow(HollowReactanglebi, 3))) / 12;
      const centroid = HollowReactangleb / 2;
      const sectionModulesValue = momentOfInertia / centroid; // Removed .toFixed(1) here

      // Convert sectionModulesValue based on the selected unit
      const convertedValue = (sectionModulesValue * HollowReactangleSectionModulusSyConversionFactors['mm³'][HollowReactangleSectionModulusSyUnits.indexOf(HollowReactangleSectionModulusSySelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = HollowReactangleSectionModulusSySelectedUnit === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m³
        : convertedValue.toFixed(3); // 3 decimals for other units

      setHollowReactangleSectionModulusSy(formattedValue);
    }
  }, [HollowReactangleb, HollowReactangled, HollowReactanglebi, HollowReactangledi, HollowReactangleSectionModulusSySelectedUnit]);

  // Torsional Constant
  const HollowReactangleTorsionalConstantUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const HollowReactangleTorsionalConstantConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [HollowReactangletorsionalConstant, setHollowReactangleTorsionalConstant] = useState(0);
  const [HollowReactangletorsionalConstantSelectedUnit, setHollowReactangleTorsionalConstantSelectedUnit] = useState('mm⁴');

  useEffect(() => {
    const value = HollowReactangleb;
    if (!isNaN(value)) {
      const torsional = (2 * Math.PI * Math.pow(HollowReactanglebi * (HollowReactangleb - HollowReactangledi), 2) * Math.pow(HollowReactangled - HollowReactanglebi, 2)) / (HollowReactangleb * HollowReactangledi + HollowReactangled * HollowReactanglebi - Math.pow(HollowReactangledi, 2) - Math.pow(HollowReactanglebi, 2));

      // Convert torsional based on the selected unit
      const convertedValue = (torsional * HollowReactangleTorsionalConstantConversionFactors['mm⁴'][HollowReactangleTorsionalConstantUnits.indexOf(HollowReactangletorsionalConstantSelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = HollowReactangletorsionalConstantSelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setHollowReactangleTorsionalConstant(formattedValue);
    }
  }, [HollowReactangleb, HollowReactangled, HollowReactanglebi, HollowReactangledi, HollowReactangletorsionalConstantSelectedUnit]);

  // imperial units


  const HollowReactangleImperialunits = ['in'];

  const HollowReactangleImperialconversionFactors = {
    in: [1],
  };

  const [HollowReactangleinputValueImperial, setHollowReactangleInputValueImperial] = useState(0);
  const [HollowReactangleselectedUnitImperial, setHollowReactangleSelectedUnitImperial] = useState('in');

  const handleHollowReactangleInputChangeImperial = (value) => {
    setHollowReactangleInputValueImperial(value);
  };

  const handleHollowReactangleUnitChangeImperial = (unit) => {
    setHollowReactangleSelectedUnitImperial(unit);
    const factor = HollowReactangleImperialconversionFactors[unit][HollowReactangleImperialunits.indexOf(HollowReactangleselectedUnitImperial)];
    setHollowReactangleInputValueImperial((parseFloat(HollowReactangleinputValueImperial) / factor));
  };

  const HollowReactangleHeightUnitsImperial = ['in'];

  const HollowReactangleHeightConversionFactorsImperial = {
    in: [1],
  };

  const [HollowReactangleHeightInputValueImperial, setHollowReactangleHeightInputValueImperial] = useState(0);
  const [HollowReactangleheightSelectedUnitImperial, setHollowReactangleHeightSelectedUnitImperial] = useState('in');

  const handleHollowReactangleHeightInputChangeImperial = (value) => {
    setHollowReactangleHeightInputValueImperial(value);
  };

  const handleHollowReactangleHeightUnitChangeImperial = (unit) => {
    setHollowReactangleHeightSelectedUnitImperial(unit);
    const factor = HollowReactangleHeightConversionFactorsImperial[unit][HollowReactangleHeightUnitsImperial.indexOf(HollowReactangleheightSelectedUnitImperial)];
    setHollowReactangleHeightInputValueImperial((parseFloat(HollowReactangleHeightInputValueImperial) / factor));
  };

  const HollowReactangleInnerUnitsImperial = ['in'];

  const HollowReactangleInnerConversionFactorsImperial = {
    in: [1],
  };

  const [inputHollowReactangleInnerValueImperial, setInputHollowReactangleInnerValueImperial] = useState(0);
  const [selectedHollowReactangleInnerUnitImperial, setSelectedHollowReactangleInnerUnitImperial] = useState('in');

  const handleInputHollowReactangleInnerChangeImperial = (value) => {
    setInputHollowReactangleInnerValueImperial(value);
  };

  const handleUnitHollowReactangleInnerChangeImperial = (unit) => {
    setSelectedHollowReactangleInnerUnitImperial(unit);
    const factor = HollowReactangleInnerConversionFactorsImperial[unit][HollowReactangleInnerUnitsImperial.indexOf(selectedHollowReactangleInnerUnitImperial)];
    setInputHollowReactangleInnerValueImperial((parseFloat(inputHollowReactangleInnerValueImperial) / factor));
  };

  const HollowReactangleHeightInnerUnitsImperial = ['in'];

  const HollowReactangleHeightInnerConversionFactorsImperial = {
    in: [1],
  };

  const [HollowReactangleHeightInnerInputValueImperial, setHollowReactangleHeightInnerInputValueImperial] = useState(0);
  const [HollowReactangleheightInnerSelectedUnitImperial, setHollowReactangleHeightInnerSelectedUnitImperial] = useState('in');

  const handleHollowReactangleHeightInnerInputChangeImperial = (value) => {
    setHollowReactangleHeightInnerInputValueImperial(value);
  };

  const handleHollowReactangleHeightInnerUnitChangeImperial = (unit) => {
    setHollowReactangleHeightInnerSelectedUnitImperial(unit);
    const factor = HollowReactangleHeightInnerConversionFactorsImperial[unit][HollowReactangleHeightInnerUnitsImperial.indexOf(HollowReactangleheightInnerSelectedUnitImperial)];
    setHollowReactangleHeightInnerInputValueImperial((parseFloat(HollowReactangleHeightInnerInputValueImperial) / factor));
  };

  // Area
  const HollowReactangleAreaUnitsImperial = ['in²'];
  const HollowReactangleAreaConversionFactorsImperial = {
    'in²': [1],
  };

  const [HollowReactangleareaImperial, setHollowReactangleAreaImperial] = useState(0);
  const [HollowReactangleareaUnitImperial, setHollowReactangleAreaUnitImperial] = useState('in²');

  const handleHollowReactangleAreaUnitChangeImperial = (unit) => {
    const conversionFactor = HollowReactangleAreaConversionFactorsImperial[unit][HollowReactangleAreaUnitsImperial.indexOf(HollowReactangleareaUnitImperial)];
    const convertedValue = parseFloat(HollowReactangleareaImperial) / conversionFactor;
    setHollowReactangleAreaUnitImperial(unit);
    setHollowReactangleAreaImperial(convertedValue.toString());
  };

  const calculateHollowReactangleAreaImperial = () => {
    const b = parseFloat(HollowReactangleinputValueImperial);
    const d = parseFloat(HollowReactangleHeightInputValueImperial);
    const bi = parseFloat(inputHollowReactangleInnerValueImperial);
    const di = parseFloat(HollowReactangleHeightInnerInputValueImperial);

    if (!isNaN(b) && !isNaN(d) && !isNaN(bi) && !isNaN(di)) {
      const calculatedHollowReactangleArea = (d * b) - (di * bi);
      setHollowReactangleAreaImperial(calculatedHollowReactangleArea);
    }
  };

  useEffect(() => {
    calculateHollowReactangleAreaImperial();
  }, [HollowReactangleinputValueImperial, HollowReactangleHeightInputValueImperial, inputHollowReactangleInnerValueImperial, HollowReactangleHeightInnerInputValueImperial]);

  // CentroidXc
  const HollowReactangleCentroidXcUnitsImperial = ['in'];
  const HollowReactangleCentroidXcConversionFactorsImperial = {
    in: [1],
  };
  const [HollowReactanglecentroidXcImperial, setHollowReactangleCentroidXcImperial] = useState(0);
  const [HollowReactanglecentroidXcSelectedUnitImperial, setHollowReactangleCentroidXcSelectedUnitImperial] = useState('in');

  const handleHollowReactangleCentroidXcUnitChangeImperial = (unit) => {
    const conversionFactor = HollowReactangleCentroidXcConversionFactorsImperial[unit][HollowReactangleCentroidXcUnitsImperial.indexOf(HollowReactanglecentroidXcSelectedUnitImperial)];
    const convertedValue = parseFloat(HollowReactanglecentroidXcImperial) / conversionFactor;
    setHollowReactangleCentroidXcSelectedUnitImperial(unit);
    setHollowReactangleCentroidXcImperial(convertedValue.toString());
  };

  const calculateHollowReactangleCentroidXcImperial = () => {
    const b = parseFloat(HollowReactangleinputValueImperial);
    if (!isNaN(b)) {
      const calculatedHollowReactangleCentroidXc = b / 2;
      setHollowReactangleCentroidXcImperial(calculatedHollowReactangleCentroidXc);
    }
  };

  useEffect(() => {
    calculateHollowReactangleCentroidXcImperial();
  }, [HollowReactangleinputValueImperial]);

  // CentroidYc
  const HollowReactangleCentroidYcUnitsImperial = ['in'];
  const HollowReactangleCentroidYcConversionFactorsImperial = {
    in: [1],
  };
  const [HollowReactanglecentroidYcImperial, setHollowReactangleCentroidYcImperial] = useState(0);
  const [HollowReactanglecentroidYcSelectedUnitImperial, setHollowReactangleCentroidYcSelectedUnitImperial] = useState('in');

  const handleHollowReactangleCentroidYcUnitChangeImperial = (unit) => {
    const conversionFactor = HollowReactangleCentroidYcConversionFactorsImperial[unit][HollowReactangleCentroidYcUnitsImperial.indexOf(HollowReactanglecentroidYcSelectedUnitImperial)];
    const convertedValue = parseFloat(HollowReactanglecentroidYcImperial) / conversionFactor;
    setHollowReactangleCentroidYcSelectedUnitImperial(unit);
    setHollowReactangleCentroidYcImperial(convertedValue.toString());
  };

  const calculateHollowReactangleCentroidYcImperial = () => {
    const d = parseFloat(HollowReactangleHeightInputValueImperial);
    if (!isNaN(d)) {
      const calculatedHollowReactangleCentroidYc = d / 2;
      setHollowReactangleCentroidYcImperial(calculatedHollowReactangleCentroidYc);
    }
  };

  useEffect(() => {
    calculateHollowReactangleCentroidYcImperial();
  }, [HollowReactangleHeightInputValueImperial]);

  // Moment of Inertia Ix
  const HollowReactangleMomentOfInertiaIxUnitsImperial = ['in⁴'];
  const HollowReactangleMomentOfInertiaIxConversionFactorsImperial = {
    'in⁴': [1],
  };
  const [HollowReactanglemomentOfInertiaIxImperial, setHollowReactangleMomentOfInertiaIxImperial] = useState(0);
  const [HollowReactanglemomentOfInertiaIxSelectedUnitImperial, setHollowReactangleMomentOfInertiaIxSelectedUnitImperial] = useState('in⁴');

  const handleHollowReactangleMomentOfInertiaIxUnitChangeImperial = (unit) => {
    const conversionFactor = HollowReactangleMomentOfInertiaIxConversionFactorsImperial[unit][HollowReactangleMomentOfInertiaIxUnitsImperial.indexOf(HollowReactanglemomentOfInertiaIxSelectedUnitImperial)];
    const convertedValue = parseFloat(HollowReactanglemomentOfInertiaIxImperial) / conversionFactor;
    setHollowReactangleMomentOfInertiaIxSelectedUnitImperial(unit);
    setHollowReactangleMomentOfInertiaIxImperial(convertedValue.toString());
  };

  const calculateHollowReactangleMomentOfInertiaIxImperial = () => {
    const b = parseFloat(HollowReactangleinputValueImperial);
    const d = parseFloat(HollowReactangleHeightInputValueImperial);
    const bi = parseFloat(inputHollowReactangleInnerValueImperial);
    const di = parseFloat(HollowReactangleHeightInnerInputValueImperial);

    if (!isNaN(b) && !isNaN(d) && !isNaN(bi) && !isNaN(di)) {
      const calculatedHollowReactangleMomentOfInertiaIx = (((b * Math.pow(d, 3)) - (bi * Math.pow(di, 3))) / 12).toFixed(1);
      setHollowReactangleMomentOfInertiaIxImperial(calculatedHollowReactangleMomentOfInertiaIx);
    }
  };

  useEffect(() => {
    calculateHollowReactangleMomentOfInertiaIxImperial();
  }, [HollowReactangleinputValueImperial, HollowReactangleHeightInputValueImperial, inputHollowReactangleInnerValueImperial, HollowReactangleHeightInnerInputValueImperial]);

  // Moment of Inertia Iy
  const HollowReactangleMomentOfInertiaIyUnitsImperial = ['in⁴'];
  const HollowReactangleMomentOfInertiaIyConversionFactorsImperial = {
    'in⁴': [1],
  };
  const [HollowReactanglemomentOfInertiaIyImperial, setHollowReactangleMomentOfInertiaIyImperial] = useState(0);
  const [HollowReactanglemomentOfInertiaIySelectedUnitImperial, setHollowReactangleMomentOfInertiaIySelectedUnitImperial] = useState('in⁴');

  const handleHollowReactangleMomentOfInertiaIyUnitChangeImperial = (unit) => {
    const conversionFactor = HollowReactangleMomentOfInertiaIyConversionFactorsImperial[unit][HollowReactangleMomentOfInertiaIyUnitsImperial.indexOf(HollowReactanglemomentOfInertiaIySelectedUnitImperial)];
    const convertedValue = parseFloat(HollowReactanglemomentOfInertiaIyImperial) / conversionFactor;
    setHollowReactangleMomentOfInertiaIySelectedUnitImperial(unit);
    setHollowReactangleMomentOfInertiaIyImperial(convertedValue.toString());
  };

  const calculateHollowReactangleMomentOfInertiaIyImperial = () => {
    const b = parseFloat(HollowReactangleinputValueImperial);
    const d = parseFloat(HollowReactangleHeightInputValueImperial);
    const bi = parseFloat(inputHollowReactangleInnerValueImperial);
    const di = parseFloat(HollowReactangleHeightInnerInputValueImperial);

    if (!isNaN(b) && !isNaN(d) && !isNaN(bi) && !isNaN(di)) {
      const calculatedHollowReactangleMomentOfInertiaIy = (((d * Math.pow(b, 3)) - (di * Math.pow(bi, 3))) / 12).toFixed(1);
      setHollowReactangleMomentOfInertiaIyImperial(calculatedHollowReactangleMomentOfInertiaIy);
    }
  };

  useEffect(() => {
    calculateHollowReactangleMomentOfInertiaIyImperial();
  }, [HollowReactangleinputValueImperial, HollowReactangleHeightInputValueImperial, inputHollowReactangleInnerValueImperial, HollowReactangleHeightInnerInputValueImperial]);

  // Section Modulus Sx
  const HollowReactangleSectionModulusSxUnitsImperial = ['in³'];
  const HollowReactangleSectionModulusSxConversionFactorsImperial = {
    'in³': [1],
  };
  const [HollowReactanglesectionModulusSxImperial, setHollowReactangleSectionModulusSxImperial] = useState(0);
  const [HollowReactanglesectionModulusSxSelectedUnitImperial, setHollowReactangleSectionModulusSxSelectedUnitImperial] = useState('in³');

  const handleHollowReactangleSectionModulusSxUnitChangeImperial = (unit) => {
    const conversionFactor = HollowReactangleSectionModulusSxConversionFactorsImperial[unit][HollowReactangleSectionModulusSxUnitsImperial.indexOf(HollowReactanglesectionModulusSxSelectedUnitImperial)];
    const convertedValue = parseFloat(HollowReactanglesectionModulusSxImperial) / conversionFactor;
    setHollowReactangleSectionModulusSxSelectedUnitImperial(unit);
    setHollowReactangleSectionModulusSxImperial(convertedValue.toString());
  };

  const calculateHollowReactangleSectionModulusSxImperial = () => {
    const b = parseFloat(HollowReactangleinputValueImperial);
    const d = parseFloat(HollowReactangleHeightInputValueImperial);
    const bi = parseFloat(inputHollowReactangleInnerValueImperial);
    const di = parseFloat(HollowReactangleHeightInnerInputValueImperial);
    const momentOfInertia = ((b * Math.pow(d, 3)) - (bi * Math.pow(di, 3))) / 12;
    const centroid = d / 2;

    if (!isNaN(momentOfInertia) && !isNaN(centroid) && centroid !== 0) {
      const calculatedHollowReactangleSectionModulusSx = (momentOfInertia / centroid).toFixed(2);
      setHollowReactangleSectionModulusSxImperial(calculatedHollowReactangleSectionModulusSx);
    }
  };

  useEffect(() => {
    calculateHollowReactangleSectionModulusSxImperial();
  }, [HollowReactanglemomentOfInertiaIxImperial, HollowReactanglecentroidYcImperial]);




  // Section Modulus Sx
  const HollowReactangleSectionModulusSyUnitsImperial = ['in³'];
  const HollowReactangleSectionModulusSyConversionFactorsImperial = {
    'in³': [1],
  };
  const [HollowReactanglesectionModulusSyImperial, setHollowReactangleSectionModulusSyImperial] = useState(0);
  const [HollowReactanglesectionModulusSySelectedUnitImperial, setHollowReactangleSectionModulusSySelectedUnitImperial] = useState('in³');

  const handleHollowReactangleSectionModulusSyUnitChangeImperial = (unit) => {
    const conversionFactor = HollowReactangleSectionModulusSyConversionFactorsImperial[unit][HollowReactangleSectionModulusSyUnitsImperial.indexOf(HollowReactanglesectionModulusSySelectedUnitImperial)];
    const convertedValue = parseFloat(HollowReactanglesectionModulusSyImperial) / conversionFactor;
    setHollowReactangleSectionModulusSySelectedUnitImperial(unit);
    setHollowReactangleSectionModulusSyImperial(convertedValue.toString());
  };

  const calculateHollowReactangleSectionModulusSyImperial = () => {
    const b = parseFloat(HollowReactangleinputValueImperial);
    const d = parseFloat(HollowReactangleHeightInputValueImperial);
    const bi = parseFloat(inputHollowReactangleInnerValueImperial);
    const di = parseFloat(HollowReactangleHeightInnerInputValueImperial);
    const momentOfInertia = ((d * Math.pow(b, 3)) - (di * Math.pow(bi, 3))) / 12;
    const centroid = b / 2;

    if (!isNaN(momentOfInertia) && !isNaN(centroid) && centroid !== 0) {
      const calculatedHollowReactangleSectionModulusSy = (momentOfInertia / centroid).toFixed(2);
      setHollowReactangleSectionModulusSyImperial(calculatedHollowReactangleSectionModulusSy);
    }
  };

  useEffect(() => {
    calculateHollowReactangleSectionModulusSyImperial();
  }, [HollowReactanglemomentOfInertiaIxImperial, HollowReactanglecentroidYcImperial]);



  // Torsional Constant
  const HollowReactangleTorsionalConstantUnitsImperial = ['in⁴'];
  const HollowReactangleTorsionalConstantConversionFactorsImperial = {
    'in⁴': [1],
  };
  const [HollowReactangletorsionalConstantImperial, setHollowReactangleTorsionalConstantImperial] = useState(0);
  const [HollowReactangletorsionalConstantSelectedUnitImperial, setHollowReactangleTorsionalConstantSelectedUnitImperial] = useState('in⁴');

  const handleHollowReactangleTorsionalConstantUnitChangeImperial = (unit) => {
    const conversionFactor = HollowReactangleTorsionalConstantConversionFactorsImperial[unit][HollowReactangleTorsionalConstantUnitsImperial.indexOf(HollowReactangletorsionalConstantSelectedUnitImperial)];
    const convertedValue = parseFloat(HollowReactangletorsionalConstantImperial) / conversionFactor;
    setHollowReactangleTorsionalConstantSelectedUnitImperial(unit);
    setHollowReactangleTorsionalConstantImperial(convertedValue.toString());
  };

  const calculateHollowReactangleTorsionalConstantImperial = () => {
    const b = parseFloat(HollowReactangleinputValueImperial);
    const d = parseFloat(HollowReactangleHeightInputValueImperial);
    const bi = parseFloat(inputHollowReactangleInnerValueImperial);
    const di = parseFloat(HollowReactangleHeightInnerInputValueImperial);

    if (!isNaN(b) && !isNaN(d) && !isNaN(bi) && !isNaN(di)) {
      const calculatedHollowReactangleTorsionalConstant = ((2 * Math.PI * Math.pow(bi * (b - di), 2) * Math.pow(d - bi, 2)) / (b * di + d * bi - Math.pow(di, 2) - Math.pow(bi, 2))).toFixed(2);
      setHollowReactangleTorsionalConstantImperial(calculatedHollowReactangleTorsionalConstant);
    }
  };

  useEffect(() => {
    calculateHollowReactangleTorsionalConstantImperial();
  }, [HollowReactangleinputValueImperial, HollowReactangleHeightInputValueImperial, inputHollowReactangleInnerValueImperial, HollowReactangleHeightInnerInputValueImperial]);


  // tee section calculation

  const TeeSectionunits = ['mm', 'cm', 'm'];
  // const [exponent, setExponent] = useState(4);

  const TeeSectionconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [TeeSectioninputValue, setTeeSectionInputValue] = useState(0);
  const [TeeSectionselectedUnit, setTeeSectionSelectedUnit] = useState('mm');
  const [internalTeeSectioninputValue, setInternalTeeSectioninputValue] = useState(0); // Always in mm

  const handleTeeSectionInputChange = (value) => {
    setTeeSectionInputValue(value);
    const factor = TeeSectionconversionFactors[TeeSectionselectedUnit][0];
    setInternalTeeSectioninputValue(parseFloat(value) * factor);
  };

  const handleTeeSectionUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(TeeSectioninputValue) * TeeSectionconversionFactors[TeeSectionselectedUnit][0];
    const convertedValue = newMetricValueInMM / TeeSectionconversionFactors[unit][0];
    setTeeSectionSelectedUnit(unit);
    setTeeSectionInputValue(convertedValue.toFixed(3));
  };


  const TeeSectionHeightunits = ['mm', 'cm', 'm'];

  const TeeSectionHeightConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [TeeSectionHeightInputValue, setTeeSectionHeightInputValue] = useState(0);
  const [TeeSectionheightSelectedUnit, setTeeSectionHeightSelectedUnit] = useState('mm');
  const [internalTeeSectionHeightInputValue, setInternalTeeSectionHeightInputValue] = useState(0); // Always in mm

  const handleTeeSectionHeightInputValue = (value) => {
    setTeeSectionHeightInputValue(value);
    const factor = TeeSectionHeightConversionFactors[TeeSectionheightSelectedUnit][0];
    setInternalTeeSectionHeightInputValue(parseFloat(value) * factor);
  };

  const handleTeeSectionHeightSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(TeeSectionHeightInputValue) * TeeSectionHeightConversionFactors[TeeSectionheightSelectedUnit][0];
    const convertedValue = newMetricValueInMM / TeeSectionHeightConversionFactors[unit][0];
    setTeeSectionHeightSelectedUnit(unit);
    setTeeSectionHeightInputValue(convertedValue.toFixed(3));
  };



  const TeeSectionInnerunits = ['mm', 'cm', 'm'];
  const TeeSectionInnerconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [inputTeeSectionInnerValue, setinputTeeSectionInnerValue] = useState(0);
  const [selectedTeeSectionInnerUnit, setSelectedTeeSectionInnerUnit] = useState('mm');
  const [internalinputTeeSectionInnerValue, setInternalinputTeeSectionInnerValue] = useState(0); // Always in mm

  const handleInputTeeSectioninnerChange = (value) => {
    setinputTeeSectionInnerValue(value);
    const factor = TeeSectionInnerconversionFactors[selectedTeeSectionInnerUnit][0];
    setInternalinputTeeSectionInnerValue(parseFloat(value) * factor);
  };

  const handleUnitTeeSectioninnerChange = (unit) => {
    const newMetricValueInMM = parseFloat(inputTeeSectionInnerValue) * TeeSectionInnerconversionFactors[selectedTeeSectionInnerUnit][0];
    const convertedValue = newMetricValueInMM / TeeSectionInnerconversionFactors[unit][0];
    setSelectedTeeSectionInnerUnit(unit);
    setinputTeeSectionInnerValue(convertedValue.toFixed(5));
  };

  const TeeSectionHeightInnerunits = ['mm', 'cm', 'm'];
  const TeeSectionHeightInnerConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [TeeSectionHeightInnerInputValue, setTeeSectionHeightInnerInputValue] = useState(0);
  const [TeeSectionheightinnerSelectedUnit, setTeeSectionheightinnerSelectedUnit] = useState('mm');
  const [internalTeeSectionHeightInnerInputValue, setInternalTeeSectionHeightInnerInputValue] = useState(0); // Always in mm

  const handleTeeSectionHeightinnerInputValue = (value) => {
    setTeeSectionHeightInnerInputValue(value);
    const factor = TeeSectionHeightInnerConversionFactors[TeeSectionheightinnerSelectedUnit][0];
    setInternalTeeSectionHeightInnerInputValue(parseFloat(value) * factor);
  };

  const handleTeeSectionHeightinnerSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(TeeSectionHeightInnerInputValue) * TeeSectionHeightInnerConversionFactors[TeeSectionheightinnerSelectedUnit][0];
    const convertedValue = newMetricValueInMM / TeeSectionHeightInnerConversionFactors[unit][0];
    setTeeSectionheightinnerSelectedUnit(unit);
    setTeeSectionHeightInnerInputValue(convertedValue.toFixed(5));
  };

  // Area
  const TeeSectionAreaUnits = ['mm²', 'cm²', 'm²'];
  const TeeSectionAreaConversionFactors = {
    'mm²': [1, 0.01, 0.000001],
    'cm²': [100, 1, 0.0001],
    'm²': [1000000, 10000, 1],
  };
  const [TeeSectionArea, setTeeSectionArea] = useState(0);
  const [TeeSectionAreaUnit, setTeeSectionAreaUnit] = useState('mm²');

  // CentroidXc
  const TeeSectionCentroidXcUnits = ['mm', 'cm', 'm'];

  const [TeeSectioncentroidXc, setTeeSectionCentroidXc] = useState(0);
  const [TeeSectioncentroidXcSelectedUnit, setTeeSectionCentroidXcSelectedUnit] = useState('mm');

  // CentroidYc
  const TeeSectionCentroidYcUnits = ['mm', 'cm', 'm'];

  const [TeeSectioncentroidYc, setTeeSectionCentroidYc] = useState(0);
  const [TeeSectioncentroidYcSelectedUnit, setTeeSectionCentroidYcSelectedUnit] = useState('mm');

  const TeeSectionMomentOfInertiaIxUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const TeeSectionMomentOfInertiaIxConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [TeeSectionmomentOfInertiaIx, setTeeSectionMomentOfInertiaIx] = useState(0);
  const [TeeSectionmomentOfInertiaIxSelectedUnit, setTeeSectionMomentOfInertiaIxSelectedUnit] = useState('mm⁴');


  // Moment of Inertia Iy
  const TeeSectionMomentOfInertiaIyUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const TeeSectionMomentOfInertiaIyConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };
  const [TeeSectionmomentOfInertiaIy, setTeeSectionMomentOfInertiaIy] = useState(0);
  const [TeeSectionmomentOfInertiaIySelectedUnit, setTeeSectionMomentOfInertiaIySelectedUnit] = useState('mm⁴');


  // Section Modulus Sx
  const TeeSectionSectionModulusSxUnits = ['mm³', 'cm³', 'm³'];
  const TeeSectionSectionModulusSxConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [TeeSectionsectionModulusSx, setTeeSectionSectionModulusSx] = useState(0);
  const [TeeSectionsectionModulusSxSelectedUnit, setTeeSectionSectionModulusSxSelectedUnit] = useState('mm³');

  const TeeSectionSectionModulusSyUnits = ['mm³', 'cm³', 'm³'];
  const TeeSectionSectionModulusSyConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [TeeSectionSectionModulusSy, setTeeSectionSectionModulusSy] = useState(0);
  const [TeeSectionSectionModulusSySelectedUnit, setTeeSectionSectionModulusSySelectedUnit] = useState('mm³');


  const TeeSectionb = parseFloat(internalTeeSectioninputValue);
  const TeeSectiond = parseFloat(internalTeeSectionHeightInputValue);
  const TeeSectiont = parseFloat(internalinputTeeSectionInnerValue);
  const TeeSectiontw = parseFloat(internalTeeSectionHeightInnerInputValue);

  useEffect(() => {
    const value = parseFloat(internalTeeSectioninputValue) * TeeSectionHeightConversionFactors[TeeSectionheightSelectedUnit][0];
    if (!isNaN(value)) {
      const TeeSectionareaValue = (TeeSectiont * TeeSectionb) + (TeeSectiontw * TeeSectiond);
      setTeeSectionArea((TeeSectionareaValue / TeeSectionAreaConversionFactors[TeeSectionAreaUnit][0]));
    } else {
      setTeeSectionArea('');
    }
  }, [internalTeeSectioninputValue, internalinputTeeSectionInnerValue, internalTeeSectionHeightInnerInputValue, internalTeeSectionHeightInputValue, TeeSectionAreaUnit]);


  useEffect(() => {
    const value = internalTeeSectioninputValue;
    const centroidValue = (value / 2);
    setTeeSectionCentroidXc((centroidValue / TeeSectionconversionFactors[TeeSectioncentroidXcSelectedUnit][0]));
  }, [internalTeeSectioninputValue, internalinputTeeSectionInnerValue, internalTeeSectionHeightInnerInputValue, internalTeeSectionHeightInputValue, TeeSectioncentroidXcSelectedUnit]);


  // CentriodYcUnits

  useEffect(() => {
    const TeeSectionb = parseFloat(internalTeeSectioninputValue);
    const TeeSectiond = parseFloat(internalTeeSectionHeightInputValue);
    const TeeSectiont = parseFloat(internalinputTeeSectionInnerValue);
    const TeeSectiontw = parseFloat(internalTeeSectionHeightInnerInputValue);

    const numerator = (TeeSectionb * Math.pow(TeeSectiont, 2)) + ((TeeSectiontw * TeeSectiond) * (2 * TeeSectiont + TeeSectiond));
    const denominator = 2 * (TeeSectiont * TeeSectionb + TeeSectiontw * TeeSectiond);
    const calculatedImperialTeeSectionCentroidYc = (numerator / denominator);

    const formattedValue = TeeSectionmomentOfInertiaIxSelectedUnit === 'm'
      ? calculatedImperialTeeSectionCentroidYc.toFixed(3) // 2 decimal places in scientific notation for m⁴
      : calculatedImperialTeeSectionCentroidYc.toFixed(2); // 3 decimals for other units


    setTeeSectionCentroidYc((formattedValue / TeeSectionconversionFactors[TeeSectioncentroidYcSelectedUnit][0]));
  }, [internalTeeSectioninputValue, internalinputTeeSectionInnerValue, internalTeeSectionHeightInnerInputValue, internalTeeSectionHeightInputValue, TeeSectioncentroidYcSelectedUnit]);


  useEffect(() => {
    const value = TeeSectionb;
    if (!isNaN(value)) {
      const numerator = (TeeSectionb * Math.pow(TeeSectiont, 2)) + ((TeeSectiontw * TeeSectiond) * (2 * TeeSectiont + TeeSectiond));
      const denominator = 2 * (TeeSectiont * TeeSectionb + TeeSectiontw * TeeSectiond);
      const Yc = numerator / denominator;
      const term1 = TeeSectionb * Math.pow((TeeSectiond + TeeSectiont), 3) - Math.pow(TeeSectiond, 3) * (TeeSectionb - TeeSectiontw);
      const term2 = 3;
      const Area = (TeeSectiont * TeeSectionb) + (TeeSectiontw * TeeSectiond);
      const term3 = Area * Math.pow((TeeSectiond + TeeSectiont - Yc), 2);
      const TeeSectionmomentOfInertiaIx = ((term1 / term2) - term3);

      // Convert TeeSectionmomentOfInertiaIx based on the selected unit
      const convertedValue = (TeeSectionmomentOfInertiaIx * TeeSectionMomentOfInertiaIxConversionFactors['mm⁴'][TeeSectionMomentOfInertiaIxUnits.indexOf(TeeSectionmomentOfInertiaIxSelectedUnit)]);

      // Format the output based on the selected unit
      const formattedValue = TeeSectionmomentOfInertiaIxSelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places and append 'e' for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setTeeSectionMomentOfInertiaIx(formattedValue);
    }
  }, [TeeSectionb, TeeSectiond, TeeSectiont, TeeSectiontw, TeeSectionmomentOfInertiaIxSelectedUnit]);


  useEffect(() => {
    const value = TeeSectionb;
    if (!isNaN(value)) {
      const term1 = ((TeeSectiont * Math.pow(TeeSectionb, 3)) + (TeeSectiond * Math.pow(TeeSectiontw, 3)));
      const term2 = 12;
      const TeeSectionmomentOfInertiaIy = (term1 / term2);

      const convertedValue = (TeeSectionmomentOfInertiaIy * TeeSectionMomentOfInertiaIyConversionFactors['mm⁴'][TeeSectionMomentOfInertiaIyUnits.indexOf(TeeSectionmomentOfInertiaIySelectedUnit)]);

      const formattedValue = TeeSectionmomentOfInertiaIySelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places and append 'e' for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setTeeSectionMomentOfInertiaIy(formattedValue);
    }
  }, [TeeSectionb, TeeSectiond, TeeSectiont, TeeSectiontw, TeeSectionmomentOfInertiaIySelectedUnit]);

  useEffect(() => {
    const value = TeeSectionb
    if (!isNaN(value)) {
      const numerator = (TeeSectionb * Math.pow(TeeSectiont, 2)) + ((TeeSectiontw * TeeSectiond) * (2 * TeeSectiont + TeeSectiond));
      const denominator = 2 * (TeeSectiont * TeeSectionb + TeeSectiontw * TeeSectiond);
      const Yc = numerator / denominator;
      const term1 = TeeSectionb * Math.pow((TeeSectiond + TeeSectiont), 3) - Math.pow(TeeSectiond, 3) * (TeeSectionb - TeeSectiontw);
      const term2 = 3;
      const area = (TeeSectiont * TeeSectionb) + (TeeSectiontw * TeeSectiond);
      const term3 = area * Math.pow((TeeSectiond + TeeSectiont - Yc), 2);
      const Ix = (term1 / term2) - term3;
      const Sxdenomenator = (TeeSectiond + TeeSectiont) - Yc;
      const sectionModulesValue = (Ix / Sxdenomenator).toFixed(2);
      const convertedValue = (sectionModulesValue * TeeSectionSectionModulusSxConversionFactors['mm³'][TeeSectionSectionModulusSxUnits.indexOf(TeeSectionsectionModulusSxSelectedUnit)]);

      const formattedValue = TeeSectionsectionModulusSxSelectedUnit === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places and append 'e' for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setTeeSectionSectionModulusSx(formattedValue);
    }
  }, [TeeSectionb, TeeSectiond, TeeSectiont, TeeSectiontw, TeeSectionsectionModulusSxSelectedUnit]);

  useEffect(() => {
    const value = TeeSectionb;
    if (!isNaN(value)) {
      const term1 = ((TeeSectiont * Math.pow(TeeSectionb, 3)) + (TeeSectiond * Math.pow(TeeSectiontw, 3)));
      const term2 = 12;
      const Iy = (term1 / term2).toFixed(2);
      const Xc = TeeSectionb / 2;
      const sectionModulesValue = (Iy / Xc).toFixed(4);
      const convertedValue = (sectionModulesValue * TeeSectionSectionModulusSyConversionFactors['mm³'][TeeSectionSectionModulusSyUnits.indexOf(TeeSectionSectionModulusSySelectedUnit)]);

      const formattedValue = TeeSectionSectionModulusSySelectedUnit === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places and append 'e' for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units


      setTeeSectionSectionModulusSy(formattedValue);
    }
  }, [TeeSectionb, TeeSectiond, TeeSectiont, TeeSectiontw, TeeSectionSectionModulusSySelectedUnit]);


  const TeeSectionImperialunits = ['in'];
  // const [exponent, setExponent] = useState(4);

  const TeeSectionImperialconversionFactors = {
    in: [1]
  };

  const [TeeSectionImperialinputValue, setTeeSectionImperialInputValue] = useState(0);
  const [TeeSectionImperialselectedUnit, setTeeSectionImperialSelectedUnit] = useState('in');

  const handleTeeSectionImperialInputChange = (value) => {
    setTeeSectionImperialInputValue(value);
  };

  const handleTeeSectionImperialUnitChange = (unit) => {
    setTeeSectionImperialSelectedUnit(unit);
    const TeeSectionImperialfactor = TeeSectionImperialconversionFactors[unit][TeeSectionImperialunits.indexOf(TeeSectionImperialselectedUnit)];
    setTeeSectionImperialInputValue((parseFloat(TeeSectionImperialinputValue) / TeeSectionImperialfactor).toFixed(4));
  };

  const TeeSectionImperialHeightunits = ['in'];

  const TeeSectionImperialHeightConversionFactors = {
    in: [1]
  };

  const [TeeSectionImperialHeightInputValue, setTeeSectionImperialHeightInputValue] = useState(0);
  const [TeeSectionImperialheightSelectedUnit, setTeeSectionImperialHeightSelectedUnit] = useState('in');

  const handleTeeSectionImperialHeightInputValue = (values) => {
    setTeeSectionImperialHeightInputValue(values);
  };

  const handleTeeSectionImperialHeightSelectedUnit = (units) => {
    setTeeSectionImperialHeightSelectedUnit(units);
    const Hrightfactor = TeeSectionImperialHeightConversionFactors[units][TeeSectionImperialHeightunits.indexOf(TeeSectionImperialheightSelectedUnit)];
    setTeeSectionImperialHeightInputValue((parseFloat(TeeSectionImperialHeightInputValue) / Hrightfactor).toFixed(4));
  };

  const ImperialTeeSectionInnerunits = ['in'];
  // const [exponent, setExponent] = useState(4);

  const ImperialTeeSectionInnerconversionFactors = {
    in: [1]
  };

  const [ImperialinputTeeSectionInnerValue, ImperialsetInputTeeSectionInnerValue] = useState(0);
  const [ImperialselectedTeeSectionInnerUnit, ImperialsetSelectedTeeSectionInnerUnit] = useState('in');

  const handleImperialInputTeeSectioninnerChange = (value) => {
    ImperialsetInputTeeSectionInnerValue(value);
  };

  const handleImperialUnitTeeSectioninnerChange = (unit) => {
    ImperialsetSelectedTeeSectionInnerUnit(unit);
    const TeeSectioninnerfactor = ImperialTeeSectionInnerconversionFactors[unit][ImperialTeeSectionInnerunits.indexOf(ImperialselectedTeeSectionInnerUnit)];
    ImperialsetInputTeeSectionInnerValue((parseFloat(ImperialinputTeeSectionInnerValue) / TeeSectioninnerfactor).toFixed(4));
  };

  const ImperialTeeSectionHeightInnerunits = ['in'];

  const ImperialTeeSectionHeightInnerConversionFactors = {
    in: [1]
  };

  const [ImperialTeeSectionHeightInnerInputValue, setImperialTeeSectionHeightinnerInputValue] = useState(0);
  const [ImperialTeeSectionheightinnerSelectedUnit, setImperialTeeSectionHeightinnerSelectedUnit] = useState('in');

  const handleImperialTeeSectionHeightinnerInputValue = (values) => {
    setImperialTeeSectionHeightinnerInputValue(values);
  };

  const handleImperialTeeSectionHeightinnerSelectedUnit = (units) => {
    setImperialTeeSectionHeightinnerSelectedUnit(units);
    const Hrightfactor = ImperialTeeSectionHeightInnerConversionFactors[units][ImperialTeeSectionHeightInnerunits.indexOf(ImperialTeeSectionheightinnerSelectedUnit)];
    setImperialTeeSectionHeightinnerInputValue((parseFloat(ImperialTeeSectionHeightInnerInputValue) / Hrightfactor).toFixed(4));
  };

  const ImperialTeeSectionAreaUnits = ['in²'];
  const ImperialTeeSectionAreaConversionFactors = {
    'in²': [1],
  };
  const [ImperialTeeSectionArea, setImperialTeeSectionArea] = useState(0);
  const [ImperialTeeSectionAreaUnit, setImperialTeeSectionAreaUnit] = useState('in²');

  const handleImperialTeeSectionAreaUnitChange = (unit) => {
    const currentImperialTeeSectionAreaInNewUnit = ImperialTeeSectionArea * ImperialTeeSectionAreaConversionFactors[ImperialTeeSectionAreaUnit][ImperialTeeSectionAreaUnits.indexOf(unit)];
    setImperialTeeSectionArea(currentImperialTeeSectionAreaInNewUnit);
    setImperialTeeSectionAreaUnit(unit);
  };

  const calculateImperialTeeSectionArea = () => {
    const TeeSectiond = parseFloat(TeeSectionImperialHeightInputValue);
    const TeeSectionb = parseFloat(TeeSectionImperialinputValue);
    const TeeSectiont = parseFloat(ImperialinputTeeSectionInnerValue);
    const TeeSectiontw = parseFloat(ImperialTeeSectionHeightInnerInputValue);
    const calculatedImperialTeeSectionArea = (TeeSectiont * TeeSectionb) + (TeeSectiontw * TeeSectiond);
    setImperialTeeSectionArea(calculatedImperialTeeSectionArea);
  };

  useEffect(() => {
    calculateImperialTeeSectionArea();
  }, [TeeSectionImperialinputValue, TeeSectionImperialHeightInputValue, ImperialinputTeeSectionInnerValue, ImperialTeeSectionHeightInnerInputValue]);

  // CentroidXc

  const ImperialTeeSectionCentroidXcUnits = ['in²'];
  const ImperialTeeSectionCentroidXcconversionFactors = {
    'in²': [1],
  };
  const [ImperialTeeSectionCentroidXc, setImperialTeeSectionCentroidXc] = useState(0);
  const [ImperialTeeSectionCentroidXcUnit, setImperialTeeSectionCentroidXcUnit] = useState('in²');

  const handleImperialTeeSectionCentroidXcUnitChange = (unit) => {
    const currentImperialTeeSectionCentroidXcInNewUnit = ImperialTeeSectionCentroidXc * ImperialTeeSectionCentroidXcconversionFactors[ImperialTeeSectionCentroidXcUnit][ImperialTeeSectionCentroidXcUnits.indexOf(unit)];
    setImperialTeeSectionCentroidXc(currentImperialTeeSectionCentroidXcInNewUnit);
    setImperialTeeSectionCentroidXcUnit(unit);
  };

  const calculateImperialTeeSectionCentroidXc = () => {
    const b = parseFloat(TeeSectionImperialinputValue);
    const calculatedImperialTeeSectionCentroidXc = b / 2;
    setImperialTeeSectionCentroidXc(calculatedImperialTeeSectionCentroidXc);
  };

  useEffect(() => {
    calculateImperialTeeSectionCentroidXc();
  }, [TeeSectionImperialinputValue, TeeSectionImperialHeightInputValue, ImperialinputTeeSectionInnerValue, ImperialTeeSectionHeightInnerInputValue]);


  // CentroidYc

  const ImperialTeeSectionCentroidYcUnits = ['in²'];
  const ImperialTeeSectionCentroidYcconversionFactors = {
    'in²': [1],
  };
  const [ImperialTeeSectionCentroidYc, setImperialTeeSectionCentroidYc] = useState(0);
  const [ImperialTeeSectionCentroidYcUnit, setImperialTeeSectionCentroidYcUnit] = useState('in²');

  const handleImperialTeeSectionCentroidYcUnitChange = (unit) => {
    const currentImperialTeeSectionCentroidYcInNewUnit = ImperialTeeSectionCentroidYc * ImperialTeeSectionCentroidYcconversionFactors[ImperialTeeSectionCentroidYcUnit][ImperialTeeSectionCentroidYcUnits.indexOf(unit)];
    setImperialTeeSectionCentroidYc(currentImperialTeeSectionCentroidYcInNewUnit);
    setImperialTeeSectionCentroidYcUnit(unit);
  };

  const calculateImperialTeeSectionCentroidYc = () => {
    const d = parseFloat(TeeSectionImperialHeightInputValue);
    const b = parseFloat(TeeSectionImperialinputValue);
    const t = parseFloat(ImperialinputTeeSectionInnerValue);
    const tw = parseFloat(ImperialTeeSectionHeightInnerInputValue);

    const numerator = (b * Math.pow(t, 2)) + ((tw * d) * (2 * t + d));
    const denominator = 2 * (t * b + tw * d);
    const calculatedImperialTeeSectionCentroidYc = (numerator / denominator).toFixed(3);
    setImperialTeeSectionCentroidYc(calculatedImperialTeeSectionCentroidYc);
  };

  useEffect(() => {
    calculateImperialTeeSectionCentroidYc();
  }, [TeeSectionImperialinputValue, TeeSectionImperialHeightInputValue, ImperialinputTeeSectionInnerValue, ImperialTeeSectionHeightInnerInputValue]);

  // Moment of Inertia Ix
  const [ImperialTeeSectionMomentOfInertiaIx, setImperialTeeSectionMomentOfInertiaIx] = useState(0);
  const [ImperialTeeSectionMomentOfInertiaIxUnit, setImperialTeeSectionMomentOfInertiaIxUnit] = useState('in⁴');

  const ImperialTeeSectionMomentOfInertiaIxUnits = ['in⁴'];
  const ImperialTeeSectionMomentOfInertiaIxConversionFactors = {
    'in⁴': [1]
  };

  const handleImperialTeeSectionMomentOfInertiaIxUnitChange = (unit) => {
    const currentImperialTeeSectionMomentOfInertiaIxInNewUnit = ImperialTeeSectionMomentOfInertiaIx * ImperialTeeSectionMomentOfInertiaIxConversionFactors[ImperialTeeSectionMomentOfInertiaIxUnit][ImperialTeeSectionMomentOfInertiaIxUnits.indexOf(unit)];
    setImperialTeeSectionMomentOfInertiaIx(currentImperialTeeSectionMomentOfInertiaIxInNewUnit);
    setImperialTeeSectionMomentOfInertiaIxUnit(unit);
  };

  const calculateImperialTeeSectionMomentOfInertiaIx = () => {
    const d = parseFloat(TeeSectionImperialHeightInputValue);
    const b = parseFloat(TeeSectionImperialinputValue);
    const t = parseFloat(ImperialinputTeeSectionInnerValue);
    const tw = parseFloat(ImperialTeeSectionHeightInnerInputValue);
    const numerator = (b * Math.pow(t, 2)) + ((tw * d) * (2 * t + d));
    const denominator = 2 * (t * b + tw * d);
    const Yc = numerator / denominator;
    const term1 = b * Math.pow((d + t), 3) - Math.pow(d, 3) * (b - tw);
    const term2 = 3;
    const area = (t * b) + (tw * d)
    const term3 = area * Math.pow((d + t - Yc), 2);
    const ImperialTeeSectionMomentOfInertiaIx = ((term1 / term2) - term3).toFixed(3);
    setImperialTeeSectionMomentOfInertiaIx(ImperialTeeSectionMomentOfInertiaIx);
  };

  useEffect(() => {
    calculateImperialTeeSectionMomentOfInertiaIx();
  }, [TeeSectionImperialinputValue, TeeSectionImperialHeightInputValue, ImperialinputTeeSectionInnerValue, ImperialTeeSectionHeightInnerInputValue]);

  // Moment of Inertia Iy
  const [ImperialTeeSectionMomentOfInertiaIy, setImperialTeeSectionMomentOfInertiaIy] = useState(0);
  const [ImperialTeeSectionMomentOfInertiaIyUnit, setImperialTeeSectionMomentOfInertiaIyUnit] = useState('in⁴');

  const ImperialTeeSectionMomentOfInertiaIyUnits = ['in⁴'];
  const ImperialTeeSectionMomentOfInertiaIyConversionFactors = {
    'in⁴': [1]
  };

  const handleImperialTeeSectionMomentOfInertiaIyUnitChange = (unit) => {
    const currentImperialTeeSectionMomentOfInertiaIyInNewUnit = ImperialTeeSectionMomentOfInertiaIy * ImperialTeeSectionMomentOfInertiaIyConversionFactors[ImperialTeeSectionMomentOfInertiaIyUnit][ImperialTeeSectionMomentOfInertiaIyUnits.indexOf(unit)];
    setImperialTeeSectionMomentOfInertiaIy(currentImperialTeeSectionMomentOfInertiaIyInNewUnit);
    setImperialTeeSectionMomentOfInertiaIyUnit(unit);
  };

  const calculateImperialTeeSectionMomentOfInertiaIy = () => {
    const d = parseFloat(TeeSectionImperialHeightInputValue);
    const b = parseFloat(TeeSectionImperialinputValue);
    const t = parseFloat(ImperialinputTeeSectionInnerValue);
    const tw = parseFloat(ImperialTeeSectionHeightInnerInputValue);
    const term1 = (t * Math.pow(b, 3)) + (d * Math.pow(tw, 3));
    const term2 = 12;
    const ImperialTeeSectionMomentOfInertiaIy = (term1 / term2).toFixed(3);
    setImperialTeeSectionMomentOfInertiaIy(ImperialTeeSectionMomentOfInertiaIy);
  };

  useEffect(() => {
    calculateImperialTeeSectionMomentOfInertiaIy();
  }, [TeeSectionImperialinputValue, TeeSectionImperialHeightInputValue, ImperialinputTeeSectionInnerValue, ImperialTeeSectionHeightInnerInputValue]);


  // Section Modulus Sx
  const ImperialTeeSectionSectionModulesUnits = ['in³'];
  const ImperialTeeSectionSectionModulesConversionFactors = {
    'in³': [1]
  };
  const [ImperialTeeSectionSectionModulesSx, setImperialTeeSectionSectionModules] = useState(0);
  const [ImperialTeeSectionSectionModulesSxUnit, setImperialTeeSectionSectionModulesSxUnit] = useState('in³');

  const handleImperialTeeSectionSectionModulesSxUnitChange = (unit) => {
    const currentImperialTeeSectionSectionModulesSxInNewUnit = ImperialTeeSectionSectionModulesSx * ImperialTeeSectionSectionModulesConversionFactors[ImperialTeeSectionSectionModulesSxUnit][ImperialTeeSectionSectionModulesUnits.indexOf(unit)];
    setImperialTeeSectionSectionModulesSxUnit(unit);
    setImperialTeeSectionSectionModules(currentImperialTeeSectionSectionModulesSxInNewUnit);
  };

  const calculateImperialTeeSectionSectionModulesSx = () => {
    const d = parseFloat(TeeSectionImperialHeightInputValue);
    const b = parseFloat(TeeSectionImperialinputValue);
    const t = parseFloat(ImperialinputTeeSectionInnerValue);
    const tw = parseFloat(ImperialTeeSectionHeightInnerInputValue);
    const numerator = (b * Math.pow(t, 2)) + ((tw * d) * (2 * t + d));
    const denominator = 2 * (t * b + tw * d);
    const Yc = numerator / denominator;
    const term1 = b * Math.pow((d + t), 3) - Math.pow(d, 3) * (b - tw);
    const term2 = 3;
    const term3 = ImperialTeeSectionArea * Math.pow((d + t - Yc), 2);
    const Ix = (term1 / term2) - term3;
    const Sxdenomenator = (d + t) - Yc;
    const TeeSectionsectionModulesSx = (Ix / Sxdenomenator).toFixed(2);
    setImperialTeeSectionSectionModules(TeeSectionsectionModulesSx);
  };

  useEffect(() => {
    calculateImperialTeeSectionSectionModulesSx();
  }, [TeeSectionImperialHeightInputValue, TeeSectionImperialinputValue, ImperialinputTeeSectionInnerValue, ImperialTeeSectionHeightInnerInputValue, ImperialTeeSectionArea]);

  // Section Modulus Sy
  const ImperialTeeSectionSectionModulesSyUnits = ['in³'];
  const ImperialTeeSectionSectionModulesSyConversionFactors = {
    'in³': [1]
  };

  const [ImperialTeeSectionsectionModulesSy, setImperialTeeSectionSectionModulesSy] = useState(0);
  const [ImperialTeeSectionsectionModulesSyUnit, setImperialTeeSectionSectionModulesSyUnit] = useState('in³');

  const handleImperialTeeSectionSectionModulesSyUnitChange = (unit) => {
    const currentImperialTeeSectionSectionModulesSyInNewUnit = ImperialTeeSectionsectionModulesSy * ImperialTeeSectionSectionModulesSyConversionFactors[ImperialTeeSectionsectionModulesSyUnit][ImperialTeeSectionSectionModulesSyUnits.indexOf(unit)];
    setImperialTeeSectionSectionModulesSyUnit(unit);
    setImperialTeeSectionSectionModulesSy(currentImperialTeeSectionSectionModulesSyInNewUnit);
  };

  const calculateImperialTeeSectionSectionModulesSy = () => {
    const d = parseFloat(TeeSectionImperialHeightInputValue);
    const b = parseFloat(TeeSectionImperialinputValue);
    const t = parseFloat(ImperialinputTeeSectionInnerValue);
    const tw = parseFloat(ImperialTeeSectionHeightInnerInputValue);
    const Iy = (((t) * (b * b * b)) + ((d) * (tw * tw * tw))) / 12;
    const Xc = b / 2;
    const ImperialTeeSectionsectionModulesSy = (Iy / Xc).toFixed(2);
    setImperialTeeSectionSectionModulesSy(ImperialTeeSectionsectionModulesSy);
  };

  useEffect(() => {
    calculateImperialTeeSectionSectionModulesSy();
  }, [TeeSectionImperialHeightInputValue, TeeSectionImperialinputValue, ImperialinputTeeSectionInnerValue, ImperialTeeSectionHeightInnerInputValue]);

  // channels calculation
  const Channelbunits = ['mm', 'cm', 'm'];
  const ChannelbconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [ChannelbinputValue, setChannelbInputValue] = useState(0);
  const [ChannelbselectedUnit, setChannelbSelectedUnit] = useState('mm');
  const [internalChannelbinputValue, setInternalChannelbinputValue] = useState(0); // Always in mm

  const handleChannelbInputChange = (value) => {
    setChannelbInputValue(value);
    const factor = ChannelbconversionFactors[ChannelbselectedUnit][0];
    setInternalChannelbinputValue(parseFloat(value) * factor);
  };

  const handleChannelbUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(ChannelbinputValue) * ChannelbconversionFactors[ChannelbselectedUnit][0];
    const convertedValue = newMetricValueInMM / ChannelbconversionFactors[unit][0];
    setChannelbSelectedUnit(unit);
    setChannelbInputValue(convertedValue.toFixed(3));
  };



  const Channeldunits = ['mm', 'cm', 'm'];
  const ChanneldconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [ChanneldinputValue, setChanneldInputValue] = useState(0);
  const [ChannelselectedUnit, setdChannelselectedUnit] = useState('mm');
  const [internalChanneldinputValue, setInternalChanneldinputValue] = useState(0); // Always in mm

  const handledChannelInputChange = (value) => {
    setChanneldInputValue(value);
    const factor = ChanneldconversionFactors[ChannelselectedUnit][0];
    setInternalChanneldinputValue(parseFloat(value) * factor);
  };

  const handledChannelUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(ChanneldinputValue) * ChanneldconversionFactors[ChannelselectedUnit][0];
    const convertedValue = newMetricValueInMM / ChanneldconversionFactors[unit][0];
    setdChannelselectedUnit(unit);
    setChanneldInputValue(convertedValue.toFixed(3));
  };


  const Channeltwunits = ['mm', 'cm', 'm'];
  const ChanneltwconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [ChanneltwinputValue, setChannelTwInputValue] = useState(0);
  const [ChannelTwselectedUnit, setChanneltwSelectedUnit] = useState('mm');
  const [internalChanneltwinputValue, setInternalChanneltwinputValue] = useState(0); // Always in mm

  const handleChanneltwInputChange = (value) => {
    setChannelTwInputValue(value);
    const factor = ChanneltwconversionFactors[ChannelTwselectedUnit][0];
    setInternalChanneltwinputValue(parseFloat(value) * factor);
  };

  const handleChanneltwUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(ChanneltwinputValue) * ChanneltwconversionFactors[ChannelTwselectedUnit][0];
    const convertedValue = newMetricValueInMM / ChanneltwconversionFactors[unit][0];
    setChanneltwSelectedUnit(unit);
    setChannelTwInputValue(convertedValue.toFixed(3));
  };


  const ChannelTValueunits = ['mm', 'cm', 'm'];
  const ChannelTValueconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [ChannelTValue, setChannelTValue] = useState(0);
  const [ChannelselectedTUnit, setChannelTValueSelectedUnit] = useState('mm');
  const [internalChannelTValue, setInternalChannelTValue] = useState(0); // Always in mm

  const handleChannelTValueInputChange = (value) => {
    setChannelTValue(value);
    const factor = ChannelTValueconversionFactors[ChannelselectedTUnit][0];
    setInternalChannelTValue(parseFloat(value) * factor);
  };

  const handleChannelTValueUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(ChannelTValue) * ChannelTValueconversionFactors[ChannelselectedTUnit][0];
    const convertedValue = newMetricValueInMM / ChannelTValueconversionFactors[unit][0];
    setChannelTValueSelectedUnit(unit);
    setChannelTValue(convertedValue.toFixed(3));
  };



  // Area
  const ChannelAreaUnits = ['mm²', 'cm²', 'm²'];
  const ChannelAreaConversionFactors = {
    'mm²': [1, 0.01, 0.000001],
    'cm²': [100, 1, 0.0001],
    'm²': [1000000, 10000, 1],
  };
  const [ChannelArea, setChannelArea] = useState(0);
  const [ChannelAreaUnit, setChannelAreaUnit] = useState('mm²');

  // CentroidXc
  const ChannelCentroidXcUnits = ['mm', 'cm', 'm'];

  const [ChannelcentroidXc, setChannelCentroidXc] = useState(0);
  const [ChannelcentroidXcSelectedUnit, setChannelCentroidXcSelectedUnit] = useState('mm');

  // CentroidYc
  const ChannelCentroidYcUnits = ['mm', 'cm', 'm'];

  const [ChannelcentroidYc, setChannelCentroidYc] = useState(0);
  const [ChannelcentroidYcSelectedUnit, setChannelCentroidYcSelectedUnit] = useState('mm');

  const ChannelMomentOfInertiaIxUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const ChannelMomentOfInertiaIxConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [ChannelmomentOfInertiaIx, setChannelMomentOfInertiaIx] = useState(0);
  const [ChannelmomentOfInertiaIxSelectedUnit, setChannelMomentOfInertiaIxSelectedUnit] = useState('mm⁴');


  // Moment of Inertia Iy
  const ChannelMomentOfInertiaIyUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const ChannelMomentOfInertiaIyConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };
  const [ChannelmomentOfInertiaIy, setChannelMomentOfInertiaIy] = useState(0);
  const [ChannelmomentOfInertiaIySelectedUnit, setChannelMomentOfInertiaIySelectedUnit] = useState('mm⁴');


  // Section Modulus Sx
  const ChannelSectionModulusSxUnits = ['mm³', 'cm³', 'm³'];
  const ChannelSectionModulusSxConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [ChannelsectionModulusSx, setChannelSectionModulusSx] = useState(0);
  const [ChannelsectionModulusSxSelectedUnit, setChannelSectionModulusSxSelectedUnit] = useState('mm³');

  const ChannelSectionModulusSyUnits = ['mm³', 'cm³', 'm³'];
  const ChannelSectionModulusSyConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [ChannelSectionModulusSy, setChannelSectionModulusSy] = useState(0);
  const [ChannelSectionModulusSySelectedUnit, setChannelSectionModulusSySelectedUnit] = useState('mm³');





  useEffect(() => {
    const t = parseFloat(internalChannelTValue);
    const tw = parseFloat(internalChanneltwinputValue);
    const b = parseFloat(internalChannelbinputValue);
    const d = parseFloat(internalChanneldinputValue);
    const value = parseFloat(internalChannelbinputValue) * ChanneldconversionFactors[ChannelselectedUnit][0];
    if (!isNaN(value)) {
      const ChannelareaValue = (t * b + 2 * (tw * d)).toFixed(3);
      setChannelArea((ChannelareaValue / ChannelAreaConversionFactors[ChannelAreaUnit][0]));
    } else {
      setChannelArea('');
    }
  }, [internalChannelTValue, internalChanneltwinputValue, internalChannelbinputValue, internalChanneldinputValue, ChannelAreaUnit]);


  useEffect(() => {
    const value = internalChannelbinputValue;
    const centroidValue = (value / 2);
    setChannelCentroidXc((centroidValue / ChannelbconversionFactors[ChannelcentroidXcSelectedUnit][0]));
  }, [internalChannelTValue, internalChanneltwinputValue, internalChannelbinputValue, internalChanneldinputValue, ChannelcentroidXcSelectedUnit]);


  // CentriodYcUnits

  useEffect(() => {
    const t = parseFloat(internalChannelTValue);
    const tw = parseFloat(internalChanneltwinputValue);
    const b = parseFloat(internalChannelbinputValue);
    const d = parseFloat(internalChanneldinputValue);

    const numerator = (b * (t * t)) + (2 * tw * d * (2 * t + d));
    const denominator = 2 * ((t * b) + (2 * tw * d));
    const calculatedCentroidYc = numerator / denominator;

    let formattedValue;
    switch (ChannelcentroidYcSelectedUnit) {
      case 'm':
        formattedValue = (calculatedCentroidYc / ChannelbconversionFactors[ChannelcentroidYcSelectedUnit][0]).toFixed(5);
        break;
      case 'cm':
        formattedValue = (calculatedCentroidYc / ChannelbconversionFactors[ChannelcentroidYcSelectedUnit][0]).toFixed(3);
        break;
      case 'mm':
        formattedValue = (calculatedCentroidYc / ChannelbconversionFactors[ChannelcentroidYcSelectedUnit][0]).toFixed(2);
        break;
      default:
        formattedValue = calculatedCentroidYc / ChannelbconversionFactors[ChannelcentroidYcSelectedUnit][0];
    }

    setChannelCentroidYc(formattedValue);
  }, [internalChannelTValue, internalChanneltwinputValue, internalChannelbinputValue, internalChanneldinputValue, ChannelcentroidYcSelectedUnit]);


  useEffect(() => {
    const value = b;
    if (!isNaN(value)) {
      const t = parseFloat(internalChannelTValue);
      const tw = parseFloat(internalChanneltwinputValue);
      const b = parseFloat(internalChannelbinputValue);
      const d = parseFloat(internalChanneldinputValue);
      const numeratorYc = (b * (t * t)) + (2 * tw * d * (2 * t + d));
      const denominatorYc = 2 * (t * b + 2 * tw * d);
      const Yc = numeratorYc / denominatorYc;
      const numeratorIx = b * Math.pow(d + t, 3);
      const subtractValueIx1 = Math.pow(d, 3) * (b - 2 * tw);
      const subtractValueIx2 = (t * b + 2 * tw * d) * Math.pow(d + t - Yc, 2);
      const ChannelmomentOfInertiaIx = (((numeratorIx - subtractValueIx1) / 3) - subtractValueIx2);

      let formattedValue;
      switch (ChannelmomentOfInertiaIxSelectedUnit) {
        case 'm⁴':
          formattedValue = (ChannelmomentOfInertiaIx * ChannelMomentOfInertiaIxConversionFactors['mm⁴'][ChannelMomentOfInertiaIxUnits.indexOf(ChannelmomentOfInertiaIxSelectedUnit)]).toExponential(4);
          break;
        case 'cm⁴':
          formattedValue = (ChannelmomentOfInertiaIx * ChannelMomentOfInertiaIxConversionFactors['mm⁴'][ChannelMomentOfInertiaIxUnits.indexOf(ChannelmomentOfInertiaIxSelectedUnit)]).toFixed(4);
          break;
        case 'mm⁴':
          formattedValue = (ChannelmomentOfInertiaIx * ChannelMomentOfInertiaIxConversionFactors['mm⁴'][ChannelMomentOfInertiaIxUnits.indexOf(ChannelmomentOfInertiaIxSelectedUnit)]).toFixed(2);
          break;
        default:
          formattedValue = ChannelmomentOfInertiaIx * ChannelMomentOfInertiaIxConversionFactors['mm⁴'][ChannelMomentOfInertiaIxUnits.indexOf(ChannelmomentOfInertiaIxSelectedUnit)];
      }

      setChannelMomentOfInertiaIx(formattedValue);
    }
  }, [internalChannelTValue, internalChanneltwinputValue, internalChannelbinputValue, internalChanneldinputValue, ChannelmomentOfInertiaIxSelectedUnit]);

  useEffect(() => {
    const value = b;
    if (!isNaN(value)) {

      const t = parseFloat(internalChannelTValue);
      const tw = parseFloat(internalChanneltwinputValue);
      const b = parseFloat(internalChannelbinputValue);
      const d = parseFloat(internalChanneldinputValue);
      const numeratorIy = (d + t) * Math.pow(b, 3);
      const subtractValueIy = d * Math.pow((b - 2 * tw), 3);
      const denominatorIy = 12;
      const ChannelmomentOfInertiaIy = ((numeratorIy - subtractValueIy) / denominatorIy);

      let convertedValue;
      switch (ChannelmomentOfInertiaIySelectedUnit) {
        case 'mm⁴':
          convertedValue = (ChannelmomentOfInertiaIy * ChannelMomentOfInertiaIyConversionFactors['mm⁴'][ChannelMomentOfInertiaIyUnits.indexOf(ChannelmomentOfInertiaIySelectedUnit)]).toFixed(2);
          break;
        case 'cm⁴':
          convertedValue = (ChannelmomentOfInertiaIy * ChannelMomentOfInertiaIyConversionFactors['mm⁴'][ChannelMomentOfInertiaIyUnits.indexOf(ChannelmomentOfInertiaIySelectedUnit)]).toFixed(4);
          break;
        case 'm⁴':
          convertedValue = (ChannelmomentOfInertiaIy * ChannelMomentOfInertiaIyConversionFactors['mm⁴'][ChannelMomentOfInertiaIyUnits.indexOf(ChannelmomentOfInertiaIySelectedUnit)]).toExponential(4);
          break;
        default:
          convertedValue = ChannelmomentOfInertiaIy * ChannelMomentOfInertiaIyConversionFactors['mm⁴'][ChannelMomentOfInertiaIyUnits.indexOf(ChannelmomentOfInertiaIySelectedUnit)];
      }

      setChannelMomentOfInertiaIy(convertedValue);
    }
  }, [internalChannelTValue, internalChanneltwinputValue, internalChannelbinputValue, internalChanneldinputValue, ChannelmomentOfInertiaIySelectedUnit]);


  useEffect(() => {
    const value = b
    if (!isNaN(value)) {

      const t = parseFloat(internalChannelTValue);
      const tw = parseFloat(internalChanneltwinputValue);
      const b = parseFloat(internalChannelbinputValue);
      const d = parseFloat(internalChanneldinputValue);
      const numerator = (b * t * t) + (2 * tw * d * (2 * t + d));
      const denominator = 2 * (t * b + 2 * tw * d);
      const Yc = numerator / denominator;
      const numeratorIx = b * Math.pow(d + t, 3);
      const subtractValueIx1 = Math.pow(d, 3) * (b - 2 * tw);
      const subtractValueIx2 = (t * b + 2 * tw * d) * Math.pow(d + t - Yc, 2);
      const momentOfInertiaIx = (((numeratorIx - subtractValueIx1) / 3) - subtractValueIx2).toFixed(3);
      const denominatorSx = d + t - Yc;
      const sectionModulesValue = (momentOfInertiaIx / denominatorSx).toFixed(3);
      // const convertedValue = (sectionModulesValue * ChannelSectionModulusSxConversionFactors['mm³'][ChannelSectionModulusSxUnits.indexOf(ChannelsectionModulusSxSelectedUnit)]);

      let convertedValue;
      switch (ChannelsectionModulusSxSelectedUnit) {
        case 'mm³':
          convertedValue = (sectionModulesValue * ChannelSectionModulusSxConversionFactors['mm³'][ChannelSectionModulusSxUnits.indexOf(ChannelsectionModulusSxSelectedUnit)]).toFixed(2);
          break;
        case 'cm³':
          convertedValue = (sectionModulesValue * ChannelSectionModulusSxConversionFactors['mm³'][ChannelSectionModulusSxUnits.indexOf(ChannelsectionModulusSxSelectedUnit)]).toFixed(4);
          break;
        case 'm³':
          convertedValue = (sectionModulesValue * ChannelSectionModulusSxConversionFactors['mm³'][ChannelSectionModulusSxUnits.indexOf(ChannelsectionModulusSxSelectedUnit)]).toExponential(4);
          break;
        default:
          convertedValue = sectionModulesValue * ChannelSectionModulusSxConversionFactors['mm³'][ChannelSectionModulusSxUnits.indexOf(ChannelsectionModulusSxSelectedUnit)];
      }

      setChannelSectionModulusSx(convertedValue);
    }
  }, [internalChannelTValue, internalChanneltwinputValue, internalChannelbinputValue, internalChanneldinputValue, ChannelsectionModulusSxSelectedUnit]);

  useEffect(() => {
    const value = b;
    if (!isNaN(value)) {

      const t = parseFloat(internalChannelTValue);
      const tw = parseFloat(internalChanneltwinputValue);
      const b = parseFloat(internalChannelbinputValue);
      const d = parseFloat(internalChanneldinputValue);
      const numeratorIy = (d + t) * Math.pow(b, 3);
      const subtractValueIy = d * Math.pow(b - 2 * tw, 3);
      const momentOfInertiaIy = ((numeratorIy - subtractValueIy) / 12);
      const Xc = b / 2;
      const sectionModulesValue = (momentOfInertiaIy / Xc).toFixed(4);
      const convertedValue = (sectionModulesValue * ChannelSectionModulusSyConversionFactors['mm³'][ChannelSectionModulusSyUnits.indexOf(ChannelSectionModulusSySelectedUnit)]);
      setChannelSectionModulusSy(convertedValue);
    }
  }, [internalChannelTValue, internalChanneltwinputValue, internalChannelbinputValue, internalChanneldinputValue, ChannelSectionModulusSySelectedUnit]);



  const ImperialChannelbunits = ['in'];
  const ImperialChannelbconversionFactors = {
    in: [1],
  };

  const [ImperialChannelbinputValue, setImperialChannelbInputValue] = useState(0);
  const [ImperialChannelbselectedUnit, setImperialChannelbSelectedUnit] = useState('in');

  const handleImperialChannelbInputChange = (value) => {
    setImperialChannelbInputValue(value);
  };

  const handleImperialChannelbUnitChange = (unit) => {
    setImperialChannelbSelectedUnit(unit);
    const ImperialChannelbfactor = ImperialChannelbconversionFactors[unit][ImperialChannelbunits.indexOf(ImperialChannelbselectedUnit)];
    setImperialChannelbInputValue((parseFloat(ImperialChannelbinputValue) / ImperialChannelbfactor).toFixed(4));
  };


  const ChannelImperialdunits = ['in'];
  const ChannelImperialdconversionFactors = {
    in: [1],
  };

  const [ChannelImperialdinputValue, setChannelImperialdInputValue] = useState(0);
  const [ChannelImperialdselectedUnit, setChannelImperialdSelectedUnit] = useState('in');

  const handleChannelImperialdInputChange = (value) => {
    setChannelImperialdInputValue(value);
  };

  const handleChannelImperialdUnitChange = (unit) => {
    setChannelImperialdSelectedUnit(unit);
    const ChannelImperialdfactor = ChannelImperialdconversionFactors[unit][Channeldunits.indexOf(ChannelImperialdselectedUnit)];
    setChannelImperialdInputValue((parseFloat(ChannelImperialdinputValue) / ChannelImperialdfactor).toFixed(4));
  };


  const ChannelImperialtwunits = ['in'];
  const ChannelImperialtwconversionFactors = {
    in: [1],
  };

  const [ChannelImperialtwinputValue, setChannelImperialTwInputValue] = useState(0);
  const [ChannelImperialTwselectedUnit, setChannelImperialtwSelectedUnit] = useState('in');

  const handleChannelImperialtwInputChange = (value) => {
    setChannelImperialTwInputValue(value);
  };

  const handleChannelImperialtwUnitChange = (unit) => {
    setChannelImperialtwSelectedUnit(unit);
    const ChannelImperialtwfactor = ChannelImperialtwconversionFactors[unit][ChannelImperialtwunits.indexOf(ChannelImperialTwselectedUnit)];
    setChannelImperialTwInputValue((parseFloat(ChannelImperialtwinputValue) / ChannelImperialtwfactor).toFixed(4));
  };

  const ChannelImperialTValueunits = ['in'];
  const ChannelImperialTValueconversionFactors = {
    in: [1],
  };

  const [ChannelImperialTValue, setChannelImperialTValue] = useState(0);
  const [ChannelImperialTVselectedTUnit, setChannelImperialTValueSelectedUnit] = useState('in');

  const handleChannelImperialTValueInputChange = (value) => {
    setChannelImperialTValue(value);
  };

  const handleChannelImperialTValueUnitChange = (unit) => {
    setChannelImperialTValueSelectedUnit(unit);
    const ChannelImperialTValuefactor = ChannelImperialTValueconversionFactors[unit][ChannelImperialTValueunits.indexOf(ChannelImperialTVselectedTUnit)];
    setChannelImperialTValue((parseFloat(ChannelImperialTValue) / ChannelImperialTValuefactor).toFixed(4));
  };
  const ImperialChannelAreaUnits = ['in²'];
  const ImperialChannelAreaConversionFactors = {
    'in²': [1],
  };
  const [ImperialChannelArea, setImperialChannelArea] = useState(0);
  const [ImperialChannelAreaUnit, setImperialChannelAreaUnit] = useState('in²');

  const handleImperialChannelAreaUnitChange = (unit) => {
    const currentImperialChannelAreaInNewUnit = ImperialChannelArea * ImperialChannelAreaConversionFactors[ImperialChannelAreaUnit][ImperialChannelAreaUnits.indexOf(unit)];
    setImperialChannelArea(currentImperialChannelAreaInNewUnit);
    setImperialChannelAreaUnit(unit);
  };

  const calculateImperialChannelArea = () => {
    const d = parseFloat(ChannelImperialdinputValue);
    const b = parseFloat(ImperialChannelbinputValue);
    const t = parseFloat(ChannelImperialTValue);
    const tw = parseFloat(ChannelImperialtwinputValue);
    const calculatedImperialChannelArea = (t * b + 2 * (tw * d))

    // const calculatedImperialChannelArea = (t * b + ((2*tw) * d));
    setImperialChannelArea(calculatedImperialChannelArea);
  };

  useEffect(() => {
    calculateImperialChannelArea();
  }, [ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue]);

  // CentroidXcUnits
  const ImperialChannelCentroidXcUnits = ['in'];
  const ImperialChannelCentroidXcConversionFactors = {
    in: [1],
  };
  const [ImperialChannelCentroidXc, setImperialChannelCentroidXc] = useState(0);
  const [ImperialChannelCentroidXcSelectedUnit, setImperialChannelCentroidXcSelectedUnit] = useState('in');

  // Handle change in centroid Xc unit
  const handleImperialChannelCentroidXcUnitChange = (unit) => {
    const conversionFactor = ImperialChannelCentroidXcConversionFactors[ImperialChannelCentroidXcSelectedUnit][ImperialChannelCentroidXcUnits.indexOf(unit)];
    const newImperialChannelCentroidXc = parseFloat(ImperialChannelCentroidXc) * conversionFactor;
    setImperialChannelCentroidXc(newImperialChannelCentroidXc);
    setImperialChannelCentroidXcSelectedUnit(unit);
  };

  // Calculate centroid Xc based on input value
  const calculateImperialChannelCentroidXc = (ImperialChannelbinputValue) => {
    if (!isNaN(ImperialChannelbinputValue)) {
      const ChannelcentroidXc = ImperialChannelbinputValue / 2;
      setImperialChannelCentroidXc(ChannelcentroidXc);
    }
  };

  useEffect(() => {
    calculateImperialChannelCentroidXc(ImperialChannelbinputValue);
  }, [ImperialChannelbinputValue]);

  // CentriodYcUnits

  const ImperialChannelCentroidYcUnits = ['in'];
  const ImperialChannelCentroidYcConversionFactors = {
    in: [1],
  };
  const [ImperialChannelCentroidYc, setImperialChannelCentroidYc] = useState(0);
  const [ImperialChannelCentroidYcSelectedUnit, setImperialChannelCentroidYcSelectedUnit] = useState('in');

  // Handle change in centroid Xc unit
  const handleImperialChannelCentroidYcUnitChange = (unit) => {
    const conversionFactor = ImperialChannelCentroidYcConversionFactors[ImperialChannelCentroidYcSelectedUnit][ImperialChannelCentroidYcUnits.indexOf(unit)];
    const newImperialChannelCentroidYc = parseFloat(ImperialChannelCentroidYc) * conversionFactor;
    setImperialChannelCentroidYc(newImperialChannelCentroidYc);
    setImperialChannelCentroidYcSelectedUnit(unit);
  };

  // Calculate centroid Xc based on input value
  const calculateImperialChannelCentroidYc = (ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue) => {
    const t = parseFloat(ChannelImperialTValue);
    const tw = parseFloat(ChannelImperialtwinputValue);
    const b = parseFloat(ImperialChannelbinputValue);
    const d = parseFloat(ChannelImperialdinputValue);
    if (!isNaN(b)) {
      const numerator = (b * (t * t)) + (2 * tw * d * (2 * t + d));
      const denominator = 2 * ((t * b) + (2 * tw * d));
      const ImperialChannelCentroidYc = (numerator / denominator).toFixed(2);
      setImperialChannelCentroidYc(ImperialChannelCentroidYc);
    } else {
      setImperialChannelCentroidYc('');
    }
  };

  useEffect(() => {
    calculateImperialChannelCentroidYc(ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue);
  }, [ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue]);

  // MomentOfInertia Ix

  const ImperialChannelMomentOfInertiaIxUnits = ['in⁴'];
  const ImperialChannelMomentOfInertiaIxConversionFactors = {
    'in⁴': [1],
  };

  const [ImperialChannelMomentOfInertiaIx, setImperialChannelMomentOfInertiaIx] = useState(0);
  const [ImperialChannelMomentOfInertiaIxSelectedUnit, setImperialChannelMomentOfInertiaIxSelectedUnit] = useState('in⁴');

  // Handle change in moment of inertia unit
  const handleImperialChannelMomentOfInertiaUnitChange = (newUnit) => {
    const currentFactor = ImperialChannelMomentOfInertiaIxConversionFactors[ImperialChannelMomentOfInertiaIxSelectedUnit][ImperialChannelMomentOfInertiaIxUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(ImperialChannelMomentOfInertiaIx) * currentFactor;
    setImperialChannelMomentOfInertiaIxSelectedUnit(newUnit);
    setImperialChannelMomentOfInertiaIx(convertedValue.toString());
  };

  // Calculate moment of inertia Ix
  const calculateImperialChannelMomentOfInertiaIx = (ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue) => {
    const t = parseFloat(ChannelImperialTValue);
    const tw = parseFloat(ChannelImperialtwinputValue);
    const b = parseFloat(ImperialChannelbinputValue);
    const d = parseFloat(ChannelImperialdinputValue);
    if (!isNaN(t) && !isNaN(tw) && !isNaN(b) && !isNaN(d)) {
      const numeratorYc = (b * (t * t)) + (2 * tw * d * (2 * t + d));
      const denominatorYc = 2 * (t * b + 2 * tw * d);
      const Yc = numeratorYc / denominatorYc;
      const numeratorIx = b * Math.pow(d + t, 3);
      const subtractValueIx1 = Math.pow(d, 3) * (b - 2 * tw);
      const subtractValueIx2 = (t * b + 2 * tw * d) * Math.pow(d + t - Yc, 2);
      const ImperialChannelMomentOfInertiaIx = (((numeratorIx - subtractValueIx1) / 3) - subtractValueIx2).toFixed(2);

      const currentFactor = ImperialChannelMomentOfInertiaIxConversionFactors['in⁴'][ImperialChannelMomentOfInertiaIxUnits.indexOf(ImperialChannelMomentOfInertiaIxSelectedUnit)];
      setImperialChannelMomentOfInertiaIx((ImperialChannelMomentOfInertiaIx * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateImperialChannelMomentOfInertiaIx(ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue);
  }, [ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue, ImperialChannelMomentOfInertiaIxSelectedUnit]);

  // MomentOfInertia Iy

  const ImperialChannelMomentOfInertiaIyUnits = ['in⁴'];
  const ImperialChannelMomentOfInertiaIyConversionFactors = {
    'in⁴': [1],
  };

  const [ImperialChannelMomentOfInertiaIy, setImperialChannelMomentOfInertiaIy] = useState(0);
  const [ImperialChannelMomentOfInertiaIySelectedUnit, setImperialChannelMomentOfInertiaIySelectedUnit] = useState('in⁴');

  // Handle change in moment of inertia unit
  const handleImperialChannelMomentOfInertiaIyUnitChange = (newUnit) => {
    const currentFactor = ImperialChannelMomentOfInertiaIyConversionFactors[ImperialChannelMomentOfInertiaIySelectedUnit][ImperialChannelMomentOfInertiaIyUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(ImperialChannelMomentOfInertiaIy) * currentFactor;
    setImperialChannelMomentOfInertiaIySelectedUnit(newUnit);
    setImperialChannelMomentOfInertiaIy(convertedValue.toString());
  };

  // Calculate moment of inertia Iy
  const calculateImperialChannelMomentOfInertiaIy = (ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue) => {
    const t = parseFloat(ChannelImperialTValue);
    const tw = parseFloat(ChannelImperialtwinputValue);
    const b = parseFloat(ImperialChannelbinputValue);
    const d = parseFloat(ChannelImperialdinputValue);
    if (!isNaN(t) && !isNaN(tw) && !isNaN(b) && !isNaN(d)) {
      const numeratorIy = (d + t) * Math.pow(b, 3);
      const subtractValueIy = d * Math.pow((b - 2 * tw), 3);
      const denominatorIy = 12;
      const ImperialChannelMomentOfInertiaIy = ((numeratorIy - subtractValueIy) / denominatorIy).toFixed(2);
      const currentFactor = ImperialChannelMomentOfInertiaIyConversionFactors['in⁴'][ImperialChannelMomentOfInertiaIyUnits.indexOf(ImperialChannelMomentOfInertiaIySelectedUnit)];
      setImperialChannelMomentOfInertiaIy((ImperialChannelMomentOfInertiaIy * currentFactor).toString());
    } else {
      setImperialChannelMomentOfInertiaIy('');
    }
  };

  useEffect(() => {
    calculateImperialChannelMomentOfInertiaIy(ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue);
  }, [ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue, ImperialChannelMomentOfInertiaIySelectedUnit]);



  // Section Modulus Sx
  const ImperialChannelSectionModulesUnits = ['in³'];
  const ImperialChannelSectionModulesConversionFactors = {
    'in³': [1]
  };
  const [ImperialChannelSectionModulesSx, setImperialChannelSectionModules] = useState(0);
  const [ImperialChannelSectionModulesSxUnit, setImperialChannelSectionModulesSxUnit] = useState('in³');

  const handleImperialChannelSectionModulesSxUnitChange = (unit) => {
    const currentImperialChannelSectionModulesSxInNewUnit = ImperialChannelSectionModulesSx * ImperialChannelSectionModulesConversionFactors[ImperialChannelSectionModulesSxUnit][ImperialChannelSectionModulesUnits.indexOf(unit)];
    setImperialChannelSectionModulesSxUnit(unit);
    setImperialChannelSectionModules(currentImperialChannelSectionModulesSxInNewUnit);
  };

  const calculateImperialChannelSectionModulesSx = () => {
    const t = parseFloat(ChannelImperialTValue);
    const tw = parseFloat(ChannelImperialtwinputValue);
    const b = parseFloat(ImperialChannelbinputValue);
    const d = parseFloat(ChannelImperialdinputValue);
    const numerator = (b * t * t) + (2 * tw * d * (2 * t + d));
    const denominator = 2 * (t * b + 2 * tw * d);
    const Yc = numerator / denominator;
    const numeratorIx = b * Math.pow(d + t, 3);
    const subtractValueIx1 = Math.pow(d, 3) * (b - 2 * tw);
    const subtractValueIx2 = (t * b + 2 * tw * d) * Math.pow(d + t - Yc, 2);
    const momentOfInertiaIx = (((numeratorIx - subtractValueIx1) / 3) - subtractValueIx2);
    const denominatorSx = d + t - Yc;
    const ImperialChannelsectionModulesSx = (momentOfInertiaIx / denominatorSx).toFixed(2);
    setImperialChannelSectionModules(ImperialChannelsectionModulesSx);
  };

  useEffect(() => {
    calculateImperialChannelSectionModulesSx();
  }, [ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue]);

  // Section Modulus Sy
  const ImperialChannelSectionModulesSyUnits = ['in³'];
  const ImperialChannelSectionModulesSyConversionFactors = {
    'in³': [1]
  };

  const [ImperialChannelsectionModulesSy, setImperialChannelSectionModulesSy] = useState(0);
  const [ImperialChannelsectionModulesSyUnit, setImperialChannelSectionModulesSyUnit] = useState('in³');

  const handleImperialChannelSectionModulesSyUnitChange = (unit) => {
    const currentImperialChannelSectionModulesSyInNewUnit = ImperialChannelsectionModulesSy * ImperialChannelSectionModulesSyConversionFactors[ImperialChannelsectionModulesSyUnit][ImperialChannelSectionModulesSyUnits.indexOf(unit)];
    setImperialChannelSectionModulesSyUnit(unit);
    setImperialChannelSectionModulesSy(currentImperialChannelSectionModulesSyInNewUnit);
  };

  const calculateImperialChannelSectionModulesSy = () => {
    const d = parseFloat(ChannelImperialdinputValue);
    const b = parseFloat(ImperialChannelbinputValue);
    const t = parseFloat(ChannelImperialTValue);
    const tw = parseFloat(ChannelImperialtwinputValue);

    const numeratorIy = (d + t) * Math.pow(b, 3);
    const subtractValueIy = d * Math.pow((b - 2 * tw), 3);
    const denominatorIy = 12;
    const Iy = (numeratorIy - subtractValueIy) / denominatorIy;

    const Xc = b / 2;
    const ImperialChannelsectionModulesSy = (Iy / Xc).toFixed(2);
    // setChannelSectionModulesSy(ChannelsectionModulesSy.toString());
    setImperialChannelSectionModulesSy(ImperialChannelsectionModulesSy.toString());
    // setImperialChannelSectionModulesSy(ImperialChannelsectionModulesSy);
  };

  useEffect(() => {
    calculateImperialChannelSectionModulesSy();
  }, [ChannelImperialTValue, ChannelImperialtwinputValue, ImperialChannelbinputValue, ChannelImperialdinputValue]);

  // I-Beam section calculation
  const Isectionbunits = ['mm', 'cm', 'm'];
  const IsectionbconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [IsectionbinputValue, setIsectionbInputValue] = useState(0);
  const [IsectionbselectedUnit, setIsectionbSelectedUnit] = useState('mm');
  const [IsectioninternalbinputValue, setIsectionInternalbinputValue] = useState(0); // Always in mm

  const handlebInputChange = (value) => {
    setIsectionbInputValue(value);
    const factor = IsectionbconversionFactors[IsectionbselectedUnit][0];
    setIsectionInternalbinputValue(parseFloat(value) * factor);
  };

  const handlebUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(IsectionbinputValue) * IsectionbconversionFactors[IsectionbselectedUnit][0];
    const convertedValue = newMetricValueInMM / IsectionbconversionFactors[unit][0];
    setIsectionbSelectedUnit(unit);
    setIsectionbInputValue(convertedValue.toFixed(3));
  };



  const Isectiondunits = ['mm', 'cm', 'm'];
  const IsectiondconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [IsectiondinputValue, setIsectiondInputValue] = useState(0);
  const [IsectionselectedUnit, setIsectiondselectedUnit] = useState('mm');
  const [internalIsectiondinputValue, setInternalIsectiondinputValue] = useState(0); // Always in mm

  const handledInputChange = (value) => {
    setIsectiondInputValue(value);
    const factor = IsectiondconversionFactors[IsectionselectedUnit][0];
    setInternalIsectiondinputValue(parseFloat(value) * factor);
  };

  const handledUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(IsectiondinputValue) * IsectiondconversionFactors[IsectionselectedUnit][0];
    const convertedValue = newMetricValueInMM / IsectiondconversionFactors[unit][0];
    setIsectiondselectedUnit(unit);
    setIsectiondInputValue(convertedValue.toFixed(3));
  };

  const IsectionTValueunits = ['mm', 'cm', 'm'];
  const IsectionTValueconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [IsectionTValue, setIsectionTValue] = useState(0);
  const [IsectionselectedIsectionTVUnit, setIsectionTValueSelectedUnit] = useState('mm');
  const [internalIsectionTValue, setinternalIsectionTValue] = useState(0); // Always in mm

  const handleIsectionTValueInputChange = (value) => {
    setIsectionTValue(value);
    const factor = IsectionTValueconversionFactors[IsectionselectedIsectionTVUnit][0];
    setinternalIsectionTValue(parseFloat(value) * factor);
  };

  const handleIsectionTValueUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(IsectionTValue) * IsectionTValueconversionFactors[IsectionselectedIsectionTVUnit][0];
    const convertedValue = newMetricValueInMM / IsectionTValueconversionFactors[unit][0];
    setIsectionTValueSelectedUnit(unit);
    setIsectionTValue(convertedValue.toFixed(5));
  };

  const Isectiontwunits = ['mm', 'cm', 'm'];
  const IsectiontwconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [IsectiontwinputValue, setIsectiontwinputValue] = useState(0);
  const [IsectionTwselectedUnit, setIsectiontwinputValueSelectedUnit] = useState('mm');
  const [internalIsectiontwinputValue, setInternalIsectiontwinputValue] = useState(0); // Always in mm

  const handleIsectiontwInputChange = (value) => {
    setIsectiontwinputValue(value);
    const factor = IsectiontwconversionFactors[IsectionTwselectedUnit][0];
    setInternalIsectiontwinputValue(parseFloat(value) * factor);
  };

  const handleIsectiontwUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(IsectiontwinputValue) * IsectiontwconversionFactors[IsectionTwselectedUnit][0];
    const convertedValue = newMetricValueInMM / IsectiontwconversionFactors[unit][0];
    setIsectiontwinputValueSelectedUnit(unit);
    setIsectiontwinputValue(convertedValue.toFixed(5));
  };




  const IsectionRunits = ['mm', 'cm', 'm'];
  const IsectionRconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };
  const [IsectionRinputValue, setIsectionRinputValue] = useState(0);
  const [IsectionRselectedUnit, setIsectionRSelectedUnit] = useState('mm');
  const [internalIsectionRinputValue, setInternalIsectionRinputValue] = useState(0); // Always in mm

  const handleIsectionRInputChange = (value) => {
    setIsectionRinputValue(value);
    const factor = IsectionRconversionFactors[IsectionRselectedUnit][0];
    setInternalIsectionRinputValue(parseFloat(value) * factor);
  };

  const handleIsectionRUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(IsectionRinputValue) * IsectionRconversionFactors[IsectionRselectedUnit][0];
    const convertedValue = newMetricValueInMM / IsectionRconversionFactors[unit][0];
    setIsectionRSelectedUnit(unit);
    setIsectionRinputValue(convertedValue.toFixed(5));
  };

  // Area
  const IsectionAreaUnits = ['mm²', 'cm²', 'm²'];
  const IsectionAreaConversionFactors = {
    'mm²': [1, 0.01, 0.000001],
    'cm²': [100, 1, 0.0001],
    'm²': [1000000, 10000, 1],
  };
  const [IsectionArea, setIsectionArea] = useState(0);
  const [IsectionAreaUnit, setIsectionAreaUnit] = useState('mm²');

  // CentroidXc
  const IsectionCentroidXcUnits = ['mm', 'cm', 'm'];

  const [IsectioncentroidXc, setIsectionCentroidXc] = useState(0);
  const [IsectioncentroidXcSelectedUnit, setIsectionCentroidXcSelectedUnit] = useState('mm');

  // CentroidYc
  const IsectionCentroidYcUnits = ['mm', 'cm', 'm'];

  const [IsectioncentroidYc, setIsectionCentroidYc] = useState(0);
  const [IsectioncentroidYcSelectedUnit, setIsectionCentroidYcSelectedUnit] = useState('mm');

  const IsectionMomentOfInertiaIxUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const IsectionMomentOfInertiaIxConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [IsectionmomentOfInertiaIx, setIsectionMomentOfInertiaIx] = useState(0);
  const [IsectionmomentOfInertiaIxSelectedUnit, setIsectionMomentOfInertiaIxSelectedUnit] = useState('mm⁴');


  // Moment of Inertia Iy
  const IsectionMomentOfInertiaIyUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const IsectionMomentOfInertiaIyConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };
  const [IsectionmomentOfInertiaIy, setIsectionMomentOfInertiaIy] = useState(0);
  const [IsectionmomentOfInertiaIySelectedUnit, setIsectionMomentOfInertiaIySelectedUnit] = useState('mm⁴');


  // Section Modulus Sx
  const IsectionSectionModulusSxUnits = ['mm³', 'cm³', 'm³'];
  const IsectionSectionModulusSxConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [IsectionsectionModulusSx, setIsectionSectionModulusSx] = useState(0);
  const [IsectionsectionModulusSxSelectedUnit, setIsectionSectionModulusSxSelectedUnit] = useState('mm³');

  const IsectionSectionModulusSyUnits = ['mm³', 'cm³', 'm³'];
  const IsectionSectionModulusSyConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [IsectionSectionModulusSy, setIsectionSectionModulusSy] = useState(0);
  const [IsectionSectionModulusSySelectedUnit, setIsectionSectionModulusSySelectedUnit] = useState('mm³');


  const IsectionTorsionalConstantUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const IsectionTorsionalConstantConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };
  const [IsectionTorsionalConstant, setIsectionTorsionalConstant] = useState(0);
  const [IsectionTorsionalConstantSelectedUnit, setIsectionTorsionalConstantSelectedUnit] = useState('mm⁴');



  const Isectiont = parseFloat(internalIsectionTValue);
  const Isectiontw = parseFloat(internalIsectiontwinputValue);
  const IsectionR = parseFloat(internalIsectionRinputValue);
  const Isectionb = parseFloat(IsectioninternalbinputValue);
  const Isectiond = parseFloat(internalIsectiondinputValue);

  useEffect(() => {
    const value = parseFloat(IsectioninternalbinputValue) * IsectiondconversionFactors[IsectionselectedUnit][0];
    if (!isNaN(value)) {
      const IsectionareaValue = (Isectiontw * Isectiond) + 2 * (Isectiont * Isectionb);
      setIsectionArea((IsectionareaValue / IsectionAreaConversionFactors[IsectionAreaUnit][0]));
    } else {
      setIsectionArea('');
    }
  }, [internalIsectionTValue, internalIsectiontwinputValue, IsectioninternalbinputValue, internalIsectiondinputValue, IsectionAreaUnit]);


  useEffect(() => {
    const b = IsectioninternalbinputValue;
    const IsectioncentroidXcValue = b / 2;
    setIsectionCentroidXc((IsectioncentroidXcValue / IsectionbconversionFactors[IsectioncentroidXcSelectedUnit][0]));
  }, [internalIsectionTValue, internalIsectiontwinputValue, IsectioninternalbinputValue, internalIsectiondinputValue, IsectioncentroidXcSelectedUnit]);


  // CentriodYcUnits

  useEffect(() => {
    const Isectiont = parseFloat(internalIsectionTValue);
    const Isectiond = parseFloat(internalIsectiondinputValue);

    const calculatedIsectionCentroidYc = (Isectiond / 2) - (-Isectiont);
    const convertedValue = ((calculatedIsectionCentroidYc / IsectionbconversionFactors[IsectioncentroidYcSelectedUnit][0]));

    const formattedValue = IsectioncentroidYcSelectedUnit === 'm'
      ? convertedValue.toFixed(4) // 2 decimal places in scientific notation for m⁴
      : convertedValue.toFixed(3); // 3 decimals for other units

    setIsectionCentroidYc(formattedValue);
  }, [internalIsectionTValue, internalIsectiontwinputValue, IsectioninternalbinputValue, internalIsectiondinputValue, IsectioncentroidYcSelectedUnit]);


  useEffect(() => {
    const value = Isectionb;
    if (!isNaN(value)) {
      const firstValueIx = (Isectionb * Math.pow((Isectiond - (-2 * Isectiont)), 3));
      const subtractedValueIx = (Isectionb - Isectiontw) * Math.pow(Isectiond, 3);
      const IsectionmomentOfInertiaIx = ((firstValueIx - subtractedValueIx) / 12).toFixed(3);
      const convertedValue = (IsectionmomentOfInertiaIx * IsectionMomentOfInertiaIxConversionFactors['mm⁴'][IsectionMomentOfInertiaIxUnits.indexOf(IsectionmomentOfInertiaIxSelectedUnit)]);
      const formattedValue = IsectionmomentOfInertiaIxSelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setIsectionMomentOfInertiaIx(formattedValue);
    }
  }, [Isectionb, Isectiond, Isectiont, Isectiontw, IsectionmomentOfInertiaIxSelectedUnit]);

  useEffect(() => {
    const value = Isectionb;
    if (!isNaN(value)) {
      const numeratorIy = (Math.pow(Isectionb, 3) * Isectiont) / 6;
      const denominatorIy = (Math.pow(Isectiontw, 3) * Isectiond) / 12;
      const IsectionmomentOfInertiaIy = (numeratorIy + denominatorIy).toFixed(2);

      const convertedValue = (IsectionmomentOfInertiaIy * IsectionMomentOfInertiaIyConversionFactors['mm⁴'][IsectionMomentOfInertiaIyUnits.indexOf(IsectionmomentOfInertiaIySelectedUnit)]);
      const formattedValue = IsectionmomentOfInertiaIySelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setIsectionMomentOfInertiaIy(formattedValue);
    }
  }, [Isectionb, Isectiond, Isectiont, Isectiontw, IsectionmomentOfInertiaIySelectedUnit]);


  useEffect(() => {
    const value = Isectionb
    if (!isNaN(value)) {
      const firstValueIx = (Isectionb * Math.pow((Isectiond - (-2 * Isectiont)), 3));
      const subtractedValueIx = (Isectionb - Isectiontw) * Math.pow(Isectiond, 3);
      const Ix = (firstValueIx - subtractedValueIx) / 12;
      const Yc = (Isectiond / 2) - (-Isectiont);
      const sectionModulesValue = (Ix / Yc).toFixed(2);
      const convertedValue = (sectionModulesValue * IsectionSectionModulusSxConversionFactors['mm³'][IsectionSectionModulusSxUnits.indexOf(IsectionsectionModulusSxSelectedUnit)]);
      const formattedValue = IsectionsectionModulusSxSelectedUnit === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m³
        : convertedValue.toFixed(3); // 3 decimals for other units

      setIsectionSectionModulusSx(formattedValue);
    }
  }, [Isectionb, Isectiond, Isectiont, Isectiontw, IsectionsectionModulusSxSelectedUnit]);
  useEffect(() => {
    const value = Isectionb;
    if (!isNaN(value)) {
      const Xc = Isectionb / 2
      const numeratorIy = (Math.pow(Isectionb, 3) * Isectiont) / 6;
      const denominatorIy = (Math.pow(Isectiontw, 3) * Isectiond) / 12;
      const Iy = numeratorIy + denominatorIy;
      const sectionModulesValue = (Iy / Xc).toFixed(3);
      const convertedValue = (sectionModulesValue * IsectionSectionModulusSyConversionFactors['mm³'][IsectionSectionModulusSyUnits.indexOf(IsectionSectionModulusSySelectedUnit)]);
      const formattedValue = IsectionSectionModulusSySelectedUnit === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m³
        : convertedValue.toFixed(3); // 3 decimals for other units

      setIsectionSectionModulusSy(formattedValue);
    }
  }, [Isectionb, Isectiond, Isectiont, Isectiontw, IsectionSectionModulusSySelectedUnit]);

  useEffect(() => {
    const value = Isectionb;
    if (!isNaN(value)) {
      const k1 = ((Isectionb * Math.pow(Isectiont, 3)) / 3) - (0.21 * Math.pow(Isectiont, 4)) - (0.0175 * (Math.pow(Isectiont, 8) / Math.pow(Isectionb, 4)))
      const k2 = (Isectiond * Math.pow(Isectiontw, 3)) / 3
      // const D = Math.pow((t -(-R)),2)
      // const D = R*tw
      const Dnumenator = (Math.pow((Isectiont - (-IsectionR)), 2) + IsectionR * Isectiontw + Math.pow(Isectiontw, 2) / 4)
      const Ddenomenator = 2 * IsectionR + Isectiont
      const D = Dnumenator / Ddenomenator
      if (Isectiont < Isectiontw) {
        var t1 = Isectiont
        var t2 = Isectiontw
      } else {
        var t1 = Isectiontw
        var t2 = Isectiont
      }
      const alpha = (t1 / t2) * (0.15 + (0.1 * (IsectionR / Isectiont)))
      const K = 2 * k1 + k2 + (2 * alpha * Math.pow(D, 4))
      const IsectionTorsionalConstant = K.toFixed(3);
      const convertedValue = (IsectionTorsionalConstant * IsectionTorsionalConstantConversionFactors['mm⁴'][IsectionTorsionalConstantUnits.indexOf(IsectionTorsionalConstantSelectedUnit)]);
      const formattedValue = IsectionTorsionalConstantSelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setIsectionTorsionalConstant(formattedValue);
    }
  }, [Isectionb, Isectiond, Isectiont, Isectiontw, IsectionR, IsectionTorsionalConstantSelectedUnit]);


  const IsectionImperialbunits = ['in'];
  const IsectionImperialbconversionFactors = {
    in: [1],
  };

  const [IsectionImperialbinputValue, setIsectionImperialbInputValue] = useState(0);
  const [IsectionImperialbselectedUnit, setIsectionImperialbSelectedUnit] = useState('in');

  const handleIsectionImperialbInputChange = (value) => {
    setIsectionImperialbInputValue(value);
  };

  const handleIsectionImperialbUnitChange = (unit) => {
    setIsectionImperialbSelectedUnit(unit);
    const bfactor = IsectionImperialbconversionFactors[unit][IsectionImperialbunits.indexOf(IsectionImperialbselectedUnit)];
    setIsectionImperialbInputValue((parseFloat(IsectionImperialbinputValue) / bfactor).toFixed(4));
  };


  const IsectionImperialdunits = ['in'];
  const IsectionImperialdconversionFactors = {
    in: [1],
  };

  const [IsectionImperialdinputValue, setIsectionImperialdInputValue] = useState(0);
  const [IsectionImperialdselectedUnit, setIsectionImperialdSelectedUnit] = useState('in');

  const handleIsectionImperialdInputChange = (value) => {
    setIsectionImperialdInputValue(value);
  };

  const handleIsectionImperialdUnitChange = (unit) => {
    setIsectionImperialdSelectedUnit(unit);
    const dfactor = IsectionImperialdconversionFactors[unit][IsectionImperialdunits.indexOf(IsectionImperialdselectedUnit)];
    setIsectionImperialdInputValue((parseFloat(IsectionImperialdinputValue) / dfactor).toFixed(4));
  };


  const IsectionImperialtwunits = ['in'];
  const IsectionImperialtwconversionFactors = {
    in: [1],
  };

  const [IsectionImperialtwinputValue, setIsectionImperialTwInputValue] = useState(0);
  const [IsectionImperialTWSelectedValue, setIsectionImperialtwSelectedUnit] = useState('in');

  const handleIsectionImperialtwInputChange = (value) => {
    setIsectionImperialTwInputValue(value);
  };

  const handleIsectionImperialtwUnitChange = (unit) => {
    setIsectionImperialtwSelectedUnit(unit);
    const IsectionImperialtwfactor = IsectionImperialtwconversionFactors[unit][IsectionImperialtwunits.indexOf(IsectionImperialTWSelectedValue)];
    setIsectionImperialTwInputValue((parseFloat(IsectionImperialtwinputValue) / IsectionImperialtwfactor).toFixed(4));
  };

  const IsectionImperialRunits = ['in'];
  const IsectionImperialRconversionFactors = {
    in: [1],
  };

  const [IsectionImperialRinputValue, setIsectionImperialRInputValue] = useState(0);
  const [IsectionImperialRSelectedValue, setIsectionImperialRSelectedUnit] = useState('in');

  const handleIsectionImperialRInputChange = (value) => {
    setIsectionImperialRInputValue(value);
  };

  const handleIsectionImperialRUnitChange = (unit) => {
    setIsectionImperialRSelectedUnit(unit);
    const IsectionImperialRfactor = IsectionImperialRconversionFactors[unit][IsectionImperialRunits.indexOf(IsectionImperialRSelectedValue)];
    setIsectionImperialRInputValue((parseFloat(IsectionImperialRinputValue) / IsectionImperialRfactor).toFixed(4));
  };

  const IsectionImperialtunits = ['in'];
  const IsectionImperialtconversionFactors = {
    in: [1],
  };

  const [IsectionImperialt, setIsectionImperialt] = useState(0);
  const [IsectionImperialtselectedTUnit, setIsectionImperialtSelectedUnit] = useState('in');

  const handleIsectionImperialtInputChange = (value) => {
    setIsectionImperialt(value);
  };

  const handleIsectionImperialtUnitChange = (unit) => {
    setIsectionImperialtSelectedUnit(unit);
    const IsectionImperialtfactor = IsectionImperialtconversionFactors[unit][IsectionImperialtunits.indexOf(IsectionImperialtselectedTUnit)];
    setIsectionImperialt((parseFloat(IsectionImperialt) / IsectionImperialtfactor).toFixed(4));
  };


  const IsectionImperialAreaUnits = ['in²'];
  const IsectionImperialAreaconversionFactors = {
    'in²': [1],
  };

  const [IsectionImperialArea, setIsectionImperialArea] = useState(0);
  const [IsectionImperialAreaUnit, setIsectionImperialAreaUnit] = useState('in²');

  // Handler for changing area unit
  const handleIsectionImperialAreaUnitChange = (unit) => {
    setIsectionImperialAreaUnit(unit);
    const currentFactor = IsectionImperialAreaconversionFactors[IsectionImperialAreaUnit][IsectionImperialAreaUnits.indexOf(unit)];
    setIsectionImperialArea((prevIsectionImperialArea) => prevIsectionImperialArea * currentFactor);
  };

  // Calculate area based on inputs
  const calculateIsectionImperialArea = () => {
    const t = parseFloat(IsectionImperialt);
    const tw = parseFloat(IsectionImperialtwinputValue);
    const b = parseFloat(IsectionImperialbinputValue);
    const d = parseFloat(IsectionImperialdinputValue);
    const areaValue = (tw * d) + 2 * (t * b);
    setIsectionImperialArea(areaValue);
  };

  // Effect to recalculate area whenever inputs change
  useEffect(() => {
    calculateIsectionImperialArea();
  }, [IsectionImperialt, IsectionImperialtwinputValue, IsectionImperialbinputValue, IsectionImperialdinputValue]);



  // CentroidXcUnits
  const IsectionImperialCentroidXcUnits = ['in'];
  const IsectionImperialCentroidXcconversionFactors = {
    in: [1],
  };

  const [IsectionImperialCentroidXc, setIsectionImperialCentroidXc] = useState(0);
  const [IsectionImperialCentroidXcSelectedUnit, setIsectionImperialCentroidXcSelectedUnit] = useState('in');

  // Handler for changing centroid Xc unit
  const handleIsectionImperialCentroidXcUnitChange = (unit) => {
    setIsectionImperialCentroidXcSelectedUnit(unit);
    const currentFactor = IsectionImperialCentroidXcconversionFactors[unit][IsectionImperialCentroidXcUnits.indexOf(IsectionImperialCentroidXcSelectedUnit)];
    setIsectionImperialCentroidXc((prevIsectionImperialCentroidXc) => prevIsectionImperialCentroidXc * currentFactor);
  };

  // Calculate centroid Xc based on input value
  const calculateIsectionImperialCentroidXc = () => {
    const b = parseFloat(IsectionImperialbinputValue);
    const IsectionImperialCentroidXc = b / 2;
    setIsectionImperialCentroidXc(IsectionImperialCentroidXc);
  };

  // Effect to recalculate centroid Xc whenever binputValue changes
  useEffect(() => {
    calculateIsectionImperialCentroidXc();
  }, [IsectionImperialbinputValue]);


  // CentriodYcUnits
  const IsectionImperialCentriodYcUnits = ['in'];
  const IsectionImperialCentriodYcconversionFactors = {
    in: [1],
  };
  const [IsectionImperialCentriodYc, setIsectionImperialCentriodYc] = useState(0);
  const [IsectionImperialCentriodYcSelectedUnit, setIsectionImperialCentriodYcSelectedUnit] = useState('in');

  // Handler for changing centroid Yc unit
  const handleIsectionImperialCentriodYcUnitChange = (unit) => {
    setIsectionImperialCentriodYcSelectedUnit(unit);
    const currentFactor = IsectionImperialCentriodYcconversionFactors[unit][IsectionImperialCentriodYcUnits.indexOf(IsectionImperialCentriodYcSelectedUnit)];
    setIsectionImperialCentriodYc((prevIsectionImperialCentriodYc) => prevIsectionImperialCentriodYc * currentFactor);
  };

  // Calculate centroid Yc based on input values
  const calculateIsectionImperialCentriodYc = () => {

    const t = parseFloat(IsectionImperialt);
    const d = parseFloat(IsectionImperialdinputValue);
    const IsectionImperialCentriodYc = (d / 2) - (-t);
    setIsectionImperialCentriodYc(IsectionImperialCentriodYc);
  };

  // Effect to recalculate centroid Yc whenever TValue or dinputValue changes
  useEffect(() => {
    calculateIsectionImperialCentriodYc();
  }, [IsectionImperialt, IsectionImperialdinputValue]);



  // MomentOfInertia Ix
  const [IsectionImperialMomentOfInertiaIx, setIsectionImperialMomentOfInertiaIx] = useState(0);
  const [IsectionImperialMomentOfInertiaIxSelectedUnit, setIsectionImperialMomentOfInertiaIxSelectedUnit] = useState('in⁴');

  // Handler for changing moment of inertia Ix unit
  const IsectionImperialMomentOfInertiaIxUnits = ['in⁴', 'cm⁴', 'm⁴', 'in⁴'];
  const IsectionImperialMomentOfInertiaIxconversionFactors = {
    'in⁴': [1]
  };
  const handleIsectionImperialMomentOfInertiaUnitChange = (unit) => {
    setIsectionImperialMomentOfInertiaIxSelectedUnit(unit);
    const currentFactor = IsectionImperialMomentOfInertiaIxconversionFactors[unit][IsectionImperialMomentOfInertiaIxUnits.indexOf(IsectionImperialMomentOfInertiaIxSelectedUnit)];
    setIsectionImperialMomentOfInertiaIx((prevIsectionImperialMomentOfInertiaIx) => prevIsectionImperialMomentOfInertiaIx / currentFactor);
  };

  // Calculate moment of inertia Ix based on input values
  const calculateIsectionImperialMomentOfInertiaIx = () => {
    const t = parseFloat(IsectionImperialt);
    const tw = parseFloat(IsectionImperialtwinputValue);
    const b = parseFloat(IsectionImperialbinputValue);
    const d = parseFloat(IsectionImperialdinputValue);

    const firstValueIx = (b * Math.pow((d - (-2 * t)), 3));
    const subtractedValueIx = (b - tw) * Math.pow(d, 3);
    const momentOfInertiaIxValue = ((firstValueIx - subtractedValueIx) / 12).toFixed(2);

    setIsectionImperialMomentOfInertiaIx(momentOfInertiaIxValue);
  };

  // Effect to recalculate moment of inertia Ix whenever inputs change
  useEffect(() => {
    calculateIsectionImperialMomentOfInertiaIx();
  }, [IsectionImperialt, IsectionImperialtwinputValue, IsectionImperialbinputValue, IsectionImperialdinputValue]);

  // IsectionImperialMomentOfInertia Iy
  const IsectionImperialMomentOfInertiaIyUnits = ['in⁴', 'cm⁴', 'm⁴', 'in⁴'];
  const IsectionImperialMomentOfInertiaIyconversionFactors = {
    'in³': [1]
  };

  const [IsectionImperialMomentOfInertiaIy, setIsectionImperialMomentOfInertiaIy] = useState(0);
  const [IsectionImperialMomentOfInertiaIySelectedUnit, setIsectionImperialMomentOfInertiaIySelectedUnit] = useState('in⁴');

  // Handler for changing moment of inertia Iy unit
  const handleIsectionImperialMomentOfInertiaIyUnitChange = (unit) => {
    setIsectionImperialMomentOfInertiaIySelectedUnit(unit);
    const currentFactor = IsectionImperialMomentOfInertiaIyconversionFactors[unit][IsectionImperialMomentOfInertiaIyUnits.indexOf(IsectionImperialMomentOfInertiaIySelectedUnit)];
    setIsectionImperialMomentOfInertiaIy((prevIsectionImperialMomentOfInertiaIy) => prevIsectionImperialMomentOfInertiaIy / currentFactor);
  };

  // Calculate moment of inertia Iy based on input values
  const calculateIsectionImperialMomentOfInertiaIy = () => {

    const t = parseFloat(IsectionImperialt);
    const tw = parseFloat(IsectionImperialtwinputValue);
    const b = parseFloat(IsectionImperialbinputValue);
    const d = parseFloat(IsectionImperialdinputValue);

    const numeratorIy = (Math.pow(b, 3) * t) / 6;
    const denominatorIy = (Math.pow(tw, 3) * d) / 12;
    const momentOfInertiaIyValue = (numeratorIy + denominatorIy).toFixed(2);

    setIsectionImperialMomentOfInertiaIy(momentOfInertiaIyValue);
  };

  // Effect to recalculate moment of inertia Iy whenever inputs change
  useEffect(() => {
    calculateIsectionImperialMomentOfInertiaIy();
  }, [IsectionImperialt, IsectionImperialtwinputValue, IsectionImperialbinputValue, IsectionImperialdinputValue]);



  // SectionModules Sx
  const IsectionImperialSectionModulesSxUnits = ['in³', 'cm³', 'm³', 'in³'];
  const IsectionImperialSectionModulesSxConversionUnit = {
    'in³': [1]
  };
  const [IsectionImperialSectionModulesSx, setIsectionImperialSectionModulesSx] = useState(0);
  const [IsectionImperialSectionModulesSxSelectedUnit, setIsectionImperialSectionModulesSxSelectedUnit] = useState('in³');
  const handleIsectionImperialSectionModulesSxUnitChange = (unit) => {
    setIsectionImperialSectionModulesSxSelectedUnit(unit);
    const IsectionImperialSectionModulesSxFactor = IsectionImperialSectionModulesSxConversionUnit[unit][IsectionImperialSectionModulesSxUnits.indexOf(IsectionImperialSectionModulesSxSelectedUnit)];
    setIsectionImperialSectionModulesSx((parseFloat(IsectionImperialSectionModulesSx) / IsectionImperialSectionModulesSxFactor));
  };
  const calculateIsectionImperialSectionModulesSxInputChangeValue = () => {

    const t = parseFloat(IsectionImperialt);
    const tw = parseFloat(IsectionImperialtwinputValue);
    const b = parseFloat(IsectionImperialbinputValue);
    const d = parseFloat(IsectionImperialdinputValue);

    const firstValueIx = (b * Math.pow((d - (-2 * t)), 3));
    const subtractedValueIx = (b - tw) * Math.pow(d, 3);
    const Ix = (firstValueIx - subtractedValueIx) / 12;
    const Yc = (d / 2) - (-t);
    const IsectionImperialSectionModulesSx = (Ix / Yc).toFixed(2);
    setIsectionImperialSectionModulesSx(IsectionImperialSectionModulesSx);
  };
  useEffect(() => {
    calculateIsectionImperialSectionModulesSxInputChangeValue();
  }, [IsectionImperialMomentOfInertiaIy, IsectioncentroidYc]);


  // Sy

  const IsectionImperialSectionModulesSyUnits = ['in³', 'cm³', 'm³', 'in³'];
  const IsectionImperialSectionModulesSyConversionUnit = {
    'in³': [1]
  };
  const [IsectionImperialSectionModulesSy, setIsectionImperialSectionModulesSy] = useState(0);
  const [IsectionImperialSectionModulesSySelectedUnit, setIsectionImperialSectionModulesSySelectedUnit] = useState('in³');

  // Handler for changing IsectionImperialSectionModulesSy unit
  const handleIsectionImperialSectionModulesSyUnitChange = (unit) => {
    setIsectionImperialSectionModulesSySelectedUnit(unit);
    const currentFactor = IsectionImperialSectionModulesSyConversionUnit[unit][IsectionImperialSectionModulesSyUnits.indexOf(IsectionImperialSectionModulesSySelectedUnit)];
    setIsectionImperialSectionModulesSy((prevIsectionImperialSectionModulesSy) => prevIsectionImperialSectionModulesSy * currentFactor);
  };

  // Calculate IsectionImperialSectionModulesSy based on MomentOfInertiaIy and CentroidXc
  const calculateIsectionImperialSectionModulesSy = () => {
    const t = parseFloat(IsectionImperialt);
    const tw = parseFloat(IsectionImperialtwinputValue);
    const b = parseFloat(IsectionImperialbinputValue);
    const d = parseFloat(IsectionImperialdinputValue);
    const Xc = b / 2
    const numeratorIy = (Math.pow(b, 3) * t) / 6;
    const denominatorIy = (Math.pow(tw, 3) * d) / 12;
    const Iy = numeratorIy + denominatorIy;
    if (Xc !== 0) {

      const IsectionImperialSectionModulesSyValue = (Iy / Xc).toFixed(2);
      setIsectionImperialSectionModulesSy(IsectionImperialSectionModulesSyValue);
    } else {
      setIsectionImperialSectionModulesSy(0);
    }
  };

  // Effect to recalculate IsectionImperialSectionModulesSy whenever MomentOfInertiaIy or CentroidXc changes
  useEffect(() => {
    calculateIsectionImperialSectionModulesSy();
  });

  const IsectionImperialTorsionalConstantUnits = ['in⁴'];
  const IsectionImperialTorsionalConstantConversionFactors = {
    'in⁴': [1]
  };
  const [IsectionImperialTorsionalConstant, setIsectionImperialTorsionalConstant] = useState();
  const [IsectionImperialTorsionalConstantSelectedUnit, setIsectionImperialTorsionalConstantSelectedUnit] = useState('in⁴');

  // Handler for changing IsectionImperialTorsionalConstant unit
  const handleIsectionImperialTorsionalConstantUnitChange = (unit) => {
    setIsectionImperialTorsionalConstantSelectedUnit(unit);
    const currentFactor = IsectionImperialTorsionalConstantConversionFactors[unit][IsectionImperialTorsionalConstantUnits.indexOf(IsectionImperialTorsionalConstantSelectedUnit)];
    setIsectionImperialTorsionalConstant((prevIsectionImperialTorsionalConstant) => prevIsectionImperialTorsionalConstant * currentFactor);
  };
  const tIsectionImperial = parseFloat(IsectionImperialt);
  const RIsectionImperial = parseFloat(IsectionImperialRinputValue);
  const twIsectionImperial = parseFloat(IsectionImperialtwinputValue);
  const bIsectionImperial = parseFloat(IsectionImperialbinputValue);
  const dIsectionImperial = parseFloat(IsectionImperialdinputValue);

  // Calculate IsectionImperialTorsionalConstant based on MomentOfInertiaIy and CentroidXc
  const calculateIsectionImperialTorsionalConstant = () => {
    const k1 = ((bIsectionImperial * Math.pow(tIsectionImperial, 3)) / 3) - (0.21 * Math.pow(tIsectionImperial, 4)) - (0.0175 * (Math.pow(tIsectionImperial, 8) / Math.pow(bIsectionImperial, 4)))
    const k2 = (dIsectionImperial * Math.pow(twIsectionImperial, 3)) / 3
    // const D = Math.pow((t -(-R)),2)
    // const D = R*tw
    const Dnumenator = (Math.pow((tIsectionImperial - (-RIsectionImperial)), 2) + RIsectionImperial * twIsectionImperial + Math.pow(twIsectionImperial, 2) / 4)
    const Ddenomenator = 2 * RIsectionImperial + tIsectionImperial
    const D = Dnumenator / Ddenomenator
    if (tIsectionImperial < twIsectionImperial) {
      var t1 = tIsectionImperial
      var t2 = twIsectionImperial
    } else {
      var t1 = twIsectionImperial
      var t2 = tIsectionImperial
    }
    const alpha = (t1 / t2) * (0.15 + (0.1 * (RIsectionImperial / tIsectionImperial)))
    const K = 2 * k1 + k2 + (2 * alpha * Math.pow(D, 4))
    const TorsionalConstant = K.toFixed(3);
    const convertedValue = (TorsionalConstant * IsectionImperialTorsionalConstantConversionFactors['in⁴'][IsectionImperialTorsionalConstantUnits.indexOf(IsectionImperialTorsionalConstantSelectedUnit)]);
    setIsectionImperialTorsionalConstant(convertedValue);
  };
  useEffect(() => {
    calculateIsectionImperialTorsionalConstant();
  });

  // Lsection calculation


  const Lsectionbunits = ['mm', 'cm', 'm'];
  const LsectionbconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [LsectionbinputValue, setLsectionbInputValue] = useState(0);
  const [LsectionbselectedUnit, setLsectionbSelectedUnit] = useState('mm');
  const [internalLsectionbinputValue, setInternalLsectionbinputValue] = useState(0); // Always in mm

  const handlebLsectionInputChange = (value) => {
    setLsectionbInputValue(value);
    const factor = LsectionbconversionFactors[LsectionbselectedUnit][0];
    setInternalLsectionbinputValue(parseFloat(value) * factor);
  };

  const handlebLsectionUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(LsectionbinputValue) * LsectionbconversionFactors[LsectionbselectedUnit][0];
    const convertedValue = newMetricValueInMM / LsectionbconversionFactors[unit][0];
    setLsectionbSelectedUnit(unit);
    setLsectionbInputValue(convertedValue.toFixed(3));
  };



  const Lsectiondunits = ['mm', 'cm', 'm'];
  const LsectiondconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [LsectiondinputValue, setLsectiondInputValue] = useState(0);
  const [selectedUnit, setdselectedUnit] = useState('mm');
  const [internalLsectiondinputValue, setInternalLsectiondinputValue] = useState(0); // Always in mm

  const handledLsectionInputChange = (value) => {
    setLsectiondInputValue(value);
    const factor = LsectiondconversionFactors[selectedUnit][0];
    setInternalLsectiondinputValue(parseFloat(value) * factor);
  };

  const handledLsectionUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(LsectiondinputValue) * LsectiondconversionFactors[selectedUnit][0];
    const convertedValue = newMetricValueInMM / LsectiondconversionFactors[unit][0];
    setdselectedUnit(unit);
    setLsectiondInputValue(convertedValue.toFixed(3));
  };
  const LsectionTValueunits = ['mm', 'cm', 'm'];
  const LsectionTValueconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [LsectionTValue, setLsectionTValue] = useState(0);
  const [selectedLsectionTVUnit, setLsectionTValueSelectedUnit] = useState('mm');
  const [internalLsectionTValue, setInternalLsectionTValue] = useState(0); // Always in mm

  const handleLsectionTValueInputChange = (value) => {
    setLsectionTValue(value);
    const factor = LsectionTValueconversionFactors[selectedLsectionTVUnit][0];
    setInternalLsectionTValue(parseFloat(value) * factor);
  };

  const handleLsectionTValueUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(LsectionTValue) * LsectionTValueconversionFactors[selectedLsectionTVUnit][0];
    const convertedValue = newMetricValueInMM / LsectionTValueconversionFactors[unit][0];
    setLsectionTValueSelectedUnit(unit);
    setLsectionTValue(convertedValue.toFixed(3));
  };

  const LsectionAreaUnits = ['mm²', 'cm²', 'm²'];
  const LsectionAreaConversionFactors = {
    'mm²': [1, 0.01, 0.000001],
    'cm²': [100, 1, 0.0001],
    'm²': [1000000, 10000, 1],
  };
  const [LsectionArea, setLsectionArea] = useState(0);
  const [LsectionAreaUnit, setLsectionAreaUnit] = useState('mm²');

  // CentroidXc
  const LsectionCentroidXcUnits = ['mm', 'cm', 'm'];

  const [LsectioncentroidXc, setLsectionCentroidXc] = useState(0);
  const [LsectioncentroidXcSelectedUnit, setLsectionCentroidXcSelectedUnit] = useState('mm');

  // CentroidYc
  const LsectionCentroidYcUnits = ['mm', 'cm', 'm'];

  const [LsectioncentroidYc, setLsectionCentroidYc] = useState(0);
  const [LsectioncentroidYcSelectedUnit, setLsectionCentroidYcSelectedUnit] = useState('mm');

  const LsectionMomentOfInertiaIxUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const LsectionMomentOfInertiaIxConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [LsectionmomentOfInertiaIx, setLsectionMomentOfInertiaIx] = useState(0);
  const [LsectionmomentOfInertiaIxSelectedUnit, setLsectionMomentOfInertiaIxSelectedUnit] = useState('mm⁴');


  // Moment of Inertia Iy
  const LsectionMomentOfInertiaIyUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const LsectionMomentOfInertiaIyConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };
  const [LsectionmomentOfInertiaIy, setLsectionMomentOfInertiaIy] = useState(0);
  const [LsectionmomentOfInertiaIySelectedUnit, setLsectionMomentOfInertiaIySelectedUnit] = useState('mm⁴');


  // Section Modulus Sx
  const LsectionSectionModulusSxUnits = ['mm³', 'cm³', 'm³'];
  const LsectionSectionModulusSxConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [LsectionsectionModulusSx, setLsectionSectionModulusSx] = useState(0);
  const [LsectionsectionModulusSxSelectedUnit, setLsectionSectionModulusSxSelectedUnit] = useState('mm³');

  const LsectionSectionModulusSyUnits = ['mm³', 'cm³', 'm³'];
  const LsectionSectionModulusSyConversionFactors = {
    'mm³': [1, 1e-3, 1e-9],
    'cm³': [1e3, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };
  const [LsectionSectionModulusSy, setLsectionSectionModulusSy] = useState(0);
  const [LsectionSectionModulusSySelectedUnit, setLsectionSectionModulusSySelectedUnit] = useState('mm³');





  const Lsectiont = parseFloat(internalLsectionTValue);
  const Lsectionb = parseFloat(internalLsectionbinputValue);
  const Lsectiond = parseFloat(internalLsectiondinputValue);

  useEffect(() => {
    const value = parseFloat(internalLsectionbinputValue) * LsectiondconversionFactors[selectedUnit][0];
    if (!isNaN(value)) {
      const LsectionareaValue = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
      setLsectionArea((LsectionareaValue / LsectionAreaConversionFactors[LsectionAreaUnit][0]));
    } else {
      setLsectionArea('');
    }
  }, [internalLsectionTValue, internalLsectionbinputValue, internalLsectiondinputValue, LsectionAreaUnit]);


  useEffect(() => {
    const b = internalLsectionbinputValue;
    const numerator = (b * b) + (Lsectiond * Lsectiont) - (Lsectiont * Lsectiont);
    const denominator = 2 * (b - (-Lsectiond) - Lsectiont);
    const LsectioncentroidXcValue = numerator / denominator;
    const convertedValue = (LsectioncentroidXcValue / LsectionbconversionFactors[LsectioncentroidXcSelectedUnit][0]);

    let formattedValue;
    switch (LsectioncentroidXcSelectedUnit) {
      case 'mm':
        formattedValue = convertedValue.toFixed(2);
        break;
      case 'cm':
        formattedValue = convertedValue.toFixed(3);
        break;
      case 'm':
        formattedValue = convertedValue.toFixed(4);
        break;
      default:
        formattedValue = convertedValue;
    }

    setLsectionCentroidXc(formattedValue);
  }, [internalLsectionTValue, internalLsectionbinputValue, internalLsectiondinputValue, LsectioncentroidXcSelectedUnit]);

  // CentriodYcUnits

  useEffect(() => {
    const Lsectiont = parseFloat(internalLsectionTValue);
    const Lsectionb = parseFloat(internalLsectionbinputValue);
    const Lsectiond = parseFloat(internalLsectiondinputValue);

    const numerator = (Lsectiond * Lsectiond) + (Lsectionb * Lsectiont) - (Lsectiont * Lsectiont);
    const denominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
    const calculatedLsectionCentroidYc = numerator / denominator;
    const convertedValue = (calculatedLsectionCentroidYc / LsectionbconversionFactors[LsectioncentroidYcSelectedUnit][0]);

    let formattedValue;
    switch (LsectioncentroidYcSelectedUnit) {
      case 'm':
        formattedValue = convertedValue.toFixed(5);
        break;
      case 'cm':
        formattedValue = convertedValue.toFixed(4);
        break;
      case 'mm':
        formattedValue = convertedValue.toFixed(2);
        break;
      default:
        formattedValue = convertedValue;
    }

    setLsectionCentroidYc(formattedValue);
  }, [internalLsectionTValue, internalLsectionbinputValue, internalLsectiondinputValue, LsectioncentroidYcSelectedUnit]);


  useEffect(() => {
    const value = Lsectionb;
    if (!isNaN(value)) {
      const ycValuenumerator = (Lsectiond * Lsectiond) + (Lsectionb * Lsectiont) - (Lsectiont * Lsectiont);
      const ycValuedenominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
      const ycValue = ycValuenumerator / ycValuedenominator;
      const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
      const numerator = (Lsectionb * Math.pow(Lsectiond, 3)) - ((Lsectionb - Lsectiont) * Math.pow(Lsectiond - Lsectiont, 3));
      const LsectionmomentOfInertiaIx = ((numerator / 3) - (area * Math.pow(Lsectiond - ycValue, 2)));
      const convertedValue = (LsectionmomentOfInertiaIx * LsectionMomentOfInertiaIxConversionFactors['mm⁴'][LsectionMomentOfInertiaIxUnits.indexOf(LsectionmomentOfInertiaIxSelectedUnit)]);

      let formattedValue;
      switch (LsectionmomentOfInertiaIxSelectedUnit) {
        case 'm⁴':
          formattedValue = convertedValue.toExponential(4);
          break;
        case 'cm⁴':
          formattedValue = convertedValue.toFixed(4);
          break;
        case 'mm⁴':
          formattedValue = convertedValue.toFixed(2);
          break;
        default:
          formattedValue = convertedValue;
      }

      setLsectionMomentOfInertiaIx(formattedValue);
    }
  }, [Lsectionb, Lsectiond, Lsectiont, LsectionmomentOfInertiaIxSelectedUnit]);

  useEffect(() => {
    const value = Lsectionb;
    if (!isNaN(value)) {

      const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
      const xcnumerator = (Lsectionb * Lsectionb) + (Lsectiond * Lsectiont) - (Lsectiont * Lsectiont);
      const xcdenominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
      const xc = xcnumerator / xcdenominator;
      const numerator = (Lsectiond * Math.pow(Lsectionb, 3)) - ((Lsectiond - Lsectiont) * Math.pow(Lsectionb - Lsectiont, 3));
      const LsectionmomentOfInertiaIy = ((numerator / 3) - (area * Math.pow(Lsectionb - xc, 2)));
      const convertedValue = (LsectionmomentOfInertiaIy * LsectionMomentOfInertiaIyConversionFactors['mm⁴'][LsectionMomentOfInertiaIyUnits.indexOf(LsectionmomentOfInertiaIySelectedUnit)]);

      let formattedValue;
      switch (LsectionmomentOfInertiaIySelectedUnit) {
        case 'm⁴':
          formattedValue = convertedValue.toExponential(4);
          break;
        case 'cm⁴':
          formattedValue = convertedValue.toFixed(4);
          break;
        case 'mm⁴':
          formattedValue = convertedValue.toFixed(2);
          break;
        default:
          formattedValue = convertedValue;
      }

      setLsectionMomentOfInertiaIy(formattedValue);
    }
  }, [Lsectionb, Lsectiond, Lsectiont, LsectionmomentOfInertiaIySelectedUnit]);


  useEffect(() => {
    const value = Lsectionb;
    if (!isNaN(value)) {
      const ycnumerator = (Lsectiond * Lsectiond) + (Lsectionb * Lsectiont) - (Lsectiont * Lsectiont);
      const ycdenominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
      const yc = ycnumerator / ycdenominator;
      const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
      const numerator = (Lsectionb * Math.pow(Lsectiond, 3)) - ((Lsectionb - Lsectiont) * Math.pow(Lsectiond - Lsectiont, 3));
      const ix = (numerator / 3) - (area * Math.pow(Lsectiond - yc, 2));
      const sectionModulesValue = (ix / (Lsectiond - yc));
      const convertedValue = (sectionModulesValue * LsectionSectionModulusSxConversionFactors['mm³'][LsectionSectionModulusSxUnits.indexOf(LsectionsectionModulusSxSelectedUnit)]);

      let formattedValue;
      switch (LsectionsectionModulusSxSelectedUnit) {
        case 'm³':
          formattedValue = convertedValue.toExponential(4);
          break;
        case 'cm³':
          formattedValue = convertedValue.toFixed(4);
          break;
        case 'mm³':
          formattedValue = convertedValue.toFixed(2);
          break;
        default:
          formattedValue = convertedValue;
      }

      setLsectionSectionModulusSx(formattedValue);
    }
  }, [Lsectionb, Lsectiond, Lsectiont, LsectionsectionModulusSxSelectedUnit]);


  useEffect(() => {
    const value = Lsectionb;
    if (!isNaN(value)) {
      const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
      const xcnumerator = (Lsectionb * Lsectionb) + (Lsectiond * Lsectiont) - (Lsectiont * Lsectiont);
      const xcdenominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
      const xc = xcnumerator / xcdenominator;
      const numerator = (Lsectiond * Math.pow(Lsectionb, 3)) - ((Lsectiond - Lsectiont) * Math.pow(Lsectionb - Lsectiont, 3));
      const iy = (numerator / 3) - (area * Math.pow(Lsectionb - xc, 2));
      const sectionModulesValue = (iy / (Lsectionb - xc));
      const convertedValue = (sectionModulesValue * LsectionSectionModulusSyConversionFactors['mm³'][LsectionSectionModulusSyUnits.indexOf(LsectionSectionModulusSySelectedUnit)]);

      let formattedValue;
      switch (LsectionSectionModulusSySelectedUnit) {
        case 'm³':
          formattedValue = convertedValue.toExponential(4);
          break;
        case 'cm³':
          formattedValue = convertedValue.toFixed(4);
          break;
        case 'mm³':
          formattedValue = convertedValue.toFixed(2);
          break;
        default:
          formattedValue = convertedValue;
      }

      setLsectionSectionModulusSy(formattedValue);
    }
  }, [Lsectionb, Lsectiond, Lsectiont, LsectionSectionModulusSySelectedUnit]);


  const LsectionImperialbunits = ['in'];
  const LsectionImperialbconversionFactors = {
    in: [1],
  };

  const [LsectionImperialbinputValue, setLsectionImperialbInputValue] = useState(0);
  const [LsectionImperialbselectedUnit, setLsectionImperialbSelectedUnit] = useState('in');

  const handleLsectionImperialbInputChange = (value) => {
    setLsectionImperialbInputValue(value);
  };

  const handleLsectionImperialbUnitChange = (unit) => {
    setLsectionImperialbSelectedUnit(unit);
    const LsectionImperialbfactor = LsectionImperialbconversionFactors[unit][LsectionImperialbunits.indexOf(LsectionImperialbselectedUnit)];
    setLsectionImperialbInputValue((parseFloat(LsectionImperialbinputValue) / LsectionImperialbfactor).toFixed(3));
  };


  const LsectionImperialdunits = ['in'];
  const LsectionImperialdconversionFactors = {
    in: [1],
  };

  const [LsectionImperialdinputValue, setLsectionImperialdInputValue] = useState(0);
  const [LsectionImperialdselectedUnit, setLsectionImperialSdelectedUnit] = useState('in');

  const handledLsectionImperialInputChange = (value) => {
    setLsectionImperialdInputValue(value);
  };

  const handledLsectionImperialUnitChange = (unit) => {
    setLsectionImperialSdelectedUnit(unit);
    const dfactor = LsectionImperialdconversionFactors[unit][LsectionImperialdunits.indexOf(LsectionImperialdselectedUnit)];
    setLsectionImperialdInputValue((parseFloat(LsectionImperialdinputValue) / dfactor).toFixed(3));
  };

  const LsectionImperialTValueunits = ['in'];
  const LsectionImperialTValueconversionFactors = {
    in: [1],
  };

  const [LsectionImperialTValue, setLsectionImperialTValue] = useState(0);
  const [LsectionImperialTVselectedTUnit, setLsectionImperialTValueSelectedUnit] = useState('in');

  const handleLsectionImperialTValueInputChange = (value) => {
    setLsectionImperialTValue(value);
  };

  const handleLsectionImperialTValueUnitChange = (unit) => {
    setLsectionImperialTValueSelectedUnit(unit);
    const LsectionImperialTValuefactor = LsectionImperialTValueconversionFactors[unit][LsectionImperialTValueunits.indexOf(LsectionImperialTVselectedTUnit)];
    setLsectionImperialTValue((parseFloat(LsectionImperialTValue) / LsectionImperialTValuefactor).toFixed(3));
  };

  const LsectionImperialAreaUnits = ['in²'];
  const LsectionImperialAreaConversionFactors = {
    'in²': [1],
  };
  const [LsectionImperialArea, setLsectionImperialArea] = useState(0);
  const [LsectionImperialAreaUnit, setLsectionImperialAreaUnit] = useState('in²');
  const handleLsectionImperialAreaUnitChange = (unit) => {
    const LsectionImperialAreaFactor = LsectionImperialAreaConversionFactors[unit][LsectionImperialAreaUnits.indexOf(LsectionImperialAreaUnit)];
    setLsectionImperialAreaUnit(unit);
    setLsectionImperialArea((parseFloat(LsectionImperialArea) / LsectionImperialAreaFactor));
  };
  const calculateLsectionImperialArea = () => {
    const Lsectiont = parseFloat(LsectionImperialTValue);
    const Lsectionb = parseFloat(LsectionImperialbinputValue);
    const Lsectiond = parseFloat(LsectionImperialdinputValue);
    const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
    setLsectionImperialArea(area);
  };
  useEffect(() => {
    calculateLsectionImperialArea();
  }, [LsectionImperialTValue, LsectionImperialbinputValue, LsectionImperialdinputValue]);

  // CentroidXcUnits
  const LsectionImperialCentroidXcUnits = ['in'];
  const LsectionImperialCentroidXcConversionFactors = {
    in: [1],
  };
  const [LsectionImperialCentroidXc, setLsectionImperialCentroidXc] = useState(0);
  const [LsectionImperialCentroidXcSelectedUnit, setLsectionImperialCentroidXcSelectedUnit] = useState('in');
  const handleLsectionImperialCentroidXcUnitChange = (unit) => {
    const CentroidFactor = LsectionImperialCentroidXcConversionFactors[unit][LsectionImperialCentroidXcUnits.indexOf(LsectionImperialCentroidXcSelectedUnit)];
    setLsectionImperialCentroidXcSelectedUnit(unit);
    setLsectionImperialCentroidXc((parseFloat(LsectionImperialCentroidXc) / CentroidFactor));
  };
  const calculateLsectionImperialCentroidXc = () => {
    const Lsectiont = parseFloat(LsectionImperialTValue);
    const Lsectionb = parseFloat(LsectionImperialbinputValue);
    const Lsectiond = parseFloat(LsectionImperialdinputValue);
    const numerator = (Lsectionb * Lsectionb) + (Lsectiond * Lsectiont) - (Lsectiont * Lsectiont);
    const denominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
    const LsectionImperialCentroidXc = (numerator / denominator).toFixed(2);
    setLsectionImperialCentroidXc(LsectionImperialCentroidXc);
  };
  useEffect(() => {
    calculateLsectionImperialCentroidXc();
  }, [LsectionImperialbinputValue, LsectionImperialTValue, LsectionImperialdinputValue]);

  // CentroidYcUnits
  const LsectionImperialCentroidYcUnits = ['in'];
  const LsectionImperialCentroidYcConversionFactors = {
    in: [1],
  };
  const [LsectionImperialCentroidYc, setLsectionImperialCentroidYc] = useState(0);
  const [LsectionImperialCentroidYcSelectedUnit, setLsectionImperialCentroidYcSelectedUnit] = useState('in');
  const handleLsectionImperialCentroidYcUnitChange = (unit) => {
    const LsectionImperialCentroidYcFactor = LsectionImperialCentroidYcConversionFactors[unit][LsectionImperialCentroidYcUnits.indexOf(LsectionImperialCentroidYcSelectedUnit)];
    setLsectionImperialCentroidYcSelectedUnit(unit);
    setLsectionImperialCentroidYc((parseFloat(LsectionImperialCentroidYc) / LsectionImperialCentroidYcFactor));
  };
  const calculateLsectionImperialCentroidYc = () => {
    const Lsectiont = parseFloat(LsectionImperialTValue);
    const Lsectionb = parseFloat(LsectionImperialbinputValue);
    const Lsectiond = parseFloat(LsectionImperialdinputValue);
    const numerator = (Lsectiond * Lsectiond) + (Lsectionb * Lsectiont) - (Lsectiont * Lsectiont);
    const denominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
    const centroidYc = (numerator / denominator).toFixed(2);
    setLsectionImperialCentroidYc(centroidYc);
  };
  useEffect(() => {
    calculateLsectionImperialCentroidYc();
  }, [LsectionImperialTValue, LsectionImperialbinputValue, LsectionImperialdinputValue]);

  // MomentOfInertia Ix
  const LsectionImperialMomentOfInertiaIxUnits = ['in⁴'];
  const LsectionImperialMomentOfInertiaIxConversionFactors = {
    'in⁴': [1],
  };
  const [LsectionImperialMomentOfInertiaIx, setLsectionImperialMomentOfInertiaIx] = useState(0);
  const [LsectionImperialMomentOfInertiaIxSelectedUnit, setLsectionImperialMomentOfInertiaIxSelectedUnit] = useState('in⁴');
  const handleLsectionImperialMomentOfInertiaIxUnitChange = (unit) => {
    const LsectionImperialMomentOfInertiaIxFactor = LsectionImperialMomentOfInertiaIxConversionFactors[unit][LsectionImperialMomentOfInertiaIxUnits.indexOf(LsectionImperialMomentOfInertiaIxSelectedUnit)];
    setLsectionImperialMomentOfInertiaIxSelectedUnit(unit);
    setLsectionImperialMomentOfInertiaIx((parseFloat(LsectionImperialMomentOfInertiaIx) / LsectionImperialMomentOfInertiaIxFactor));
  };
  const calculateLsectionImperialMomentOfInertiaIx = () => {
    const Lsectiont = parseFloat(LsectionImperialTValue);
    const Lsectionb = parseFloat(LsectionImperialbinputValue);
    const Lsectiond = parseFloat(LsectionImperialdinputValue);
    const ycnumerator = (Lsectiond * Lsectiond) + (Lsectionb * Lsectiont) - (Lsectiont * Lsectiont);
    const ycdenominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
    const yc = ycnumerator / ycdenominator;
    const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
    const numerator = (Lsectionb * Math.pow(Lsectiond, 3)) - ((Lsectionb - Lsectiont) * Math.pow(Lsectiond - Lsectiont, 3));
    const momentOfInertiaIx = ((numerator / 3) - (area * Math.pow(Lsectiond - yc, 2))).toFixed(2);
    setLsectionImperialMomentOfInertiaIx(momentOfInertiaIx);
  };
  useEffect(() => {
    calculateLsectionImperialMomentOfInertiaIx();
  }, [LsectionImperialTValue, LsectionImperialbinputValue, LsectionImperialdinputValue, LsectionArea, LsectioncentroidYc]);

  // MomentOfInertia Iy
  const LsectionImperialMomentOfInertiaIyUnits = ['in⁴'];
  const LsectionImperialMomentOfInertiaIyConversionFactors = {
    'in⁴': [1],
  };

  const [LsectionImperialMomentOfInertiaIy, setLsectionImperialMomentOfInertiaIy] = useState(0);
  const [LsectionImperialMomentOfInertiaIySelectedUnit, setLsectionImperialMomentOfInertiaIySelectedUnit] = useState('in⁴');

  const handleLsectionImperialMomentOfInertiaIyUnitChange = (newUnit) => {
    if (LsectionImperialMomentOfInertiaIy === '') return;

    const currentFactor = LsectionImperialMomentOfInertiaIyConversionFactors[LsectionImperialMomentOfInertiaIySelectedUnit][LsectionImperialMomentOfInertiaIyUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(LsectionImperialMomentOfInertiaIy) * currentFactor;
    setLsectionImperialMomentOfInertiaIySelectedUnit(newUnit);
    setLsectionImperialMomentOfInertiaIy(convertedValue.toString());
  };

  const calculateLsectionImperialMomentOfInertiaIy = () => {
    const Lsectiont = parseFloat(LsectionImperialTValue);
    const Lsectionb = parseFloat(LsectionImperialbinputValue);
    const Lsectiond = parseFloat(LsectionImperialdinputValue);

    if (!isNaN(Lsectiont) && !isNaN(Lsectionb) && !isNaN(Lsectiond)) {
      const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
      const xcnumerator = (Lsectionb * Lsectionb) + (Lsectiond * Lsectiont) - (Lsectiont * Lsectiont);
      const xcdenominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
      const xc = xcnumerator / xcdenominator;
      const numerator = (Lsectiond * Math.pow(Lsectionb, 3)) - ((Lsectiond - Lsectiont) * Math.pow(Lsectionb - Lsectiont, 3));
      const LsectionImperialMomentOfInertiaIy = ((numerator / 3) - (area * Math.pow(Lsectionb - xc, 2))).toFixed(2);
      const currentFactor = LsectionImperialMomentOfInertiaIyConversionFactors['in⁴'][LsectionImperialMomentOfInertiaIyUnits.indexOf(LsectionImperialMomentOfInertiaIySelectedUnit)];
      setLsectionImperialMomentOfInertiaIy((LsectionImperialMomentOfInertiaIy * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateLsectionImperialMomentOfInertiaIy();
  }, [LsectionImperialTValue, LsectionImperialbinputValue, LsectionImperialdinputValue, LsectionImperialMomentOfInertiaIySelectedUnit]);

  // SectionModulus Sx
  const LsectionImperialSectionModulusSxUnits = ['in³'];
  const LsectionImperialSectionModulusSxConversionFactors = {
    'in³': [1],
  };
  const [LsectionImperialSectionModulusSx, setLsectionImperialSectionModulusSx] = useState(0);
  const [LsectionImperialSectionModulusSxSelectedUnit, setLsectionImperialSectionModulusSxSelectedUnit] = useState('in³');

  const handleLsectionImperialSectionModulusSxUnitChange = (newUnit) => {
    if (LsectionImperialSectionModulusSx === '') return;

    const currentFactor = LsectionImperialSectionModulusSxConversionFactors[LsectionImperialSectionModulusSxSelectedUnit][LsectionImperialSectionModulusSxUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(LsectionImperialSectionModulusSx) * currentFactor;
    setLsectionImperialSectionModulusSxSelectedUnit(newUnit);
    setLsectionImperialSectionModulusSx(convertedValue.toString());
  };

  const calculateLsectionImperialSectionModulusSx = () => {
    const Lsectiont = parseFloat(LsectionImperialTValue);
    const Lsectionb = parseFloat(LsectionImperialbinputValue);
    const Lsectiond = parseFloat(LsectionImperialdinputValue);

    if (!isNaN(Lsectiont) && !isNaN(Lsectionb) && !isNaN(Lsectiond)) {
      const ycnumerator = (Lsectiond * Lsectiond) + (Lsectionb * Lsectiont) - (Lsectiont * Lsectiont);
      const ycdenominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
      const yc = ycnumerator / ycdenominator;
      const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
      const numerator = (Lsectionb * Math.pow(Lsectiond, 3)) - ((Lsectionb - Lsectiont) * Math.pow(Lsectiond - Lsectiont, 3));
      const ix = (numerator / 3) - (area * Math.pow(Lsectiond - yc, 2));
      const LsectionImperialSectionModulusSx = (ix / (Lsectiond - yc)).toFixed(2);
      const currentFactor = LsectionImperialSectionModulusSxConversionFactors['in³'][LsectionImperialSectionModulusSxUnits.indexOf(LsectionImperialSectionModulusSxSelectedUnit)];
      setLsectionImperialSectionModulusSx((LsectionImperialSectionModulusSx * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateLsectionImperialSectionModulusSx();
  }, [LsectionImperialTValue, LsectionImperialbinputValue, LsectionImperialdinputValue, LsectionImperialSectionModulusSxSelectedUnit]);

  // SectionModulus Sy

  const LsectionImperialSectionModulusSyUnits = ['in³'];
  const LsectionImperialSectionModulusSyConversionFactors = {
    'in³': [1],
  };

  const [LsectionImperialSectionModulusSy, setLsectionImperialSectionModulusSy] = useState(0);
  const [LsectionImperialSectionModulusSySelectedUnit, setLsectionImperialSectionModulusSySelectedUnit] = useState('in³');

  const handleLsectionImperialSectionModulusSyUnitChange = (newUnit) => {
    if (LsectionImperialSectionModulusSy === '') return;

    const currentFactor = LsectionImperialSectionModulusSyConversionFactors[LsectionImperialSectionModulusSySelectedUnit][LsectionImperialSectionModulusSyUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(LsectionImperialSectionModulusSy) * currentFactor;
    setLsectionImperialSectionModulusSySelectedUnit(newUnit);
    setLsectionImperialSectionModulusSy(convertedValue.toString());
  };

  const calculateLsectionImperialSectionModulusSy = () => {
    const Lsectiont = parseFloat(LsectionImperialTValue);
    const Lsectionb = parseFloat(LsectionImperialbinputValue);
    const Lsectiond = parseFloat(LsectionImperialdinputValue);

    if (!isNaN(Lsectiont) && !isNaN(Lsectionb) && !isNaN(Lsectiond)) {
      const area = Lsectiont * (Lsectionb - (-Lsectiond) - Lsectiont);
      const xcnumerator = (Lsectionb * Lsectionb) + (Lsectiond * Lsectiont) - (Lsectiont * Lsectiont);
      const xcdenominator = 2 * (Lsectionb - (-Lsectiond) - Lsectiont);
      const xc = xcnumerator / xcdenominator;
      const numerator = (Lsectiond * Math.pow(Lsectionb, 3)) - ((Lsectiond - Lsectiont) * Math.pow(Lsectionb - Lsectiont, 3));
      const iy = (numerator / 3) - (area * Math.pow(Lsectionb - xc, 2));
      const LsectionImperialSectionModulusSy = (iy / (Lsectionb - xc)).toFixed(2);
      const currentFactor = LsectionImperialSectionModulusSyConversionFactors['in³'][LsectionImperialSectionModulusSyUnits.indexOf(LsectionImperialSectionModulusSySelectedUnit)];
      setLsectionImperialSectionModulusSy((LsectionImperialSectionModulusSy * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateLsectionImperialSectionModulusSy();
  }, [LsectionImperialTValue, LsectionImperialbinputValue, LsectionImperialdinputValue, LsectionImperialSectionModulusSySelectedUnit]);


  // Solid circle calculation

  const SolidCircleunits = ['mm', 'cm', 'm'];
  const SolidCircleconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [SolidCircleinputValue, setSolidCircleInputValue] = useState(0);
  const [SolidCircleselectedUnit, setSolidCircleSelectedUnit] = useState('mm');
  const [internalSolidCircleinputValue, setInternalSolidCircleinputValue] = useState(0); // Always in mm


  const handleSolidCircleInputChange = (value) => {
    setSolidCircleInputValue(value);
    const factor = SolidCircleconversionFactors[SolidCircleselectedUnit][0];
    setInternalSolidCircleinputValue(parseFloat(value) * factor);
  };

  const handleSolidCircleUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(SolidCircleinputValue) * SolidCircleconversionFactors[SolidCircleselectedUnit][0];
    const convertedValue = newMetricValueInMM / SolidCircleconversionFactors[unit][0];
    setSolidCircleSelectedUnit(unit);
    setSolidCircleInputValue(convertedValue.toFixed(3));
  };


  const SolidCircleCentroidUnits = ['mm', 'cm', 'm'];

  const [SolidCircleCentroid, setSolidCircleCentroid] = useState(0);
  const [SolidCircleCentroidSelectedunit, setSolidCircleCentroidSelectedUnit] = useState('mm');



  // MomentOfInertia
  const SolidCircleMomentOfInertiaUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const SolidCircleMomentOfInertiaConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12,],
    'cm⁴': [1e4, 1, 1e-8,],
    'm⁴': [1e12, 1e8, 1,],
  };
  const [SolidCirclemomentOfInertia, setSolidCircleMomentOfInertia] = useState(0);
  const [SolidCirclemomentOfInertiaSelectedUnit, setSolidCircleMomentOfInertiaSelectedUnit] = useState('mm⁴');


  const SolidCircleSectionModulesUnits = ['mm³', 'cm³', 'm³'];
  const SolidCircleSectionModulesConversionFactors = {
    'mm³': [1, 0.001, 1e-9],
    'cm³': [1000, 1, 1e-6],
    'm³': [1e9, 1e6, 1,],
  };
  const [SolidCirclesectionModules, setSolidCircleSectionModules] = useState(0);
  const [SolidCirclesectionModulesSelectedUnit, setSolidCircleSectionModulesSelectedUnit] = useState('mm³');


  const SolidCircleTorsionalConstantUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const SolidCircleTorsionalConstantConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [SolidCircletorsionalConstant, setSolidCircleTorsionalConstant] = useState(0);
  const [SolidCircletorsionalConstantSelectedUnit, setSolidCircleTorsionalConstantSelectedUnit] = useState('mm⁴');

  useEffect(() => {
    const value = internalSolidCircleinputValue;
    const SolidCirclecentroidValue = (value);
    setSolidCircleCentroid((SolidCirclecentroidValue / SolidCircleconversionFactors[SolidCircleCentroidSelectedunit][0]));
  }, [internalSolidCircleinputValue, SolidCircleCentroidSelectedunit]);

  useEffect(() => {
    const value = internalSolidCircleinputValue;
    if (!isNaN(value)) {
      const pi = Math.PI
      const inertia = ((pi * Math.pow(value, 4)) / 4);
      const convertedValue = (inertia * SolidCircleMomentOfInertiaConversionFactors['mm⁴'][SolidCircleMomentOfInertiaUnits.indexOf(SolidCirclemomentOfInertiaSelectedUnit)]);

      let formattedValue;
      switch (SolidCirclemomentOfInertiaSelectedUnit) {
        case 'mm⁴':
          formattedValue = convertedValue.toFixed(1);
          break;
        case 'cm⁴':
          formattedValue = convertedValue.toFixed(4);
          break;
        case 'm⁴':
          formattedValue = convertedValue.toExponential(4);
          break;
        default:
          formattedValue = convertedValue;
      }

      setSolidCircleMomentOfInertia(formattedValue);
    }
  }, [internalSolidCircleinputValue, SolidCirclemomentOfInertiaSelectedUnit]);

  useEffect(() => {
    const value = internalSolidCircleinputValue;
    if (!isNaN(value)) {
      const pi = Math.PI
      const momentOfInertiaValue = ((pi * Math.pow(value, 4)) / 4);
      const SolidCirclesectionModulesValue = (momentOfInertiaValue / value);
      const convertedValue = (SolidCirclesectionModulesValue * SolidCircleSectionModulesConversionFactors['mm³'][SolidCircleSectionModulesUnits.indexOf(SolidCirclesectionModulesSelectedUnit)]);

      let formattedValue;
      switch (SolidCirclesectionModulesSelectedUnit) {
        case 'm³':
          formattedValue = convertedValue.toExponential(4);
          break;
        case 'cm³':
          formattedValue = convertedValue.toFixed(4);
          break;
        case 'mm³':
          formattedValue = convertedValue.toFixed(2);
          break;
        default:
          formattedValue = convertedValue;
      }

      setSolidCircleSectionModules(formattedValue);
    }
  }, [internalSolidCircleinputValue, SolidCirclesectionModulesSelectedUnit]);

  useEffect(() => {
    const value = internalSolidCircleinputValue;
    if (!isNaN(value)) {
      const pi = Math.PI
      const torsional = ((pi * Math.pow(value, 4)) / 2);
      const convertedValue = (torsional * SolidCircleTorsionalConstantConversionFactors['mm⁴'][SolidCircleTorsionalConstantUnits.indexOf(SolidCircletorsionalConstantSelectedUnit)]);

      let formattedValue;
      switch (SolidCircletorsionalConstantSelectedUnit) {
        case 'm⁴':
          formattedValue = convertedValue.toExponential(4);
          break;
        case 'cm⁴':
          formattedValue = convertedValue.toFixed(4);
          break;
        case 'mm⁴':
          formattedValue = convertedValue.toFixed(1);
          break;
        default:
          formattedValue = convertedValue;
      }

      setSolidCircleTorsionalConstant(formattedValue);
    }
  }, [internalSolidCircleinputValue, SolidCircletorsionalConstantSelectedUnit]);






  const SolidCircleImperialunits = ['in'];
  const SolidCircleImperialconversionFactors = {
    in: [1],
  };

  const [SolidCircleImperialinputValue, setSolidCircleImperialInputValue] = useState(0);
  const [SolidCircleImperialselectedUnit, setSolidCircleImperialSelectedUnit] = useState('in');

  const handleSolidCircleImperialInputChange = (value) => {
    setSolidCircleImperialInputValue(value);
  };

  const handleSolidCircleImperialUnitChange = (unit) => {
    setSolidCircleImperialSelectedUnit(unit);
    const factor = SolidCircleImperialconversionFactors[unit][SolidCircleImperialunits.indexOf(SolidCircleImperialselectedUnit)];
    setSolidCircleImperialInputValue((parseFloat(SolidCircleImperialinputValue) / factor).toFixed(3));
  };

  const SolidCircleImperialCentroidUnits = ['in'];
  const SolidCircleImperialCentroidconversionFactors = {
    in: [1],
  };

  const [SolidCircleImperialCentroid, setSolidCircleImperialCentroid] = useState(0);
  const [SolidCircleImperialCentroidSelectedunit, setSolidCircleImperialCentroidSelectedUnit] = useState('in');

  const handleSolidCircleImperialCentroidUnitChange = (unit) => {
    setSolidCircleImperialCentroidSelectedUnit(unit);
    const SolidCircleImperialCentroidFactor = SolidCircleImperialCentroidconversionFactors[unit][SolidCircleImperialCentroidUnits.indexOf(SolidCircleImperialCentroidSelectedunit)];
    setSolidCircleImperialCentroid((parseFloat(SolidCircleImperialCentroid) / SolidCircleImperialCentroidFactor));
  };

  const calculateSolidCircleImperialCentroidInputChangeValue = () => {
    const Input = parseFloat(SolidCircleImperialinputValue)
    const SolidCircleImperialCentroid = Input
    setSolidCircleImperialCentroid(SolidCircleImperialCentroid);
  };
  useEffect(() => {
    calculateSolidCircleImperialCentroidInputChangeValue();
  }, [SolidCircleImperialinputValue]);


  // MomentOfInertia
  const SolidCircleImperialMomentOfInertiaUnits = ['in⁴'];
  const SolidCircleImperialMomentOfInertiaConversionFactors = {
    'in⁴': [1],
  };
  const [SolidCircleImperialMomentOfInertia, setSolidCircleImperialMomentOfInertia] = useState(0);
  const [SolidCircleImperialMomentOfInertiaSelectedUnit, setSolidCircleImperialMomentOfInertiaSelectedUnit] = useState('in⁴');

  const handleSolidCircleImperialMomentOfInertiaUnitChange = (newUnit) => {
    const currentFactor = SolidCircleImperialMomentOfInertiaConversionFactors[SolidCircleImperialMomentOfInertiaSelectedUnit][SolidCircleImperialMomentOfInertiaUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(SolidCircleImperialMomentOfInertia) * currentFactor;
    setSolidCircleImperialMomentOfInertiaSelectedUnit(newUnit);
    setSolidCircleImperialMomentOfInertia(convertedValue.toString());
  };

  const calculateSolidCircleImperialMomentOfInertia = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const value = SolidCircleImperialinputValue;
      const pi = Math.PI
      const SolidCircleImperialMomentOfInertia = ((pi * Math.pow(value, 4)) / 4).toFixed(2);

      // const SolidCircleImperialMomentOfInertia = ((pi * Math.pow(SolidCircleImperialinputValue, 4)) / 4).toFixed(1);
      const currentFactor = SolidCircleImperialMomentOfInertiaConversionFactors['in⁴'][SolidCircleImperialMomentOfInertiaUnits.indexOf(SolidCircleImperialMomentOfInertiaSelectedUnit)];
      setSolidCircleImperialMomentOfInertia((SolidCircleImperialMomentOfInertia * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateSolidCircleImperialMomentOfInertia(SolidCircleImperialinputValue);
  }, [SolidCircleImperialinputValue, SolidCircleImperialMomentOfInertiaSelectedUnit]);


  const SolidCircleImperialSectionModulesUnits = ['in³'];
  const SolidCircleImperialSectionModulesConversionFactors = {
    'in³': [1],
  };
  const [SolidCircleImperialSectionModules, setSolidCircleImperialSectionModules] = useState(0);
  const [SolidCircleImperialSectionModulesSelectedUnit, setSolidCircleImperialSectionModulesSelectedUnit] = useState('in³');

  const handleSolidCircleImperialSectionModulesUnitChange = (newUnit) => {
    const currentFactor = SolidCircleImperialSectionModulesConversionFactors[SolidCircleImperialSectionModulesSelectedUnit][SolidCircleImperialSectionModulesUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(SolidCircleImperialSectionModules) * currentFactor;
    setSolidCircleImperialSectionModulesSelectedUnit(newUnit);
    setSolidCircleImperialSectionModules(convertedValue.toString());
  };

  const calculateSolidCircleImperialSectionModules = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const pi = Math.PI
      // ((pi * Math.pow(SolidCircleImperialinputValue, 4)) / 4)
      const SolidCircleImperialSectionModules = (((pi * Math.pow(SolidCircleImperialinputValue, 4)) / 4) / SolidCircleImperialinputValue).toFixed(2);
      const currentFactor = SolidCircleImperialSectionModulesConversionFactors['in³'][SolidCircleImperialSectionModulesUnits.indexOf(SolidCircleImperialSectionModulesSelectedUnit)];
      setSolidCircleImperialSectionModules((SolidCircleImperialSectionModules * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateSolidCircleImperialSectionModules(SolidCircleImperialinputValue);
  }, [SolidCircleImperialinputValue, SolidCircleImperialSectionModulesSelectedUnit]);



  const SolidCircleImperialTorsionalConstantUnits = ['in⁴'];
  const SolidCircleImperialTorsionalConstantConversionFactors = {
    'in⁴': [1],
  };
  const [SolidCircleImperialTorsionalConstant, setSolidCircleImperialTorsionalConstant] = useState(0);
  const [SolidCircleImperialTorsionalConstantSelectedUnit, setSolidCircleImperialTorsionalConstantSelectedUnit] = useState('in⁴');

  const handleSolidCircleImperialTorsionalConstantUnitChange = (newUnit) => {
    const currentFactor = SolidCircleImperialTorsionalConstantConversionFactors[SolidCircleImperialTorsionalConstantSelectedUnit][SolidCircleImperialTorsionalConstantUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(SolidCircleImperialTorsionalConstant) * currentFactor;
    setSolidCircleImperialTorsionalConstantSelectedUnit(newUnit);
    setSolidCircleImperialTorsionalConstant(convertedValue.toString());
  };

  const calculateSolidCircleImperialTorsionalConstant = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const torsional = ((Math.PI * Math.pow(parsedValue, 4)) / 2).toFixed(2);
      const currentFactor = SolidCircleImperialTorsionalConstantConversionFactors['in⁴'][SolidCircleImperialTorsionalConstantUnits.indexOf(SolidCircleImperialTorsionalConstantSelectedUnit)];
      setSolidCircleImperialTorsionalConstant((torsional * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateSolidCircleImperialTorsionalConstant(SolidCircleImperialinputValue);
  }, [SolidCircleImperialinputValue, SolidCircleImperialTorsionalConstantSelectedUnit]);


  // Hollow Circle calculation

  const HollowCircleunits = ['mm', 'cm', 'm'];
  // const [exponent, setExponent] = useState(4);

  const HollowCircleconversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [HollowCircleinputValue, setHollowCircleInputValue] = useState(0);
  const [HollowCircleselectedUnit, setHollowCircleSelectedUnit] = useState('mm');
  const [internalHollowCircleinputValue, setInternalHollowCircleinputValue] = useState(0); // Always in mm

  const handleHollowCircleInputChange = (value) => {
    setHollowCircleInputValue(value);
    const factor = HollowCircleconversionFactors[HollowCircleselectedUnit][0];
    setInternalHollowCircleinputValue(parseFloat(value) * factor);
  };

  const handleHollowCircleUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(HollowCircleinputValue) * HollowCircleconversionFactors[HollowCircleselectedUnit][0];
    const convertedValue = newMetricValueInMM / HollowCircleconversionFactors[unit][0];
    setHollowCircleSelectedUnit(unit);
    setHollowCircleInputValue(convertedValue);
  };


  const HollowCircleHeightunits = ['mm', 'cm', 'm'];

  const HollowCircleHeightConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const [HollowCircleHeightInputValue, setHollowCircleHeightInputValue] = useState(0);
  const [HollowCircleheightSelectedUnit, setHollowCircleHeightSelectedUnit] = useState('mm');
  const [internalHollowCircleHeightInputValue, setInternalHollowCircleHeightInputValue] = useState(0); // Always in mm

  const handleHollowCircleHeightInputValue = (value) => {
    setHollowCircleHeightInputValue(value);
    const factor = HollowCircleHeightConversionFactors[HollowCircleheightSelectedUnit][0];
    setInternalHollowCircleHeightInputValue(parseFloat(value) * factor);
  };

  const handleHollowCircleHeightSelectedUnit = (unit) => {
    const newMetricValueInMM = parseFloat(HollowCircleHeightInputValue) * HollowCircleHeightConversionFactors[HollowCircleheightSelectedUnit][0];
    const convertedValue = newMetricValueInMM / HollowCircleHeightConversionFactors[unit][0];
    setHollowCircleHeightSelectedUnit(unit);
    setHollowCircleHeightInputValue(convertedValue.toFixed(4));
  };

  const HollowCircleCentroidUnits = ['mm', 'cm', 'm'];

  const [HollowCircleCentroid, setHollowCircleCentroid] = useState(0);
  const [HollowCircleCentroidSelectedunit, setHollowCircleCentroidSelectedUnit] = useState('mm');



  // MomentOfInertia
  const HollowCircleMomentOfInertiaUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const HollowCircleMomentOfInertiaConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12,],
    'cm⁴': [1e4, 1, 1e-8,],
    'm⁴': [1e12, 1e8, 1,],
  };
  const [HollowCirclemomentOfInertia, setHollowCircleMomentOfInertia] = useState(0);
  const [HollowCirclemomentOfInertiaSelectedUnit, setHollowCircleMomentOfInertiaSelectedUnit] = useState('mm⁴');


  const HollowCircleSectionModulesUnits = ['mm³', 'cm³', 'm³'];
  const HollowCircleSectionModulesConversionFactors = {
    'mm³': [1, 0.001, 1e-9],
    'cm³': [1000, 1, 1e-6],
    'm³': [1e9, 1e6, 1,],
  };
  const [HollowCirclesectionModules, setHollowCircleSectionModules] = useState(0);
  const [HollowCirclesectionModulesSelectedUnit, setHollowCircleSectionModulesSelectedUnit] = useState('mm³');



  useEffect(() => {
    const value = internalHollowCircleinputValue;
    const HollowCirclecentroidValue = value;
    setHollowCircleCentroid((HollowCirclecentroidValue / HollowCircleconversionFactors[HollowCircleCentroidSelectedunit][0]));
  }, [internalHollowCircleinputValue, HollowCircleCentroidSelectedunit]);

  useEffect(() => {
    const value = internalHollowCircleinputValue;
    if (!isNaN(value)) {
      const pi = Math.PI;
      const parsedB = internalHollowCircleinputValue;
      const parsedD = internalHollowCircleHeightInputValue;
      const HollowCirclemomentOfInertiamultiplyValue = ((Math.pow(parsedB, 4) - Math.pow(parsedD, 4)));
      const divdeinertiaValue = 4;
      const inertia = ((pi * HollowCirclemomentOfInertiamultiplyValue) / divdeinertiaValue).toFixed(1);


      // Convert inertia based on the selected unit
      const convertedValue = inertia * HollowCircleMomentOfInertiaConversionFactors['mm⁴'][HollowCircleMomentOfInertiaUnits.indexOf(HollowCirclemomentOfInertiaSelectedUnit)];

      // Format the output based on the selected unit
      const formattedValue = HollowCirclemomentOfInertiaSelectedUnit === 'm⁴'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m⁴
        : convertedValue.toFixed(3); // 3 decimals for other units

      setHollowCircleMomentOfInertia(formattedValue);
    }
  }, [internalHollowCircleinputValue, internalHollowCircleHeightInputValue, HollowCirclemomentOfInertiaSelectedUnit]);




  useEffect(() => {
    const value = internalHollowCircleinputValue;
    if (!isNaN(value)) {
      const pi = Math.PI;
      const d = parseFloat(internalHollowCircleHeightInputValue);
      const b = parseFloat(internalHollowCircleinputValue);
      const Iy = ((pi * ((Math.pow(b, 4) - Math.pow(d, 4)))) / 4);
      const HollowCirclesectionModulesValue = (Iy / b).toFixed(2)

      // Convert HollowCirclesectionModulesValue based on the selected unit
      const convertedValue = HollowCirclesectionModulesValue * HollowCircleSectionModulesConversionFactors['mm³'][HollowCircleSectionModulesUnits.indexOf(HollowCirclesectionModulesSelectedUnit)];

      // Format the output based on the selected unit
      const formattedValue = HollowCirclesectionModulesSelectedUnit === 'm³'
        ? convertedValue.toExponential(2) // 2 decimal places in scientific notation for m³
        : convertedValue.toFixed(3); // 3 decimals for other units

      setHollowCircleSectionModules(formattedValue);
    }
  }, [internalHollowCircleinputValue, internalHollowCircleHeightInputValue, HollowCirclesectionModulesSelectedUnit]);
  const pi = Math.PI;

  const HollowCircleImperialunits = ['in'];
  const HollowCircleImperialconversionFactors = {
    in: [1]
  };

  const [HollowCircleImperialinputValue, setHollowCircleImperialInputValue] = useState(0);
  const [HollowCircleImperialselectedUnit, setHollowCircleImperialSelectedUnit] = useState('in');

  const handleHollowCircleImperialInputChange = (value) => {
    setHollowCircleImperialInputValue(value);
  };

  const handleHollowCircleImperialUnitChange = (unit) => {
    setHollowCircleImperialSelectedUnit(unit);
    const factor = HollowCircleImperialconversionFactors[unit][HollowCircleImperialunits.indexOf(HollowCircleImperialselectedUnit)];
    setHollowCircleImperialInputValue((parseFloat(HollowCircleImperialinputValue) / factor).toFixed(3));
  };

  const HollowCircleImperialHeightunits = ['in'];

  const HollowCircleImperialHeightConversionFactors = {
    in: [1]
  };

  const [HollowCircleImperialHeightInputValue, setHollowCircleImperialHeightInputValue] = useState(0);
  const [HollowCircleImperialHeightSelectedUnit, setHollowCircleImperialHeightSelectedUnit] = useState('in');

  const handleHollowCircleImperialHeightInputValue = (values) => {
    setHollowCircleImperialHeightInputValue(values);
  };

  const handleHollowCircleImperialHeightSelectedUnit = (units) => {
    setHollowCircleImperialHeightSelectedUnit(units);
    const Hrightfactor = HollowCircleImperialHeightConversionFactors[units][HollowCircleImperialHeightunits.indexOf(HollowCircleImperialHeightSelectedUnit)];
    setHollowCircleImperialHeightInputValue((parseFloat(HollowCircleImperialHeightInputValue) / Hrightfactor).toFixed(4));

  }


  const [HollowCircleImperialCentroid, setHollowCircleImperialCentroid] = useState(0);
  const [HollowCircleImperialCentroidSelectedunit, setHollowCircleImperialCentroidSelectedUnit] = useState('in');

  const handleHollowCircleImperialCentroidUnitChange = (unit) => {
    setHollowCircleImperialCentroidSelectedUnit(unit);
  };

  const calculateHollowCircleImperialCentroidInputChangeValue = () => {
    const HollowCircleImperialCentroid = HollowCircleImperialinputValue;
    setHollowCircleImperialCentroid(HollowCircleImperialCentroid);
  };

  useEffect(() => {
    calculateHollowCircleImperialCentroidInputChangeValue();
  }, [HollowCircleImperialinputValue]);


  const [HollowCircleImperialmomentOfInertia, setHollowCircleImperialmomentOfInertia] = useState(0);
  const [HollowCircleImperialmomentOfInertiaSelectedUnit, setHollowCircleImperialmomentOfInertiaSelectedUnit] = useState('in⁴');

  const HollowCircleImperialMomentOfInertiaUnits = ['in⁴'];
  const HollowCircleImperialMomentOfInertiaConversionFactors = {
    'in⁴': [1],
  };

  const handleHollowCircleImperialMomentOfInertiaUnitChange = (newUnit) => {
    const currentFactor = HollowCircleImperialMomentOfInertiaConversionFactors[HollowCircleImperialmomentOfInertiaSelectedUnit][HollowCircleImperialMomentOfInertiaUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(HollowCircleImperialmomentOfInertia) * currentFactor;
    setHollowCircleImperialmomentOfInertiaSelectedUnit(newUnit);
    setHollowCircleImperialmomentOfInertia(convertedValue.toString());
  };

  const calculateHollowCircleImperialMomentOfInertia = (HollowCircleImperialinputValue, HollowCircleImperialHeightInputValue) => {
    const parsedB = parseFloat(HollowCircleImperialinputValue);
    const parsedD = parseFloat(HollowCircleImperialHeightInputValue);
    if (!isNaN(parsedB) && !isNaN(parsedD)) {
      const HollowCircleImperialmomentOfInertiamultiplyValue = ((Math.pow(parsedB, 4) - Math.pow(parsedD, 4)));
      const divdeinertiaValue = 4;
      // const HollowCircleImperialmomentOfInertia = ;
      const HollowCircleImperialmomentOfInertia = ((pi * HollowCircleImperialmomentOfInertiamultiplyValue) / divdeinertiaValue).toFixed(1);
      const currentFactor = HollowCircleImperialMomentOfInertiaConversionFactors['in⁴'][HollowCircleImperialMomentOfInertiaUnits.indexOf(HollowCircleImperialmomentOfInertiaSelectedUnit)];
      setHollowCircleImperialmomentOfInertia((HollowCircleImperialmomentOfInertia * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateHollowCircleImperialMomentOfInertia(HollowCircleImperialinputValue, HollowCircleImperialHeightInputValue);
  }, [HollowCircleImperialinputValue, HollowCircleImperialHeightInputValue, HollowCircleImperialmomentOfInertiaSelectedUnit]);


  const ImperialHollowCircleSectionModulesUnits = ['in³'];
  const HollowCircleSectionModulesConversionUnitImperial = {
    'in³': [1],
  };
  const [ImperialHollowCircleSectionModules, setImperialHollowCircleSectionModules] = useState(0);
  const [ImperialHollowCircleSectionModulesSelectedUnit, setImperialHollowCircleSectionModulesSelectedUnit] = useState('in³');
  const handleImperialHollowCircleSectionModulesUnitChange = (units) => {
    setImperialHollowCircleSectionModulesSelectedUnit(units);
    const HollowCircleSectionModulesfactors = HollowCircleSectionModulesConversionUnitImperial[units][ImperialHollowCircleSectionModulesUnits.indexOf(ImperialHollowCircleSectionModulesSelectedUnit)];
    setImperialHollowCircleSectionModules((parseFloat(ImperialHollowCircleSectionModules) / HollowCircleSectionModulesfactors));
  };

  const calculateImperialHollowCircleSectionModules = () => {
    const d = parseFloat(HollowCircleImperialHeightInputValue);
    const b = parseFloat(HollowCircleImperialinputValue);
    const Inertia = parseFloat(HollowCircleImperialmomentOfInertia);
    if (!isNaN(d) && !isNaN(Inertia)) {
      const Iy = ((pi * ((Math.pow(b, 4) - Math.pow(d, 4)))) / 4);
      // const Xc = inputValue
      const ImperialHollowCircleSectionModules = (Iy / b).toFixed(2)
      setImperialHollowCircleSectionModules(ImperialHollowCircleSectionModules);
    }
  };

  useEffect(() => {
    calculateImperialHollowCircleSectionModules();
  }, [HollowCircleImperialmomentOfInertia, HollowCircleImperialHeightInputValue]);


  return (
    <>
      <Helmet>
        <title>OOK Calculator – Precision Engineering & AI Writing Suite</title>
        <meta
          name="description"
          content="The ultimate toolkit for professionals. Access high-precision beam property calculators alongside advanced AI detection, humanization, and plagiarism tools. Designed for structural engineers, students, and content creators needing accurate results with a premium experience."
        />
        <link rel="canonical" href="https://www.ook-calculator.com/" />
      </Helmet>


      <div className='Background-Black'></div>
      <section className='background-white'>
        <div className="position-relative">
          {/* Image section */}
          <div className="position-relative overflow-hidden" style={{ height: '85vh' }}>
            <picture>
              <source type="image/webp" srcSet={backgroundWebP} />
              <img
                src={backgroundWebP}
                alt="Professional Structural Engineering Workspace with OOK Calculators"
                className="h-100"
                fetchpriority="high"
                decoding="async"
                style={{
                  objectFit: 'cover',
                  objectPosition: 'center',
                  width: '123%',
                  transform: 'translateX(-9%)'
                }}
              />
            </picture>
            <div className="Overlay-Black-header"></div>
          </div>


          {/* Text section */}
          <div className="container text-left text-white position-absolute top-50 translate-middle BeamProperties"
            style={{
              left: '48%'
            }}
          >
            <h1 className="display-4" style={{ fontWeight: '600' }}>Beam Properties Calculator</h1>
            <p className="fs-5">
              Powerful tool for engineers, architects, and researchers to evaluate
              <br />
              the behaviour of different kinds of Cross Sections.
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
              OOK Beam Properties Calculator
            </h2>

            <div className={`content ${expanded ? "expanded" : ""}`}>
              <p className="first-important lead mb-4" style={{ fontWeight: "500" }}>
                Specialized tool designed for the analysis and design
                <br />
                of structural elements, such as beams, columns,
                <br />
                shafts, and members of varying cross-sections.
              </p>
              <p className="second-important lead mb-4" style={{ fontWeight: "500" }}>
                This calculator makes it easier to analyze and optimize section properties in detail for
                <br />
                structural integrity, efficiency, and safety in a variety of construction projects.
              </p>
              <p className=" lead" style={{ fontWeight: "500" }}>
                Easily calculate and visualize the geometric properties of various cross-section shapes,
                <br />
                including Area, Centroid, Moment of Inertia, Section Modulus, and
                <br />
                Torsional Constant that are essential for structural analysis and design.
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
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option1")}>Square</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option2")}>Rectangle</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option3")}>Hollow Rectangle/Square</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option4")}>T-Section</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option5")}>C-Section</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option6")}><span style={{ fontFamily: 'none' }}>I</span>-Beam</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option7")}>L-Section</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option8")}>Solid Circle</button>
                <button className="btn mb-2 text-center" onClick={() => handleOptionChange("option9")}>Hollow Circle or Pipe</button>
              </div>
            </div>

            {/* Center Section */}
            <div className="col-12 flex-grow-2  col-lg-5 col-md-12 col-sm-12 col-xs-12 text-center py-3 structure-analysis-calculator-calculator-center bg-white justify-content-center align-items-center d-flex" >
              {selectedOption === 'option1' && <img src={Square} alt="Square" className="img-fluid" style={{ height: 'auto', width: 'auto' }} />}
              {selectedOption === 'option2' && <img src={Rectangle} alt="Rectangle" className="img-fluid" style={{ height: 'auto', width: 'auto' }} />}
              {selectedOption === 'option3' && <img src={HollowRectangle} alt="Hollow Rectangle" className="img-fluid custom-img" style={{ height: 'auto', width: 'auto' }} />}
              {selectedOption === 'option4' && <img src={TSection} alt="Tee Section" className="img-fluid" style={{ height: 'auto', width: 'auto' }} />}
              {selectedOption === 'option5' && <img src={CChannel} alt="Channel Section" className="img-fluid custom-transform" style={{ height: 'auto', width: 'auto' }} />}
              {selectedOption === 'option6' && <img src={IBeam} alt="I-beam" className="img-fluid custom-img" style={{ height: 'auto', width: 'auto' }} />}
              {selectedOption === 'option7' && <img src={LSection} alt="L Section" className="img-fluid" style={{ height: 'auto', width: 'auto' }} />}
              {selectedOption === 'option8' && <img src={Circle} alt="Solid Circle" className="img-fluid custom-circle" style={{ height: '450px', width: '450px' }} />}
              {selectedOption === 'option9' && <img src={HollowCircle} alt="Hollow Circle" className="img-fluid" style={{ height: 'auto', width: 'auto' }} />}
            </div>

            {/* Right Section */}
            <div className="col-12 flex-grow-1  col-lg-3 col-md-12 col-sm-12 col-xs-12 text-center py-3 bemProperties structure-analysis-calculator-calculator-right BeamPropertiesInputs" >
              <h2 className="text-white mt-3">Input</h2>
              <div className="mt-3">
                {selectedOption === 'option1' && (
                  <>

                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>
                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Side (a) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={isNaN(SquaremetricInputValue) ? "" : SquaremetricInputValue}
                              onChange={(e) => handleMetricInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Square Side"
                            />
                            <select
                              className='Calculator-select-option'
                              value={SquaremetricSelectedUnit}
                              onChange={(e) => handleMetricUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label='Square side unit'
                            >
                              {SquareMetricUnits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}




                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Side (a) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={isNaN(SquareImperialinputValue) ? "" : SquareImperialinputValue}
                              onChange={(e) => SquarehandleImperialInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Square Side"
                            />
                            <select
                              className='Calculator-select-option'
                              value={isNaN(SquareImperialselectedUnit) ? "" : SquareImperialselectedUnit}
                              onChange={(e) => SquarehandleImperialUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label='Square side unit'
                            >
                              {SquareImperialunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}
                  </>
                )}
                {selectedOption === 'option2' && (
                  <>
                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>

                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={RectangleinputValue}
                              onChange={(e) => handleInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Rectangle Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={RectangleselectedUnit}
                              onChange={(e) => handleUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Rectangle Width Unit"
                            >
                              {Rectangleunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Height (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={RectangleHeightInputValue}
                              onChange={(e) => handleRectangleHeightInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="Rectangle Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={RectangleheightSelectedUnit}
                              onChange={(e) => handleRectangleHeightSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="Rectangle Height Unit"
                            >
                              {RectangleHeightunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <br />


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div rectangle'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}


                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={RectangleinputValueImperial}
                              onChange={(e) => handleInputChangeImperial(e.target.value)}
                              id="input" name="value"
                              aria-label="Rectangle Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={RectangleselectedUnitImperial}
                              onChange={(e) => handleUnitChangeImperial(e.target.value)}
                              id="select" name="unit"
                              aria-label="Rectangle Width Unit"
                            >
                              {RectangleunitsImperial.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Height (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={RectangleHeightInputValueImperial}
                              onChange={(e) => handleRectangleHeightInputValueImperial(e.target.value)}
                              id="input" name="value"
                              aria-label="Rectangle Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={RectangleheightSelectedUnitImperial}
                              onChange={(e) => handleRectangleHeightSelectedUnitImperial(e.target.value)}
                              id="select" name="unit"
                              aria-label="Rectangle Height Unit"
                            >
                              {RectangleHeightunitsImperial.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <br />


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div rectangle'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}
                  </>
                )}

                {selectedOption === 'option3' && (
                  <>
                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>
                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowReactangleinputValue}
                              onChange={(e) => handleHollowReactangleInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="HollowReactangle Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowReactangleselectedUnit}
                              onChange={(e) => handleHollowReactangleUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="HollowReactangle Width Unit"
                            >
                              {HollowReactangleUnits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Depth (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowReactangleHeightInputValue}
                              onChange={(e) => handleHollowReactangleHeightInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="HollowReactangle Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowReactangleheightSelectedUnit}
                              onChange={(e) => handleHollowReactangleHeightSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="HollowReactangle Height Unit"
                            >
                              {HollowReactangleHeightunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>
                            Inner width (b
                            <span className='LowerPower'>i</span>
                            ) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={inputHollowReactangleInnerValue}
                              onChange={(e) => handleInputHollowReactangleinnerChange(e.target.value)}
                              id="input" name="value"
                              aria-label="HollowReactangle Inner Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={selectedHollowReactangleInnerUnit}
                              onChange={(e) => handleUnitHollowReactangleinnerChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="HollowReactangle Inner Width Unit"
                            >
                              {HollowReactangleInnerunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>
                            Inner depth (d
                            <span className='LowerPower'>i</span>
                            ) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowReactangleHeightInnerInputValue}
                              onChange={(e) => handleHollowReactangleHeightinnerInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="HollowReactangle inner Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowReactangleheightinnerSelectedUnit}
                              onChange={(e) => handleHollowReactangleHeightinnerSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="HollowReactangle inner Height Unit"
                            >
                              {HollowReactangleHeightInnerunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div HollowReactangle'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>

                      </>
                    )}
                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowReactangleinputValueImperial}
                              onChange={(e) => handleHollowReactangleInputChangeImperial(e.target.value)}
                              id="input" name="value"
                              aria-label="HollowReactangle Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowReactangleselectedUnitImperial}
                              onChange={(e) => handleHollowReactangleUnitChangeImperial(e.target.value)}
                              id="select" name="unit"
                              aria-label="HollowReactangle Width Unit"
                            >
                              {HollowReactangleImperialunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Depth (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowReactangleHeightInputValueImperial}
                              onChange={(e) => handleHollowReactangleHeightInputChangeImperial(e.target.value)}
                              id="input" name="value"
                              aria-label="HollowReactangle Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowReactangleheightSelectedUnitImperial}
                              onChange={(e) => handleHollowReactangleHeightUnitChangeImperial(e.target.value)}
                              id="select" name="unit"
                              aria-label="HollowReactangle Height Unit"
                            >
                              {HollowReactangleHeightUnitsImperial.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>
                            Inner width (b
                            <span className='LowerPower'>i</span>
                            ) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={inputHollowReactangleInnerValueImperial}
                              onChange={(e) => handleInputHollowReactangleInnerChangeImperial(e.target.value)}
                              id="input" name="value"
                              aria-label="HollowReactangle inner Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={selectedHollowReactangleInnerUnitImperial}
                              onChange={(e) => handleUnitHollowReactangleInnerChangeImperial(e.target.value)}
                              id="select" name="unit"
                              aria-label="HollowReactangle Inner Width Unit"
                            >
                              {HollowReactangleInnerUnitsImperial.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>
                            Inner depth (d
                            <span className='LowerPower'>i</span>
                            ) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowReactangleHeightInnerInputValueImperial}
                              onChange={(e) => handleHollowReactangleHeightInnerInputChangeImperial(e.target.value)}
                              id="input" name="value"
                              aria-label="HollowReactangle inner Height"

                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowReactangleheightInnerSelectedUnitImperial}
                              onChange={(e) => handleHollowReactangleHeightInnerUnitChangeImperial(e.target.value)}
                              id="select" name="unit"
                              aria-label="HollowReactangle inner Height Unit"
                            >
                              {HollowReactangleHeightInnerUnitsImperial.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div HollowReactangle'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}
                  </>
                )}
                {selectedOption === 'option4' && (
                  <>
                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>
                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b):</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={TeeSectioninputValue}
                              onChange={(e) => handleTeeSectionInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="T Section Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={TeeSectionselectedUnit}
                              onChange={(e) => handleTeeSectionUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="T Section Width Unit"
                            >
                              {TeeSectionunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Depth (d):</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={TeeSectionHeightInputValue}
                              onChange={(e) => handleTeeSectionHeightInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="T Section Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={TeeSectionheightSelectedUnit}
                              onChange={(e) => handleTeeSectionHeightSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="T Section Height Unit"
                            >
                              {TeeSectionHeightunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Flange Thickness (t):</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={inputTeeSectionInnerValue}
                              onChange={(e) => handleInputTeeSectioninnerChange(e.target.value)}
                              id="input" name="value"
                              aria-label="T Section Flange Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={selectedTeeSectionInnerUnit}
                              onChange={(e) => handleUnitTeeSectioninnerChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="T Section Flange Thickness Unit"
                            >
                              {TeeSectionInnerunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Web Thickness (t
                            <span className='LowerPowerminus2px'>w</span>
                            ):</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={TeeSectionHeightInnerInputValue}
                              onChange={(e) => handleTeeSectionHeightinnerInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="T Section Web thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={TeeSectionheightinnerSelectedUnit}
                              onChange={(e) => handleTeeSectionHeightinnerSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="T Section Web thickness Unit"
                            >
                              {TeeSectionHeightInnerunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div TeeSection'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}
                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b):</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={TeeSectionImperialinputValue}
                              onChange={(e) => handleTeeSectionImperialInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="T Section Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={TeeSectionImperialselectedUnit}
                              onChange={(e) => handleTeeSectionImperialUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="T Section Width Unit"
                            >
                              {TeeSectionImperialunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Depth (d):</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={TeeSectionImperialHeightInputValue}
                              onChange={(e) => handleTeeSectionImperialHeightInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="T Section Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={TeeSectionImperialheightSelectedUnit}
                              onChange={(e) => handleTeeSectionImperialHeightSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="T Section Height Unit"
                            >
                              {TeeSectionImperialHeightunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Flange Thickness (t):</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ImperialinputTeeSectionInnerValue}
                              onChange={(e) => handleImperialInputTeeSectioninnerChange(e.target.value)}
                              id="input" name="value"
                              aria-label="T Section Flange Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ImperialselectedTeeSectionInnerUnit}
                              onChange={(e) => handleImperialUnitTeeSectioninnerChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="T Section Flange Thickness Unit"
                            >
                              {ImperialTeeSectionInnerunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Web Thickness (t
                            <span className='LowerPowerminus2px'>w</span>
                            ):</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ImperialTeeSectionHeightInnerInputValue}
                              onChange={(e) => handleImperialTeeSectionHeightinnerInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="T Section Web thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ImperialTeeSectionheightinnerSelectedUnit}
                              onChange={(e) => handleImperialTeeSectionHeightinnerSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="T Section Web thickness Unit"
                            >
                              {ImperialTeeSectionHeightInnerunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div TeeSection'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}


                  </>
                )}
                {selectedOption === 'option5' && (
                  <>

                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>

                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Height (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ChannelbinputValue}
                              onChange={(e) => handleChannelbInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Channel Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ChannelbselectedUnit}
                              onChange={(e) => handleChannelbUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Channel Height Unit"
                            >
                              {Channelbunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Flange (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ChanneldinputValue}
                              onChange={(e) => handledChannelInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Channel Flange"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ChannelselectedUnit}
                              onChange={(e) => handledChannelUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Channel Flange Unit"
                            >
                              {Channeldunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Web Thickness (tw) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ChannelTValue}
                              onChange={(e) => handleChannelTValueInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Channel Web Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ChannelselectedTUnit}
                              onChange={(e) => handleChannelTValueUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Channel Web Thickness Unit"
                            >
                              {ChannelTValueunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Flange Thickness (t) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ChanneltwinputValue}
                              onChange={(e) => handleChanneltwInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Channel Flange Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ChannelTwselectedUnit}
                              onChange={(e) => handleChanneltwUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Channel Flange Thickness Unit"
                            >
                              {Channeltwunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div Channel'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>)}

                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Height (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ImperialChannelbinputValue}
                              onChange={(e) => handleImperialChannelbInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Channel Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ImperialChannelbselectedUnit}
                              onChange={(e) => handleImperialChannelbUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Channel Height Unit"
                            >
                              {ImperialChannelbunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Flange (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ChannelImperialdinputValue}
                              onChange={(e) => handleChannelImperialdInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Channel Flange"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ChannelImperialdselectedUnit}
                              onChange={(e) => handleChannelImperialdUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Channel Flange Unit"
                            >
                              {ChannelImperialdunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Web Thickness (tw) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ChannelImperialTValue}
                              onChange={(e) => handleChannelImperialTValueInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Channel Web Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ChannelImperialTVselectedTUnit}
                              onChange={(e) => handleChannelImperialTValueUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Channel Web Thickness Unit"
                            >
                              {ChannelImperialTValueunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Flange Thickness (t) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={ChannelImperialtwinputValue}
                              onChange={(e) => handleChannelImperialtwInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Channel Flange Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={ChannelImperialTwselectedUnit}
                              onChange={(e) => handleChannelImperialtwUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Channel Flange Thickness Unit"
                            >
                              {ChannelImperialtwunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div Channel'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}
                  </>
                )}
                {selectedOption === 'option6' && (
                  <>
                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>
                    {/* {result} */}
                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectionbinputValue}
                              onChange={(e) => handlebInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionbselectedUnit}
                              onChange={(e) => handlebUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Width Unit"
                            >
                              {Isectionbunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Web Height (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectiondinputValue}
                              onChange={(e) => handledInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionselectedUnit}
                              onChange={(e) => handledUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Height Unit"
                            >
                              {Isectiondunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Flange Thickness (t) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectionTValue}
                              onChange={(e) => handleIsectionTValueInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Flange Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionselectedIsectionTVUnit}
                              onChange={(e) => handleIsectionTValueUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Flange Thickness Unit"
                            >
                              {IsectionTValueunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Web Thickness (t<span className='LowerPowerminus2px torsinalConstantFormula'>w</span>) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectiontwinputValue}
                              onChange={(e) => handleIsectiontwInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Web Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionTwselectedUnit}
                              onChange={(e) => handleIsectiontwUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Web Thickness Unit"
                            >
                              {Isectiontwunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Radius (r) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectionRinputValue}
                              onChange={(e) => handleIsectionRInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Radius"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionRselectedUnit}
                              onChange={(e) => handleIsectionRUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Radius Unit"
                            >
                              {IsectionRunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div ISection'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}


                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectionImperialbinputValue}
                              onChange={(e) => handleIsectionImperialbInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionImperialbselectedUnit}
                              onChange={(e) => handleIsectionImperialbUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Width Unit"
                            >
                              {IsectionImperialbunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Web Height (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectionImperialdinputValue}
                              onChange={(e) => handleIsectionImperialdInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionImperialdselectedUnit}
                              onChange={(e) => handleIsectionImperialdUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Height Unit"
                            >

                              {IsectionImperialdunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Flange Thickness (t) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectionImperialt}
                              onChange={(e) => handleIsectionImperialtInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Flange Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionImperialtselectedTUnit}
                              onChange={(e) => handleIsectionImperialtUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Flange Thickness Unit"
                            >
                              {IsectionImperialtunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Web Thickness (t<span className='LowerPowerminus2px torsinalConstantFormula'>w</span>) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectionImperialtwinputValue}
                              onChange={(e) => handleIsectionImperialtwInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Web Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionImperialTWSelectedValue}
                              onChange={(e) => handleIsectionImperialtwUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Web Thickness Unit"
                            >
                              {IsectionImperialtwunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Radius (r) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={IsectionImperialRinputValue}
                              onChange={(e) => handleIsectionImperialRInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Radius"
                            />
                            <select
                              className='Calculator-select-option'
                              value={IsectionImperialRSelectedValue}
                              onChange={(e) => handleIsectionImperialRUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Radius Unit"
                            >
                              {IsectionImperialRunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div ISection'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}
                  </>
                )}
                {selectedOption === 'option7' && (
                  <>
                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>
                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={LsectionbinputValue}
                              onChange={(e) => handlebLsectionInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={LsectionbselectedUnit}
                              onChange={(e) => handlebLsectionUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Width Unit"
                            >
                              {Lsectionbunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Height (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={LsectiondinputValue}
                              onChange={(e) => handledLsectionInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={selectedUnit}
                              onChange={(e) => handledLsectionUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Height Unit"
                            >
                              {Lsectiondunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Thickness (t) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={LsectionTValue}
                              onChange={(e) => handleLsectionTValueInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Flange Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={selectedLsectionTVUnit}
                              onChange={(e) => handleLsectionTValueUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Flange Thickness Unit"
                            >
                              {LsectionTValueunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div LSection'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>

                      </>
                    )}
                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Width (b) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={LsectionImperialbinputValue}
                              onChange={(e) => handleLsectionImperialbInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Width"
                            />
                            <select
                              className='Calculator-select-option'
                              value={LsectionImperialbselectedUnit}
                              onChange={(e) => handleLsectionImperialbUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Width Unit"
                            >
                              {LsectionImperialbunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Height (d) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={LsectionImperialdinputValue}
                              onChange={(e) => handledLsectionImperialInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Height"
                            />
                            <select
                              className='Calculator-select-option'
                              value={LsectionImperialdselectedUnit}
                              onChange={(e) => handledLsectionImperialUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Height Unit"
                            >
                              {LsectionImperialdunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Thickness (t) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={LsectionImperialTValue}
                              onChange={(e) => handleLsectionImperialTValueInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="I Section Flange Thickness"
                            />
                            <select
                              className='Calculator-select-option'
                              value={LsectionImperialTVselectedTUnit}
                              onChange={(e) => handleLsectionImperialTValueUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="I Section Flange Thickness Unit"
                            >
                              {LsectionImperialTValueunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div LSection'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>
                    )}
                  </>
                )}
                {selectedOption === 'option8' && (
                  <>
                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>

                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Radius (R) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={SolidCircleinputValue}
                              onChange={(e) => handleSolidCircleInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Solid Circle Radius"
                            />
                            <select
                              className='Calculator-select-option'
                              value={SolidCircleselectedUnit}
                              onChange={(e) => handleSolidCircleUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Solid Circle Radius Unit"
                            >
                              {SolidCircleunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div  circle'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>)}

                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Radius (R) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={SolidCircleImperialinputValue}
                              onChange={(e) => handleSolidCircleImperialInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Solid Circle Radius"
                            />
                            <select
                              className='Calculator-select-option'
                              value={SolidCircleImperialselectedUnit}
                              onChange={(e) => handleSolidCircleImperialUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Solid Circle Radius Unit"
                            >
                              {SolidCircleImperialunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div circle'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>)}
                  </>
                )}
                {selectedOption === 'option9' && (
                  <>
                    <div style={{
                      width: '100%',
                      display: 'flex',
                      justifyContent: 'space-evenly',
                      padding: '1vw 0'
                    }}>
                      <button className='metricBtn'
                        onClick={() => toggleMetricOrImperial("option1")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Metric
                      </button>
                      <button className='ImperialBtn'
                        onClick={() => toggleMetricOrImperial("option2")}
                        style={{
                          padding: '0.2vw 1vw',
                          fontSize: '1.2vw',
                          borderRadius: '10px',
                          backgroundColor: '#fff'
                        }}
                      >
                        Imperial
                      </button>
                    </div>
                    {MetricOrImperial === 'option1' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Radius (R) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowCircleinputValue}
                              onChange={(e) => handleHollowCircleInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Hollow Circle Radius"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowCircleselectedUnit}
                              onChange={(e) => handleHollowCircleUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Hollow Circle Radius Unit"
                            >
                              {HollowCircleunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Inner radius (Ri) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowCircleHeightInputValue}
                              onChange={(e) => handleHollowCircleHeightInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="Hollow Circle Inner Radius"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowCircleheightSelectedUnit}
                              onChange={(e) => handleHollowCircleHeightSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="Hollow Circle Inner Radius Unit"
                            >
                              {HollowCircleHeightunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div Hollow-circle'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>)}

                    {MetricOrImperial === 'option2' && (
                      <>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Radius (R) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowCircleImperialinputValue}
                              onChange={(e) => handleHollowCircleImperialInputChange(e.target.value)}
                              id="input" name="value"
                              aria-label="Hollow Circle Radius"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowCircleImperialselectedUnit}
                              onChange={(e) => handleHollowCircleImperialUnitChange(e.target.value)}
                              id="select" name="unit"
                              aria-label="Hollow Circle Radius Unit"
                            >
                              {HollowCircleImperialunits.map((unit) => (
                                <option key={unit} value={unit}>
                                  {unit}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <p className='claculator-conversation-title'>Inner radius (Ri) :</p>
                          <div className='Calculator-Side-A'>
                            <input
                              className='calculator-input'
                              type="number"
                              value={HollowCircleImperialHeightInputValue}
                              onChange={(e) => handleHollowCircleImperialHeightInputValue(e.target.value)}
                              id="input" name="value"
                              aria-label="Hollow Circle Inner Radius"
                            />
                            <select
                              className='Calculator-select-option'
                              value={HollowCircleImperialHeightSelectedUnit}
                              onChange={(e) => handleHollowCircleImperialHeightSelectedUnit(e.target.value)}
                              id="select" name="unit"
                              aria-label="Hollow Circle Inner Radius Unit"
                            >
                              {HollowCircleImperialHeightunits.map((units) => (
                                <option key={units} value={units}>
                                  {units}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>


                        <div className='structure-analysis-calculator-calculator-right-show-hidden-btn-div Hollow-circle'>
                          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={toggleClass3}>{isActive3 ? ' Hide ' : ' Solve '}</button>
                        </div>
                      </>)}
                  </>
                )}
              </div>
            </div>
          </div>
        </section>
        {selectedOption === 'option1' && (
          <>

            {MetricOrImperial === 'option1' && (
              <>
                <br />
                <br />
                <div className={isActive3 ? 'show  Sectionmodules  ' : 'hidden  Sectionmodules  '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(Squarearea) ? "" : Squarearea}
                            readOnly
                            id="Squarearea" name="value"
                            aria-label="Square Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={SquareareaUnit}
                            onChange={(e) => setSquareAreaUnit(e.target.value)}
                            id="SquareareaUnit" name="unit"
                            aria-label="Square Area Unit"
                          >
                            {SquareAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title text-white'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{}}>c</span>

                        <span className='equalesto'>=</span>

                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>

                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(Squarecentroid) ? "" : Squarecentroid}
                            readOnly
                            id="Squarecentroid" name="value"
                            aria-label="Square Centroid"
                          />
                          <select
                            className='Calculator-select-option'
                            value={SquarecentroidUnit}
                            onChange={(e) => setSquareCentroidUnit(e.target.value)}
                            id="SquarecentroidUnit" name="unit"
                            aria-label="Square Centroid Unit"
                          >
                            {SquareSquareCentroidUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of inertia:</p>


                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title text-white'>
                        I
                        <span className='LowerPower sectionmodulesSolutionLowerPower'>x</span>
                        <span className='equalesto'>=</span>
                        I
                        <span className='LowerPower sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(SquaremomentOfInertia) ? "" : SquaremomentOfInertia}
                            id="SquaremomentOfInertia" name="value"
                            readOnly
                            aria-label="Square Moment of Inertia"
                          />
                          <select
                            className='Calculator-select-option'
                            value={SquaremomentOfInertiaUnit}
                            onChange={(e) => setSquareMomentOfInertiaUnit(e.target.value)}
                            id="SquaremomentOfInertiaUnit" name="unit"
                            aria-label="Square Moment of Inertia Unit"
                          >
                            {SquareSquareMomentOfInertiaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modules:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title text-white'>
                        S
                        <span className='LowerPower sectionmodulesSolutionLowerPower'>x</span>
                        <span className='equalesto'>=</span>
                        S
                        <span className='LowerPower sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(SquaresectionModules) ? "" : SquaresectionModules}
                            readOnly
                            id="SquaresectionModules" name="value"
                            aria-label="Square Section Modules"
                          />
                          <select
                            className='Calculator-select-option'
                            value={SquaresectionModulesUnit}
                            onChange={(e) => setSquareSectionModulesUnit(e.target.value)}
                            id="SquaresectionModulesUnit" name="unit"
                            aria-label="Square Section Modules Unit"
                          >
                            {SquareSectionModulesUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title text-white'>
                        K :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(SquaretorsionalConstant) ? "" : SquaretorsionalConstant}
                            readOnly
                            id="SquaretorsionalConstant" name="value"
                            aria-label="Square Torsional Constant"
                          />
                          <select
                            className='Calculator-select-option'
                            value={SquaretorsionalConstantUnit}
                            onChange={(e) => setSquareTorsionalConstantUnit(e.target.value)}
                            id="SquaretorsionalConstantUnit" name="unit"
                            aria-label="Square Torsional Constant Unit"
                          >
                            {SquareTorsionalConstantUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </div>
                </div >

              </>
            )}
            {MetricOrImperial === 'option2' && (
              <>
                <br />
                <br />
                <div className={isActive3 ? 'show  Sectionmodules  ' : 'hidden  Sectionmodules  '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>
                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(SquareImperialArea) ? "" : SquareImperialArea}
                            onChange={(e) => calculateSquareImperialAreaInputChangeValue(e.target.value)}
                            aria-label="Square Imperial Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={isNaN(SquareImperialAreaUnit) ? "" : SquareImperialAreaUnit}
                            onChange={(e) => handleSquareImperialAreaUnitChange(e.target.value)}
                            id="SquareImperialAreaUnit" name="unit"
                            aria-label="Square Imperial Area Unit"
                          >
                            {ImperialSquareAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title text-white'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower text-white' style={{}}>c</span>

                        <span className='equalesto text-white'>=</span>

                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower text-white' style={{ left: '-3px' }}>c </span>

                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(SquareImperialSquareCentroid) ? "" : SquareImperialSquareCentroid}
                            onChange={(e) => calculateSquareImperialSquareCentroidInputChangeValue(e.target.value)}
                            aria-label="Square Imperial Centroid"
                          />
                          <select
                            className='Calculator-select-option'
                            value={isNaN(SquareImperialSquareCentroidSelectedunit) ? "" : SquareImperialSquareCentroidSelectedunit}
                            onChange={(e) => handleSquareImperialSquareCentroidUnitChange(e.target.value)}
                            id="SquareImperialSquareCentroidUnit" name="unit"
                            aria-label="Square Imperial Centroid Unit"
                          >
                            {SquareImperialunits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of inertia:</p>


                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title text-white'>
                        I
                        <span className='LowerPower sectionmodulesSolutionLowerPower'>x</span>
                        <span className='equalesto'>=</span>
                        I
                        <span className='LowerPower sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(SquareImperialSquaremomentOfInertia) ? "" : SquareImperialSquaremomentOfInertia}
                            onChange={(e) => calculateSquareImperialSquareMomentOfInertia(e.target.value)}
                            aria-label="Square Imperial Moment of Inertia"
                          />
                          <select
                            className='Calculator-select-option'
                            value={isNaN(SquareImperialSquaremomentOfInertiaSelectedUnit) ? "" : SquareImperialSquaremomentOfInertiaSelectedUnit}
                            onChange={(e) => handleSquareImperialSquareMomentOfInertiaUnitChange(e.target.value)}
                            id="SquareImperialSquaremomentOfInertiaUnit" name="unit"
                            aria-label="Square Imperial Moment of Inertia Unit"
                          >
                            {ImperialSquareSquareMomentOfInertiaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modules:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title text-white'>
                        S
                        <span className='LowerPower sectionmodulesSolutionLowerPower'>x</span>
                        <span className='equalesto'>=</span>
                        S
                        <span className='LowerPower sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(SquareImperialSectionModules) ? "" : SquareImperialSectionModules}
                            onChange={(e) => calculateSquareImperialSectionModules(e.target.value)}
                            aria-label="Square Imperial Section Modules"
                          />
                          <select
                            className='Calculator-select-option'
                            value={isNaN(SquareImperialSectionModulesSelectedUnit) ? "" : SquareImperialSectionModulesSelectedUnit}
                            onChange={(e) => handleSquareImperialSectionModulesUnitChange(e.target.value)}
                            id="SquareImperialSectionModulesUnit" name="unit"
                            aria-label="Square Imperial Section Modules Unit"
                          >
                            {ImperialSquareSectionModulesUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title text-white'>
                        K :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={isNaN(SquareImperialTorsionalConstant) ? "" : SquareImperialTorsionalConstant}
                            onChange={(e) => calculateSquareImperialTorsionalConstant(e.target.value)}
                            aria-label="Square Imperial Torsional Constant"
                          />
                          <select
                            className='Calculator-select-option'
                            value={isNaN(SquareImperialTorsionalConstantSelectedUnit) ? "" : SquareImperialTorsionalConstantSelectedUnit}
                            onChange={(e) => handleSquareImperialTorsionalConstantUnitChange(e.target.value)}
                            id="SquareImperialTorsionalConstantUnit" name="unit"
                            aria-label="Square Imperial Torsional Constant Unit"
                          >
                            {ImperialSquareTorsionalConstantUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </div>
                </div >
              </>
            )}
          </>
        )}

        {selectedOption === 'option2' && (

          <>
            {MetricOrImperial === 'option1' && (
              <>
                <br />
                <br />
                <div className={isActive3 ? 'show Sectionmodules  ' : 'hidden Sectionmodules  '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw', width: '100%' }}>
                        Area:
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={RectangleArea}
                            readOnly
                            aria-label="Rectangle Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectangleAreaUnit}
                            onChange={(e) => setRectangleAreaUnit(e.target.value)}
                            id="RectangleAreaUnit" name="unit"
                            aria-label="Rectangle Area Unit"
                          >
                            {RectangleAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={RectangleCentroidXc}
                            readOnly
                            aria-label="Rectangle Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectangleCentroidXcSelectedUnit}
                            onChange={(e) => setRectangleCentroidXcSelectedUnit(e.target.value)}
                            id="RectangleCentroidXcUnit" name="unit"
                            aria-label="Rectangle Centroid Xc Unit"
                          >
                            {RectangleCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={RectangleCentriodYc}
                            readOnly
                            aria-label="Rectangle Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectangleCentriodYcSelectedUnit}
                            onChange={(e) => setRectangleCentriodYcSelectedUnit(e.target.value)}
                            id="RectangleCentriodYcUnit" name="unit"
                            aria-label="Rectangle Centroid Yc Unit"
                          >
                            {RectangleCentriodYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={RectanglemomentOfInertiaIx}
                            readOnly
                            aria-label="Rectangle Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectanglemomentOfInertiaIxSelectedUnit}
                            onChange={(e) => setRectangleMomentOfInertiaIxSelectedUnit(e.target.value)}
                            id="RectangleMomentOfInertiaIxUnit" name="unit"
                            aria-label="Rectangle Moment of Inertia Ix Unit"
                          >
                            {RectangleMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={RectanglemomentOfInertiaIy}
                            readOnly
                            aria-label="Rectangle Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectanglemomentOfInertiaIySelectedUnit}
                            onChange={(e) => setRectangleMomentOfInertiaIySelectedUnit(e.target.value)}
                            id="RectangleMomentOfInertiaIyUnit" name="unit"
                            aria-label="Rectangle Moment of Inertia Iy Unit"
                          >
                            {RectangleMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(RectangleSectionModulesSx)}
                            readOnly
                            aria-label="Rectangle Section Modules Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectangleSectionModulesSxSelectedUnit}
                            onChange={(e) => setRectangleSectionModulesSxSelectedUnit(e.target.value)}
                            id="RectangleSectionModulesSxUnit" name="unit"
                            aria-label="Rectangle Section Modules Sx Unit"
                          >
                            {RectangleSectionModulesSxUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px RectanglesectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(RectangleSectionModulesSy)}
                            readOnly
                            aria-label="Rectangle Section Modules Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectangleSectionModulesSelectedUnitSy}
                            onChange={(e) => setRectangleSectionModulesSelectedUnitSy(e.target.value)}
                            id="RectangleSectionModulesSyUnit" name="unit"
                            aria-label="Rectangle Section Modules Sy Unit"
                          >
                            {RectangleSectionModulesSyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>


                    </div>

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        K :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={RectangletorsionalConstant}
                            readOnly
                            aria-label="Rectangle Torsional Constant"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectangletorsionalConstantSelectedUnit}
                            onChange={(e) => setRectangleTorsionalConstantSelectedUnit(e.target.value)}
                            id="RectangleTorsionalConstantUnit" name="unit"
                            aria-label="Rectangle Torsional Constant Unit"
                          >
                            {RectangleTorsionalConstantUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </div>
                </div >
              </>
            )}
            {MetricOrImperial === 'option2' && (
              <>
                <br />
                <br />
                <div className={isActive3 ? 'show Sectionmodules  ' : 'hidden Sectionmodules  '} style={{ height: '45vw' }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='' style={{ borderRadius: '10px', width: '90%', margin: 'auto' }}>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={RectangleAreaImperial}
                            readOnly
                            aria-label="Rectangle Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={RectangleAreaUnitImperial}
                            onChange={(e) => handleAreaUnitChangeImperial(e.target.value)}
                            id="RectangleAreaUnitImperial" name="unit"
                            aria-label="Rectangle Area Unit Imperial"
                          >
                            {RectangleAreaUnitsImperial.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ReactangleCentroidXcImperial}
                            readOnly
                            aria-label="Rectangle Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ReactangleCentroidXcSelectedUnitImperial}
                            onChange={(e) => handleReactangleCentroidXcUnitChangeImperial(e.target.value)}
                            id="RectangleCentroidXcUnitImperial" name="unit"
                            aria-label="Rectangle Centroid Xc Unit Imperial"
                          >
                            {ReactangleCentroidXcUnitsImperial.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ReactangleCentriodYcImperial}
                            readOnly
                            aria-label="Rectangle Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ReactangleCentriodYcSelectedUnitImperial}
                            onChange={(e) => handleReactangleCentriodYcUnitChangeImperial(e.target.value)}
                            id="RectangleCentriodYcUnitImperial" name="unit"
                            aria-label="Rectangle Centroid Yc Unit Imperial"
                          >
                            {ReactangleCentriodYcUnitsImperial.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ReactanglemomentOfInertiaIxImperial}
                            readOnly
                            aria-label="Rectangle Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ReactanglemomentOfInertiaIxSelectedUnitImperial}
                            onChange={(e) => handleReactangleMomentOfInertiaIxUnitChangeImperial(e.target.value)}
                            id="RectangleMomentOfInertiaIxUnitImperial" name="unit"
                            aria-label="Rectangle Moment of Inertia Ix Unit Imperial"
                          >
                            {ReactangleMomentOfInertiaIxUnitsImperial.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ReactanglemomentOfInertiaIyImperial}
                            readOnly
                            aria-label="Rectangle Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ReactanglemomentOfInertiaIySelectedUnitImperial}
                            onChange={(e) => handleReactangleMomentOfInertiaIyUnitChangeImperial(e.target.value)}
                            aria-label="Rectangle Moment of Inertia Iy Unit Imperial"
                          >
                            {ReactangleMomentOfInertiaIyUnitsImperial.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ReactangleSectionModulesSxImperial}
                            readOnly
                            aria-label="Rectangle Section Modules Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ReactangleSectionModulesSxSelectedUnitImperial}
                            onChange={(e) => handleReactangleSectionModulesSxUnitChangeImperial(e.target.value)}
                            id="RectangleSectionModulesSxUnitImperial" name="unit"
                            aria-label="Rectangle Section Modules Sx Unit Imperial"
                          >
                            {ReactangleSectionModulesSxUnitsImperial.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ReactangleSectionModulesSyImperial}
                            readOnly
                            aria-label="Rectangle Section Modules Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ReactangleSectionModulesSySelectedUnitImperial}
                            onChange={(e) => handleReactangleSectionModulesSyUnitChangeImperial(e.target.value)}
                            id="RectangleSectionModulesSyUnitImperial" name="unit"
                            aria-label="Rectangle Section Modules Sy Unit Imperial"
                          >
                            {ReactangleSectionModulesSyUnitsImperial.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        K :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ReactangletorsionalConstantImperial}
                            onChange={(e) => calculateReactangleTorsionalConstantImperial(e.target.value)}
                            aria-label="Rectangle Torsional Constant"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ReactangletorsionalConstantSelectedUnitImperial}
                            onChange={(e) => handleReactangleTorsionalConstantUnitChangeImperial(e.target.value)}
                            id="RectangleTorsionalConstantUnitImperial" name="unit"
                            aria-label="Rectangle Torsional Constant Unit Imperial"
                          >
                            {ReactangleTorsionalConstantUnitsImperial.map((unit) => (
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

              </>
            )}
          </>
        )}

        {selectedOption === 'option3' && (
          <>
            {MetricOrImperial === 'option1' && (
              <>
                <div className={isActive3 ? 'show  Sectionmodules  ' : 'hidden  Sectionmodules  '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactangleArea}
                            readOnly
                            aria-label="Hollow Rectangle Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactangleAreaUnit}
                            onChange={(e) => setHollowReactangleAreaUnit(e.target.value)}
                            id="HollowRectangleAreaUnit" name="unit"
                            aria-label="Hollow Rectangle Area Unit"
                          >
                            {HollowReactangleAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactanglecentroidXc}
                            readOnly
                            aria-label="Hollow Rectangle Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglecentroidXcSelectedUnit}
                            onChange={(e) => setHollowReactangleCentroidXcSelectedUnit(e.target.value)}
                            id="HollowRectangleCentroidXcUnit" name="unit"
                            aria-label="Hollow Rectangle Centroid Xc Unit"
                          >
                            {HollowReactangleCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactanglecentroidYc}
                            readOnly
                            aria-label="Hollow Rectangle Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglecentroidYcSelectedUnit}
                            onChange={(e) => setHollowReactangleCentroidYcSelectedUnit(e.target.value)}
                            id="HollowRectangleCentroidYcUnit" name="unit"
                            aria-label="Hollow Rectangle Centroid Yc Unit"
                          >
                            {HollowReactangleCentroidYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactanglemomentOfInertiaIx}
                            readOnly
                            aria-label="Hollow Rectangle Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglemomentOfInertiaIxSelectedUnit}
                            onChange={(e) => setHollowReactangleMomentOfInertiaIxSelectedUnit(e.target.value)}
                            id="HollowRectangleMomentOfInertiaIxUnit" name="unit"
                            aria-label="Hollow Rectangle Moment of Inertia Ix Unit"
                          >
                            {HollowReactangleMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactanglemomentOfInertiaIy}
                            readOnly
                            aria-label="Hollow Rectangle Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglemomentOfInertiaIySelectedUnit}
                            onChange={(e) => setHollowReactangleMomentOfInertiaIySelectedUnit(e.target.value)}
                            id="HollowRectangleMomentOfInertiaIyUnit" name="unit"
                            aria-label="Hollow Rectangle Moment of Inertia Iy Unit"
                          >
                            {HollowReactangleMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(HollowReactanglesectionModulusSx)}
                            readOnly
                            aria-label="Hollow Rectangle Section Modules Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglesectionModulusSxSelectedUnit}
                            onChange={(e) => setHollowReactangleSectionModulusSxSelectedUnit(e.target.value)}
                            id="HollowRectangleSectionModulesSxUnit" name="unit"
                            aria-label="Hollow Rectangle Section Modules Sx Unit"
                          >
                            {HollowReactangleSectionModulusSxUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(HollowReactangleSectionModulusSy)}
                            readOnly
                            aria-label="Hollow Rectangle Section Modules Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactangleSectionModulusSySelectedUnit}
                            onChange={(e) => setHollowReactangleSectionModulusSySelectedUnit(e.target.value)}
                            id="HollowRectangleSectionModulesSyUnit" name="unit"
                            aria-label="Hollow Rectangle Section Modules Sy Unit"
                          >
                            {HollowReactangleSectionModulusSyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>


                    </div>
                  </div>
                </div >
              </>
            )}


            {MetricOrImperial === 'option2' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules' : 'hidden Sectionmodules'} style={{ height: '45vw' }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='' style={{ borderRadius: '10px', width: '90%', margin: 'auto' }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactangleareaImperial}
                            onChange={(e) => calculateHollowReactangleAreaImperial(e.target.value)}
                            aria-label="Hollow Rectangle Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactangleareaUnitImperial}
                            onChange={(e) => handleHollowReactangleAreaUnitChangeImperial(e.target.value)}
                            id="HollowRectangleAreaUnitImperial" name="unit"
                            aria-label="Hollow Rectangle Area Unit Imperial"
                          >
                            {HollowReactangleAreaUnitsImperial.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactanglecentroidXcImperial}
                            onChange={(e) => calculateHollowReactangleCentroidXcImperial(e.target.value)}
                            aria-label="Hollow Rectangle Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglecentroidXcSelectedUnitImperial}
                            onChange={(e) => handleHollowReactangleCentroidXcUnitChangeImperial(e.target.value)}
                            id="HollowRectangleCentroidXcUnitImperial" name="unit"
                            aria-label="Hollow Rectangle Centroid Xc Unit Imperial"
                          >
                            {HollowReactangleCentroidXcUnitsImperial.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactanglecentroidYcImperial}
                            onChange={(e) => calculateHollowReactangleCentroidYcImperial(e.target.value)}
                            aria-label="Hollow Rectangle Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglecentroidYcSelectedUnitImperial}
                            onChange={(e) => handleHollowReactangleCentroidYcUnitChangeImperial(e.target.value)}
                            id="HollowRectangleCentroidYcUnitImperial" name="unit"
                            aria-label="Hollow Rectangle Centroid Yc Unit Imperial"
                          >
                            {HollowReactangleCentroidYcUnitsImperial.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactanglemomentOfInertiaIxImperial}
                            onChange={(e) => calculateHollowReactangleMomentOfInertiaIxImperial(e.target.value)}
                            aria-label="Hollow Rectangle Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglemomentOfInertiaIxSelectedUnitImperial}
                            onChange={(e) => handleHollowReactangleMomentOfInertiaIxUnitChangeImperial(e.target.value)}
                            id="HollowRectangleMomentOfInertiaIxUnitImperial" name="unit"
                            aria-label="Hollow Rectangle Moment of Inertia Ix Unit Imperial"
                          >
                            {HollowReactangleMomentOfInertiaIxUnitsImperial.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={HollowReactanglemomentOfInertiaIyImperial}
                            onChange={(e) => calculateHollowReactangleMomentOfInertiaIyImperial(e.target.value)}
                            aria-label="Hollow Rectangle Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglemomentOfInertiaIySelectedUnitImperial}
                            onChange={(e) => handleHollowReactangleMomentOfInertiaIyUnitChangeImperial(e.target.value)}
                            id="HollowRectangleMomentOfInertiaIyUnitImperial" name="unit"
                            aria-label="Hollow Rectangle Moment of Inertia Iy Unit Imperial"
                          >
                            {HollowReactangleMomentOfInertiaIyUnitsImperial.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(HollowReactanglesectionModulusSxImperial)}
                            onChange={(e) => calculateHollowReactangleSectionModulusSxImperial(e.target.value)}
                            aria-label="Hollow Rectangle Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglesectionModulusSxSelectedUnitImperial}
                            onChange={(e) => handleHollowReactangleSectionModulusSxUnitChangeImperial(e.target.value)}
                            id="HollowRectangleSectionModulesSxUnitImperial" name="unit"
                            aria-label="Hollow Rectangle Section Modules Sx Unit Imperial"
                          >
                            {HollowReactangleSectionModulusSxUnitsImperial.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(HollowReactanglesectionModulusSyImperial)}
                            onChange={(e) => calculateHollowReactangleSectionModulusSyImperial(e.target.value)}
                            aria-label="Hollow Rectangle Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={HollowReactanglesectionModulusSySelectedUnitImperial}
                            onChange={(e) => handleHollowReactangleSectionModulusSyUnitChangeImperial(e.target.value)}
                            id="HollowRectangleSectionModulesSyUnitImperial" name="unit"
                            aria-label="Hollow Rectangle Section Modules Sy Unit Imperial"
                          >
                            {HollowReactangleSectionModulusSyUnitsImperial.map((unit) => (
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
              </>
            )}
          </>
        )}
        {selectedOption === 'option4' && (
          <>
            {MetricOrImperial === 'option1' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={TeeSectionArea}
                            readOnly
                            aria-label="Tee Section Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={TeeSectionAreaUnit}
                            onChange={(e) => setTeeSectionAreaUnit(e.target.value)}
                            id="TeeSectionAreaUnit" name="unit"
                            aria-label="Tee section Area unit"
                          >
                            {TeeSectionAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={TeeSectioncentroidXc}
                            readOnly
                            aria-label="Tee Section Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={TeeSectioncentroidXcSelectedUnit}
                            onChange={(e) => setTeeSectionCentroidXcSelectedUnit(e.target.value)}
                            id="TeeSectionCentroidXcUnit" name="unit"
                            aria-label="Tee Section Centroid Xc Unit"
                          >
                            {TeeSectionCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={TeeSectioncentroidYc}
                            readOnly
                            aria-label="Tee Section Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={TeeSectioncentroidYcSelectedUnit}
                            onChange={(e) => setTeeSectionCentroidYcSelectedUnit(e.target.value)}
                            id="TeeSectionCentroidYcUnit" name="unit"
                            aria-label="Tee Section Centroid Yc Unit"
                          >
                            {TeeSectionCentroidYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={TeeSectionmomentOfInertiaIx}
                            readOnly
                            aria-label="Tee Section Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={TeeSectionmomentOfInertiaIxSelectedUnit}
                            onChange={(e) => setTeeSectionMomentOfInertiaIxSelectedUnit(e.target.value)}
                            id="TeeSectionMomentOfInertiaIxUnit" name="unit"
                          >
                            {TeeSectionMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={TeeSectionmomentOfInertiaIy}
                            readOnly
                            aria-label="Tee Section Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={TeeSectionmomentOfInertiaIySelectedUnit}
                            onChange={(e) => setTeeSectionMomentOfInertiaIySelectedUnit(e.target.value)}
                            id="TeeSectionMomentOfInertiaIyUnit" name="unit"
                            aria-label="Tee Section Moment of Inertia Iy Unit"
                          >
                            {TeeSectionMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>


                    </div>

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(TeeSectionsectionModulusSx)}
                            readOnly
                            aria-label="Tee Section Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={TeeSectionsectionModulusSxSelectedUnit}
                            onChange={(e) => setTeeSectionSectionModulusSxSelectedUnit(e.target.value)}
                            id="TeeSectionSectionModulesSxUnit" name="unit"
                            aria-label="Tee Section Section Modules Sx Unit"
                          >
                            {TeeSectionSectionModulusSxUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(TeeSectionSectionModulusSy)}
                            readOnly
                            aria-label="Tee Section Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={TeeSectionSectionModulusSySelectedUnit}
                            onChange={(e) => setTeeSectionSectionModulusSySelectedUnit(e.target.value)}
                            id="TeeSectionSectionModulesSyUnit" name="unit"
                            aria-label="Tee Section Section Modules Sy Unit"
                          >
                            {TeeSectionSectionModulusSyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </div>
                </div >

              </>)}
            {MetricOrImperial === 'option2' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialTeeSectionArea}
                            onChange={(e) => calculateImperialTeeSectionArea(e.target.value)}
                            aria-label="Imperial Tee Section Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialTeeSectionAreaUnit}
                            onChange={(e) => handleImperialTeeSectionAreaUnitChange(e.target.value)}
                            id="ImperialTeeSectionAreaUnit" name="unit"
                            aria-label="Imperial Tee Section Area Unit"
                          >
                            {ImperialTeeSectionAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialTeeSectionCentroidXc}
                            onChange={(e) => calculateImperialTeeSectionCentroidXc(e.target.value)}
                            aria-label="Imperial Tee Section Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialTeeSectionCentroidXcUnit}
                            onChange={(e) => handleImperialTeeSectionCentroidXcUnitChange(e.target.value)}
                            id="ImperialTeeSectionCentroidXcUnit" name="unit"
                            aria-label="Imperial Tee Section Centroid Xc Unit"
                          >
                            {ImperialTeeSectionCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialTeeSectionCentroidYc}
                            onChange={(e) => calculateImperialTeeSectionCentroidYc(e.target.value)}
                            aria-label="Imperial Tee Section Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialTeeSectionCentroidYcUnit}
                            onChange={(e) => handleImperialTeeSectionCentroidYcUnitChange(e.target.value)}
                            id="ImperialTeeSectionCentroidYcUnit" name="unit"
                            aria-label="Imperial Tee Section Centroid Yc Unit"
                          >
                            {ImperialTeeSectionCentroidYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialTeeSectionMomentOfInertiaIx}
                            onChange={(e) => calculateImperialTeeSectionMomentOfInertiaIx(e.target.value)}
                            aria-label="Imperial Tee Section Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialTeeSectionMomentOfInertiaIxUnit}
                            onChange={(e) => handleImperialTeeSectionMomentOfInertiaIxUnitChange(e.target.value)}
                            id="ImperialTeeSectionMomentOfInertiaIxUnit" name="unit"
                            aria-label="Imperial Tee Section Moment of Inertia Ix Unit"
                          >
                            {ImperialTeeSectionMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialTeeSectionMomentOfInertiaIy}
                            onChange={(e) => calculateImperialTeeSectionMomentOfInertiaIy(e.target.value)}
                            aria-label="Imperial Tee Section Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialTeeSectionMomentOfInertiaIyUnit}
                            onChange={(e) => handleImperialTeeSectionMomentOfInertiaIyUnitChange(e.target.value)}
                            id="ImperialTeeSectionMomentOfInertiaIyUnit" name="unit"
                            aria-label="Imperial Tee Section Moment of Inertia Iy Unit"
                          >
                            {ImperialTeeSectionMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(ImperialTeeSectionSectionModulesSx)}
                            onChange={(e) => calculateImperialTeeSectionSectionModulesSx(e.target.value)}
                            aria-label="Imperial Tee Section Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialTeeSectionSectionModulesSxUnit}
                            onChange={(e) => handleImperialTeeSectionSectionModulesSxUnitChange(e.target.value)}
                            id="ImperialTeeSectionSectionModulesSxUnit" name="unit"
                            aria-label="Imperial Tee Section Section Modules Sx Unit"
                          >
                            {ImperialTeeSectionSectionModulesUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px TeeSectionsectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={(ImperialTeeSectionsectionModulesSy)}
                            onChange={(e) => calculateImperialTeeSectionSectionModulesSy(e.target.value)}
                            aria-label="Imperial Tee Section Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialTeeSectionsectionModulesSyUnit}
                            onChange={(e) => handleImperialTeeSectionSectionModulesSyUnitChange(e.target.value)}
                            id="ImperialTeeSectionSectionModulesSyUnit" name="unit"
                            aria-label="Imperial Tee Section Section Modules Sy Unit"
                          >
                            {ImperialTeeSectionSectionModulesSyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </div>
                </div >

              </>
            )}
          </>
        )}
        {selectedOption === 'option5' && (
          <>
            {MetricOrImperial === 'option1' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>
                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ChannelArea}
                            readOnly
                            aria-label="Channel Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ChannelAreaUnit}
                            onChange={(e) => setChannelAreaUnit(e.target.value)}
                            id="ChannelAreaUnit" name="unit"
                            aria-label="Channel Area Unit"
                          >
                            {ChannelAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ChannelcentroidXc}
                            readOnly
                            aria-label="Channel Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ChannelcentroidXcSelectedUnit}
                            onChange={(e) => setChannelCentroidXcSelectedUnit(e.target.value)}
                            id="ChannelCentroidXcUnit" name="unit"
                            aria-label="Channel Centroid Xc Unit"
                          >
                            {ChannelCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ChannelcentroidYc}
                            readOnly
                            aria-label="Channel Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ChannelcentroidYcSelectedUnit}
                            onChange={(e) => setChannelCentroidYcSelectedUnit(e.target.value)}
                            id="ChannelCentroidYcUnit" name="unit"
                            aria-label="Channel Centroid Yc Unit"
                          >
                            {ChannelCentroidYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ChannelmomentOfInertiaIx}
                            readOnly
                            aria-label="Channel Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ChannelmomentOfInertiaIxSelectedUnit}
                            onChange={(e) => setChannelMomentOfInertiaIxSelectedUnit(e.target.value)}
                            id="ChannelMomentOfInertiaIxUnit" name="unit"
                            aria-label="Channel Moment of Inertia Ix Unit"
                          >
                            {ChannelMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ChannelmomentOfInertiaIy}
                            readOnly
                            aria-label="Channel Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ChannelmomentOfInertiaIySelectedUnit}
                            onChange={(e) => setChannelMomentOfInertiaIySelectedUnit(e.target.value)}
                            id="ChannelMomentOfInertiaIyUnit" name="unit"
                            aria-label="Channel Moment of Inertia Iy Unit"
                          >
                            {ChannelMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ChannelsectionModulusSx}
                            readOnly
                            aria-label="Channel Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ChannelsectionModulusSxSelectedUnit}
                            onChange={(e) => setChannelSectionModulusSxSelectedUnit(e.target.value)}
                            id="ChannelSectionModulusSxUnit" name="unit"
                            aria-label="Channel Section Modulus Sx Unit"
                          >
                            {ChannelSectionModulusSxUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ChannelSectionModulusSy}
                            readOnly
                            aria-label="Channel Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ChannelSectionModulusSySelectedUnit}
                            onChange={(e) => setChannelSectionModulusSySelectedUnit(e.target.value)}
                            id="ChannelSectionModulusSyUnit" name="unit"
                            aria-label="Channel Section Modulus Sy Unit"
                          >
                            {ChannelSectionModulusSyUnits.map((unit) => (
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
                  </div>
                </div >
              </>)}

            {MetricOrImperial === 'option2' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>
                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialChannelArea}
                            onChange={(e) => calculateImperialChannelArea(e.target.value)}
                            aria-label="Imperial Channel Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialChannelAreaUnit}
                            onChange={(e) => handleImperialChannelAreaUnitChange(e.target.value)}
                            id="ImperialChannelAreaUnit" name="unit"
                            aria-label="Imperial Channel Area Unit"
                          >
                            {ImperialChannelAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialChannelCentroidXc}
                            onChange={(e) => calculateImperialChannelCentroidXc(e.target.value)}
                            aria-label="Imperial Channel Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialChannelCentroidXcSelectedUnit}
                            onChange={(e) => handleImperialChannelCentroidXcUnitChange(e.target.value)}
                            id="ImperialChannelCentroidXcUnit" name="unit"
                            aria-label="Imperial Channel Centroid Xc Unit"
                          >
                            {ImperialChannelCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialChannelCentroidYc}
                            onChange={(e) => calculateImperialChannelCentroidYc(e.target.value)}
                            aria-label="Imperial Channel Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialChannelCentroidYcSelectedUnit}
                            onChange={(e) => handleImperialChannelCentroidYcUnitChange(e.target.value)}
                            id="ImperialChannelCentroidYcUnit" name="unit"
                            aria-label="Imperial Channel Centroid Yc Unit"
                          >
                            {ImperialChannelCentroidYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialChannelMomentOfInertiaIx}
                            onChange={(e) => calculateImperialChannelMomentOfInertiaIx(e.target.value)}
                            aria-label="Imperial Channel Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialChannelMomentOfInertiaIxSelectedUnit}
                            onChange={(e) => handleImperialChannelMomentOfInertiaUnitChange(e.target.value)}
                            id="ImperialChannelMomentOfInertiaIxUnit" name="unit"
                            aria-label="Imperial Channel Moment of Inertia Ix Unit"
                          >
                            {ImperialChannelMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialChannelMomentOfInertiaIy}
                            onChange={(e) => calculateImperialChannelMomentOfInertiaIy(e.target.value)}
                            aria-label="Imperial Channel Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialChannelMomentOfInertiaIySelectedUnit}
                            onChange={(e) => handleImperialChannelMomentOfInertiaIyUnitChange(e.target.value)}
                            id="ImperialChannelMomentOfInertiaIyUnit" name="unit"
                            aria-label="Imperial Channel Moment of Inertia Iy Unit"
                          >
                            {ImperialChannelMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialChannelSectionModulesSx}
                            onChange={(e) => calculateImperialChannelSectionModulesSx(e.target.value)}
                            aria-label="Imperial Channel Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialChannelSectionModulesSxUnit}
                            onChange={(e) => handleImperialChannelSectionModulesSxUnitChange(e.target.value)}
                            id="ChannelSectionModulusSxUnit" name="unit"
                            aria-label="Channel Section Modulus Sx Unit"
                          >
                            {ImperialChannelSectionModulesUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={ImperialChannelsectionModulesSy}
                            onChange={(e) => calculateImperialChannelSectionModulesSy(e.target.value)}
                            aria-label="Imperial Channel Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={ImperialChannelsectionModulesSyUnit}
                            onChange={(e) => handleImperialChannelSectionModulesSyUnitChange(e.target.value)}
                            id="ChannelSectionModulusSyUnit" name="unit"
                            aria-label="Channel Section Modulus Sy Unit"
                          >
                            {ImperialChannelSectionModulesSyUnits.map((unit) => (
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
                  </div>
                </div >
              </>
            )}
          </>
        )}
        {selectedOption === 'option6' && (
          <>
            {MetricOrImperial === 'option1' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionArea}
                            readOnly
                            aria-label="I-Section Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionAreaUnit}
                            onChange={(e) => setIsectionAreaUnit(e.target.value)}
                          >
                            {IsectionAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectioncentroidXc}
                            readOnly
                            aria-label="I-Section Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectioncentroidXcSelectedUnit}
                            onChange={(e) => setIsectionCentroidXcSelectedUnit(e.target.value)}
                            id="IsectionCentroidXcUnit"
                            name="unit"
                            aria-label="I-Section Centroid Xc Unit"
                          >
                            {IsectionCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectioncentroidYc}
                            readOnly
                            aria-label="I-Section Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectioncentroidYcSelectedUnit}
                            onChange={(e) => setIsectionCentroidYcSelectedUnit(e.target.value)}
                            id="IsectionCentroidYcUnit"
                            name="unit"
                            aria-label="I-Section Centroid Yc Unit"
                          >
                            {IsectionCentroidYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionmomentOfInertiaIx}
                            readOnly
                            aria-label="I-Section Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionmomentOfInertiaIxSelectedUnit}
                            onChange={(e) => setIsectionMomentOfInertiaIxSelectedUnit(e.target.value)}
                            id="IsectionMomentOfInertiaIxUnit"
                            name="unit"
                            aria-label="I-Section Moment of Inertia Ix Unit"
                          >
                            {IsectionMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionmomentOfInertiaIy}
                            readOnly
                            aria-label="I-Section Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionmomentOfInertiaIySelectedUnit}
                            onChange={(e) => setIsectionMomentOfInertiaIySelectedUnit(e.target.value)}
                            id="IsectionMomentOfInertiaIyUnit"
                            name="unit"
                            aria-label="I-Section Moment of Inertia Iy Unit"
                          >
                            {IsectionMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionsectionModulusSx}
                            readOnly
                            aria-label="I-Section Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionsectionModulusSxSelectedUnit}
                            onChange={(e) => setIsectionSectionModulusSxSelectedUnit(e.target.value)}
                            aria-label="I-section Section Modulus Sx unit"
                          >
                            {IsectionSectionModulusSxUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionSectionModulusSy}
                            readOnly
                            aria-label="I-Section Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionSectionModulusSySelectedUnit}
                            onChange={(e) => setIsectionSectionModulusSySelectedUnit(e.target.value)}
                            aria-label="I-section Section Modulus Sy unit"
                          >
                            {IsectionSectionModulusSyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        K :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionTorsionalConstant}
                            readOnly
                            aria-label="I-Section Torsional Constant"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionTorsionalConstantSelectedUnit}
                            onChange={(e) => setIsectionTorsionalConstantSelectedUnit(e.target.value)}
                            aria-label="I-section Torsional Constant unit"
                          >
                            {IsectionTorsionalConstantUnits.map((unit) => (
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
                  </div>
                </div >
              </>
            )}


            {MetricOrImperial === 'option2' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionImperialArea}
                            onChange={(e) => calculateIsectionImperialArea(e.target.value)}
                            aria-label="I-Section Imperial Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionImperialAreaUnit}
                            onChange={(e) => handleIsectionImperialAreaUnitChange(e.target.value)}
                            aria-label="I-Section Imperial Area unit"
                          >
                            {IsectionImperialAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw', fontWeight: '600' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionImperialCentroidXc}
                            onChange={(e) => calculateIsectionImperialCentroidXc(e.target.value)}
                            aria-label="I-Section Imperial Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionImperialCentroidXcSelectedUnit}
                            onChange={(e) => handleIsectionImperialCentroidXcUnitChange(e.target.value)}
                            aria-label="I-Section Imperial Centroid Xc unit"
                          >
                            {IsectionImperialCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionImperialCentriodYc}
                            onChange={(e) => calculateIsectionImperialCentriodYc(e.target.value)}
                            aria-label="I-Section Imperial Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionImperialCentriodYcSelectedUnit}
                            onChange={(e) => handleIsectionImperialCentriodYcUnitChange(e.target.value)}
                            aria-label="I-Section Imperial Centroid Yc unit"
                          >
                            {IsectionImperialCentriodYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw', fontWeight: '600' }}>Moment of Inertia :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionImperialMomentOfInertiaIx}
                            onChange={(e) => calculateIsectionImperialMomentOfInertiaIx(e.target.value)}
                            aria-label="I-Section Imperial Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionImperialMomentOfInertiaIxSelectedUnit}
                            onChange={(e) => handleIsectionImperialMomentOfInertiaUnitChange(e.target.value)}
                            aria-label="I-Section Imperial Moment of Inertia Ix unit"
                          >
                            {IsectionImperialMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionImperialMomentOfInertiaIy}
                            onChange={(e) => calculateIsectionImperialMomentOfInertiaIy(e.target.value)}
                            aria-label="I-Section Imperial Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionImperialMomentOfInertiaIySelectedUnit}
                            onChange={(e) => handleIsectionImperialMomentOfInertiaIyUnitChange(e.target.value)}
                            aria-label="I-Section Imperial Moment of Inertia Iy unit"
                          >
                            {IsectionImperialMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw', fontWeight: '600' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionImperialSectionModulesSx}
                            onChange={(e) => calculateIsectionImperialSectionModulesSxInputChangeValue(e.target.value)}
                            aria-label="I-Section Imperial Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionImperialSectionModulesSxSelectedUnit}
                            onChange={(e) => handleIsectionImperialSectionModulesSxUnitChange(e.target.value)}
                            aria-label="I-Section Imperial Section Modulus Sx unit"
                          >
                            {IsectionImperialSectionModulesSxUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionImperialSectionModulesSy}
                            onChange={(e) => calculateIsectionImperialSectionModulesSy(e.target.value)}
                            aria-label="I-Section Imperial Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionImperialSectionModulesSySelectedUnit}
                            onChange={(e) => handleIsectionImperialSectionModulesSyUnitChange(e.target.value)}
                            aria-label="I-Section Imperial Section Modulus Sy unit"
                          >
                            {IsectionImperialSectionModulesSyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw', fontWeight: '600' }}>Torsional Constant :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        K :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={IsectionImperialTorsionalConstant}
                            onChange={(e) => calculateIsectionImperialTorsionalConstant(e.target.value)}
                            aria-label="I-Section Imperial Torsional Constant"
                          />
                          <select
                            className='Calculator-select-option'
                            value={IsectionImperialTorsionalConstantSelectedUnit}
                            onChange={(e) => handleIsectionImperialTorsionalConstantUnitChange(e.target.value)}
                            aria-label="I-Section Imperial Torsional Constant unit"
                          >
                            {IsectionImperialTorsionalConstantUnits.map((unit) => (
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
                  </div>
                </div >
              </>
            )}
          </>
        )}
        {selectedOption === 'option7' && (
          <>
            {MetricOrImperial === 'option1' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionArea}
                            readOnly
                            aria-label="L-Section Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionAreaUnit}
                            onChange={(e) => setLsectionAreaUnit(e.target.value)}
                          >
                            {LsectionAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectioncentroidXc}
                            readOnly
                            aria-label="L-Section Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectioncentroidXcSelectedUnit}
                            onChange={(e) => setLsectionCentroidXcSelectedUnit(e.target.value)}
                            aria-label="L-Section Centroid Xc unit"
                          >
                            {LsectionCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectioncentroidYc}
                            readOnly
                            aria-label="L-Section Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectioncentroidYcSelectedUnit}
                            onChange={(e) => setLsectionCentroidYcSelectedUnit(e.target.value)}
                            aria-label="L-Section Centroid Yc unit"
                          >
                            {LsectionCentroidYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionmomentOfInertiaIx}
                            readOnly
                            aria-label="L-Section Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionmomentOfInertiaIxSelectedUnit}
                            onChange={(e) => setLsectionMomentOfInertiaIxSelectedUnit(e.target.value)}
                            aria-label="L-Section Moment of Inertia Ix unit"
                          >
                            {LsectionMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionmomentOfInertiaIy}
                            readOnly
                            aria-label="L-Section Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionmomentOfInertiaIySelectedUnit}
                            onChange={(e) => setLsectionMomentOfInertiaIySelectedUnit(e.target.value)}
                            aria-label="L-Section Moment of Inertia Iy unit"
                          >
                            {LsectionMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionsectionModulusSx}
                            readOnly
                            aria-label="L-Section Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionsectionModulusSxSelectedUnit}
                            onChange={(e) => setLsectionSectionModulusSxSelectedUnit(e.target.value)}
                            aria-label="L-Section Section Modulus Sx unit"
                          >
                            {LsectionSectionModulusSxUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionSectionModulusSy}
                            readOnly
                            aria-label="L-Section Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionSectionModulusSySelectedUnit}
                            onChange={(e) => setLsectionSectionModulusSySelectedUnit(e.target.value)}
                            aria-label="L-Section Section Modulus Sy unit"
                          >
                            {LsectionSectionModulusSyUnits.map((unit) => (
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
                  </div>
                </div >
              </>
            )}
            {MetricOrImperial === 'option2' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <div className='Section-properties-Solutions' style={{
                    borderRadius: '10px',
                    width: '90%',
                    margin: 'auto',
                  }}>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='text-white bold-heading-solution claculator-conversation-title' style={{ fontSize: '1.8vw', fontWeight: '600' }}>
                        Area: </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionImperialArea}
                            onChange={(e) => calculateLsectionImperialArea(e.target.value)}
                            aria-label="L-Section Imperial Area"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionImperialAreaUnit}
                            onChange={(e) => handleLsectionImperialAreaUnitChange(e.target.value)}
                            aria-label="L-Section Imperial Area unit"
                          >
                            {LsectionImperialAreaUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />
                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        X
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionImperialCentroidXc}
                            onChange={(e) => calculateLsectionImperialCentroidXc(e.target.value)}
                            aria-label="L-Section Imperial Centroid Xc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionImperialCentroidXcSelectedUnit}
                            onChange={(e) => handleLsectionImperialCentroidXcUnitChange(e.target.value)}
                            aria-label="L-Section Imperial Centroid Xc unit"
                          >
                            {LsectionImperialCentroidXcUnits.map((unit) => (
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
                        Y
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionImperialCentroidYc}
                            onChange={(e) => calculateLsectionImperialCentroidYc(e.target.value)}
                            aria-label="L-Section Imperial Centroid Yc"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionImperialCentroidYcSelectedUnit}
                            onChange={(e) => handleLsectionImperialCentroidYcUnitChange(e.target.value)}
                            aria-label="L-Section Imperial Centroid Yc unit"
                          >
                            {LsectionImperialCentroidYcUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    <br />

                    <h2 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of Inertia :</h2>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionImperialMomentOfInertiaIx}
                            onChange={(e) => calculateLsectionImperialMomentOfInertiaIx(e.target.value)}
                            aria-label="L-Section Imperial Moment of Inertia Ix"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionImperialMomentOfInertiaIxSelectedUnit}
                            onChange={(e) => handleLsectionImperialMomentOfInertiaIxUnitChange(e.target.value)}
                            aria-label="L-Section Imperial Moment of Inertia Ix unit"
                          >
                            {LsectionImperialMomentOfInertiaIxUnits.map((unit) => (
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
                        I
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionImperialMomentOfInertiaIy}
                            onChange={(e) => calculateLsectionImperialMomentOfInertiaIy(e.target.value)}
                            aria-label="L-Section Imperial Moment of Inertia Iy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionImperialMomentOfInertiaIySelectedUnit}
                            onChange={(e) => handleLsectionImperialMomentOfInertiaIyUnitChange(e.target.value)}
                            aria-label="L-Section Imperial Moment of Inertia Iy unit"
                          >
                            {LsectionImperialMomentOfInertiaIyUnits.map((unit) => (
                              <option key={unit} value={unit}>
                                {unit}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    <br />

                    <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modulus :</p>

                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <p className='claculator-conversation-title'>
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>x </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionImperialSectionModulusSx}
                            onChange={(e) => calculateLsectionImperialSectionModulusSx(e.target.value)}
                            aria-label="L-Section Imperial Section Modulus Sx"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionImperialSectionModulusSxSelectedUnit}
                            onChange={(e) => handleLsectionImperialSectionModulusSxUnitChange(e.target.value)}
                            aria-label="L-Section Imperial Section Modulus Sx unit"
                          >
                            {LsectionImperialSectionModulusSxUnits.map((unit) => (
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
                        S
                        <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y </span>
                        :
                      </p>
                      <div className='Calculator-Side-A'>
                        <div className='input-and-select-div'>
                          <input
                            className='calculator-input'
                            type="number"
                            value={LsectionImperialSectionModulusSy}
                            onChange={(e) => calculateLsectionImperialSectionModulusSy(e.target.value)}
                            aria-label="L-Section Imperial Section Modulus Sy"
                          />
                          <select
                            className='Calculator-select-option'
                            value={LsectionImperialSectionModulusSySelectedUnit}
                            onChange={(e) => handleLsectionImperialSectionModulusSyUnitChange(e.target.value)}
                            aria-label="L-Section Imperial Section Modulus Sy unit"
                          >
                            {LsectionImperialSectionModulusSyUnits.map((unit) => (
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
                  </div>
                </div >
              </>
            )}
          </>
        )}
        {selectedOption === 'option8' && (
          <>

            {MetricOrImperial === 'option1' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      X
                      <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' >c</span>

                      <span className='equalesto'>=</span>

                      Y
                      <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }} >c </span>

                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={SolidCircleCentroid}
                          readOnly
                          aria-label="Solid Circle Centroid"
                        />
                        <select
                          className='Calculator-select-option'
                          value={SolidCircleCentroidSelectedunit}
                          onChange={(e) => setSolidCircleCentroidSelectedUnit(e.target.value)}
                          aria-label="Solid Circle Centroid unit"
                        >
                          {SolidCircleCentroidUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of inertia:</p>


                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      I
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >x</span>
                      <span className='equalesto'>=</span>
                      I
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >y </span>
                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={(SolidCirclemomentOfInertia)}
                          readOnly
                          aria-label="Solid Circle Moment of Inertia"
                        />
                        <select
                          className='Calculator-select-option'
                          value={SolidCirclemomentOfInertiaSelectedUnit}
                          onChange={(e) => setSolidCircleMomentOfInertiaSelectedUnit(e.target.value)}
                          aria-label="Solid Circle Moment of Inertia unit"
                        >
                          {SolidCircleMomentOfInertiaUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modules:</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      S
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >x</span>
                      <span className='equalesto'>=</span>
                      S
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >y </span>
                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={SolidCirclesectionModules}
                          readOnly
                          aria-label="Solid Circle Section Modules"
                        />
                        <select
                          className='Calculator-select-option'
                          value={SolidCirclesectionModulesSelectedUnit}
                          onChange={(e) => setSolidCircleSectionModulesSelectedUnit(e.target.value)}
                          aria-label="Solid Circle Section Modules unit"
                        >
                          {SolidCircleSectionModulesUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant :</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      K :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={SolidCircletorsionalConstant}
                          readOnly
                          aria-label="Solid Circle Torsional Constant"
                        />
                        <select
                          className='Calculator-select-option'
                          value={SolidCircletorsionalConstantSelectedUnit}
                          onChange={(e) => setSolidCircleTorsionalConstantSelectedUnit(e.target.value)}
                          aria-label="Solid Circle Torsional Constant unit"
                        >
                          {SolidCircleTorsionalConstantUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
              </>)}

            {MetricOrImperial === 'option2' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>

                  <br />
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      X
                      <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' >c</span>

                      <span className='equalesto'>=</span>

                      Y
                      <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }} >c </span>

                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={SolidCircleImperialCentroid}
                          onChange={(e) => calculateSolidCircleImperialCentroidInputChangeValue(e.target.value)}
                          aria-label="Solid Circle Imperial Centroid"
                        />
                        <select
                          className='Calculator-select-option'
                          value={SolidCircleImperialCentroidSelectedunit}
                          onChange={(e) => handleSolidCircleImperialCentroidUnitChange(e.target.value)}
                        >
                          {SolidCircleImperialCentroidUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of inertia:</p>


                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      I
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >x</span>
                      <span className='equalesto'>=</span>
                      I
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >y </span>
                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={(SolidCircleImperialMomentOfInertia)}
                          onChange={(e) => calculateSolidCircleImperialMomentOfInertia(e.target.value)}
                          aria-label="Solid Circle Imperial Moment of Inertia"
                        />
                        <select
                          className='Calculator-select-option'
                          value={SolidCircleImperialMomentOfInertiaSelectedUnit}
                          onChange={(e) => handleSolidCircleImperialMomentOfInertiaUnitChange(e.target.value)}
                        >
                          {SolidCircleImperialMomentOfInertiaUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modules:</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      S
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >x</span>
                      <span className='equalesto'>=</span>
                      S
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >y </span>
                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={SolidCircleImperialSectionModules}
                          onChange={(e) => calculateSolidCircleImperialSectionModules(e.target.value)}
                          aria-label="Solid Circle Imperial Section Modules"
                        />
                        <select
                          className='Calculator-select-option'
                          value={SolidCircleImperialSectionModulesSelectedUnit}
                          onChange={(e) => handleSolidCircleImperialSectionModulesUnitChange(e.target.value)}
                        >
                          {SolidCircleImperialSectionModulesUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant :</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      K :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={SolidCircleImperialTorsionalConstant}
                          onChange={(e) => calculateSolidCircleImperialTorsionalConstant(e.target.value)}
                          aria-label="Solid Circle Imperial Torsional Constant"
                        />
                        <select
                          className='Calculator-select-option'
                          value={SolidCircleImperialTorsionalConstantSelectedUnit}
                          onChange={(e) => handleSolidCircleImperialTorsionalConstantUnitChange(e.target.value)}
                        >
                          {SolidCircleImperialTorsionalConstantUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
              </>)}
          </>
        )}
        {selectedOption === 'option9' && (
          <>
            {MetricOrImperial === 'option1' && (
              <>
                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>
                  <br />
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      X
                      <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' >c</span>

                      <span className='equalesto'>=</span>

                      Y
                      <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>

                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={HollowCircleCentroid}
                          readOnly
                          aria-label="Hollow Circle Centroid"
                        />
                        <select
                          className='Calculator-select-option'
                          value={HollowCircleCentroidSelectedunit}
                          onChange={(e) => setHollowCircleCentroidSelectedUnit(e.target.value)}
                        >
                          {HollowCircleCentroidUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of inertia:</p>


                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      I
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >x</span>
                      <span className='equalesto'>=</span>
                      I
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >y </span>
                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={(HollowCirclemomentOfInertia)}
                          readOnly
                          aria-label="Hollow Circle Moment of Inertia"
                        />
                        <select
                          className='Calculator-select-option'
                          value={HollowCirclemomentOfInertiaSelectedUnit}
                          onChange={(e) => setHollowCircleMomentOfInertiaSelectedUnit(e.target.value)}
                        >
                          {HollowCircleMomentOfInertiaUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modules:</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      S
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >x</span>
                      <span className='equalesto'>=</span>
                      S
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >y </span>
                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={HollowCirclesectionModules}
                          readOnly
                          aria-label="Hollow Circle Section Modules"
                        />
                        <select
                          className='Calculator-select-option'
                          value={HollowCirclesectionModulesSelectedUnit}
                          onChange={(e) => setHollowCircleSectionModulesSelectedUnit(e.target.value)}
                        >
                          {HollowCircleSectionModulesUnits.map((unit) => (
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
                </div>
              </>)}

            {MetricOrImperial === 'option2' && (
              <>

                <div className={isActive3 ? 'show Sectionmodules ' : 'hidden Sectionmodules '} style={{
                  height: '45vw',
                }}>
                  <br />
                  <br />
                  <h2 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h2>
                  <br />
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      X
                      <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' >c</span>

                      <span className='equalesto'>=</span>

                      Y
                      <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>

                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={HollowCircleImperialCentroid}
                          onChange={(e) => calculateHollowCircleImperialCentroidInputChangeValue(e.target.value)}
                          aria-label="Hollow Circle Imperial Centroid"
                        />
                        <select
                          className='Calculator-select-option'
                          value={HollowCircleImperialCentroidSelectedunit}
                          onChange={(e) => handleHollowCircleImperialCentroidUnitChange(e.target.value)}
                        >
                          {HollowCircleImperialunits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of inertia:</p>


                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      I
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >x</span>
                      <span className='equalesto'>=</span>
                      I
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >y </span>
                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={(HollowCircleImperialmomentOfInertia)}
                          onChange={(e) => calculateHollowCircleImperialMomentOfInertia(e.target.value)}
                          aria-label="Hollow Circle Imperial Moment of Inertia"
                        />
                        <select
                          className='Calculator-select-option'
                          value={HollowCircleImperialmomentOfInertiaSelectedUnit}
                          onChange={(e) => handleHollowCircleImperialMomentOfInertiaUnitChange(e.target.value)}
                        >
                          {HollowCircleImperialMomentOfInertiaUnits.map((unit) => (
                            <option key={unit} value={unit}>
                              {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                  <br />
                  <p className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modules:</p>

                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <p className='claculator-conversation-title'>
                      S
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >x</span>
                      <span className='equalesto'>=</span>
                      S
                      <span className='LowerPower sectionmodulesSolutionLowerPower' >y </span>
                      :
                    </p>
                    <div className='Calculator-Side-A'>
                      <div className='input-and-select-div'>
                        <input
                          className='calculator-input'
                          type="number"
                          value={ImperialHollowCircleSectionModules}
                          onChange={(e) => calculateImperialHollowCircleSectionModules(e.target.value)}
                          aria-label="Imperial Hollow Circle Section Modules"
                        />
                        <select
                          className='Calculator-select-option'
                          value={ImperialHollowCircleSectionModulesSelectedUnit}
                          onChange={(e) => handleImperialHollowCircleSectionModulesUnitChange(e.target.value)}
                        >
                          {ImperialHollowCircleSectionModulesUnits.map((unit) => (
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
                </div>
              </>)}

          </>
        )}

        <br />
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
        <AreaOfSection />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <PrincipleAxis />
        <br />
        <br />
        <br />
        <br />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <br />
        <Centroid />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <br />
        <br />
        <AreaMomentsOfInertia />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <br />
        <SectionModulus />
        <hr className="Beam-properties-calculator-hr"></hr>
        <br />
        <br />
        <br />
        <br />
        <TorsionalConstant />


        <section className='cse-header-top'>
          <Link smooth="true" duration={500} offset={-70} onClick={scrollToTop} aria-label="Scroll to top">
            <GrLinkTop className='' />
          </Link>
        </section>
      </section >

    </>
  )
}
