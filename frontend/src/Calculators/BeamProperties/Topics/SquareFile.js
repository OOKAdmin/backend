import React, { useState, useEffect } from 'react'

export default function SquareFile(props) {
  const MetricUnits = ['mm', 'cm', 'm'];
  const MetricConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const AreaUnits = ['mm²', 'cm²', 'm²'];
  const AreaConversionFactors = {
    'mm²': [1, 0.01, 0.000001],
    'cm²': [100, 1, 0.0001],
    'm²': [1000000, 10000, 1],
  };

  const CentroidUnits = ['mm', 'cm', 'm'];
  const CentroidConversionFactors = {
    mm: [1, 0.1, 0.001],
    cm: [10, 1, 0.01],
    m: [1000, 100, 1],
  };

  const MomentOfInertiaUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const MomentOfInertiaConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const SectionModulesUnits = ['mm³', 'cm³', 'm³'];
  const SectionModulesConversionFactors = {
    'mm³': [1, 0.001, 1e-9],
    'cm³': [1000, 1, 1e-6],
    'm³': [1e9, 1e6, 1],
  };

  const TorsionalConstantUnits = ['mm⁴', 'cm⁴', 'm⁴'];
  const TorsionalConstantConversionFactors = {
    'mm⁴': [1, 1e-4, 1e-12],
    'cm⁴': [1e4, 1, 1e-8],
    'm⁴': [1e12, 1e8, 1],
  };

  const [metricInputValue, setMetricInputValue] = useState('0');
  const [metricSelectedUnit, setMetricSelectedUnit] = useState('mm');
  const [internalMetricValue, setInternalMetricValue] = useState(0); // Always in mm

  const [area, setArea] = useState(0);
  const [areaUnit, setAreaUnit] = useState('mm²');

  const [centroid, setCentroid] = useState(0);
  const [centroidUnit, setCentroidUnit] = useState('mm');

  const [momentOfInertia, setMomentOfInertia] = useState(0);
  const [momentOfInertiaUnit, setMomentOfInertiaUnit] = useState('mm⁴');

  const [sectionModules, setSectionModules] = useState(0);
  const [sectionModulesUnit, setSectionModulesUnit] = useState('mm³');

  const [torsionalConstant, setTorsionalConstant] = useState(0);
  const [torsionalConstantUnit, setTorsionalConstantUnit] = useState('mm⁴');

  const handleMetricInputChange = (value) => {
    setMetricInputValue(value);
    const factor = MetricConversionFactors[metricSelectedUnit][0];
    setInternalMetricValue(parseFloat(value) * factor);
  };

  const handleMetricUnitChange = (unit) => {
    const newMetricValueInMM = parseFloat(metricInputValue) * MetricConversionFactors[metricSelectedUnit][0];
    const convertedValue = newMetricValueInMM / MetricConversionFactors[unit][0];
    setMetricSelectedUnit(unit);
    setMetricInputValue(convertedValue);
  };

  useEffect(() => {
    const value = parseFloat(metricInputValue) * MetricConversionFactors[metricSelectedUnit][0];
    if (!isNaN(value)) {
      const areaValue = (value * value).toFixed(5);
      setArea((areaValue / AreaConversionFactors[areaUnit][0]));
    } else {
      setArea('');
    }
  }, [metricInputValue, areaUnit, metricSelectedUnit]);

  useEffect(() => {
    const value = internalMetricValue;
    const centroidValue = (value / 2).toFixed(2);
    setCentroid((centroidValue / CentroidConversionFactors[centroidUnit][0]));
  }, [internalMetricValue, centroidUnit]);

  useEffect(() => {
    const value = internalMetricValue;
    if (!isNaN(value)) {
      const inertia = ((value ** 4) / 12).toFixed(3);
      const convertedValue = (inertia * MomentOfInertiaConversionFactors['mm⁴'][MomentOfInertiaUnits.indexOf(momentOfInertiaUnit)]);
      setMomentOfInertia(convertedValue);
    }
  }, [internalMetricValue, momentOfInertiaUnit]);

  useEffect(() => {
    const value = internalMetricValue;
    if (!isNaN(value)) {
      const centroidValue = (value / 2);
      const momentOfInertiaValue = ((value ** 4) / 12);
      const sectionModulesValue = (momentOfInertiaValue / centroidValue).toFixed(4);
      const convertedValue = (sectionModulesValue * SectionModulesConversionFactors['mm³'][SectionModulesUnits.indexOf(sectionModulesUnit)]);
      setSectionModules(convertedValue);
    }
  }, [internalMetricValue, sectionModulesUnit]);

  useEffect(() => {
    const value = internalMetricValue;
    if (!isNaN(value)) {
      const torsional = (9 * Math.pow(value, 4)) / 64;
      const convertedValue = (torsional * TorsionalConstantConversionFactors['mm⁴'][TorsionalConstantUnits.indexOf(torsionalConstantUnit)]);
      setTorsionalConstant(convertedValue);
    }
  }, [internalMetricValue, torsionalConstantUnit]);










  const Imperialunits = ['in'];
  const ImperialFactors = {
    in: [1],
  };

  const [ImperialinputValue, setImperialInputValue] = useState(0);
  const [ImperialselectedUnit, setImperialSelectedUnit] = useState('in');

  const handleImperialInputChange = (value) => {
    setImperialInputValue(value);
  };

  const handleImperialUnitChange = (unit) => {
    setImperialSelectedUnit(unit);
    const Imperialfactor = ImperialFactors[unit][Imperialunits.indexOf(ImperialselectedUnit)];
    setImperialInputValue((parseFloat(ImperialinputValue) / Imperialfactor).toFixed(3));
  };

  const ImperialAreaUnits = ['in²'];
  const ImperialAreaConversionFactors = {
    'in²': [1],
  };
  const [ImperialArea, setImperialArea] = useState(0);
  const [ImperialAreaUnit, setImperialAreaUnit] = useState('in²');

  const handleImperialAreaUnitChange = (unit) => {
    const currentAreaInNewUnit = ImperialArea * ImperialAreaConversionFactors[ImperialAreaUnit][ImperialAreaUnits.indexOf(unit)];
    setImperialArea(currentAreaInNewUnit);
    setImperialAreaUnit(unit);
  };

  const calculateImperialAreaInputChangeValue = (e) => {
    const ImperialinputValue = parseFloat(e.target.value);
    if (!isNaN(ImperialinputValue)) {
      const ImperialArea = (ImperialinputValue * ImperialinputValue);
      setImperialArea(ImperialArea);
    } else {
      setImperialArea('');
    }
    setImperialInputValue(e.target.value);
  };

  useEffect(() => {
    if (ImperialinputValue !== '') {
      calculateImperialAreaInputChangeValue({ target: { value: ImperialinputValue } });
    }
  }, [ImperialinputValue]);



  const [ImperialCentroid, setImperialCentroid] = useState(0);
  const [ImperialCentroidSelectedunit, setImperialCentroidSelectedUnit] = useState('in');

  const handleImperialCentroidUnitChange = (unit) => {
    setImperialCentroidSelectedUnit(unit);
  };

  const calculateImperialCentroidInputChangeValue = () => {
    const Input = ImperialinputValue;
    const ImperialCentroid = Input / 2;
    setImperialCentroid(ImperialCentroid);
  };

  useEffect(() => {
    calculateImperialCentroidInputChangeValue();
  }, [ImperialinputValue]);



  const ImperialMomentOfInertiaUnits = ['in⁴'];
  const ImperialMomentOfInertiaConversionFactors = {
    'in⁴': [1],
  };
  const [ImperialmomentOfInertia, setImperialMomentOfInertia] = useState(0);
  const [ImperialmomentOfInertiaSelectedUnit, setImperialMomentOfInertiaSelectedUnit] = useState('in⁴');

  const handleImperialMomentOfInertiaUnitChange = (newUnit) => {
    const currentFactor = ImperialMomentOfInertiaConversionFactors[ImperialmomentOfInertiaSelectedUnit][ImperialMomentOfInertiaUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(ImperialmomentOfInertia) * currentFactor;
    setImperialMomentOfInertiaSelectedUnit(newUnit);
    setImperialMomentOfInertia(convertedValue.toString());
  };

  const calculateImperialMomentOfInertia = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const inertia = ((parsedValue ** 4) / 12).toFixed(3);
      const currentFactor = ImperialMomentOfInertiaConversionFactors['in⁴'][ImperialMomentOfInertiaUnits.indexOf(ImperialmomentOfInertiaSelectedUnit)];
      setImperialMomentOfInertia((inertia * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateImperialMomentOfInertia(ImperialinputValue);
  }, [ImperialinputValue, ImperialmomentOfInertiaSelectedUnit]);

  const ImperialSectionModulesUnits = ['in³'];
  const ImperialSectionModulesConversionFactors = {
    'in³': [1],
  };
  const [ImperialSectionModules, setImperialSectionModules] = useState(0);
  const [ImperialSectionModulesSelectedUnit, setImperialSectionModulesSelectedUnit] = useState('in³');

  const handleImperialSectionModulesUnitChange = (newUnit) => {
    const currentFactor = ImperialSectionModulesConversionFactors[ImperialSectionModulesSelectedUnit][ImperialSectionModulesUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(ImperialSectionModules) * currentFactor;
    setImperialSectionModulesSelectedUnit(newUnit);
    setImperialSectionModules(convertedValue.toString());
  };

  const calculateImperialSectionModules = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const centroidValue = (parsedValue / 2).toFixed(3);
      const momentOfInertiaValue = ((parsedValue ** 4) / 12).toFixed(3);
      const sectionModules = (momentOfInertiaValue / centroidValue).toFixed(3);
      const currentFactor = ImperialSectionModulesConversionFactors['in³'][ImperialSectionModulesUnits.indexOf(ImperialSectionModulesSelectedUnit)];
      setImperialSectionModules((sectionModules * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateImperialSectionModules(ImperialinputValue);
  }, [ImperialinputValue, ImperialSectionModulesSelectedUnit]);


  const ImperialTorsionalConstantUnits = ['in⁴'];
  const ImperialTorsionalConstantConversionFactors = {
    'in⁴': [1],
  };
  const [ImperialTorsionalConstant, setImperialTorsionalConstant] = useState(0);
  const [ImperialTorsionalConstantSelectedUnit, setImperialTorsionalConstantSelectedUnit] = useState('in⁴');

  const handleImperialTorsionalConstantUnitChange = (newUnit) => {
    const currentFactor = ImperialTorsionalConstantConversionFactors[ImperialTorsionalConstantSelectedUnit][ImperialTorsionalConstantUnits.indexOf(newUnit)];
    const convertedValue = parseFloat(ImperialTorsionalConstant) * currentFactor;
    setImperialTorsionalConstantSelectedUnit(newUnit);
    setImperialTorsionalConstant(convertedValue.toString());
  };

  const calculateImperialTorsionalConstant = (value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      const torsional = ((9 * Math.pow(parsedValue, 4)) / 64).toFixed(3);
      const currentFactor = ImperialTorsionalConstantConversionFactors['in⁴'][ImperialTorsionalConstantUnits.indexOf(ImperialTorsionalConstantSelectedUnit)];
      setImperialTorsionalConstant((torsional * currentFactor).toString());
    }
  };

  useEffect(() => {
    calculateImperialTorsionalConstant(ImperialinputValue);
  }, [ImperialinputValue, ImperialTorsionalConstantSelectedUnit]);




  return (
    <>
      <div style={{
        width: '100%',
        display: 'flex',
        justifyContent: 'space-evenly',
        padding: '1vw 0'
      }}>
        <button className='metricBtn'
          onClick={() => props.MetricOrImperialBtn("option1")}
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
          onClick={() => props.MetricOrImperialBtn("option2")}
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
      {props.MetricOrImperial === 'option1' && (
        <>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Side (a) :</h3>
            <div className='Calculator-Side-A'>
              <input
                className='calculator-input'
                type="number"
                value={isNaN(metricInputValue) ? "" : metricInputValue}
                onChange={(e) => handleMetricInputChange(e.target.value)}
              />
              <select
                className='Calculator-select-option'
                value={metricSelectedUnit}
                onChange={(e) => handleMetricUnitChange(e.target.value)}
              >
                {MetricUnits.map((unit) => (
                  <option key={unit} value={unit}>
                    {unit}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn ' onClick={props.toggleBreaks}>{props.togglefunction ? ' Hide ' : ' Solve '}</button>
          <div className={props.togglefunction ? 'show  Sectionmodules  ' : 'hidden  Sectionmodules  '} style={{
            height: '45vw',
          }}>
            <br />
            <br />
            <h3 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h3>

            <br />
            <br />
            <div className='' style={{
              borderRadius: '10px',
              width: '90%',
              margin: 'auto',
            }}>
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Area:</h3>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  Area: </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(area) ? "" : area}
                      readOnly
                    />
                    <select
                      className='Calculator-select-option'
                      value={areaUnit}
                      onChange={(e) => setAreaUnit(e.target.value)}
                    >
                      {AreaUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <br />
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</h3>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  X
                  <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{}}>c</span>

                  <span className='equalesto'>=</span>

                  Y
                  <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>

                  :
                </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(centroid) ? "" : centroid}
                      readOnly
                    />
                    <select
                      className='Calculator-select-option'
                      value={centroidUnit}
                      onChange={(e) => setCentroidUnit(e.target.value)}
                    >
                      {CentroidUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <br />
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of inertia:</h3>


              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  I
                  <span className='LowerPower sectionmodulesSolutionLowerPower'>x</span>
                  <span className='equalesto'>=</span>
                  I
                  <span className='LowerPower sectionmodulesSolutionLowerPower'>y </span>
                  :
                </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(momentOfInertia) ? "" : momentOfInertia}
                      readOnly
                    />
                    <select
                      className='Calculator-select-option'
                      value={momentOfInertiaUnit}
                      onChange={(e) => setMomentOfInertiaUnit(e.target.value)}
                    >
                      {MomentOfInertiaUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <br />
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modules:</h3>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  S
                  <span className='LowerPower sectionmodulesSolutionLowerPower'>x</span>
                  <span className='equalesto'>=</span>
                  S
                  <span className='LowerPower sectionmodulesSolutionLowerPower'>y </span>
                  :
                </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(sectionModules) ? "" : sectionModules}
                      readOnly

                    />
                    <select
                      className='Calculator-select-option'
                      value={sectionModulesUnit}
                      onChange={(e) => setSectionModulesUnit(e.target.value)}
                    >
                      {SectionModulesUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <br />
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant:</h3>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  K :
                </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(torsionalConstant) ? "" : torsionalConstant}
                      readOnly
                    />
                    <select
                      className='Calculator-select-option'
                      value={torsionalConstantUnit}
                      onChange={(e) => setTorsionalConstantUnit(e.target.value)}
                    >
                      {TorsionalConstantUnits.map((unit) => (
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




      {props.MetricOrImperial === 'option2' && (
        <>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <h3 className='claculator-conversation-title'>Side (a) :</h3>
            <div className='Calculator-Side-A'>
              <input
                className='calculator-input'
                type="number"
                value={isNaN(ImperialinputValue) ? "" : ImperialinputValue}
                onChange={(e) => handleImperialInputChange(e.target.value)}
              />
              <select
                className='Calculator-select-option'
                value={isNaN(ImperialselectedUnit) ? "" : ImperialselectedUnit}
                onChange={(e) => handleImperialUnitChange(e.target.value)}
              >
                {Imperialunits.map((unit) => (
                  <option key={unit} value={unit}>
                    {unit}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <button className='structure-analysis-calculator-calculator-right-show-hidden-btn' onClick={props.toggleBreaks}>{props.togglefunction ? ' Hide ' : ' Solve '}</button>

          <div className={props.togglefunction ? 'show  Sectionmodules  ' : 'hidden  Sectionmodules  '} style={{
            height: '45vw',
          }}>
            <br />
            <br />
            <h3 className='text-white calculator-defination-section text-center' style={{ fontSize: '3vw', }}>Section Properties Of Beam</h3>
            <br />
            <br />
            <div className='' style={{
              borderRadius: '10px',
              width: '90%',
              margin: 'auto',
            }}>
              {/* <br /> */}
              {/* <h3 className='text-white' style={{ fontSize: '1.8vw' }}>Area:</h3> */}
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Area:</h3>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  Area: </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(ImperialArea) ? "" : ImperialArea}
                      onChange={(e) => calculateImperialAreaInputChangeValue(e.target.value)}

                    />
                    <select
                      className='Calculator-select-option'
                      value={isNaN(ImperialAreaUnit) ? "" : ImperialAreaUnit}
                      onChange={(e) => handleImperialAreaUnitChange(e.target.value)}
                    >
                      {ImperialAreaUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <br />
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Centroid:</h3>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  X
                  <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{}}>c</span>

                  <span className='equalesto'>=</span>

                  Y
                  <span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-3px' }}>c </span>

                  :
                </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(ImperialCentroid) ? "" : ImperialCentroid}
                      onChange={(e) => calculateImperialCentroidInputChangeValue(e.target.value)}

                    />
                    <select
                      className='Calculator-select-option'
                      value={isNaN(ImperialCentroidSelectedunit) ? "" : ImperialCentroidSelectedunit}
                      onChange={(e) => handleImperialCentroidUnitChange(e.target.value)}
                    >
                      {Imperialunits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <br />
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Moment of inertia:</h3>


              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  I
                  <span className='LowerPower sectionmodulesSolutionLowerPower'>x</span>
                  <span className='equalesto'>=</span>
                  I
                  <span className='LowerPower sectionmodulesSolutionLowerPower'>y </span>
                  :
                </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(ImperialmomentOfInertia) ? "" : ImperialmomentOfInertia}
                      onChange={(e) => calculateImperialMomentOfInertia(e.target.value)}
                    />
                    <select
                      className='Calculator-select-option'
                      value={isNaN(ImperialmomentOfInertiaSelectedUnit) ? "" : ImperialmomentOfInertiaSelectedUnit}
                      onChange={(e) => handleImperialMomentOfInertiaUnitChange(e.target.value)}
                    >
                      {ImperialMomentOfInertiaUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <br />
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Section Modules:</h3>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  S
                  <span className='LowerPower sectionmodulesSolutionLowerPower'>x</span>
                  <span className='equalesto'>=</span>
                  S
                  <span className='LowerPower sectionmodulesSolutionLowerPower'>y </span>
                  :
                </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(ImperialSectionModules) ? "" : ImperialSectionModules}
                      onChange={(e) => calculateImperialSectionModules(e.target.value)}

                    />
                    <select
                      className='Calculator-select-option'
                      value={isNaN(ImperialSectionModulesSelectedUnit) ? "" : ImperialSectionModulesSelectedUnit}
                      onChange={(e) => handleImperialSectionModulesUnitChange(e.target.value)}
                    >
                      {ImperialSectionModulesUnits.map((unit) => (
                        <option key={unit} value={unit}>
                          {unit}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <br />
              <h3 className='text-white bold-heading-solution' style={{ fontSize: '1.8vw' }}>Torsional Constant:</h3>

              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <h3 className='claculator-conversation-title'>
                  K :
                </h3>
                <div className='Calculator-Side-A'>
                  <div className='input-and-select-div'>
                    <input
                      className='calculator-input'
                      type="number"
                      value={isNaN(ImperialTorsionalConstant) ? "" : ImperialTorsionalConstant}
                      onChange={(e) => calculateImperialTorsionalConstant(e.target.value)}

                    />
                    <select
                      className='Calculator-select-option'
                      value={isNaN(ImperialTorsionalConstantSelectedUnit) ? "" : ImperialTorsionalConstantSelectedUnit}
                      onChange={(e) => handleImperialTorsionalConstantUnitChange(e.target.value)}
                    >
                      {ImperialTorsionalConstantUnits.map((unit) => (
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
  )
}
