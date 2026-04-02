import React, { useState } from 'react'
// import axios from 'axios';
export default function OutputsofDeflection(props) {
    // const [result, setResult] = useState(null);
    // const [data, setData] = useState('hello world')
    // const handleCalculate = async () => {
    //     const response = await axios.post('/api/calculate', { data });
    //     setResult(response.data.result);
    // };

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
        props.toggleBreaks();
        // sendData={sendData}
        props.sendData()
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
    
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hiddenBeamDeflection' : 'showBeamDeflection'}

    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
    ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
      ${props.LoadBendingDeflectionclasses}

  `;
    const DropDowmTworightMain = `
    ScrollTransactionTwoBeamDeflactionDropDown rightsideTwoBeamDeflactionDropDown
  
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hiddenBeamDeflection' : 'showBeamDeflection'}

    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
  ${props.LoadBendingDeflectionclasses}
  `;
    const DropDowmthirdMain = `
    ScrollTransactionTwoBeamDeflactionDropDown rightsideTwoBeamDeflactionDropDown top405
  
    ${!showfirstDiv && !showSecondDiv && !showThirdDiv ? 'hiddenBeamDeflection' : 'showBeamDeflection'}

    ${showfirstDiv ? 'ScrollTransactionOne' : ''}
  ${showSecondDiv ? 'ScrollTransactionTwo' : ''}
  ${showThirdDiv ? 'ScrollTransactionThree' : ''}
  ${props.LoadBendingDeflectionclasses}
  `;
    return (
        <>
            <button className='structure-analysis-calculator-calculator-right-show-hidden-btn' onClick={handleCombinedClick}>Analyze Beam</button>
            <div>
                <div className={`${DropDowmOneMain} ${props.AddYoungsModulesClass ? "YoungModules" : ""} ${props.AddCrossSectionClass ? "CrossSection" : ""} ${props.AddloadClass ? "LoadPage" : ""}  ${props.AddSupportClass ? "SupportPage" : ""}`} style={{
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
                        {props.reactionData}
                        <br />
                        <br />
                    </div>
                </div>
            </div>
            <div>
                <div  className={`${DropDowmOnerightMain} ${props.AddYoungsModulesClass ? "YoungModules" : ""} ${props.AddCrossSectionClass ? "CrossSection" : ""} ${props.AddloadClass ? "LoadPage" : ""}  ${props.AddSupportClass ? "SupportPage" : ""}`} style={{
                    background: '#fff',
                    height: '30vw',
                    left: '55%',
                    transform: 'translate(0%, 0%)'
                }}>
                    <br />
                    <h2 className='DeflectionCalculatorTopic' style={{ fontSize: '3vw', color: 'black', textAlign: 'center' }}>Shear Force</h2>
                    {props.shearForceChartData}
                </div>
            </div>
<br/>
<br/>
<br/>
<br/>

            <div>
                <div className={`${DropDowmTwoMain} ${props.AddYoungsModulesClass ? "YoungModules" : ""} ${props.AddCrossSectionClass ? "CrossSection" : ""} ${props.AddloadClass ? "LoadPage" : ""}  ${props.AddSupportClass ? "SupportPage" : ""}`} style={{
                    background: '#fff',
                    height: '30vw',
                    left: '0%',
                    transform: 'translate(0%, 0%)',
                    top: '255%'
                }}>
                    <br />
                    <h2 className='DeflectionCalculatorTopic' style={{ fontSize: '3vw', color: 'black', textAlign: 'center' }}>Bending Moment</h2>
                    {props.bendingMomentChartData}
                </div>
            </div>
            <div>
                <div className={`${DropDowmTworightMain} ${props.AddYoungsModulesClass ? "YoungModules" : ""} ${props.AddCrossSectionClass ? "CrossSection" : ""} ${props.AddloadClass ? "LoadPage" : ""}  ${props.AddSupportClass ? "SupportPage" : ""}`} style={{
                    background: '#fff',
                    height: '30vw',
                    left: '55%',
                    transform: 'translate(0%, 0%)',
                    top: '255%'
                }}>
                    <br />
                    <h2 className='DeflectionCalculatorTopic' style={{ fontSize: '3vw', color: 'black', textAlign: 'center' }}>Deflection</h2>
                    {props.deflectionChartData}

                </div>
            </div >
        </>

    )
}