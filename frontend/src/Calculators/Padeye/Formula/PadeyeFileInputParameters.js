import React, { useState } from 'react'

export default function PadeyeFileInputParameters() {

    const [PadeyeInputParametersDropDown, setPadeyeInputParametersDropDown] = useState(false);

    const TogglePadeyeInputParametersDropDown = () => {
        setPadeyeInputParametersDropDown(!PadeyeInputParametersDropDown);
    };
    
    return (
        <>

            <div className='DropDownMainDiv'
                style={{
                    padding: '2% 4%',
                    margin: 'auto',
                    width: '90%',
                    background: 'rgb(247 247 247)',
                    borderRadius: '12px',
                }}
            >
                <h3 className='calculator-info-blue-section-main-topic InnerDropDownHeading' style={{
                    padding: '0',
                    margin: 'auto',
                    width: '100%',
                    justifyContent: 'left',
                    textTransform: 'uppercase',
                    fontWeight: '300',
                    fontSize: '3vw',
                    important: true
                }}>
                    Input parameters :
                </h3>
                <button
                    className='gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore InnerDropDownBtn'
                    onClick={TogglePadeyeInputParametersDropDown}
                    style={{ margin: '0.8vw 0', background: '#fff',
                        opacity:'1', transform:'none'
                     }}
                >Discover more</button>
            </div>
            <br />
            <br />

            <div className={PadeyeInputParametersDropDown ? ' Padeye-InputParameters-DropDown active ' : ' Padeye-InputParameters-DropDown '}>
                <br />
                <br />
                <br />
                <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{
                    padding: '0',
                    margin: 'auto',
                    width: '90%',
                    justifyContent: 'left',
                    textTransform: 'uppercase',
                    fontWeight: '300',
                    fontSize: '3.65vw',
                    important: true,
                }}>
                    PADEYE MATERIAL PARAMETER :
                </h3>
                <br />

                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', width: '100%', margin: 'auto' }}>

                    <div>
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Yield Strength : </p>
                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Stress at which material start deform plastically (permanent changes in shape even after the load is removed).</p>
                        <br />
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Tensile Strength :</p>
                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Maximum amount of tensile stress that a material can sustain before failure.</p>
                        <br />
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Modulus of elasticity :</p>
                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Material property which indicates how much a material will deform when subjected to load. Materials having high elastic modulus can resist deformation more effectively, while with a low elastic modulus are more flexible and deform more easily under the same load.</p>
                        <br />
                    </div>
                </div>
                <br />
                <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown'  style={{
                    padding: '0',
                    margin: 'auto',
                    width: '90%',
                    justifyContent: 'left',
                    textTransform: 'uppercase',
                    fontWeight: '300',
                    fontSize: '3.65vw',
                    important: true,
                }}>
                    WELDING PARAMETERS :
                </h3>
                <br />
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', width: '100%', margin: 'auto' }}>
                    <div>
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Weld Leg Size
                            (
                            <span className='power' style={{ fontSize: '1.5vw', top: '0vw', color: '#000' }}> t</span>
                            <span className='LowerPower' style={{ color: '#000', fontWeight: '700' }}>wc </span>

                            )
                            :
                        </p>
                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>The length of each side of the triangular cross-section of a fillet weld, measured from the root of the weld (where the two pieces of metal meet) to the toe (the outermost point of the weld) on each side.</p>
                        <br />
                    </div>
                </div>

                <br />
                <br />
                <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{
                    padding: '0',
                    margin: 'auto',
                    width: '90%',
                    justifyContent: 'left',
                    textTransform: 'uppercase',
                    fontWeight: '300',
                    fontSize: '3.65vw',
                    important: true,
                }}>
                    padeye load parameters :
                </h3>
                <br />
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', width: '100%', margin: 'auto' }}>

                    <div>
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Angle of load with vertical (
                            
                            <span className='power' style={{ fontSize: '1.5vw', top: '2px', color: '#000', padding: '0px 0px',fontWeight: '500' }}> θ </span>

                            ) 
                            : </p>
                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Angle of sling with respect to padeye in a plane parallel to the padeye.</p>
                        <br />
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Out of plane angle
                            (
                                
                            <span className='power' style={{ fontSize: '1.5vw', top: '2px', color: '#000', padding: '0px 0px',fontWeight: '500' }}>Φ</span>

                            )
                             :</p>
                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Angle of sling with respect to padeye in a plane perpendicular to the padeye.</p>
                        <br />
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Dynamic load factor
                            (
                                
                                <span className='power' style={{ fontSize: '1.5vw', top: '2px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300' }}>  DLF</span>
                            <span className='LowerPower' style={{ color: '#000', fontWeight: '700' }}></span>

                            ) 
                            : </p>
                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Used to account for the dynamic forces that can be experienced by a padeye due to sudden accelerations, decelerations, and other dynamic forces that can significantly increase the stresses on the padeye beyond the static load during a lifting operation.</p>
                        <br />
                        <br />
                        <br />
                    </div>
                </div>
            </div>
        </>
  )
}
