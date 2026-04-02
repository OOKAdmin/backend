import React, { useState, useEffect, useRef } from 'react'

import Areaimg from '../../../images/Beam-Properties-Area-Side-Image.webp'

export default function OutputParameterFile() {
    const [isActive, setIsActive] = useState(true);

    const toggleClass = () => {
        setIsActive(!isActive);
    };

    // 
    const textRef = useRef(null);
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setIsVisible(true);
                    } else {
                        setIsVisible(false);
                    }
                });
            },
            { threshold: 1 }
        );

        if (textRef.current) {
            observer.observe(textRef.current);
        }

        return () => {
            if (textRef.current) {
                observer.unobserve(textRef.current);
            }
        };
    }, []);
    // scroll transtion

    const targetRef = useRef(null);

    // Function to handle the button click
    const handleScroll = () => {
        targetRef.current.scrollIntoView({ behavior: 'smooth' });
    };

    const handleCombinedClick = () => {
        handleScroll();
        toggleClass();
    };
    return (
        <>
            <section className='structure-analysis-calculator-formula-dropdown-section'>
                <div className="container-fluid text-white bg-black py-4 align-items-center justify-content-center d-flex" style={{ maxWidth: '85%', margin: '0 auto', height: '70vh' }}>
                    <div className="row justify-content-evenly py-4 align-items-center" >
                        {/* Text Column */}
                        <div className="col-lg-6 col-md-12 mb-4 py-0 px-0">
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>Outputs Parameters:</h3>
                            <p ref={textRef} />
                            <br />
                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore  ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>

                        </div>

                        {/* Image Column */}
                        <div className="col-lg-4 col-md-12">
                            <img
                                src={Areaimg} // Replace with your image URL
                                alt="Placeholder"
                                className="img-fluid"
                                style={{ borderRadius: '20px' }}
                            />
                        </div>
                    </div>
                </div>
                <br />
                <br />
                <br />
                <br />

                <br />
                <br />
                <br />
                <div className={isActive ? 'mae-calculator-info PadeyeDropDown ' : 'mae-calculator-info PadeyeDropDown  active'} style={{ border: '2px solid', borderRadius: '30px' }} ref={targetRef}>
                    <br />
                    <br />
                    <br />
                    <div>
                        <br />
                        <br />
                        <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Reaction forces :
                        </h3>
                        <br />
                        <p className="calculator-info-blue-section-main-topic DefinatinOfDropDown" style={{ padding: '0px', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '1.8vw', }}>Forces exerted by a structure's supports or connections in response to applied loads. These are the forces that balance the applied loads, ensuring that the structure remains in equilibrium and static. Reaction forces depend upon the type of support.</p>
                        <br />
                        <br />
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline' }}>

                            <div>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '1.7vw', display: 'block' }}>Fixed support produces reaction forces in horizontal reaction
                                    (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-1px' }}>x</span>)
                                    , vertical reaction
                                    (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' >y</span>)
                                    , and moment
                                    (M). 
                                    {/* <div style={{ marginTop: '5px' }} /> */}
                                    <span> Because it restricts translation in both horizontal(x) and vertical(y) directions as well as rotational movement.</span></p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '1.7vw', display: 'block' }}>Pin support produce reaction forces in both horizontal
                                    (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-1px' }}>x</span>),
                                    vertical
                                    (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y</span>)
                                    reaction because it restrict translation in both horizontal(x) and vertical(y) direction but allow rotation</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '1.7vw', display: 'block' }}>Roller support produce only vertical reaction
                                    (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-1px' }}>y</span>)
                                    because it allows rotation and translation in the horizontal(x) direction but does not allow translation in the vertical(y) direction.</p>

                                <br />
                                <br />
                            </div>
                        </div>

                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Beam Deflection :
                        </p>
                        <br/>
                        <p className="calculator-info-blue-section-main-topic DefinatinOfDropDown" style={{ padding: '0px', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '2.2vw', }}>Beam bends or sags when subjected to loads such as forces, moments, or even its weight.</p>
                        <br />
                        <p className="calculator-info-blue-section-main-topic TopicOfDropdown" style={{ padding: '0px', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', }}>Formula :</p>
                        <br />
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline' }}>
                            <div style={{ width: '100%' }}>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '2vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '100', marginRight: '5px' }}>  δ </span>

                                    = (FL
                                    <span className='power' style={{ fontSize: '1.2vw', marginRight: '2px' }}>  3</span>
                                    )/3EI
                                </p>

                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '2vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '100', marginRight: '5px' }}>  δ </span>
                                    is the deflection
                                </p>

                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>F is the applied load	</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>L is the length of the beam</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>E is the modulus of elasticity of the material</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>I is the moment of inertia of the cross-sectional area of the beam</p>
                                <br />
                                <br />
                            </div>
                        </div>

                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Factors affecting Beam Deflection :
                        </p>
                        <br/>
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline' }}>

                            <div style={{ width: '100%' }}>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>Material Properties : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Young’s modulus (E) of the material affect how much beam will deflect.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>	Cross-Sectional Shape : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Cross-Sectional Shape with larger moments of inertia are stiffer and deflect less.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>Length of the Beam : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Deflection is directly proportional to the beam's length.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>Load type : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Amount & distribution of load such as point load, uniformly distributed load, and varying load affect the deflection.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>Support Conditions : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Support such as Fixed, Roller each have different deflection characteristics due to their support constraints.  </p>
                                <br />


                                <br />
                            </div>
                        </div>


                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Deflection diagram :
                        </p>
                        <br/>
                        <p className="calculator-info-blue-section-main-topic DefinatinOfDropDown" style={{ padding: '0px', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '1.8vw', }}>Graphical representation of the displacement or deflection a beam or structure undergoes when subjected to external forces and moments. Provides essential information for engineers about the behaviour of a beam under load or ensure that structural elements can withstand expected loads without experiencing excessive deflection.</p>
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline' }}>

                            <div style={{ width: '100%' }}>
                                <br />
                                <br />
                            </div>
                        </div>


                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Shear stress :
                        </p>
                        <br/>
                        <p className="calculator-info-blue-section-main-topic DefinatinOfDropDown" style={{ padding: '0px', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '1.8vw', }}>Internal force that acts parallel to the cross-sectional area of the beam. When a beam is subjected to transverse loads causing one layer of the material to slide or deform relative to adjacent layers. Also defined as the force per unit area acting parallel to a plane within the material.</p>
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline' }}>

                            <div style={{ width: '100%' }}>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>Formula :</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span style={{ fontFamily: 'none', marginRight: '5px' }}>τ </span>
                                    = F/A
                                </p>
                                <div style={{ marginTop: '20px' }}></div>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '1.8vw' }}><span style={{ marginRight: '8px' }}>SI unit : </span> Pascal (Pa) or newton per square meter (N/m²)</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.5vw' }}>Shear Force Diagram : </p>
                                <div style={{ marginTop: '8px' }} />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', }}>Graphical representation of the shear force variation along a beam or structural member's length when it is subjected to external loads. The position of the beam is plotted along the horizontal axis and the magnitude of the shear force is plotted along the vertical axis. A useful tool for structural analysis and design because it helps us to determine the maximum shear force and its location and how shear forces vary along a beam's length.</p>
                                <br />
                                <br />

                            </div>
                        </div>


                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Bending moment :
                        </p>
                        <br/>
                        <p className="calculator-info-blue-section-main-topic DefinatinOfDropDown" style={{ padding: '0px', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '1.8vw', }}>Measure of the internal bending forces experienced within a structural member, such as a beam or column, when subjected to external loads or moments.</p>
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline' }}>

                            <div style={{ width: '100%' }}>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>	Sign convention : </p>
                                <div style={{ marginTop: '8px' }} />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', }}>Sign convention helps to determine the type of bending (sagging or hogging) at a given section of the beam. Bending moments are considered positive when they cause concave bending on the top of the beam called sagging and convex bending on the bottom of the beam called hogging. </p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>Formula :</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>M=∑Fxd</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>where:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>M = Bending moment at the point (in units of force times distance)</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>F = Applied force perpendicular to the beam axis</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>d = Perpendicular distance from the point to the line of action of the force</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '1.8vw' }}><span style={{ marginRight: '8px' }}>SI unit : </span>Newton meter (Nm)</p>
                                <br />
                                <div style={{ marginTop: '2px' }} />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw' }}>Bending moment diagram : </p>
                                <div style={{ marginTop: '8px' }} />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', }}>Graphical representation of how the bending moment varies along the length of the structural member. Plotted between the length of the member along the horizontal axis and the magnitude of the bending moment along the vertical axis. Bending moment diagrams are essential for analysing the behaviour of the member under load and for designing structures to resist bending stresses.</p>

                            </div>
                        </div>


                    </div>
                </div>
            </section>
        </>
    )
}
