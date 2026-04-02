import React, { useState, useEffect, useRef } from 'react'

// Blackbox image
import Areaimg from '../../../images/Beam-Properties-Area-Side-Image.webp'

// type of loads images
import PointLoadImg from '../../../images/BeamDeflaction-Blackbox-point-load.png'
import UniformdistributedLoadImg from '../../../images/BeamDeflaction-Blackbox-Distributed-support.png'
import NonuniformdistributedLoadImg from '../../../images/BeamDeflaction-Blackbox-non-distributed-support.png'

// types of support images
import FixedImage from '../../../images/BeamDeflaction-Blackbox-Fixed-support.png'
import PinnedImage from '../../../images/BeamDeflaction-Blackbox-Pined-support.png'
import RollerImage from '../../../images/BeamDeflaction-Blackbox-roller-support.png'
export default function InputsParametersFile() {
    const [isActive, setIsActive] = useState(true);

    const toggleClass = () => {
        setIsActive(!isActive);
    };
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
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>Inputs Parameters:</h3>
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
                <div className={isActive ? 'mae-calculator-info PadeyeDropDown ' : 'mae-calculator-info PadeyeDropDown  active'} style={{ border: '2px solid', borderRadius: '30px' }} ref={targetRef}>
                    <br />
                    <br />
                    <br />
                    <div>
                        <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Inputs parameters :
                        </h3>
                        <br />

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>

                            <div>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>	Length of the Beam : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Deflection is directly proportional to the beam's length</p>
                                <br />
                                <br />
                                <div style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
                                    <hr style={{ width: '80%' }} />
                                </div>
                                <br />
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>	Young’s modulus (<span className='power' style={{ fontSize: '1.5vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300' }}>  E</span>) :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Property that indicates how much a material will deform when subjected to load.</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Materials having high elastic modulus can resist deformation more effectively, While with a low elastic modulus are more flexible and deform more easily under the same load.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Young’s Modulus is calculated by :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>E = σ/ε</p>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>Where:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '78%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '1.5vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '200', marginRight: '5px' }}>  σ is the stress </span>
                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '78%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '1.5vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '200', marginRight: '5px' }}>  ε is the strain </span>
                                </p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2vw' }}>Units of Young’s Modulus</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left' }}>S.I Units: Pascal (Pa) and Mega-Pascal (Mpa).</p>
                                <br />
                                <br />
                                <div style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
                                    <hr style={{ width: '80%' }} />
                                </div>
                                <br />
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Second moment of area :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Measure of cross-section resistance to bending due to its shape.</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>A larger second moment of area means greater resistance to bending stress and beam deflection.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Area moment of inertia can be calculated by :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '1.5vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300', marginRight: '5px', marginLeft: '8px' }}> Iy = ∫x
                                        <span className='power' style={{ fontSize: '1vw', marginRight: '2px' }}>  2 </span>

                                        dA</span>
                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '1.5vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300', marginRight: '5px', marginLeft: '8px' }}> Ix = ∫y
                                        <span className='power' style={{ fontSize: '1vw', marginRight: '2px' }}>  2 </span>

                                        dA</span>

                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.5vw' }}>Where:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '78%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '1.5vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300', marginRight: '5px' }}> distance from the x axis to area dA</span>
                                </p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2vw' }}>Units of area moment of inertia</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left' }}>S.I Units: millimeter
                                    <span className='power' style={{ fontSize: '1.2vw', marginRight: '2px' }}>  4 </span>
                                    (mm
                                    <span className='power' style={{ fontSize: '1.2vw', marginRight: '2px' }}>  4</span>
                                    ).</p>
                                <br />
                                <br />
                                <div style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
                                    <hr style={{ width: '80%' }} />
                                </div>
                                <br />
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Load :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>External force or weight acts on a structure that can cause stress and strain in the structure.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Load is calculated by :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}><br />
                                    <span className='power' style={{ fontSize: '2vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300', marginRight: '5px', marginLeft: '8px' }}> F=σ⋅A</span>
                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.8vw' }}>Where:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '78%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '2vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300', marginRight: '5px' }}> F is the load applied.</span>
                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '78%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '2vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300', marginRight: '5px' }}> σ is the stress.</span>
                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '78%', justifyContent: 'left', fontSize: '1.8vw' }}>
                                    <span className='power' style={{ fontSize: '2vw', top: '0px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300', marginRight: '5px' }}>A is the cross-sectional area.</span>
                                </p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2vw' }}>Units of load</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left' }}>S.I Units: Newton (N).</p>
                                <br />
                                <br />
                                <br />
                                <br />
                                <div style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
                                    <hr style={{ width: '80%' }} />
                                </div>
                                <br />
                            </div>
                        </div>
                        <br />
                        <br />


                        <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left',fontWeight: '300', fontSize: '3.65vw', important: true, textTransform:'uppercase' }}>
                            Type of load acting in beams that cause deflection in beams :
                        </h3>
                        <br />

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>

                            <div>
                                <div style={{ display: 'flex', }}>
                                    <div>
                                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Point Load: </p>
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left' }}>Act over a very small area of a structural element but considered as a single point.</p>
                                    </div>
                                    <img src={PointLoadImg} alt='' style={{ width: '20vw' }} />
                                </div>
                                <br />
                                <br />
                                <div style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
                                    <hr style={{ width: '80%' }} />
                                </div>
                                <br />
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>	Distributed load  :</p>
                                <br />
                                <div style={{ display: 'flex', }}>
                                    <div>
                                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.7vw' }}>1.	Uniform distributed Loads</p>
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '75%', justifyContent: 'left' }}>Constant load along the length of the beam i.e. the beam experiences the same load from the start and end point of beams.</p>
                                    </div>
                                    <img src={UniformdistributedLoadImg} alt='' style={{ width: '20vw' }} />
                                </div>
                                <br />
                                <div style={{ display: 'flex', }}>
                                    <div>
                                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '1.7vw' }}>2.	Non-uniform distributed Loads</p>
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '75%', justifyContent: 'left' }}>Varying load along the length of the beam i.e. the beam experiences different loads from the start and end point of the beams.</p>
                                    </div>
                                    <img src={NonuniformdistributedLoadImg} alt='' style={{ width: '20vw' }} />

                                </div>
                                <br />
                            </div>
                        </div>
                        <br />
                        <br />
                        <div style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
                            <hr style={{ width: '80%' }} />
                        </div>
                        <br />
                        <br />

                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Support  :
                        </p>
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '2vw', important: true }}>
                            Essential components in a structure that provide stability and resistance to forces under various loads acting on it.</p>
                        <br />
                        <p className="calculator-info-blue-section-main-topic TopicOfDropdown" style={{ padding: '0px', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '2.5vw', }}>Type of supports:</p>
                        <br />

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div>
                                <div style={{ display: 'flex', }}>
                                    <div>
                                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Fixed support : </p>
                                        <br />
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left' }}>Restrict all translation in horizontal
                                            (x) and vertical
                                            (y) direction as well as rotational movement of the structural member.
                                        </p>
                                        <div style={{ marginTop: '15px' }}></div>
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left', display: 'block' }}>
                                            Fixed support produces reaction forces in horizontal reaction
                                            (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-1px' }}>x</span>)
                                            , vertical reaction
                                            (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y</span>)
                                            , and moment
                                            (M).
                                        </p>
                                        <div style={{ marginTop: '15px' }}></div>

                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left' }}>
                                            Fixed support is used in situation where complete rigidity is required such as cantilever beams, fixed-end beams, and frame structures.
                                        </p>
                                    </div>
                                    <img src={FixedImage} alt='' style={{ width: '24vw', height: '14vw', marginTop: '2vw' }} />

                                </div>
                                <br />
                                <br />
                                <div style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
                                    <hr style={{ width: '80%' }} />
                                </div>
                                <br />
                                <br />
                                <div style={{ display: 'flex', }}>
                                    <div>
                                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>	Pin support :</p>
                                        <br />
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left' }}>Restrict translation in both horizontal(x) and vertical(y) direction but allow rotation of structural member.
                                        </p>
                                        <div style={{ marginTop: '15px' }}></div>
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left', display: 'block' }}>
                                            Pin support produces reaction forces in both horizontal
                                            (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower' style={{ left: '-1px' }}>x</span>)
                                            , and vertical
                                            (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y</span>)
                                            reactions.
                                        </p>
                                        <div style={{ marginTop: '15px' }}></div>
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left' }}>
                                            Commonly used in bridges, trusses, and other structures that require rotation but restrict translational movement.
                                        </p>
                                    </div>
                                    <img src={PinnedImage} alt='' style={{ width: '19vw', height: '10vw', marginTop: '3vw' }} />

                                </div>
                                <br />
                                <br />
                                <div style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
                                    <hr style={{ width: '80%' }} />
                                </div>
                                <br />
                                <br />

                                <div style={{ display: 'flex', }}>
                                    <div>
                                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '80%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>	Roller support :</p>
                                        <br />
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left' }}>Allows rotation and translation in horizontal (x) direction but does not allow translation in vertical (y) direction.
                                        </p>
                                        <div style={{ marginTop: '15px' }}></div>
                                        <div style={{ marginTop: '15px' }}></div>
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left' }}>
                                            Produce only vertical reaction (R<span className='LowerPowerminus2px sectionmodulesSolutionLowerPower'>y</span>).
                                        </p>
                                        <div style={{ marginTop: '15px' }}></div>
                                        <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '76%', justifyContent: 'left' }}>
                                            Used in bridges and large structures which allows thermal expansion and contraction to distribute the load.
                                        </p>
                                    </div>
                                    <img src={RollerImage} alt='' style={{ width: '19vw', height: '10vw', marginTop: '3vw' }} />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </>
    )
}
