import React, { useState, useEffect, useRef } from 'react'

// Side Image
import Areaimg from '../../../images/Beam-Properties-Area-Side-Image.webp'

// Images
import SquareImg from '../../../images/torsional constant/square.png'
import Rectangle from '../../../images/torsional constant/rectangle.png'
import Hollowwalledrectangle from '../../../images/torsional constant/hollow rect.png'
import Isection from '../../../images/torsional constant/isection.jpg'
import Circle from '../../../images/torsional constant/circle.png'

// icons
import { LiaLessThanSolid } from "react-icons/lia";

export default function TorsionalConstant() {
    const [isActive, setIsActive] = useState(true);

    const toggleClass = () => {
        setIsActive(!isActive);
    };
    // black box tranaction
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
                <div className='container-fluid text-white bg-black py-4 align-items-center justify-content-center d-flex' style={{ maxWidth: '85%', margin: '0 auto', height: '70vh' }}>
                    <div className="row justify-content-evenly py-4 align-items-center" >
                        <div className='col-lg-6 col-md-12 mb-4 py-0 px-0 '>
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>Torsional Constant:</h2>
                            <br ref={textRef} />
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere ${isVisible ? 'scrolled' : ''}`} >
                                Geometrical property of bar cross section that describes
                                its resistance to torsional deformation during torsion.
                            </h2>
                            <br />
                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>
                        </div>

                        <div className="col-lg-4 col-md-12">
                            <img
                                src={Areaimg}
                                alt="Placeholder"
                                className="img-fluid"
                                style={{ borderRadius: '20px' }}
                                loading='lazy'
                                width="600"
                                height="400"
                            />
                        </div>
                    </div>
                </div>
                <br />
                <br />
                <br />
                <br />
                <h3 className='calculator-defination-section text-center first-important heading' style={{ fontSize: '2vw' }}>
                    Significance of torsional constant
                </h3>
                <h4 className='calculator-defination-section text-center second-important' style={{ marginTop: '15px', fontSize: '1.2vw' }}>
                    Crucial for designing shafts that transmit power in machinery.
                </h4>
                <h4 className='calculator-defination-section text-center ' style={{ marginTop: '10px', fontSize: '1.2vw' }}>
                    Knowing the torsional constant allows for the design of
                </h4>
                <h4 className='calculator-defination-section text-center ' style={{ fontSize: '1.2vw' }}>
                    shafts that can safely transfer torque without
                </h4>
                <h4 className='calculator-defination-section text-center ' style={{ fontSize: '1.2vw' }}>
                    excessive twisting or failure.
                </h4>
                <br />
                <h3 className='calculator-defination-section text-center  heading' style={{ fontSize: '2vw', color: '#000' }}>
                    Torsional constant can be calculated by :
                </h3>
                <h4 className='calculator-defination-section text-center ' style={{ marginTop: '15px', fontSize: '1.2vw' }}>
                    ϕ=TL/kg or K =TL/ ϕg
                </h4>
                <br />
                <br />
                <h3 className='calculator-defination-section text-center  heading' style={{ color: '#000', textTransform: 'uppercase' }}>
                    In this equation:
                </h3>
                <div className='d-flex flex-column justify-content-center  align-items-center'>
                    <h4 className='calculator-defination-section ' style={{ textAlign: 'left', width: '45%', color: '#1d1d1dbf', marginTop: '15px' }}>
                        ϕ — Angle of twist.
                    </h4>
                    <h4 className='calculator-defination-section ' style={{ textAlign: 'left', width: '45%', color: '#1d1d1dbf' }}>
                        T — Torque applied to the beam.
                    </h4>
                    <h4 className='calculator-defination-section ' style={{ textAlign: 'left', width: '45%', color: '#1d1d1dbf' }}>
                        L — Shaft length.
                    </h4>
                    <h4 className='calculator-defination-section ' style={{ textAlign: 'left', width: '45%', color: '#1d1d1dbf' }}>
                        G — Shear modulus of the shaft material.
                    </h4>
                    <h4 className='calculator-defination-section ' style={{ textAlign: 'left', width: '45%', color: '#1d1d1dbf' }}>
                        K — Beam torsional constant, the one we get with this calculator.
                    </h4>

                    <br />
                    <br />
                    <h4 className='calculator-defination-section text-center ' style={{ width: '50%', color: '#1d1d1dbf' }}>
                        The torsional constant in this formula has the same function as the polar moment (polar moment of inertia - describes a cross section resistance to torsion due to its shape or measure the strength (max. applicable torque) of shaft )circular beam, but we termed it here with the
                        K symbol to differentiate both.

                    </h4>
                </div>
                <br ref={targetRef} />

                {isActive ? (
                    <div></div>
                ) : (
                    <div>
                        <br />
                        <br />
                        <br />
                        <br />
                    </div>
                )}
                <div className={isActive ? 'mae-calculator-info custom-container py-5 ' : 'mae-calculator-info custom-container py-5 active'}>
                    <table className="table custom-table text-center align-middle">
                        <tbody>
                            <tr>
                                <td style={{ width: '30%' }}>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Cross Section Shape</p>
                                </td>
                                <td style={{ width: '40%' }}>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Formula</p>
                                </td>
                                <td style={{ width: '30%' }}>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Section Name</p>
                                </td>
                            </tr>
                            <tr>

                                <td >
                                    <img src={Circle} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' style={{ width: '10vw' }} />

                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        K
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            (π.r
                                            <span className='power'>4</span>
                                            )
                                            <hr />
                                            2
                                        </span>
                                    </p>
                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Circle</p>
                                </td>

                            </tr>

                            <tr>

                                <td >
                                    <img src={SquareImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' style={{ width: '10vw', height: '10vw' }} />

                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        K
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            9
                                            <hr />
                                            64
                                        </span>
                                        <span style={{ marginLeft: '8px' }}>a
                                            <span className='power'>4</span>
                                        </span>
                                    </p>
                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Square</p>
                                </td>

                            </tr>
                            <tr>

                                <td >
                                    <img src={Rectangle} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' style={{ width: '10vw', height: '10vw' }} />

                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        K
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            ab
                                            <span className='power'>3</span>
                                            <hr />
                                            3
                                        </span>
                                        <span className='equalesto'>−</span>

                                        <span style={{ textAlign: 'center' }}>
                                            0.21b
                                            <span className='power'>4</span>
                                            +0.0175
                                        </span>
                                        <span style={{ textAlign: 'center', marginLeft: '5px' }}>b
                                            <span className='power'>8</span>
                                            <hr />
                                            a
                                            <span className='power'>4</span>
                                        </span>
                                    </p>
                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Rectangle</p>
                                </td>

                            </tr>

                            <tr>

                                <td >
                                    <img src={Isection} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' style={{ width: '20vw', height: '20vw' }} />

                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        K
                                        <span className='equalesto'>=</span>
                                        2K
                                        <span className='LowerPowerminus2px torsinalConstantFormula'>1</span>
                                        +K
                                        <span className='LowerPower'>2</span>
                                        +2αD
                                        <span className='power'>4</span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        K<span className='LowerPowerminus2px torsinalConstantFormula'>1</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            (bt
                                            <span className='power'>3</span>
                                            )
                                            <hr />
                                            3
                                        </span>
                                        <span className='equalesto'>–</span>
                                        0.21t<span className='power'>4</span>
                                        <span className='equalesto'>+</span>
                                        0.0175
                                        <span style={{ textAlign: 'center', marginLeft: '5px' }}>
                                            t<span className='power'>8</span>
                                            <hr />
                                            b
                                            <span className='power'>4</span>
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        K<span className='LowerPower'>2</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            d.tw
                                            <span className='power'>3</span>
                                            <hr />
                                            3
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        D
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            (t+r)
                                            <span className='power'>2</span>
                                            +
                                            rtw
                                            +
                                            (tw
                                            <span className='power'>2</span>
                                            /
                                            4)
                                            {/* <span className='LowerPowerminus2px torsinalConstantFormula'>1</span> */}
                                            <hr />
                                            2r+t
                                        </span>

                                    </p>

                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        α
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            t<span className='LowerPowerminus2px torsinalConstantFormula'>1</span>
                                            <hr />
                                            t<span className='LowerPowerminus2px torsinalConstantFormula'>2</span>

                                        </span>
                                        (0.15
                                        <span className='equalesto'>+</span>
                                        0.1
                                        <span style={{ textAlign: 'center', marginRight: '5px' }}>
                                            r<hr />b
                                        </span >
                                        )
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        If t <LiaLessThanSolid style={{ fontSize: '1vw' }} /> tw, then
                                        <span style={{ textAlign: 'center', marginLeft: '10px' }}>
                                            t<span className='LowerPowerminus2px torsinalConstantFormula'>1</span>
                                            <hr />
                                            t<span className='LowerPowerminus2px torsinalConstantFormula'>2</span>

                                        </span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            t
                                            <hr />
                                            tw
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        If tw <LiaLessThanSolid style={{ fontSize: '1vw' }} /> t, then
                                        <span style={{ textAlign: 'center', marginLeft: '10px' }}>
                                            t<span className='LowerPowerminus2px torsinalConstantFormula'>2</span>
                                            <hr />
                                            t<span className='LowerPowerminus2px torsinalConstantFormula'>1</span>

                                        </span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            tw
                                            <hr />
                                            t
                                        </span>
                                    </p>
                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '><span style={{ fontFamily: 'none' }}>I</span>-Beam</p>
                                </td>

                            </tr>
                        </tbody>
                    </table>
                </div>

                {isActive ? (
                    <div></div>
                ) : (
                    <div>
                        <br />
                        <br />
                        <br />
                        <br />
                    </div>
                )}
                <h3 className='calculator-defination-section text-center first-important heading' style={{ fontSize: '2vw' }}>
                    Units of torsional constant
                </h3>
                <h4 className='calculator-defination-section text-center second-important' style={{ marginTop: '15px', fontSize: '1.2vw' }}>
                    S.I Units of torsional constant is mm
                    <span className='power'>4</span>
                    .
                </h4>
                <h4 className='calculator-defination-section text-center second-important' style={{ marginTop: '10px', fontSize: '1.2vw' }}>
                    Imperial units of torsional constant is in
                    <span className='power'>4</span>.
                </h4>
                <br />
                <br />
            </section>
        </>
    )
}
