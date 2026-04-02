import React, { useState, useEffect, useRef } from 'react'
// Side Image
import Areaimg from '../../../images/Beam-Properties-Area-Side-Image.webp'

// Images
import SquareImg from '../../../images/section/square.png'
import RectImg from '../../../images/section/rectangle.png'
import HollowRectImg from '../../../images/section/hollow rect.png'
import TImg from '../../../images/section/t section.png'
import channel from '../../../images/section/c channel.png'
import IImg from '../../../images/section/i section.png'
import Angle from '../../../images/section/l angle.png'
import circle from '../../../images/section/circle.png'
import hollowcircle from '../../../images/section/hollow circle.png'

export default function SectionModulus() {
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
                <div className='container-fluid text-white bg-black py-4 align-items-center justify-content-center d-flex' style={{ maxWidth: '85%', margin: '0 auto', height: '70vh' }}>
                    <div className="row justify-content-evenly py-4 align-items-center" >
                        <div className='col-lg-6 col-md-12 mb-4 py-0 px-0 '>
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}  >Section Modulus:</h2>
                            <br ref={textRef} />
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere ${isVisible ? 'scrolled' : ''}`} >
                                Geometrical property of cross
                                section used to design beam or flexural member. Measure
                                section's ability to resist<br /> bending or flexural deformation.
                            </h2>
                            <br />
                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>

                        </div>


                        <div className="col-lg-4 col-md-12">
                            <img
                                src={Areaimg} // Replace with your image URL
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
                <p className='calculator-defination-section text-center first-important heading' style={{ fontSize: '2vw', color: '#000' }}>
                    Significance of section modulus
                </p>
                <h4 className='calculator-defination-section text-center second-important' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    It tells us about the strength of the beam, higher section modulus means high strength of the beam.
                </h4>
                <h4 className='calculator-defination-section text-center '>
                    Higher section modulus indicates greater resistance to bending, making it a crucial factor in designing structures.
                </h4>
                <br />
                <br />
                <p className='calculator-defination-section text-center  heading' style={{ fontSize: '2vw', color: '#000' }}>
                    <span style={{ fontFamily: 'none' }}>I</span>nput to calculate the section
                </p>
                <h4 className='calculator-defination-section text-center ' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    Dimensions of cross-sections
                </h4>
                <h4 className='calculator-defination-section text-center '>
                    Maximum distance from the neutral axis to the surface of the member.
                </h4>

                <br />
                <p  className='calculator-defination-section text-center  heading' style={{ fontSize: '2vw', color: '#000' }}>
                    Section modulus can be calculated by :
                </p>
                <br />
                <h4 className='calculator-defination-section text-center ' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    S = I/c
                </h4>
                <h4 className='calculator-defination-section text-center ' style={{ marginTop: '15px', fontSize: '1.2vw', color: '#1d1d1dbf', fontSize: '1.5vw' }}>
                    Where
                </h4>
                <h4 className='calculator-defination-section text-center ' style={{ marginTop: '10px', color: '#1d1d1dbf' }}>
                    <span style={{fontFamily:'none'}}>I</span> is the second moment of inertia (area moment of inertia)
                </h4>
                <h4 className='calculator-defination-section text-center ' style={{ color: '#1d1d1dbf' }}>
                    &
                </h4>
                <h4 className='calculator-defination-section text-center ' style={{ color: '#1d1d1dbf' }}>
                    C is the maximum distance from the neutral axis to the surface of the member.
                </h4>
                <br ref={targetRef} />
                {isActive ? (
                    <div></div>
                ) : (
                    <div>
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
                                    <img src={SquareImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />

                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        S
                                        <span className='LowerPower'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span>
                                            a
                                            <span className='power'>3</span>
                                            <hr />
                                            6
                                        </span>
                                    </p>
                                </td>
                                <td >
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Square</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={RectImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text' >
                                        S
                                        <span className='LowerPower'>
                                            x
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            b.d
                                            <span className='power'>2</span>
                                            <hr></hr>
                                            6
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>
                                            y
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            d.b
                                            <span className='power'>2</span>
                                            <hr></hr>
                                            6
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Rectangle</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={HollowRectImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text' style={{ textAlign: 'center' }}>
                                        S
                                        <span className='LowerPowerminus2px'>
                                            x
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            I
                                            <span className='LowerPower'>x</span>
                                            <hr></hr>
                                            Y
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text' style={{ textAlign: 'center' }}>
                                        S
                                        <span className='LowerPowerminus2px'>
                                            y
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            I
                                            <span className='LowerPower'>y</span>
                                            <hr></hr>
                                            X
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Hollow Rectangle</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={TImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text' >
                                        S
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I
                                            <span className='LowerPower'>x</span>
                                            <hr />
                                            d+t-Y
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>

                                        S
                                        <span className='LowerPower'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            I
                                            <span className='LowerPower'>y</span>
                                            <hr />
                                            X
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>T-Section</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={channel} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text' >
                                        S
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I
                                            <span className='LowerPower'>x</span>
                                            <hr />
                                            d+t-Y
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I                                    <span className='LowerPower'>y</span>
                                            <hr />
                                            X
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>C-Section</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={IImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I
                                            <span className='LowerPower'>x</span>
                                            <hr />
                                            Y
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I<span className='LowerPower'>y</span>

                                            <hr />
                                            X
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '><span style={{ fontFamily: 'none' }}>I</span>-Beam</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={Angle} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>X</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I
                                            <span className='LowerPower'>x</span>
                                            <hr />
                                            d-Y
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>

                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I<span className='LowerPower'>y</span>

                                            <hr />
                                            b-X
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '> L-Section</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={circle} className='structure-analysis-calculator-formula-dropdown-section-img inertiaSolidCircleimg' alt=''
                                        style={{
                                            width: '10vw',
                                            height: '10vw',
                                        }} />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px  formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>y</span>
                                        <span className='equalesto'>=</span>
                                        S
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I
                                            <span className='LowerPower'>x</span>

                                            <hr />
                                            Y
                                            <span className='LowerPower'>c</span>
                                        </span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            π.R
                                            <span className='power'>3</span>
                                            <hr />
                                            4
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Circle</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={hollowcircle} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px  formula-text'>
                                        S
                                        <span className='LowerPowerminus2px'>y</span>
                                        <span className='equalesto'>=</span>
                                        S
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            I
                                            <span className='LowerPower'>x</span>

                                            <hr />
                                            Y
                                            <span className='LowerPower'>c</span>
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Hollow Circle</p>
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
                    </div>
                )}
                <h3 className='calculator-defination-section text-center  heading' style={{ fontSize: '2vw', color: '#000' }}>
                    Units of section modulus
                </h3>
                <h4 className='calculator-defination-section text-center ' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    S.I Units of section modulus is mm
                    <span className='power'>3</span>.
                </h4>
                <h4 className='calculator-defination-section text-center ' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    Imperial units of section modulus is in
                    <span className='power'>3</span>.
                </h4>
                <br />
                <br />
            </section>
        </>
    )
}
