import React, { useState, useEffect, useRef } from 'react'
// Side Image
import Areaimg from '../../../images/Beam-Properties-Area-Side-Image.webp'

// Images
import SquareImg from '../../../images/section/square.png'
import ReactangleImg from '../../../images/section/rectangle.png'
import HollowReactangleImg from '../../../images/section/hollow rect.png'
import TeeSectionImg from '../../../images/section/t section.png'
import ChannelImg from '../../../images/section/c channel.png'
import IsectionImg from '../../../images/section/i section.png'
import LsectionImg from '../../../images/section/l angle.png'
import SolidCircleImg from '../../../images/section/circle.png'
import HollowcircleImg from '../../../images/section/hollow circle.png'

export default function AreaMomentsOfInertia() {
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
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`} style={{ lineHeight: 'initial' }}>Area Moments of Inertia (Ix, Iy):</h2>
                            <br ref={textRef} />
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere ${isVisible ? 'scrolled' : ''}`} >
                                Geometrical property that reflects how the area of a cross-section is distributed relative to a particular axis and measure of cross-section resistance to bending due to its shape.
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
                   Significance of area moment of inertia
                </h3>
                <h4 className='calculator-defination-section text-center second-important' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    Resistance of an area against the applied moment
                </h4>
                <h4 className='calculator-defination-section text-center' style={{ fontSize: '1.2vw', }}>
                    (bending or twisting moment) about an axis.
                </h4>
                <br />
                <br />
                <h3 className='calculator-defination-section text-center heading' style={{ fontSize: '2vw', color: '#000' }}>
                    <span style={{fontFamily:'none'}}>I</span>nputs to calculate area moment of inertia
                </h3>
                <h4 className='calculator-defination-section text-center' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    Dimensions of cross-sections
                </h4>
                <br />
                <br />
                <h3 className='calculator-defination-section text-center heading' style={{ fontSize: '2vw', color: '#000' }}>
                    Moment of inertia can be calculated by :
                </h3>
                <br />
                <h4 className='calculator-defination-section text-center' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    I
                    <span className='LowerPower'>x</span>
                    = ∫y
                    <span className='power'>2</span>
                    dA & I
                    <span className='LowerPower'>y</span>
                    = ∫x
                    <span className='power'>2</span>
                    dA
                </h4>
                <h3 className='calculator-defination-section text-center' style={{ marginTop: '15px', color: '#1d1d1dbf', fontSize: '1.5vw' }}>
                    Where
                </h3>
                <h4 className='calculator-defination-section text-center' style={{ marginTop: '10px', color: '#1d1d1dbf', fontSize: '1.2vw', }}>
                    y = distance from the x-axis to area dA
                </h4>
                <h4 className='calculator-defination-section text-center' style={{ color: '#1d1d1dbf', fontSize: '1.2vw', }}>
                    x = distance from the y-axis to area dA
                </h4>
                
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
                <br ref={targetRef} />
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
                                <td>
                                    <img src={SquareImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram" />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        I
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        I
                                        <span className='LowerPower'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span>a
                                            <span className='power'>4</span>
                                            <hr />
                                            12
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>Square</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={ReactangleImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram" />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text' >
                                        I
                                        <span className='LowerPower'>
                                            x
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            b.d
                                            <span className='power'>3</span>
                                            <hr></hr>
                                            12
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        I
                                        <span className='LowerPowerminus2px'>
                                            y
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            d.b
                                            <span className='power'>3</span>
                                            <hr></hr>
                                            12
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>Rectangle</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={HollowReactangleImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text' style={{ textAlign: 'center' }}>
                                        I
                                        <span className='LowerPower'>
                                            x
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            (bd
                                            <span className='power'>3</span>
                                            <span className='minus'>-</span>
                                            b
                                            <span className='LowerPower'>
                                                i
                                            </span>
                                            d
                                            <span className='LowerPower'>
                                                i
                                            </span>
                                            <span className='power'>3</span>)
                                            <hr></hr>
                                            12
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text' style={{ textAlign: 'center' }}>
                                        I
                                        <span className='LowerPower'>
                                            y
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            (d.b
                                            <span className='power'>3</span>
                                            <span className='minus'>-</span>
                                            d
                                            <span className='LowerPower'>
                                                i
                                            </span>
                                            .b
                                            <span className='LowerPower'>
                                                i
                                            </span>
                                            <span className='power'>3</span>)
                                            <hr></hr>
                                            12
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Hollow Rectangle</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={TeeSectionImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text '>
                                        I
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            (b(d+t)
                                            <span className='power'>3</span>
                                            -
                                            d
                                            <span className='power'>3</span>
                                            (b-t
                                            <span className='LowerPower'>w</span>
                                            ))
                                            <hr />
                                            3
                                        </span>
                                        <span style={{ padding: '0 5px', textAlign: 'center' }}>                                        -
                                            A(d+t-y
                                            <span className='LowerPower'>c</span>

                                            )
                                            <span className='power'>2</span>
                                        </span>
                                    </p>

                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>

                                        I
                                        <span className='LowerPower'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            (t.b
                                            <span className='power'>3</span>
                                            +d.t
                                            <span className='LowerPower'>w</span>
                                            <span className='power'>2</span>
                                            )
                                            <hr />
                                            12
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>T-Section</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={ChannelImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text padding-margin-bottom-0' >
                                        I
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            (b(d+t)
                                            <span className='power'>3</span>
                                            -
                                            d
                                            <span className='power'>3</span>
                                            (b-2t
                                            <span className='LowerPower'>w</span>
                                            ))
                                            <hr />
                                            3
                                        </span>
                                        <span style={{ padding: '0 5px', textAlign: 'center' }}>-
                                            A(d+t-y
                                            <span className='LowerPower'>c</span>

                                            )
                                            <span className='power'>2</span>
                                        </span>
                                    </p>

                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        I
                                        <span className='LowerPowerminus2px'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            ((d+t)b
                                            <span className='power'>3</span>
                                            –
                                            d(b-2t
                                            <span className='LowerPower'>w</span>
                                            )
                                            <span className='power'>3</span>
                                            )
                                            <hr />
                                            12
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>C-Section</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={IsectionImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        I
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            (b(d+2t)
                                            <span className='power'>3</span>
                                            -
                                            (b-t
                                            <span className='LowerPower'>w</span>
                                            )d
                                            <span className='power'>3</span>)
                                            <hr />
                                            12
                                        </span>
                                    </p>
                                    <br />

                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        I
                                        <span className='LowerPower'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{
                                            display: 'flex',
                                            justifyContent: 'center',
                                            alignItems: 'center',
                                        }}>
                                            <span style={{ textAlign: 'center' }}>
                                                (b
                                                <span className='power'>3</span>
                                                .t)
                                                <hr />
                                                6
                                            </span>
                                            <span style={{ padding: '0 5px' }}>+</span>
                                            <span style={{ textAlign: 'center' }}>
                                                (t
                                                <span className='LowerPower'>w</span>
                                                <span className='power'>3</span>
                                                .t)
                                                <hr />
                                                12
                                            </span>
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '><span style={{fontFamily:'none'}}>I</span>-Beam</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={LsectionImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        I
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            (b.d
                                            <span className='power'>3</span>
                                            -(b-t)(d-t)
                                            <span className='power'>3</span>
                                            )
                                            <hr />
                                            3
                                        </span>
                                        <span style={{ padding: '0 5px', textAlign: 'center' }}>
                                            – A(d-y
                                            <span className='LowerPower'>c</span>
                                            )
                                            <span className='power'>2</span>
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        I
                                        <span className='LowerPower'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            (d.b
                                            <span className='power'>3</span>
                                            -(d-t)(b-t)
                                            <span className='power'>3</span>
                                            )
                                            <hr />
                                            3
                                        </span>
                                        <span style={{ padding: '0 5px', textAlign: 'center' }}>
                                            – A(d-x
                                            <span className='LowerPower'>c</span>
                                            )
                                            <span className='power'>2</span>
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '> L-Section</p>
                                </td>

                            </tr>

                            <tr>
                                <td>
                                    <img src={SolidCircleImg} className='structure-analysis-calculator-formula-dropdown-section-img inertiaSolidCircleimg' alt=''
                                        style={{
                                            width: '10vw',
                                            height: '10vw',
                                        }} />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px  formula-text'>
                                        I
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        I
                                        <span className='LowerPower'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            π.R
                                            <span className='power'>4</span>
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
                                    <img src={HollowcircleImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt='' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        I
                                        <span className='LowerPowerminus2px'>x</span>
                                        <span className='equalesto'>=</span>
                                        I
                                        <span className='LowerPower'>y</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            π(R
                                            <span className='power'>4</span>
                                            –
                                            R
                                            <span className='LowerPowerminus2px'>i</span>
                                            <span className='power'>4</span>
                                            )
                                            <hr />
                                            4
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
                <h3 className='calculator-defination-section text-center heading' style={{ fontSize: '2vw', color: '#000' }}>
                    Units of area moment of inertia
                </h3>
                <h4 className='calculator-defination-section text-center' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    S.I Units of area moment of inertia is mm
                    <span className='power'>4</span>.
                </h4>
                <h4 className='calculator-defination-section text-center' style={{ marginTop: '15px', fontSize: '1.2vw', }}>
                    Imperial units of area moment of inertia is in
                    <span className='power'>4</span>.
                </h4>
                <br />
                <br />
            </section>
        </>
    )
}
