import React, { useState, useEffect, useRef } from 'react'
// Side Image
import Areaimg from '../../../images/Beam-Properties-Area-Side-Image.webp'

// Images
import SquareImg from '../../../images/centroid/square.png'
import ReactangleImg from '../../../images/centroid/rectangle.png'
import HollowReactangleImg from '../../../images/centroid/hollow rect.png'
import TsectionImg from '../../../images/centroid/t section.png'
import Cchannelimg from '../../../images/centroid/c channel.png'
import Isection from '../../../images/centroid/i section.png'
import Langle from '../../../images/centroid/l angle.png'
import Circle from '../../../images/centroid/circle.png'
import HollowCircle from '../../../images/centroid/hollow circle.png'
export default function Centroid() {
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
                <div className='container-fluid text-white bg-black py-4 align-items-center justify-content-center d-flex' >
                    <div className="row justify-content-evenly py-4 align-items-center" >
                        <div className='col-lg-6 col-md-12 mb-4 py-0 px-0 '>
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}  >Centroid:</h2>
                            <br ref={textRef} />
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere ${isVisible ? 'scrolled' : ''}`} >Center of mass for the geometric shape
                                or point within the section where the area could be
                                balanced without any rotation.
                            </h2>
                            <br />

                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>

                        </div>

                        <div className="col-lg-4 col-md-12">
                            <img
                                src={Areaimg} // Replace with your image URL
                                alt="Placeholder"
                                className="img-fluid Areaimg"
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
                <p className='calculator-defination-section text-center first-important heading' >
                    Significances of centroid
                </p>
                <br />
                <h4 className='calculator-defination-section text-center second-important ' >
                    Used to Calculate the moment of inertia.
                </h4>
                <h4 className='calculator-defination-section text-center Margin10x0'>
                    Centroid is crucial for designing stable structures.
                </h4>
                <h4 className='calculator-defination-section text-center' >
                    Determining the location of the neutral axis of section or member.
                </h4>
                <br />
                <br />
                <br />
                <p className='calculator-defination-section text-center heading' >
                   <span className='FontFamilyNone'>I</span>nputs to calculate centroid
                </p>
                <h4 className='calculator-defination-section text-center' >
                    Dimensions of cross-sections
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
                                        Y
                                        <span className='LowerPowerminus2px'>c</span>
                                        <span className='equalesto'>=</span>
                                        X
                                        <span className='LowerPower'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span>a<hr />2</span>
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
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        x
                                        <span className='LowerPower'>
                                            c
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            b
                                            <hr></hr>
                                            2
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        y
                                        <span className='LowerPowerminus2px'>
                                            c
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            d
                                            <hr></hr>
                                            2
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>Rectangle</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={HollowReactangleImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram" />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        x
                                        <span className='LowerPower'>
                                            c
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            b
                                            <hr></hr>
                                            2
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        y
                                        <span className='LowerPowerminus2px'>
                                            c
                                        </span>
                                        <span className='equalesto'>=</span>

                                        <span>
                                            d
                                            <hr></hr>
                                            2
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>Hollow Rectangle</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={TsectionImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram" />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        y
                                        <span className='LowerPowerminus2px'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }} >
                                            (b.t
                                            <span className='power'>2</span>
                                            +t
                                            <span className='LowerPowerminus2px'>w</span>
                                            .d(2t+d))
                                            <hr />
                                            2(t.b+t
                                            <span className='LowerPowerminus2px'>w</span>
                                            .d)
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        x
                                        <span className='LowerPower'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            b
                                            <hr />
                                            2
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>T-Section</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={Cchannelimg} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram" />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        y
                                        <span className='LowerPowerminus2px'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            <span>
                                                (b.t
                                                <span className='power'>2</span>
                                                +2t
                                                <span className='LowerPowerminus2px'>w</span>
                                                .d(2t+d))
                                            </span>
                                            <span>
                                                <hr />
                                            </span>
                                            <span>
                                                2(t.b+2t
                                                <span className='LowerPowerminus2px'>w</span>
                                                .d)
                                            </span>
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        x
                                        <span className='LowerPower'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            b
                                            <hr />
                                            2
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>C-Section</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={Isection} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram" />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        y
                                        <span className='LowerPowerminus2px'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            d
                                            <hr />
                                            2+t
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        x
                                        <span className='LowerPower'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span style={{ textAlign: 'center' }}>
                                            b
                                            <hr />
                                            2
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'><span style={{fontFamily:'none'}}>I</span>-Beam</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={Langle} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram" />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        y
                                        <span className='LowerPowerminus2px'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span>
                                            (d
                                            <span className='power'>2</span>
                                            +b.t−t
                                            <span className='power'>2</span>
                                            ) <hr /> (2.(b+d−t))
                                        </span>
                                    </p>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        x
                                        <span className='LowerPower'>c</span>
                                        <span className='equalesto'>=</span>
                                        <span>
                                            (b
                                            <span className='power'>2</span>
                                            +d.t−t
                                            <span className='power'>2</span>
                                            ) <hr /> (2.(b+d−t))
                                        </span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>L-Section</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={Circle} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram" />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px  formula-text'>
                                        Y
                                        <span className='LowerPowerminus2px'>c</span>
                                        <span className='equalesto'>=</span>
                                        X
                                        <span className='LowerPower'>c</span>
                                        <span className='equalesto'>=</span>
                                        R
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>Circle</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={HollowCircle} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Square Diagram"
                                        style={{
                                            width: '21vw',
                                            height: '18vw',
                                        }}
                                    />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        Y
                                        <span className='LowerPowerminus2px'>c</span>
                                        <span className='equalesto'>=</span>
                                        X
                                        <span className='LowerPower'>c</span>
                                        <span className='equalesto'>=</span>
                                        R
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>Hollow Circle</p>
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
                <p className='calculator-defination-section text-center first-important heading' style={{ fontSize: '2vw' }}>
                    Units of centroid
                </p>
                <h4 className='calculator-defination-section text-center second-important' style={{ marginTop: '15px', fontSize: '1.2vw' }}>
                    S.I Units of centroid is mm.
                </h4>
                <h4 className='calculator-defination-section text-center second-important' style={{ marginTop: '15px', fontSize: '1.2vw' }}>
                    Imperial of centroid is in.
                </h4>
                <br />
                <br />
                <br />
            </section>
        </>
    )
}
