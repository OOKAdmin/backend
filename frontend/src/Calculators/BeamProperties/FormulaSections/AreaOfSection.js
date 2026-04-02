import React, { useState, useEffect, useRef } from 'react'

// Side Image
import Areaimg from '../../../images/Beam-Properties-Area-Side-Image.webp'

// Images
import SquareImg from '../../../images/Beam-Properties-Area-Square.png'
import RectangleImg from '../../../images/Beam-Properties-Rectangle.png'
import HollowReactangleImg from '../../../images/Beam-Properties-Area-Hollow-Rect.png'
import TeeSectionImg from '../../../images/Beam-Properties-Area-T-Section.png'
import ChannelImg from '../../../images/Beam-Properties-Area-C-Channel.png'
import IsectionImg from '../../../images/Beam-Properties-Area-I-Section.png'
import LsectionImg from '../../../images/Beam-Properties-Area-L-Angle.png'
import SolidCircleImg from '../../../images/Beam-Properties-Area-Circle.png'
import HollowcircleImg from '../../../images/Beam-Properties-Area-Hollow-Circle.png'
export default function AreaOfSection() {
    const [isActive, setIsActive] = useState(true);

    const toggleClass = () => {
        setIsActive(!isActive);
    };
    // Black box tranaction
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
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>Area of section(A):</h2>
                            <p ref={textRef} />
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere  ${isVisible ? 'scrolled' : ''}`}>
                                Total amount of space inside the section.
                            </h2>
                            <br />

                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore  ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>

                        </div>

                        {/* Image Column */}
                        <div className="col-lg-4 col-md-12">
                            <img
                                loading='lazy'
                                src={Areaimg}
                                alt="Placeholder"
                                className="img-fluid Areaimg"
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
                <h3 className='calculator-defination-section text-center first-important heading'>
                    Significance of area of sections
                </h3>
                <h4 className='calculator-defination-section defination text-center second-important'>
                    Used for calculations of stress, strain, and moment of inertia.
                </h4>
                <br />
                <br />
                <h3 className='calculator-defination-section text-center  heading' >
                    <span>I</span>nputs to calculate the area of section
                </h3>
                <h4 className='calculator-defination-section text-center ' >
                    Dimension of cross-section
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
                                <td className='tabletdwith30'>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Cross Section Shape</p>
                                </td>
                                <td className='tabletdwith40'>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Formula</p>
                                </td>
                                <td className='tabletdwith30'>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Section Name</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={SquareImg} className='structure-analysis-calculator-formula-dropdown-section-img SquareImg' alt="Square Diagram" width="600" height="400" loading='lazy' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>A = a<sup>2</sup></p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px'>Square</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={RectangleImg} className='structure-analysis-calculator-formula-dropdown-section-img RectangleImg' alt="Rectangle Diagram" width="600" height="400" loading='lazy'/>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>A = d.b</p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Rectangle</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={HollowReactangleImg} className='structure-analysis-calculator-formula-dropdown-section-img HollowReactangleImg' alt="Rectangle Diagram" width="600" height="400" loading='lazy'/>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>A
                                        <span className='equalesto'>=</span>
                                        (d.b) – (d<span className='LowerPower'>i </span> <span className='forthedot'>.</span>b<span className='LowerPower'>i</span>)
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Hollow Rectangle</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={TeeSectionImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Rectangle Diagram" width="600" height="400" loading='lazy'/>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>A
                                        =
                                        t.b
                                        +t
                                        <span className='LowerPowerminus2px'>w</span>
                                        .d</p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>T-Section</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={ChannelImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Rectangle Diagram" width="600" height="400" loading='lazy'/>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        A
                                        <span className='equalesto'>=</span>
                                        t.b+2t
                                        <span className='LowerPowerminus2px'>w</span>

                                        .d</p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>C-Section</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img src={IsectionImg} className='structure-analysis-calculator-formula-dropdown-section-img' alt="Rectangle Diagram" width="600" height="400" loading='lazy'/>
                                </td>
                                <td>

                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        A
                                        <span className='equalesto'>=</span>
                                        t
                                        <span className='LowerPowerminus2px'>w</span>

                                        .d+2t.b</p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '><span className='FontFamilyNone'>I</span>-Beam</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img className='structure-analysis-calculator-formula-dropdown-section-img LsectionImg' src={LsectionImg} alt='' width="600" height="400" loading='lazy'/>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        A
                                        <span className='equalesto'>=</span>
                                        t(b+d−t)</p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>L-Section</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img className='structure-analysis-calculator-formula-dropdown-section-img SolidCircleImg' src={SolidCircleImg} alt='' width="600" height="400" loading='lazy'/>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        A
                                        <span className='equalesto'>=</span>
                                        π.r
                                        <span className='power'>2</span>
                                    </p>
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px '>Circle</p>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <img className='structure-analysis-calculator-formula-dropdown-section-img HollowcircleImg' src={HollowcircleImg} alt=''  width="600" height="400" loading='lazy' />
                                </td>
                                <td>
                                    <p className='calculator-info-blue-section-main-topic margin-bottom-10px formula-text'>
                                        A
                                        <span className='equalesto'>=</span>
                                        π(r-r
                                        <span className='LowerPower'>i</span>
                                        )
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
                <p className='calculator-defination-section text-center first-important heading unitsheading' >
                    Units of area of cross-section
                </p>
                <h4 className='calculator-defination-section text-center  units' >
                    S.I Units of area of cross-section is mm
                    <span className='power'>2</span>

                </h4>
                <h4 className='calculator-defination-section text-center  units' >
                    Imperial Units of the area of cross-section is in
                    <span className='power'>2</span>
                </h4>
                <br />
                <br />
                <br />
                <br />
            </section>
        </>
    )
}
