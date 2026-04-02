import React, { useState, useEffect, useRef } from 'react';
import PadeyeBlackbox from '../../../images/PadEye-Blackbox-Side-image.png';

export default function WhatisNetForce() {
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

                <div className="container-fluid py-4 align-items-center justify-content-center d-flex" style={{ maxWidth: '85%', margin: '0 auto', height: '70vh' }}>
                    <div className="row justify-content-evenly py-4 align-items-center" >
                        {/* Text Column */}
                        <div className="col-lg-8 col-md-12 mb-4 py-0 px-0">
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>What is Net Force?</h3>
                            <br ref={textRef} />
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere  ${isVisible ? 'scrolled' : ''}`}>
                                Net force is the total vector sum of all forces that have been individually applied to an object. This function identifies the object’s state whether it stays still, moves uniformly or accelerates. It is important to calculate the net force so we understand as well as predict an object’s motion, as governed by Newton’s Second Law of Motion (Fnet= ma)
                            </h3>
                            <br />

                            <p className='gbp-h3 structure-analysis-calculator-information-h3' style={{ padding: '0', margin: 'auto', width: '100%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '2.65vw', important: true }}>
                                How Forces Interact:
                            </p>
                            <br />

                            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                                <div>
                                    <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '100%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>1.	Add Forces in the Same Direction:  </p>
                                    <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '100%', justifyContent: 'left' }}>When a couple of forces are going to act along the same line and in the same direction, we just add their magnitudes. </p>
                                    <br />
                                    <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '100%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>2.	Subtract Forces in Opposite Directions:</p>
                                    <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '100%', justifyContent: 'left' }}>When forces are acting fiendishly, their magnitudes are deducting. </p>
                                    <br />
                                    <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '100%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>3.	Combine Perpendicular Forces Using Vectors:</p>
                                    <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '100%', justifyContent: 'left' }}> If forces are at an angle, we can combine them by vector addition that can break down the forces into components and use Pythagorean theorem or trigonometry.</p>
                                    <br />
                                </div>
                            </div>
                        </div>
                        <div className="col-lg-2 col-md-12">
                            {/* <img
                                    src={PadeyeBlackbox} // Replace with your image URL
                                    alt="Placeholder"
                                    className="img-fluid"
                                    style={{ borderRadius: '20px' }}
                                /> */}
                        </div>
                    </div>


                </div>
                <br />
                <br />

                <br />
                <br />
            </section>
        </>
    )
}
