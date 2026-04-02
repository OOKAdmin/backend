import React, { useEffect, useState, useRef } from 'react';

//images
import SlingBlackboximg from '../../../images/Padeye-Sling-blackbox-side-image.jpeg'

export default function SlingsFile() {
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
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>Slings:</h3>
                            <br ref={textRef} />
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere  ${isVisible ? 'scrolled' : ''}`}>
                            Essential component designed to move large and heavy loads that would be extremely difficult or impossible to move manually.
                            </h3>
                            <br />
                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore  ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>

                        </div>

                        {/* Image Column */}
                        <div className="col-lg-4 col-md-12">
                            <img
                                src={SlingBlackboximg} // Replace with your image URL
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
                <div className={isActive ? 'mae-calculator-info PadeyeDropDown  ' : 'mae-calculator-info PadeyeDropDown   active'} style={{ border: '2px solid', borderRadius: '30px' }} ref={targetRef}>
                    <br />
                    <br />
                    <div>
                        <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Input parameter :
                        </h3>
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div style={{ width: '100%' }}>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Sling diameter (

                                    <span className='power' style={{ fontSize: '1.5vw', top: '-0.1vw', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300' }}>  D</span>
                                    <span className='LowerPower' style={{ color: '#000', fontWeight: '300' }}>sling </span>

                                    ) : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Sling diameter directly influences the strength and its load-bearing capacity</p>
                                <br />

                            </div>
                        </div>


                        <br />
                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Factor affecting sling selection :
                        </p>

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Weight of the Load : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Sling has a specific load-carrying capacity. Always ensure the selected sling’s capacity exceeds the load weight.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Sling angle :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>As the angle decreases form, the tension on sling increases, which could potentially exceed the sling's rated capacity and cause a failure. Sling angle should not be less than 30 deg.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Type of Lift :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Nature of the lift, such as vertical, choker, or basket lift, also affects sling selection. Each type has different load capacities and application methods suited to specific lifting scenarios.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Environmental Conditions :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Environmental factors, such as temperature, chemicals, and moisture, can impact the material properties of the sling.</p>
                                <br />
                            </div>
                        </div>

                        <br />
                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Sling Material :
                        </p>

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div style={{ width: '100%' }}>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Wire Rope Slings : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Suitable for heavy-duty applications, high strength, and resistance to abrasion and heat.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Chain Slings :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Durable, adjustable, ideal for high-temperature environments and rugged conditions.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Synthetic Slings (
                                    <span className='power' style={{ fontSize: '1.5vw', top: '1px', color: '#000', fontFamily: "'Kanit', sans-serif !important", fontWeight: '300' }}>  Nylon and Polyester</span>

                                    ) :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Lightweight, flexible, and non-conductive. Nylon slings absorb shock loads but are susceptible to chemical damage. Polyester slings are resistant to most chemicals and UV light.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Metal Mesh Slings :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>High temperature and abrasion resistance, good for handling abrasive materials.</p>
                                <br />
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </>
    )
}
