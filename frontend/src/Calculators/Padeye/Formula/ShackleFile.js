import React, { useEffect, useState, useRef } from 'react';

//images
import ShackelBlackBoxImg from '../../../images/Padeye-shackle-blackbox-side-image.png'
import InputParametersOfShackle from './InputParametersOfShackle';
export default function ShackleFile() {
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
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>Shackle:</h3>
                            <br ref={textRef} />
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere  ${isVisible ? 'scrolled' : ''}`}>
                            U-shaped, load-bearing connecting device designed to be strong, durable, and capable of withstanding high stress and loads. Shackles are typically made of metal, and they come in a variety of shapes and sizes.
                            </h3>
                            <br />

                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore  ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>

                        </div>

                        {/* Image Column */}
                        <div className="col-lg-4 col-md-12">
                            <img
                                src={ShackelBlackBoxImg} // Replace with your image URL
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
                <div className={isActive ? 'mae-calculator-info PadeyeDropDown ' : 'mae-calculator-info PadeyeDropDown  active'} style={{ border: '2px solid', borderRadius: '30px' }} ref={targetRef}>
                    <br />
                    <br />
                    <br />
                    <div>
                        <InputParametersOfShackle />
                        <br />
                        <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Shackles are used in a wide variety of applications, including :
                        </h3>
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Lifting and rigging : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Shackles are used to lift and move heavy objects. They are typically used with other lifting equipment, such as cranes and hoists.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Towing :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Shackles are used to tow vehicles and other objects. They are typically used in conjunction with chains or tow straps.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Securing :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Shackles are used to secure objects in place. They are typically used to secure cargo, equipment, and livestock.</p>
                                <br />
                            </div>
                        </div>


                        <br />
                        <p className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Choosing the Right Shackle :
                        </p>
                        <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>
                            When choosing a shackle, it is essential to consider the following factors:
                        </p>

                        <br />
                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Load capacity : </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Shackle swl (Safe Working Load) should be higher than the weight of the load.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Type of load :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Shackle must be compatible with the type of load. For example, a chain shackle should not be used to tow a vehicle.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Environmental Conditions :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Shackle must withstand the environment in which it will be used. For example, a shackle used in a marine environment must be made of stainless steel.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Price :</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Shackles can range from a few dollars to hundreds of dollars. It is vital to choose a shackle that fits your budget.</p>
                                <br />
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </>
    )
}
