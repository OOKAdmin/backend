import React, { useEffect, useState, useRef } from 'react';

//images
import PadeyeBlackbox from '../../../images/PadEye-Blackbox-Side-image.png'
import PadeyeFileInputParameters from './PadeyeFileInputParameters';
import PadeyeFileOutputparameters from './PadeyeFileOutputparameters';
//files
export default function PadeyeFile() {
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
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>Pad-Eye:</h3>
                            <br ref={textRef} />
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere  ${isVisible ? 'scrolled' : ''}`}>
                            Net force is the vector sum of all individual forces acting on an object. It determines whether the object remains at rest, moves uniformly, or accelerates. Calculating the net force is crucial for understanding and predicting an object’s motion, as governed by Newton’s Second Law of Motion (Fnet= ma).
                            </h3>
                            <br />
                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore  ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>

                        </div>

                        {/* Image Column */}
                        <div className="col-lg-4 col-md-12">
                            <img
                                src={PadeyeBlackbox} // Replace with your image URL
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
                        <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Factors affecting pad-eye selection are :
                        </h3>
                        <br />
                        <br />

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Load Capacity: </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>A critical factor in the selection and use of padeyes, as it determines how much weight the padeye can safely support.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Material Selection:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Padeye material must be crossive resistant, ductile, and have good strength and durability.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Shackle Compatibility:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>SWL (Safe Working Load) of the shackle should equal to the SWL of the padeye. Padeye hole diameter must match the shackle pin diameter, with a clearance not exceeding 3% of the pin diameter.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Weld Type and Size:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>The type and size of welds must be suitable to handle the load that will be applied to the padeye. Welds must be designed to distribute stress evenly and avoid creating weak points.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Lift Configuration:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Depend upon the type of lift, such as straight lifting or multi-point lifts, will affect the padeye design. Such as vertical padeyes are used for straight lifts, while angular designs may be needed for multi-point lifts to prevent lateral bending moments.</p>
                                <br />
                            </div>
                        </div>
                        <br />
                        <br />

                        <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', textTransform: 'uppercase', fontWeight: '300', fontSize: '3.65vw', important: true }}>
                            Applications of pad-eye :
                        </h3>
                        <br />

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Lifting and Rigging Operation: </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Padeyes are used for lifting heavier or complex loads. Designed to withstand the forces associated during lifting. </p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Towing & Mooring:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Padeyes are used for towing vessel or securing them at docks.</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Securing Cargo:</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Padeyes on decks serve as attachment points for cargo and preventing movement it during transportation and ensuring goods are transported safely.</p>
                                <br />
                            </div>
                        </div>
                        <br />
                        <br />
                        <PadeyeFileInputParameters />
                        <br />
                        <br />
                        <PadeyeFileOutputparameters />

                    </div>

                </div>
            </section>
        </>
    )
}
