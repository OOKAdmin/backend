import React, { useState, useEffect, useRef } from 'react'

import Areaimg from '../../../images/Beam-Properties-Area-Side-Image.webp'


export default function PrincipleAxis() {
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
    return (
        <>
            <section className='structure-analysis-calculator-formula-dropdown-section'>
                <div className='container-fluid text-white bg-black py-4 align-items-center justify-content-center d-flex' >
                    <div className="row justify-content-evenly py-4 align-items-center" >
                        <div className='col-lg-6 col-md-12 mb-4 py-0 px-0 '>
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}  >Principal axis:</h2>
                            <br ref={textRef} />
                            <h2 className={`gbp-h3 structure-analysis-calculator-information-h3 defination scrollfromhere ${isVisible ? 'scrolled' : ''}`} >Main axes of cross section or
                                members which are perpendicular and intersect each other at the center of the area or centroid.
                            </h2>
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
            </section>
        </>
    )
}
