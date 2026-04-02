import React, { useState, useEffect, useRef } from 'react';
import PadeyeBlackbox from '../../../images/PadEye-Blackbox-Side-image.png';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

export default function Howtocalculateforces() {
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
                        <div className="col-lg-8 col-md-12 mb-4 py-0 px-0">
                            <h3 className={`gbp-h3 structure-analysis-calculator-information-h3 ${isVisible ? 'scrolled' : ''}`}>How to calculate net force?</h3>
                            <br ref={textRef} />
                            <br />
                            <button className={`gbp-h3 structure-analysis-calculator-information-h3-button-Discovermore  ${isVisible ? 'scrolled' : ''}`} onClick={handleCombinedClick}>Discover more</button>

                        </div>

                        {/* Image Column */}
                        <div className="col-lg-2 col-md-12">
                            <img
                                src={PadeyeBlackbox} // Replace with your image URL
                                alt="Placeholder"
                                className="img-fluid"
                                style={{ borderRadius: '20px', opacity: '0' }}
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
                            How to calculate net force?
                        </h3>
                        <br />

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left' }}>The net force (commonly written as Fnet) can be mathematically expressed in the most basic situation as:</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', color: '#000' }}>
                                    F<sub>net</sub> = Σ F<sub>i</sub>
                                </p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left' }}>where each Fi is an individual force (with its direction considered). For one-dimensional problems, forces in the same direction add together, while those in opposite directions subtract from one another:</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', color: '#000' }}>
                                    F<sub>net</sub> = F<sub>1</sub> + F<sub>2</sub> + F<sub>3</sub> + … + F<sub>N</sub>
                                </p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left' }}>If acting along a straight line, assign positive and negative signs to indicate direction (e.g., right as positive, left as negative).</p>
                                <br />
                            </div>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
                            <div>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Forces at Angles and Vector Decomposition</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>When forces act at angles, a process called vector decomposition is required. For any force FF at an angle θ\theta to the horizontal:</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>F<sub>x</sub> = F · cos(θ)</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}> F<sub>y</sub> = F · sin(θ)</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>Here, Fx and Fy are the horizontal and vertical components, respectively. Sum all x- and y-components separately to get the total net force in each direction, then recombine using the Pythagorean theorem if needed:</p>
                                <br />
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>F<sub>net</sub> = √((F<sub>x,net</sub>)<sup>2</sup> + (F<sub>y,net</sub>)<sup>2</sup>)</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>
                                    Angle of net force: θ<sub>net</sub> = arctan(
                                    <span style={{ display: "inline-flex", flexDirection: "column", alignItems: "center" }}>
                                        <span style={{ borderBottom: "1px solid white", padding: "0 4px" }}>F<sub>y,net</sub></span>
                                        <span>F<sub>x,net</sub></span>
                                    </span>
                                    )
                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left' }}>This approach is fundamental for problems where forces do not align with the main axes (such as on inclined planes or where multiple forces with different angles are present).</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', display: 'flex' }}><span style={{ fontWeight: 'bold', width: '25%' }}>For example: </span> A 50 N force acts horizontally to the right, and a 30 N force acts vertically upward on an object. Find the net force and its direction.</p>

                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', display: 'block' }}>

                                    <span className="font-bold">Given:</span>
                                    <ul className="list-disc list-inside space-y-1">
                                        <li>Force 1 (F₁) = 50 N at 0°</li>
                                        <li>Force 2 (F₂) = 30 N at 90°</li>
                                    </ul>


                                    <span className="font-bold">Solution:</span>
                                    <ul className="list-disc list-inside space-y-2">
                                        <li>
                                            Fₓ = 50 cos(0°) + 30 cos(90°) = 50 + 0 = <span className="font-semibold">50 N</span>
                                        </li>
                                        <li>
                                            Fᵧ = 50 sin(0°) + 30 sin(90°) = 0 + 30 = <span className="font-semibold">30 N</span>
                                        </li>
                                        <li>
                                            Net Force = √(50² + 30²) = √(2500 + 900) = √3400 = <span className="font-semibold">58.3 N</span>
                                        </li>
                                        <li>
                                            Direction = tan⁻¹(30/50) = <span className="font-semibold">31.0° above horizontal</span>
                                        </li>
                                    </ul>


                                    <span className="text-xl font-bold">
                                        Answer: <span className="text-green-400">58.3 N at 31.0° above horizontal</span>
                                    </span>
                                </p>

                                <br />
                                <br />

                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Common Net Force Scenarios in Real Life</p>

                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', display: 'block' }}>
                                    <span className="text-2xl font-bold">
                                        Real-world problems frequently involve net force analysis:
                                    </span>
                                    <ul className="list-disc list-inside space-y-3">
                                        <li>
                                            <span className="font-bold">Tug of War:</span> Two teams pull a rope in opposite directions; the net
                                            force determines the rope's movement.
                                        </li>
                                        <li>
                                            <span className="font-bold">Box on a Surface:</span> Applied force, friction, gravity, and normal
                                            reaction force combine to give the net force dictating motion.
                                        </li>
                                        <li>
                                            <span className="font-bold">Inclined Plane:</span> Gravity splits into parallel and perpendicular
                                            components, friction acts opposite, and sometimes an additional applied
                                            force changes the overall net force.
                                        </li>
                                        <li>
                                            <span className="font-bold">Elevator Movement:</span> Forces include gravity, cable tension, and friction.
                                        </li>
                                        <li>
                                            <span className="font-bold">Object in Free Fall:</span> The only force is gravity (neglecting air
                                            resistance), so net force equals gravitational force.
                                        </li>
                                        <li>
                                            <span className="font-bold">Car under Braking:</span> Applied braking force (friction), air resistance,
                                            and sometimes downhill or uphill components.
                                        </li>
                                    </ul>
                                </p>


                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Frequently Asked Questions</p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', display: 'block' }}>
                                    <div className="space-y-4">
                                        <div>
                                            <p className="font-semibold">What happens when net force is zero?</p>
                                            <p>
                                                According to Newton's first law of motion, if an object is either at rest
                                                or moving in a straight line with constant velocity when the net force
                                                is zero.
                                            </p>
                                        </div>
                                        <div>
                                            <p className="font-semibold">How do I handle forces in 3D space?</p>
                                            <p>
                                                For three-dimensional problems, calculate net force components in x, y,
                                                and z directions, then find the magnitude using:
                                            </p>
                                            <p className="mt-1 font-mono text-green-400">Fₙₑₜ = √(Fx² + Fy² + Fz²)</p>
                                        </div>
                                        <div>
                                            <p className="font-semibold">Can net force be negative?</p>
                                            <p>
                                                Net force is a vector quantity with both magnitude and direction. While
                                                the magnitude cannot be negative, the direction can be represented as
                                                negative depending on the chosen coordinate system.
                                            </p>
                                        </div>
                                    </div>
                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', display: 'block' }}>
                                    Net force magnitude is always positive, but components can be negative depending on the chosen coordinate system and direction conventions.
                                </p>
                                <br/>
                                <p className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', display: 'block', fontSize: '2.2vw', important: true  }}>
                                    What's the difference between force and net force?
                                </p>
                                <p className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '85%', justifyContent: 'left', display: 'block' }}>
                                    Force refers to individual pushes or pulls, while net force is the vector sum of all forces acting on an object.
                                </p>


                            </div>
                        </div>
                        <br />
                        <br />
                    </div>

                </div>
            </section >
        </>
    )
}
