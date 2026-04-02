import React, { useState, useEffect, useRef } from 'react';
import { Helmet } from 'react-helmet';
// css
import '../Css/AboutUS.css';
import 'swiper/css';
import 'swiper/css/navigation'; // If using Navigation module
import 'swiper/css/pagination'; // If using Pagination module
// CSS
import '../Css/BeamProperties.css'
import '../Css/BeamDeflection.css'
import '../Css/NumberLine.css'
import '../Css/AboutUS.css'
import '../Css/Navbar.css'
import '../Css/Padeye.css'



// modules
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Swiper, SwiperSlide } from 'swiper/react';
import { Navigation, Pagination, Autoplay } from 'swiper/modules';


// icons 
import { GiProgression } from "react-icons/gi";
import { GiDiceTarget } from "react-icons/gi";
import { GrVulnerability } from "react-icons/gr";
import { FaLinkedin } from "react-icons/fa";
import { GrLinkTop } from "react-icons/gr";


// images
// import Background from '../images/AboutUs-BackgroundImg.jpg';
import backgroundJPG from '../images/AboutUs-BackgroundImg.jpg';   // Replace with actual path
import backgroundWebP from '../images/AboutUs-BackgroundImg.webp'; // Replace with actual path

import DamanAnand from '../images/DamanAnand.png';
import HimanshuGupta from '../images/HimanshuGupta.png';
import Megha from '../images/Megha.png';
import RajeevYadav from '../images/rajeev.jpg';

export default function AboutUs() {
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  const [expanded, setExpanded] = useState(false);
  const [Vision, setVision] = useState(false);
  const [mission, setMission] = useState(false);
  const [capabilities, setCapabilities] = useState(false);
  const [OurTeam, setOurTeam] = useState(false);
  const [Member1, setMember1] = useState(false);
  const [Member2, setMember2] = useState(false);
  const [Member3, setMember3] = useState(false);
  const [WorkWithUs, setWorkWithUs] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 100 && !expanded) {
        setExpanded(true);
      }
      if (window.scrollY > 600 && !Vision) {
        setVision(true);
      }
      if (window.scrollY > 1000 && !mission) {
        setMission(true);
      }
      if (window.scrollY > 1600 && !capabilities) {
        setCapabilities(true);
      }
      if (window.scrollY > 2000 && !OurTeam) {
        setOurTeam(true);
      }
      if (window.scrollY > 2400 && !Member1) {
        setMember1(true);
      }
      if (window.scrollY > 2400 && !Member2) {
        setMember2(true);
      }
      if (window.scrollY > 2400 && !Member3) {
        setMember3(true);
      }
      if (window.scrollY > 2900 && !WorkWithUs) {
        setWorkWithUs(true);
      }
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [expanded, Vision, mission, capabilities, OurTeam, Member1, Member2, Member3, WorkWithUs]);

  const ABOUTUSIntroHeadingRef = useRef(null);
  const ABOUTUSIntroRef = useRef(null);
  const AboutusSpecialityVisionRef = useRef(null);
  const AboutusSpecialityVisionheadingRef = useRef(null);
  const AboutusSpecialityVisiondefinationRef = useRef(null);
  const AboutusSpecialityMissionRef = useRef(null);
  const AboutusSpecialityMissionheadingRef = useRef(null);
  const AboutusSpecialityMissiondefinationRef = useRef(null);
  const AboutusSpecialityCapabilitiesRef = useRef(null);
  const AboutusSpecialityCapabilitiesheadingRef = useRef(null);
  const AboutusSpecialityCapabilitiesdefinationRef = useRef(null);
  const AboutusOurTeamHeadingRef = useRef(null);
  const AboutusOurTeamRef = useRef(null);
  const AboutusMember1Ref = useRef(null);
  const AboutusMember2Ref = useRef(null);
  const AboutusMember3Ref = useRef(null);
  const AboutusMember4Ref = useRef(null);

  useEffect(() => {
    const handleScroll = () => {
      const ABOUTUSIntro = document.querySelector('.Aboutus-information');
      const AboutusSpecialityVision = document.querySelector('.vision');
      const AboutusSpecialityMission = document.querySelector('.mission');
      const AboutusSpecialityCapabilities = document.querySelector('.Capabilities');
      const AboutusSpecialityOurTeam = document.querySelector('.OurTeam');
      const AboutusSpecialityMember1 = document.querySelector('.Member-1');
      const AboutusSpecialityMember2 = document.querySelector('.Member-2');
      const AboutusSpecialityMember3 = document.querySelector('.Member-3');
      const AboutusSpecialityMember4 = document.querySelector('.Member-4');
      // const AboutusSpecialityOurTeam= document.querySelector('.OurTeam');
      const ABOUTUSIntroPosition = ABOUTUSIntro.getBoundingClientRect().top;
      const AboutusSpecialityVisionPosition = AboutusSpecialityVision.getBoundingClientRect().top;
      const AboutusSpecialityMissionPosition = AboutusSpecialityMission.getBoundingClientRect().top;
      const AboutusSpecialityCapabilitiesPosition = AboutusSpecialityCapabilities.getBoundingClientRect().top;
      const AboutusSpecialityOurTeamPosition = AboutusSpecialityOurTeam.getBoundingClientRect().top;
      const AboutusSpecialityMember1Position = AboutusSpecialityMember1.getBoundingClientRect().top;
      const AboutusSpecialityMember2Position = AboutusSpecialityMember2.getBoundingClientRect().top;
      const AboutusSpecialityMember3Position = AboutusSpecialityMember3.getBoundingClientRect().top;
      const AboutusSpecialityMember4Position = AboutusSpecialityMember4.getBoundingClientRect().top;
      const screenHeight = window.innerHeight;

      if (ABOUTUSIntroPosition < screenHeight) {
        ABOUTUSIntroRef.current.classList.add('expanded');
        ABOUTUSIntroHeadingRef.current.classList.add('expanded');
      }
      if (AboutusSpecialityVisionPosition < screenHeight) {
        AboutusSpecialityVisionRef.current.classList.add('expanded');
        AboutusSpecialityVisionheadingRef.current.classList.add('expanded');
        AboutusSpecialityVisiondefinationRef.current.classList.add('expanded');
      }
      if (AboutusSpecialityMissionPosition < screenHeight) {
        AboutusSpecialityMissionRef.current.classList.add('expanded');
        AboutusSpecialityMissionheadingRef.current.classList.add('expanded');
        AboutusSpecialityMissiondefinationRef.current.classList.add('expanded');
      }
      if (AboutusSpecialityCapabilitiesPosition < screenHeight) {
        AboutusSpecialityCapabilitiesRef.current.classList.add('expanded');
        AboutusSpecialityCapabilitiesheadingRef.current.classList.add('expanded');
        AboutusSpecialityCapabilitiesdefinationRef.current.classList.add('expanded');
      }
      if (AboutusSpecialityOurTeamPosition < screenHeight) {
        AboutusOurTeamHeadingRef.current.classList.add('expanded');
        AboutusOurTeamRef.current.classList.add('expanded');
      }
      if (AboutusSpecialityMember1Position < screenHeight) {
        AboutusMember1Ref.current.classList.add('expanded');
      }
      if (AboutusSpecialityMember2Position < screenHeight) {
        AboutusMember2Ref.current.classList.add('expanded');
      }
      if (AboutusSpecialityMember3Position < screenHeight) {
        AboutusMember3Ref.current.classList.add('expanded');
      }
      if (AboutusSpecialityMember4Position < screenHeight) {
        AboutusMember4Ref.current.classList.add('expanded');
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);
  return (
    <>

      <Helmet>
        <title>About OOK – Engineering Tools for Modern Solutions</title>
        <meta
          name="description"
          content="Discover OOK's mission to revolutionize engineering and education by offering precise, reliable, and innovative tools for complex calculations."
        />
        <link rel="canonical" href="https://www.ookcalculator.com/AboutUs" />

      </Helmet>


      <div className='Background-Black'></div>
      <section className='Aboutus-header-section'>
        <div className="position-relative">
          {/* Image section */}
          <div className="container text-left text-black position-absolute top-40 translate-middle"
            style={{
              left: '50%',
              zIndex: '10'
            }}
          >
            <h1 className="display-1 text-white Aboutus-header-section-backgroundtext-title" >ABOUT US</h1>
          </div>
          <div className="position-relative overflow-hidden" style={{ height: '85vh' }}>
            <div className="Black-overlay"></div>
            <picture>
              <source type="image/webp" srcSet={backgroundWebP} />
              <img
                loading="lazy"
                src={backgroundJPG}
                alt="Background"
                className="h-100"
                width="600"
                height="400"
                style={{
                  objectFit: 'cover',
                  objectPosition: 'center',
                  width: '100%'
                }}
  fetchpriority="high"
  decoding="async"
              />
            </picture>
          </div>

          {/* Text section */}
        </div>
      </section>

      <br />
      <br />
      <br />
      <br />

      <section className='calculator-definition-section text-center py-5 Aboutus-information' >
        <div className={`container information-section-of-BeamProperties calculator-defination-div AboutUs`}>

          <h3 className={`display-4 mb-4 Beam-properties-calculator-heading AboutUs `} ref={ABOUTUSIntroHeadingRef} style={{ fontWeight: '600' }}>
            <span>About US</span>
          </h3>
          <div className={` content calculator-defination-section-div AboutUs `} ref={ABOUTUSIntroRef}>
            <br />
            <h2 className=' first-important lead mb-4 Aboutus-information-information'>OOK was founded on 2nd Oct 2024 with a vision to revolutionize engineering and education by offering precise,<br /> reliable, and innovative tools for complex problem-solving. Our mission is to drive advancements in<br /> technology and learning, ensuring excellence and innovation in every aspect of <br />our work and transforming the future of these fields.</h2>
          </div>
        </div>
      </section>

      <section className='Aboutus-Speciality'>
        <div className={`calculator-defination-div AboutUsVision`} >
          <div className='vision'>
            <h3 className={`Beam-properties-calculator-heading AboutUsVision `} ref={AboutusSpecialityVisionRef}>

              <div className='-animation'>
                <motion.div
                  style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}
                >
                  <GiProgression className='Vision-Mission-Capabilities-icons' style={{ fontSize: '5vw', }} />
                </motion.div>
              </div>
            </h3>
            <div className='vision-text'>
              <h2 className={`text-center Beam-properties-calculator-heading AboutUsVision `} ref={AboutusSpecialityVisionheadingRef}>
                <span>VISION</span></h2>
              <div className={`calculator-defination-section-div AboutUsVision `} ref={AboutusSpecialityVisiondefinationRef}>
                <h2 className='AboutUsVision-h2 text-center'>Our vision is to provide groundbreaking and dependable engineering calculators, consistently advancing technological frontiers to empower global society. We strive to facilitate problem-solving and foster accessibility worldwide, revolutionizing engineering</h2>
              </div>
            </div>
          </div>
        </div>
        <br />
        <hr style={{ width: '40%' }} />
        <div className={`calculator-defination-div AboutUsVision`}>
          <div className='mission'>
            <h3 className={`Beam-properties-calculator-heading AboutUsVision `} ref={AboutusSpecialityMissionRef}>

              <div className='-animation'>
                <motion.div
                  style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}
                >
                  <GiDiceTarget className='Vision-Mission-Capabilities-icons' style={{ fontSize: '5vw', }} />
                </motion.div>
              </div>
            </h3>
            <div className='vision-text'>
              <h2 className={`text-center Beam-properties-calculator-heading AboutUsVision `} ref={AboutusSpecialityMissionheadingRef}>
                <span>MISSION</span></h2>
              <div className={`calculator-defination-section-div AboutUsVision `} ref={AboutusSpecialityMissiondefinationRef}>
                <h2 className='AboutUsVision-h2 text-center'>Our mission is to deliver precise and efficient results, enabling engineers and students to attain excellence. We are dedicated to providing innovative solutions that enhance productivity and facilitate success in every aspect of their work</h2>
              </div>
            </div>
          </div>
        </div>
        <br />
        <hr style={{ width: '40%' }} />
        <div className={`calculator-defination-div AboutUsVision Capabilities`}>
          <div className='mission'>
            <h3 className={`text-center Beam-properties-calculator-heading AboutUsVision `} ref={AboutusSpecialityCapabilitiesRef}>

              <div className='-animation'>
                <motion.div
                  style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}
                >
                  <GrVulnerability className='Vision-Mission-Capabilities-icons' style={{ fontSize: '5vw', }} />
                </motion.div>
              </div>
            </h3>
            <div className='vision-text'>
              <h2 className={`text-center Beam-properties-calculator-heading AboutUsVision `} ref={AboutusSpecialityCapabilitiesheadingRef}>
                <span>CAPABILITIES</span></h2>
              <div className={`calculator-defination-section-div AboutUsVision `} ref={AboutusSpecialityCapabilitiesdefinationRef}>
                <h2 className='AboutUsVision-h2 text-center'>Structure Design | Engineering Calculators | Web Designing | Consultancy</h2>
              </div>
            </div>
          </div>
        </div>
      </section>

      <br />
      <br />

      <section className='calculator-definition-section text-center py-5 Aboutus-Meetourteam'>
        <div className={`container information-section-of-BeamProperties calculator-defination-div AboutUs OurTeam`}>
          <h3 className={`display-4 mb-4 Beam-properties-calculator-heading AboutUs OurTeam `} ref={AboutusOurTeamHeadingRef} style={{ fontWeight: '600' }}>
            <span>OUR TEAM</span>
          </h3>
          <div className={`content calculator-defination-section-div AboutUs OurTeam `} ref={AboutusOurTeamRef}>

            <h2 className=' first-important lead mb-4 Aboutus-information-information'>Our team, featuring highly skilled engineers, software developers, and industry experts,<br /> is devoted to creating innovative and dependable engineering solutions.<br /> We are committed to addressing the diverse needs of both<br /> professionals and students, driving excellence and<br /> technological advancement in all our endeavors.</h2>
          </div>
        </div>
        <br />
        <br />
        <br />
        <div className='Aboutus-Meetourteam-card-main'>
          <Swiper
            modules={[Navigation, Pagination, Autoplay]}
            // navigation
            pagination={{ clickable: true }}
            autoplay={false}
            loop={true}
            spaceBetween={3}
            breakpoints={{
              // when window width is >= 640px
              640: {
                slidesPerView: 1,
                spaceBetween: 20,
              },
              // when window width is >= 768px
              768: {
                slidesPerView: 2,
                spaceBetween: 30,
              },
              // when window width is >= 1024px
              1120: {
                slidesPerView: 3,
                spaceBetween: 40,
              },
            }}
          >
            <SwiperSlide>
              <div className={`calculator-defination-div Member1 Member-1`}>
                <div className={`calculator-defination-section-div Member1 `} ref={AboutusMember1Ref}>
                  <h4> <Link style={{ float: 'right', marginRight: '15px', color: '#2275b4', width: "fit-content", height: '35px' }} to={`https://www.linkedin.com/in/daman-anand-33b395170`}><FaLinkedin></FaLinkedin></Link></h4>
                  <div className='Aboutus-Meetourteam-card Meghaanand'>
                    <div className='Aboutus-Meetourteam-card-img-main'>
                      <div className='Aboutus-Meetourteam-card-img'>
                        <img src={DamanAnand} alt='' />
                      </div>
                    </div>
                    <div>
                      <h2 className='Aboutus-Meetourteam-card-text-name' style={{ float: 'right', width: '100%', textAlign: 'center' }}>Daman Anand
                      </h2>
                    </div>
                    <h2 className='Aboutus-Meetourteam-card-text-information position' style={{ fontSize: '1.5vw', fontStyle: 'italic', fontWeight: '300' }}>Founder</h2>
                    <h2 className='Aboutus-Meetourteam-card-text-information'>With a Bachelor's degree in Mechanical & Automation Engineering, He is a skilled Design and Structural Engineer at Dimension Consultants Pte Ltd.<br /><div style={{ marginTop: '0.5vw' }} />His goal is to utilize research and marketing skills, along with providing calculators, to continually foster mutually beneficial relationships between the company and its users.</h2>

                  </div>
                </div>
              </div>
            </SwiperSlide>
            <SwiperSlide>
              <div className={`calculator-defination-div Member1 Member-2`}>
                <div className={`calculator-defination-section-div Member1 `} ref={AboutusMember2Ref}>
                  <h4> <Link style={{ float: 'right', marginRight: '15px', color: '#2275b4', width: "fit-content", height: '35px' }} to={`https://www.linkedin.com/in/himanshu-gupta-780121198`}><FaLinkedin></FaLinkedin></Link></h4>
                  <div className='Aboutus-Meetourteam-card Meghaanand'>
                    <div className='Aboutus-Meetourteam-card-img-main'>
                      <div className='Aboutus-Meetourteam-card-img'>
                        <img src={HimanshuGupta} alt='' />
                      </div>
                    </div>
                    <div>
                      <h2 className='Aboutus-Meetourteam-card-text-name' style={{ float: 'right', width: '100%', textAlign: 'center' }}>Himanshu Gupta</h2>
                    </div>
                    <h2 className='Aboutus-Meetourteam-card-text-information position' style={{ fontSize: '1.5vw', fontStyle: 'italic', fontWeight: '300' }}>Quality Assurance</h2>
                    <h2 className='Aboutus-Meetourteam-card-text-information'>Himanshu has a Bachelor's Degree in Mechanical & Automation Engineering, He is an experienced Mechanical Designer at Dimension Consultants Pte Ltd.<br /><div style={{ marginTop: '0.5vw' }} />Himanshu responsibilities include result verification and handling technical data. He brings exceptional technical skills and a positive attitude to support both our users and the development team.</h2>

                  </div>
                </div>
              </div>
            </SwiperSlide>
            <SwiperSlide>
              <div className={`calculator-defination-div Member1 Member-3`}>
                <div className={`calculator-defination-section-div Member1 `} ref={AboutusMember3Ref}>
                  <h4> <Link style={{ float: 'right', marginRight: '15px', color: '#2275b4', width: "fit-content", height: '35px', opacity: '0', cursor: 'auto' }} to={``}><FaLinkedin></FaLinkedin></Link></h4>
                  <div className='Aboutus-Meetourteam-card Meghaanand'>
                    <div className='Aboutus-Meetourteam-card-img-main'>
                      <div className='Aboutus-Meetourteam-card-img'>
                        <img src={Megha} alt='' style={{ transform: 'translate(0px, 14%)' }} />
                      </div>
                    </div>
                    <div>
                      <h2 className='Aboutus-Meetourteam-card-text-name' style={{ float: 'right', width: '100%', textAlign: 'center' }}>Megha</h2>
                    </div>
                    <h2 className='Aboutus-Meetourteam-card-text-information position' style={{ fontSize: '1.5vw', fontStyle: 'italic', fontWeight: '300' }}>Product Developer</h2>
                    <h2 className='Aboutus-Meetourteam-card-text-information'>With a strong background in programming, She is a versatile individual who brings extensive knowledge and experience.<br /><div style={{ marginTop: '0.5vw' }} />Megha is dedicated to the development and ongoing maintenance of various OOK Calculators. </h2>

                  </div>
                </div>
              </div>
            </SwiperSlide>
            <SwiperSlide>
              <div className={`calculator-defination-div Member1 Member-4`}>
                <div className={`calculator-defination-section-div Member1 `} ref={AboutusMember4Ref}>
                  <h4> <Link style={{ float: 'right', marginRight: '15px', color: '#2275b4', width: "fit-content", height: '35px', opacity: '0', cursor: 'auto' }} to={``}><FaLinkedin></FaLinkedin></Link></h4>
                  <div className='Aboutus-Meetourteam-card Meghaanand'>
                    <div className='Aboutus-Meetourteam-card-img-main'>
                      <div className='Aboutus-Meetourteam-card-img'>
                        <img src={RajeevYadav} alt='' />
                      </div>
                    </div>
                    <div>
                      <h2 className='Aboutus-Meetourteam-card-text-name' style={{ float: 'right', width: '100%', textAlign: 'center' }}>Rajeev Yadav</h2>
                    </div>
                    <h2 className='Aboutus-Meetourteam-card-text-information position' style={{ fontSize: '1.5vw', fontStyle: 'italic', fontWeight: '300' }}>Senior Product Developer</h2>
                    <h2 className='Aboutus-Meetourteam-card-text-information'>Rajeev Yadav is proficient in supporting web applications, managing server-side operations, and overseeing database management within the OOK Calculator project. </h2>

                  </div>
                </div>
              </div>
            </SwiperSlide>
          </Swiper>
        </div>
        <br />
        <br />
        <br />
        <section className='cse-header-top' >
          <Link smooth="true" duration={500} offset={-70} onClick={scrollToTop} aria-label="Scroll to top">
            <GrLinkTop className='' />
          </Link>
        </section>
      </section >
    </>
  )
}
