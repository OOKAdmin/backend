import React, { useState, useEffect, useRef } from 'react';
import { Helmet } from 'react-helmet';
import API_BASE from '../apiConfig';
// css
import './plagiarism.css';
// CSS
import '../Css/BeamProperties.css'
import '../Css/BeamDeflection.css'
import '../Css/NumberLine.css'
import '../Css/AboutUS.css'
import '../Css/Navbar.css'
import '../Css/Padeye.css'


// modules
import { Link } from 'react-router-dom';
import axios from 'axios';


// icons 
import { GrLinkTop } from "react-icons/gr";

// images
import AdvancedAIEnabledDetection from './plagarisim tool images/Advanced AI-Enabled Detection.png';
import InDepthRealTime from './plagarisim tool images/In-Depth & Real-Time Reports.png';
import MassiveDatabaseCoverage from './plagarisim tool images/Massive Database Coverage.png';
import PrivacyandDataProtection from './plagarisim tool images/Privacy and Data Protection.png';
import SimpleUserInterface from './plagarisim tool images/Simple User Interface.png';

import EducatorsandAcademicInstitutions from './plagarisim tool images/Educators and Academic Institutions.png';
import WritersandAuthors from './plagarisim tool images/Writers and Authors icon.png';
import ResearchersandScholars from './plagarisim tool images/Researchers and Scholars.png';
import BusinessandContentCreators from './plagarisim tool images/Business and Content Creators.png';

export default function Plagiarism() {

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  const [text, setText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/check_plagiarism`, { text });
      setAnalysis(res.data);
    } catch (err) {
      console.error(err);
      alert('An error occurred while fetching analysis.');
    } finally {
      setLoading(false);
    }
  };
  const [ballCount, setBallCount] = useState(0);
  const ballRefs = useRef([]);

  useEffect(() => {
    const spacingVH = 70;

    const calculateBallCount = () => {
      const pageHeight = document.documentElement.scrollHeight;
      const vh = window.innerHeight / 100;
      const spacingPx = spacingVH * vh; // convert spacingVH to px
      const count = Math.ceil(pageHeight / spacingPx); // cover entire page height
      setBallCount(count);
    };

    calculateBallCount();
    window.addEventListener('resize', calculateBallCount);
    window.addEventListener('scroll', calculateBallCount);

    const observer = new MutationObserver(calculateBallCount);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true });

    return () => {
      window.removeEventListener('resize', calculateBallCount);
      window.removeEventListener('scroll', calculateBallCount);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1 }
    );

    ballRefs.current.forEach(ball => ball && observer.observe(ball));

    return () => observer.disconnect();
  }, [ballCount]);


  const radius = 25;
  const stroke = 5;
  const normalizedRadius = radius - stroke / 2;
  const circumference = 2 * Math.PI * normalizedRadius;

  return (
    <>
      <Helmet>
        <title>Pro Plagiarism Checker – OOK | Deep Web Search</title>
        <meta
          name="description"
          content="Protect your intellectual property with OOK's professional Plagiarism Checker. Our deep-web scanning technology cross-references your text against a massive database of billions of websites and journals to provide a comprehensive similarity report with highlighted matches."
        />
        <link rel="canonical" href="https://www.ook-calculator.com/Plagiarism" />
      </Helmet>


      <div className="wave-section">
        <div className="content">
          <h1 className="Plagiarism-heading-title display-4" style={{ fontWeight: "600" }}>
            Plagiarism Checker
          </h1>
          <br />
          <p className='plagiarism-heading-text'>
            A completely free and accurate online plagiarism detector.<br />
            Simply copy and paste to detect copied content.
          </p>
        </div>
      </div>
      <div className="ball-section">
        {Array.from({ length: ballCount }, (_, i) => (
          <div
            key={i}
            ref={el => (ballRefs.current[i] = el)}
            className={`ball ball-${(i % 10) + 1}`}
            style={{
              position: 'absolute',
              top: `${i * 70}vh`,
              left: i % 2 === 0 ? '0%' : '97%',
              transform: 'translateX(-50%)',
              animationDelay: `${Math.random() * 5}s`,
            }}
          />
        ))}
      </div>
      <section style={{ background: 'white', margin: 'auto' }}>
        <br /><br /><br /><br /><br />
        <div className="plagiarismtoolsection container-fluid d-grid justify-content-center vh-100" style={{ width: '90%', gridTemplateColumns: '2fr 1fr' }}>
          <textarea
            className="w-90"
            style={{
              width: '100%',
              height: '90%',
              border: '2px solid black',
              borderRadius: '20px',
              padding: '10px',
              zIndex: '2'
            }}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter your text here"
            aria-label="Text to check for plagiarism"
          ></textarea>
          <div>
            <div className="container mt-4">
              <div className="d-flex flex-wrap justify-content-center gap-4">
                {[
                  { label: 'Plagiarism', color: '#f66', value: analysis?.total_percent || 0 },
                  { label: 'Unique', color: '#28a745', value: 100 - (analysis?.total_percent || 0) },
                  { label: 'Exact Match', color: '#dc3545', value: analysis?.exact_percent || 0 },
                  { label: 'Partial Match', color: '#ffc107', value: analysis?.partial_percent || 0 },
                ].map((item, index) => {
                  const value = item.value;
                  const offset = circumference - (value / 100) * circumference;
                  return (
                    <div key={index} className="text-center">
                      <svg height={radius * 2} width={radius * 2} style={{ display: "block", margin: "auto" }}>
                        <circle
                          stroke="#eee"
                          fill="transparent"
                          strokeWidth={stroke}
                          r={normalizedRadius}
                          cx={radius}
                          cy={radius}
                        />
                        <circle
                          stroke={item.color}
                          fill="transparent"
                          strokeWidth={stroke}
                          strokeLinecap="round"
                          strokeDasharray={`${circumference} ${circumference}`}
                          strokeDashoffset={offset}
                          r={normalizedRadius}
                          cx={radius}
                          cy={radius}
                          transform={`rotate(-90 ${radius} ${radius})`}
                          style={{ transition: "stroke-dashoffset 0.35s" }}
                        />
                        <text
                          x="50%"
                          y="50%"
                          dominantBaseline="middle"
                          textAnchor="middle"
                          fontSize="12"
                          fill="#333"
                        >
                          {value}%
                        </text>
                      </svg>
                      <div style={{ marginTop: '0.5rem', fontWeight: 'bold', fontSize: '0.9rem' }}>{item.label}</div>
                    </div>
                  );
                })}
              </div>
            </div>
            <br />
            <div
              className="plagiarismtoolresultslink w-90"
              style={{
                width: '100%',
                height: '50vh',
                border: '2px solid black',
                borderRadius: '20px',
                padding: '10px',
                zIndex: '2',
                marginLeft: '30px',
                position: 'relative',
                overflowY: 'auto',
                backgroundColor: '#fff',
              }}
            >
              {analysis && (
                <div className="mt-6">
                  {analysis.matches.map((item, idx) => (
                    <div key={idx} className="mb-6 rounded p-4">
                      <p className="font-semibold mb-2">📝 Sentence {idx + 1}: {item.sentence}</p>
                      <a
                        href={item.source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-700 font-semibold hover:underline"
                      >
                        View Source
                      </a>
                      <p className="text-sm text-gray-700 mt-1">{item.snippet}</p>
                      <p className="text-sm text-gray-500 mt-1">
                        Match Type: {item.match_type.toUpperCase()}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="container mt-4" style={{ marginLeft: '30px' }}>
              <button
                onClick={handleAnalyze}
                className="btn btn-primary"
                style={{ marginBottom: "20px" }}
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section style={{ background: '#fff' }}>
        <br />
        <div>
          <h2 style={{ width: '70%', margin: '0 auto', textAlign: 'center' }}>
            Boost your writing performance with our Plagiarism Checker
          </h2>
          <br /><br />
          <p style={{ width: '85%', margin: '0 auto', textAlign: 'center' }}>
            Your thoughts are unique—your writing should be as well. OOK's plagiarism detection system guarantees that your work demonstrates clarity, authenticity, and academic integrity. As a student, researcher, or content provider, our innovative technology makes it easy to assure originality and uphold credibility.
          </p>
        </div>
        <br /><br /><br /><br />
        <div className='pointsofplagarism'>
          <div className='pointsofplagarismdivs1'>
            <div className='pointsofplagarismdivsimg'><img src={AdvancedAIEnabledDetection} alt='' /></div>
            <div className='pointsofplagarismdivstext'>
              Advanced Plagiarism Detection: Our advanced Plagiarism Checker provides maximum precision by identifying paraphrased as well as direct plagiarism. It identifies contextual duplications, unsuitable citations, and repeated content with accuracy.
            </div>
          </div>
          <div className='pointsofplagarismdivs'>
            <div className='pointsofplagarismdivstext'>
              In-Depth & Real-Time Reports: Get a full and instant overview of the material through detailed real-time reports that provide information about plagiarized content, source matches, and the percentage of similarity in the written work. We offer detailed information which can be a support in editing in an on-going basis to keep the content original and in line with the publication standards.
            </div>
            <div className='pointsofplagarismdivsimg'><img src={InDepthRealTime} alt='' /></div>
          </div>
          <div className='pointsofplagarismdivs1'>
            <div className='pointsofplagarismdivsimg'><img src={MassiveDatabaseCoverage} alt='' /></div>
            <div className='pointsofplagarismdivstext'>
              Massive Database Coverage: We are constantly searching through millions of online sources and content such as blogs, news sites, academic journals and research articles, books, and plagiarism databases. We virtually eliminate the chances of duplicate findings.
            </div>
          </div>
          <div className='pointsofplagarismdivs'>
            <div className='pointsofplagarismdivstext'>
              Privacy and Data Protection: None of the data you provide us with will ever be uploaded, shared, or stored. Our safe platform does not allow random people to open the document you have uploaded. This feature is especially useful for professionals who process sensitive and other confidential documents.
            </div>
            <div className='pointsofplagarismdivsimg'><img src={PrivacyandDataProtection} alt='' /></div>
          </div>
          <div className='pointsofplagarismdivs1'>
            <div className='pointsofplagarismdivsimg'><img src={SimpleUserInterface} alt='' /></div>
            <div className='pointsofplagarismdivstext'>
              Simple User Interface: OOK Plagiarism Checker is designed to work with a user-friendly interface to provide a perfect and flawless user experience. You have the option of uploading, scanning for plagiarism and obtaining the report in a few seconds on the go, whether you are using a desktop, tablet or mobile phone. What’s more, you can upload your documents in a few clicks only, in order to run a quick check for copied parts - which is absolutely efficient and useful.
            </div>
          </div>
        </div>
      </section>

      <section style={{ background: '#fff' }}>
        <br /><br /><br /><br />
        <div>
          <h2 style={{ width: '70%', margin: '0 auto', textAlign: 'center' }}>
            Who employs our plagiarism checker?
          </h2>
        </div>
        <br /><br /><br /><br />
        <div className='pointsofplagarism'>
          <div className='pointsofplagarismdivs'>
            <div className='pointsofplagarismdivstext'>
              Educators and Academic Institutions: Foster academic integrity in your classroom and teach students the importance of source citation.
            </div>
            <div className='pointsofplagarismdivsimg'><img src={EducatorsandAcademicInstitutions} alt='' /></div>
          </div>
          <div className='pointsofplagarismdivs1'>
            <div className='pointsofplagarismdivsimg'><img src={WritersandAuthors} alt='' /></div>
            <div className='pointsofplagarismdivstext'>
              Writers and Authors: Protect your unique voice and steer clear of unintentional plagiarism by checking the originality of your work before it goes live.
            </div>
          </div>
          <div className='pointsofplagarismdivs'>
            <div className='pointsofplagarismdivstext'>
              Researchers and Scholars: Ensure the credibility of your research findings and prevent accidental plagiarism by comparing your work with existing studies.
            </div>
            <div className='pointsofplagarismdivsimg'><img src={ResearchersandScholars} alt='' /></div>
          </div>
          <div className='pointsofplagarismdivs1'>
            <div className='pointsofplagarismdivsimg'><img src={BusinessandContentCreators} alt='' /></div>
            <div className='pointsofplagarismdivstext'>
              Business and Content Creators: Keep your reputation intact and avoid copyright issues by confirming that your marketing materials, website content, and creative projects are truly original.
            </div>
          </div>
        </div>
        <section className='cse-header-top' >
          <Link smooth="true" duration={500} offset={-70} onClick={scrollToTop} aria-label="Scroll to top">
            <GrLinkTop className='' />
          </Link>
        </section>
      </section>
    </>
  );
}
