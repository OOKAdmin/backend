import { useState, useEffect } from "react";
import Navbar from "./Navbar/Navbar";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import BeamProperties from "./Calculators/BeamProperties/BeamProperties";
import Footer from "./Footer/Footer";
import Padeye from "./Calculators/Padeye/Padeye";
import BeamDeflection from "./Calculators/BeamDeflection/BeamDeflection";
import AboutUs from "./AboutUS/AboutUs";
import 'bootstrap/dist/css/bootstrap.min.css';
import Policy from "./policy/Policy";
import Plagiarism from "./plagiarism/plagiarism";
// import { unstable_HistoryRouter as HistoryRouter } from 'react-router-dom';
import Netforce from "./Calculators/Net force/Netforce";
import Blogs from "./blogs/blogs";
import BeamPropertiesBlog from "./blogs/BeamPropertiesBlog";
import BeamDeflectionBlog from "./blogs/BeamDeflectionBlog";
import NetforceBlog from "./blogs/NetforceBlog";
import PadeyeBlog from "./blogs/PadeyeBlog";
import PlagiarismCheckerBlog from "./blogs/plagarismblog";

// css
import './Css/AboutUS.css'
import './Css/BeamDeflection.css'
import './Css/BeamProperties.css'
import './Css/Navbar.css'
import './Css/NumberLine.css'
import './Css/Padeye.css'
import './Css/BeamProperties.css'

import FourPointRiggingCalculator from "./Calculators/FourPointRiggingCalculator/FourPointRiggingCalculator";
import FourPointRiggingwithSpreaderBarCalculator from "./Calculators/Four Point Rigging with Spreader Bar Calculator/FourPointRiggingwithSpreaderBarCalculator";
import LoginPage from "./LoginRegister/Login";
import RegisterPage from "./LoginRegister/Register";
import React from "react";
import HumanizerUI from "./Humanizer/Humanizer";
import AIDetectorUI from "./AIDetector/AIDetector";
function App() {
      const [user, setUser] = useState(null);

  // ✅ Load user on refresh
  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem("user"));
    if (storedUser) {
      setUser(storedUser);
    }
  }, []);
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>

          <Route path="/" element={<><Navbar /><BeamProperties /><Footer /></>} unstable_startTransition />
          <Route path="/BeamProperties" element={<><Navbar /><BeamProperties /><Footer /></>} unstable_startTransition />
          <Route path="/PadEye" element={<><Navbar /><Padeye /><Footer /></>} unstable_startTransition />
          <Route path="/BeamDeflection" element={<><Navbar /><BeamDeflection /><Footer /></>} unstable_startTransition />
          <Route path="/NetForce" element={<><Navbar /><Netforce /><Footer /></>} unstable_startTransition />
          <Route path="/FourPointRiggingCalculator" element={<><Navbar /><FourPointRiggingCalculator /><Footer /></>} unstable_startTransition />
          <Route path="/FourPointRiggingwithSpreaderBarCalculator" element={<><Navbar /><FourPointRiggingwithSpreaderBarCalculator /><Footer /></>} unstable_startTransition />

          <Route path="/AboutUs" element={<><Navbar /><AboutUs /><Footer /></>} unstable_startTransition />

          <Route path="/Policy" element={<><Navbar /><Policy /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs" element={<><Navbar /><Blogs /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/BeamPropertiesBlog" element={<><Navbar /><BeamPropertiesBlog /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/PadeyeBlog" element={<><Navbar /><PadeyeBlog /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/BeamDeflectionBlog" element={<><Navbar /><BeamDeflectionBlog /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/NetforceBlog" element={<><Navbar /><NetforceBlog /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/PlagiarismBlog" element={<><Navbar /><PlagiarismCheckerBlog /><Footer /></>} unstable_startTransition />

          <Route path="/Plagiarism" element={<><Navbar /><Plagiarism /><Footer /></>} unstable_startTransition />
          <Route path="/Humanizer" element={<><Navbar /><HumanizerUI /><Footer /></>} unstable_startTransition />
          <Route path="/AIDetector" element={<><Navbar /><AIDetectorUI /><Footer /></>} unstable_startTransition />

          {/* <Route path="/login" element={<><Navbar /><LoginPage setUser={setUser} /><Footer /></>} /> */}

{/* <Route 
  path="/login" 
  element={
    <>
      <Navbar />
      <div style={{ minHeight: "80vh" }}>
        <LoginPage setUser={setUser} />
      </div>
      <Footer />
    </>
  } 
/> */}

          {/* <Route path="/register" element={<><Navbar /><RegisterPage /><Footer /></>} /> */}

        <Route path="/login" element={<><Navbar /><LoginPage setUser={setUser} /><Footer /></>} />
        <Route path="/register" element={<><Navbar /><RegisterPage setUser={setUser} /><Footer /></>} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
