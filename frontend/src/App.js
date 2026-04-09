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

import LoginPage from "./LoginRegister/Login";
import RegisterPage from "./LoginRegister/Register";
import React from "react";
import HumanizerUI from "./Humanizer/Humanizer";
import AIDetectorUI from "./AIDetector/AIDetector";
import ForgotPasswordPage from "./LoginRegister/ForgotPassword";
import GrammarChecker from "./GrammarChecker/GrammarChecker";
import ParaphrasingTool from "./ParaphrasingTool/ParaphrasingTool";
import Summarizer from "./Summarizer/Summarizer";
import { GoogleOAuthProvider } from "@react-oauth/google";

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
    <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}>
    <div className="App">
      <BrowserRouter>
        <Routes>

          <Route path="/" element={<><Navbar user={user} setUser={setUser}/><BeamProperties /><Footer /></>} unstable_startTransition />
          <Route path="/BeamProperties" element={<><Navbar user={user} setUser={setUser}/><BeamProperties /><Footer /></>} unstable_startTransition />
          <Route path="/PadEye" element={<><Navbar user={user} setUser={setUser}/><Padeye /><Footer /></>} unstable_startTransition />
          <Route path="/BeamDeflection" element={<><Navbar user={user} setUser={setUser}/><BeamDeflection /><Footer /></>} unstable_startTransition />
          <Route path="/NetForce" element={<><Navbar user={user} setUser={setUser}/><Netforce /><Footer /></>} unstable_startTransition />
          <Route path="/AboutUs" element={<><Navbar user={user} setUser={setUser}/><AboutUs /><Footer /></>} unstable_startTransition />

          <Route path="/Policy" element={<><Navbar user={user} setUser={setUser}/><Policy /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs" element={<><Navbar user={user} setUser={setUser}/><Blogs /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/BeamPropertiesBlog" element={<><Navbar user={user} setUser={setUser}/><BeamPropertiesBlog /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/PadeyeBlog" element={<><Navbar user={user} setUser={setUser}/><PadeyeBlog /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/BeamDeflectionBlog" element={<><Navbar user={user} setUser={setUser}/><BeamDeflectionBlog /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/NetforceBlog" element={<><Navbar user={user} setUser={setUser}/><NetforceBlog /><Footer /></>} unstable_startTransition />
          <Route path="/Blogs/PlagiarismBlog" element={<><Navbar user={user} setUser={setUser}/><PlagiarismCheckerBlog /><Footer /></>} unstable_startTransition />

          <Route path="/Plagiarism" element={<><Navbar user={user} setUser={setUser}/><Plagiarism /><Footer /></>} unstable_startTransition />
          <Route path="/Humanizer" element={<><Navbar user={user} setUser={setUser}/><HumanizerUI /><Footer /></>} unstable_startTransition />
          <Route path="/AIDetector" element={<><Navbar user={user} setUser={setUser}/><AIDetectorUI /><Footer /></>} unstable_startTransition />
          <Route path="/GrammarChecker" element={<><Navbar user={user} setUser={setUser}/><GrammarChecker /><Footer /></>} unstable_startTransition />
          <Route path="/Paraphraser" element={<><Navbar user={user} setUser={setUser}/><ParaphrasingTool /><Footer /></>} unstable_startTransition />
          <Route path="/Summarizer" element={<><Navbar user={user} setUser={setUser}/><Summarizer /><Footer /></>} unstable_startTransition />



        <Route path="/login" element={<><Navbar user={user} setUser={setUser}/><LoginPage setUser={setUser} /><Footer /></>} />
        <Route path="/register" element={<><Navbar user={user} setUser={setUser}/><RegisterPage setUser={setUser} /><Footer /></>} />
        <Route path="/forgot-password" element={<><Navbar user={user} setUser={setUser}/><ForgotPasswordPage /><Footer /></>} />
        </Routes>
      </BrowserRouter>
    </div>
    </GoogleOAuthProvider>
  );
}

export default App;
