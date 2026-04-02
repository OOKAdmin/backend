import React, { useState } from "react";
import { Link } from 'react-router-dom';
import { GrLinkTop } from "react-icons/gr";
import { IoIosArrowDown } from "react-icons/io";
import { Helmet } from "react-helmet";
import './Policy.css'

export default function Policy() {
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };
  const [isActive, setIsActive] = useState(false);

  const toggleClass = () => {
    setIsActive((prev) => !prev);
  };
  return (
    <>
     <Helmet>
        <title>Privacy Policy - OOK Calculator</title>
        <meta
          name="description"
          content="Read the privacy policy for Ook Calculator, protecting your data and privacy."
        />
        <link rel="canonical" href="https://www.ookcalculator.com/Policy" />
      </Helmet>
      <div className='Background-Black'></div>
      <section className='background-white'>
        <div>
          <br />
          <br />
          <br />
          <br />
          <br />
          <br />
          <br />
        </div>
        <h1 className='text-center mb-3'>Privacy Policy</h1>
        <h6 className='text-center mb-3'>Your privacy is important to us</h6>
        <p className='text-center'>
          <span style={{ width: '60%' }}>
            This Privacy Policy sets out OOK's policy to respect and protect your privacy regarding any information<br />
            we may collect while operating our website.
          </span>
        </p>
        <p className='text-center'>Effective Date: 31-01-2025</p>


        <div className='InformationDivOfPrivacyPolicy'>
          <hr className="Policy-hr" ></hr>
          <h4 className=' mb-3'>Overview</h4>
          <p>
            This OOK's policy to respect your privacy regarding any information we may collect while operating our website. This Privacy Policy applies to <a href="https://ookcalculator.com/" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>"ookcalculator.com"</a>. We respect your privacy and are committed to protecting personally identifiable information you may provide us through the Website. We have adopted this privacy policy to explain what information may be collected on our Website, how we use this information, and under what circumstances we may disclose the information to third parties. This Privacy Policy applies only to information we collect through the Website and does not apply to our collection of information from other sources.
          </p>
          <p>
            This Privacy Policy, together with the Terms of service posted on our Website, set forth the general rules and policies governing your use of our Website. Depending on your activities when visiting our Website, you may be required to agree to additional terms of service.
          </p>
          <br />
          <br />


          <div>
            <h4 className='text-center mb-3' onClick={toggleClass} style={{ cursor: 'pointer' }}>Contents <IoIosArrowDown /></h4>
            <div className={isActive ? "InformationDivOfPrivacyPolicyDropDown InformationDivOfPrivacyPolicyDropDownActive" : "InformationDivOfPrivacyPolicyDropDown InformationDivOfPrivacyPolicyDropDownNotActive"}>
              <ul>
                <li><a href="#WebsiteVisitors">Website Visitors</a></li>
                <li><a href="#Security">Security</a></li>
                <li><a href="#LinksToExternalSites">Links To External Sites</a></li>
                <li><a href="#GoogleAdWords">OOK uses Google AdWords for remarketing</a></li>
                <li><a href="#Personally-Identifying-Information">Protection of Certain Personally-Identifying Information</a></li>
                <li><a href="#Aggregated-Statistics">Aggregated Statistics</a></li>
                <li><a href="#policy-changes">Privacy Policy Changes</a></li>
                <li><a href="#OtherPolicy">Other Policy</a></li>
                <li id="WebsiteVisitors"><a href="#contact-info">Contact Information & Credit</a></li>
              </ul>
            </div>
          </div>

          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>Website Visitors</h4>
          <p id="Security">Like most website operators, OOK collects non-personally-identifying information of the sort that web browsers and servers typically make available, such as the browser type, language preference, referring site, and the date and time of each visitor request. OOK's purpose in collecting non-personally identifying information is to better understand how OOK's visitors use its website. From time to time, OOK may release non-personally-identifying information in the aggregate, e.g., by publishing a report on trends in the usage of its website.</p>
          <p>OOK also collects potentially personally-identifying information like Internet Protocol (IP) addresses for logged in users and for users leaving comments on https://ookcalculator.com/ blog posts. OOK only discloses logged in user and commenter IP addresses under the same circumstances that it uses and discloses personally-identifying information as described below.</p>

          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>Security</h4>
          <p id="LinksToExternalSites">The security of your Personal Information is important to us, but remember that no method of transmission over the Internet, or method of electronic storage is 100% secure. While we strive to use commercially acceptable means to protect your Personal Information, we cannot guarantee its absolute security.</p>

          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>Links To External Sites</h4>
          <p>Our Service may contain links to external sites that are not operated by us. If you click on a third party link, you will be directed to that third party's site. We strongly advise you to review the Privacy Policy and terms of service of every site you visit.</p>
          <p id="GoogleAdWords" >We have no control over, and assume no responsibility for the content, privacy policies or practices of any third party sites, products or services.</p>

          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>OOK uses Google AdWords for remarketing</h4>
          <p>OOK uses the remarketing services to advertise on third party websites (including Google) to previous visitors to our site. It could mean that we advertise to previous visitors who haven't completed a task on our site, for example using the contact form to make an enquiry. This could be in the form of an advertisement on the Google search results page, or a site in the Google Display Network. Third-party vendors, including Google, use cookies to serve ads based on someone's past visits. Of course, any data collected will be used in accordance with our own privacy policy and Google's privacy policy.</p>
          <p id="Personally-Identifying-Information">You can set preferences for how Google advertises to you using the Google Ad Preferences page, and if you want to you can opt out of interest-based advertising entirely by cookie settings or permanently using a browser plugin.</p>

          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>Protection of Certain Personally-Identifying Information</h4>
          <p>OOK discloses potentially personally-identifying and personally-identifying information only to those of its employees, contractors and affiliated organizations that (i) need to know that information in order to process it on OOK's behalf or to provide services available at OOK's website, and (ii) that have agreed not to disclose it to others. Some of those employees, contractors and affiliated organizations may be located outside of your home country; by using OOK's website, you consent to the transfer of such information to them. OOK will not rent or sell potentially personally-identifying and personally-identifying information to anyone. Other than to its employees, contractors and affiliated organizations, as described above, OOK discloses potentially personally-identifying and personally-identifying information only in response to a subpoena, court order or other governmental request, or when OOK believes in good faith that disclosure is reasonably necessary to protect the property or rights of OOK, third parties or the public at large.</p>
          <p id="Aggregated-Statistics" >If you are a registered user of https://ookcalculator.com/ and have supplied your email address, OOK may occasionally send you an email to tell you about new features, solicit your feedback, or just keep you up to date with what's going on with OOK and our products. We primarily use our blog to communicate this type of information, so we expect to keep this type of email to a minimum. If you send us a request (for example via a support email or via one of our feedback mechanisms), we reserve the right to publish it in order to help us clarify or respond to your request or to help us support other users. OOK takes all measures reasonably necessary to protect against the unauthorized access, use, alteration or destruction of potentially personally-identifying and personally-identifying information.</p>

          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>Aggregated Statistics</h4>
          <p id="policy-changes">OOK may collect statistics about the behavior of visitors to its website. OOK may display this information publicly or provide it to others. However, OOK does not disclose your personally-identifying information.</p>

          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>Privacy Policy Changes</h4>
          <p id="OtherPolicy" >Although most changes are likely to be minor, OOK may change its Privacy Policy from time to time, and in OOK's sole discretion. OOK encourages visitors to frequently check this page for any changes to its Privacy Policy. Your continued use of this site after any change in this Privacy Policy will constitute your acceptance of such change.</p>

          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>Other Policies</h4>
          <p id="contact-info"><span style={{fontWeight:'bold'}}>Privacy Policy: </span>Your privacy is important to us This Privacy Policy sets out OOK's policy to respect and protect your privacy regarding any information we may collect while operating our website.</p>
          
          <hr className="Policy-hr" ></hr>

          <h4 className=' mb-3'>Contact Information & Credit</h4>
          <p>This privacy policy was created at <a href="https://privacyterms.io/" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>privacyterms.io</a> privacy policy generator. </p>
          <br />
          <br />
        </div>


        <section className='cse-header-top'>
          <Link smooth="true" duration={500} offset={-70} onClick={scrollToTop}>
            <GrLinkTop className='' />
          </Link>
        </section>
      </section>
    </>
  )
}
