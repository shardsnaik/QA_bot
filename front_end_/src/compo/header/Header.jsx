import React from "react";
import "./header.css"; // Include styles for better design
import img_log from '../../images/logo2.png'
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGit } from "@fortawesome/free-brands-svg-icons";
import { faBots } from "@fortawesome/free-brands-svg-icons";
import { faShareFromSquare, faUser } from '@fortawesome/free-solid-svg-icons';
// import { faUser } from "@fortawesome/free-solid-svg-icons";

const Header = () => {
  return (
    <header className="app-header">
      <div className="logo-section">
      <FontAwesomeIcon className="app-logo" icon={faBots} bounce size="2xl" />      
      </div>
      <div className="title-social">
      <div><h1 className="app-name">Q&A Bot v1</h1></div>
      <div>

      <nav className="social-links">
        <a href="https://github.com/shardsnaik" target="_blank" rel="noopener noreferrer">
        <FontAwesomeIcon icon={faGit} size="xl" />
        </a>
        <a href="https://www.instagram.com" target="_blank" rel="noopener noreferrer">
        <FontAwesomeIcon icon={faShareFromSquare} size="xl" />        </a>
        <a href="https://www.instagram.com" target="_blank" rel="noopener noreferrer">
        <FontAwesomeIcon icon={faUser} size="xl" />        </a>
      </nav>
      </div>
      </div>
    </header>
  );
};

export default Header;
