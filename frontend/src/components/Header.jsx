import React from "react";
import "../styles/Header.scss";

const Header = (props) => {
  return (
    <div className="header">
      <div className="main-site-link">
        <a
          href="https://bit.ly/main-project-site"
          target="_blank"
          rel="noreferrer"
        >
          <button>More Projects</button>
        </a>
      </div>
    </div>
  );
};

export default Header;
