import React from 'react';
import "../styles/Header.scss";

const Header = (props) => {
    return (
        <div class="header">
            <div class="main-site-link">
                <a href="https://bit.ly/main-project-site">
                    <button>More Projects</button>
                </a>
            </div>
        </div>
    )
}


export default Header;