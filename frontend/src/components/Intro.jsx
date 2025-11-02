import React from "react";
import "../styles/Intro.scss";

const Intro = (props) => {
  return (
    <div className="container" id="intro-container">
      <div className="row">
        <h1>Letterboxd Movie Recommendations</h1>
      </div>
      <div className="row intro-paragraph">
        <p>
          This site will gather movie ratings from any Letterboxd user and
          provide movie recommendations based on ratings data from thousands of
          other users.
        </p>
        <p>
          The more movies you've rated on Letterboxd, the better and more
          personalized the recommendations will be, but it can provide
          recommendations to any user.
        </p>
      </div>
      <div className="row byline">
        <p>
          By{" "}
          <span className="bolded">
            {" "}
            <a
              target="_blank"
              rel="noreferrer"
              href="https://www.samlearner.com/"
            >
              Sam Learner
            </a>
          </span>{" "}
          <a
            target="_blank"
            rel="noreferrer"
            href="https://letterboxd.com/samlearner/"
          >
            <img
              className="icon-img"
              src="images/letterboxd.svg"
              alt="letterboxd link"
            />
          </a>
        </p>
      </div>
    </div>
  );
};

export default Intro;
