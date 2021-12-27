import React from 'react';
import "../styles/Intro.scss";

const Intro = (props) => {

    return (
        <div className="container" id="intro-container">
            <div className="row">
                <h1>Letterboxd Movie Recommendations</h1>
            </div>
            <div className="row intro-paragraph">
                <p>
                    This site will gather movie ratings from any Letterboxd user and provide movie recommendations based on
                    ratings data from thousands of other users.
                </p>
                <p>
                    The more movies you've rated on Letterboxd, the better and more personalized
                    the recommendations will be, but it can provide recommendations to any user. If you'd like to filter out more well-known
                    movies (based on the number of Letterboxd reviews), you can do so using the last slider below.
                </p>
            </div>
        </div>
    )
}




export default Intro;