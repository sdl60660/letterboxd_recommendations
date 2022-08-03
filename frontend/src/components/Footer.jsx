import React from 'react';
import '../styles/Footer.scss';

const Footer = (props) => {
    return (
        <div className="container footer">
            <hr />
            <section>
                <p style={{fontWeight: "bold"}}>Project by <a target="_blank" rel="noreferrer" href="https://www.samlearner.com/">Sam Learner</a> |
                    <a target="_blank" rel="noreferrer" href="mailto:learnersd+letterboxd-recs@gmail.com"><img className="icon-img" src="images/email.svg" alt="email link" /></a> |
                    <a target="_blank" rel="noreferrer" href="https://twitter.com/sam_learner"><img className="icon-img" src="images/twitter.svg" alt="twitter link" /></a> |
                    <a target="_blank" rel="noreferrer" href="https://github.com/sdl60660"><img className="icon-img" src="images/github.svg" alt="github link" /></a> |
                    <a target="_blank" rel="noreferrer" href="https://letterboxd.com/samlearner/"><img className="icon-img" src="images/letterboxd.svg" alt="letterboxd link" /></a>
                </p>
                <p>Code and data for this project lives <a target="_blank" rel="noreferrer" href="https://github.com/sdl60660/letterboxd_recommendations">here</a>.</p>
            </section>
            <section>
                <p className="heading">Methodology</p>
                <p>
                    A user's "star" ratings are scraped their Letterboxd profile and assigned numerical ratings from 1
                    to 10 (accounting for half stars).
                    Their ratings are then combined with a sample of ratings from the top 4000 most active users on the
                    site to create a collaborative filtering
                    recommender model using singular value decomposition (SVD). All movies in the full dataset that the
                    user has not rated are run through the model for
                    predicted scores and the items with the top predicted scores are returned. Due to constraints in
                    time and computing power, the maxiumum sample size
                    that a user is allowed to select is 500,000 samples, though there are over five million ratings in
                    the full dataset from the top 4000 Letterboxd
                    users alone.
                </p>
            </section>
            <section>
                <p className="heading">Notes</p>
                <p>
                    The underlying model is completely blind to genres, themes, directors, cast, or any other content
                    information; it recommends only based on similarities in
                    rating patterns between other users and movies. I've found that it tends to recommend very popular
                    movies often, regardless of an individual user's taste
                    ("basically everyone who watches 12 Angry Men seems to like it, so why wouldn't you?"). To help
                    counteract that, I included a popularity filter that filters by how many times a
                    movie has been rated in the dataset, so that users can specifically find more obscure
                    recommendations. I've also found that it occasionally just completely whiffs
                    (I guess most people who watch "Taylor Swift: Reputation Stadium Tour" do like it, but it's not
                    really my thing). I think that's just the nature of the beast, to
                    some extent, particularly when working with a relatively small sample. It'll return 50
                    recommendations and that's usually enough to work with if I'm looking
                    for something to watch, even if there are a couple misses here or there.
                </p>
                <p>
                    If you opted in to your ratings being included in the model data and would like your data to be deleted, 
                    just reach out to me at the email address above and I will take care of it.
                </p>
            </section>
            <section>
                <p>Last Updated: July 2022</p>
            </section>
        </div>
    )
}


export default Footer;
