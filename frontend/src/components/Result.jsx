import React from 'react';
import "../styles/Result.scss";

const Result = ({ movie_data, predicted_rating, textColor }) => {
    const yearReleased = movie_data.year_released === 0 ? 'N/A' : movie_data.year_released.toFixed(0);
    const imageURL = movie_data.image_url === '' ? 'https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png' : `https://a.ltrbxd.com/resized/${movie_data.image_url}.jpg`;

    return (
        <li>
            <div className={"movie-rec-wrapper"}>
                <div className="movie-rec">
                    <a className="grid-item poster-container" target="_blank" rel="noreferrer" href={`https://letterboxd.com/film/${movie_data.movie_id}/`}><img className="movie-poster" src={imageURL} alt={`Movie poster for ${movie_data.movie_title}`}/></a>
                    <a className="grid-item title-link" target="_blank" rel="noreferrer" href={`https://letterboxd.com/film/${movie_data.movie_id}/`}>{movie_data.movie_title} ({yearReleased})</a>
                    <div className="grid-item predicted-rating-score" style={{color: textColor}}>{predicted_rating.toFixed(2)}</div>
                </div>
            </div>
        </li>
    )
}




export default Result;