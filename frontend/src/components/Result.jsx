import React, { useMemo } from 'react'
import '../styles/Result.scss'

import { Chip } from '@mui/material'

const Image = React.memo(function Image(props) {
    return <img {...props} alt={props.alt} />
})

const Result = ({ movie_data, predicted_rating, textColor }) => {
    const imageURL = useMemo(() => {
        let url = movie_data.image_url

        if (url === '') {
            url =
                'https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png'
        } else if (!url.includes('a.ltrbxd.com')) {
            url = `https://a.ltrbxd.com/resized/${movie_data.image_url}.jpg`
        }

        return url
    }, [movie_data.image_url])

    const yearReleased =
        movie_data.year_released === 0
            ? 'N/A'
            : movie_data.year_released.toFixed(0)

    return (
        <li>
            <div className={'movie-rec-wrapper'}>
                <div className="movie-rec" tabIndex={'0'}>
                    <div className="poster-wrapper">
                        <a
                            className="grid-item poster-container"
                            target="_blank"
                            rel="noreferrer"
                            href={`https://letterboxd.com/film/${movie_data.movie_id}/`}
                        >
                            <Image
                                className="movie-poster"
                                src={imageURL}
                                alt={`Movie poster for ${movie_data.movie_title}`}
                            />
                        </a>
                    </div>
                    <div className="recommendation-data-wrapper">
                        <div className="recommendation-movie-data-wrapper">
                            <a
                                className="grid-item title-link"
                                target="_blank"
                                rel="noreferrer"
                                href={`https://letterboxd.com/film/${movie_data.movie_id}/`}
                            >
                                {movie_data.movie_title} ({yearReleased})
                            </a>
                            <div
                                className={'genre-display'}
                                tabIndex="0"
                                aria-label={
                                    'Displays genres for associated movie'
                                }
                            >
                                {typeof window !== 'undefined' &&
                                    window.innerWidth > 650 &&
                                    movie_data.genres?.map((genre) => (
                                        <Chip
                                            key={genre}
                                            label={genre}
                                            name={genre}
                                        />
                                    ))}
                            </div>
                        </div>
                        <div
                            className="grid-item predicted-rating-score"
                            style={{ color: textColor }}
                            tabIndex="0"
                            aria-label={`Predicted rating for ${
                                movie_data.movie_title
                            }: ${predicted_rating.toFixed(2)} out of 10`}
                        >
                            {predicted_rating.toFixed(2)}
                        </div>
                    </div>
                </div>
            </div>
        </li>
    )
}

export default React.memo(Result)
