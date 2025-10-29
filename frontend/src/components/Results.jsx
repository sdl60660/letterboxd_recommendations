import React, { useState, useMemo } from 'react'
import { scaleLinear } from 'd3-scale'
import { Button } from '@mui/material'

import Result from './Result'
import ListFilters from './ListFilters'
import { exportCSV } from '../util/util'

import '../styles/Results.scss'

const colorScale = scaleLinear()
    .domain([1, 5.5, 9, 10])
    .range(['red', '#fde541', 'green', '#1F3D0C'])

const Results = ({ results, userWatchlist }) => {
    const [listDownloaded, setListDownloaded] = useState(false)

    const [filteredGenres, setFilteredGenres] = useState({included: null, excluded: ['Music']})
    const [filteredYearRange, setFilteredYearRange] = useState(null)
    const [excludeWatchlist, setExcludeWatchlist] = useState(true)

    const displayedResults = useMemo(() => {
        if (!results) {
            return []
        }

        let output = results.slice()

        // filter on genres
        const includeSet = new Set(filteredGenres.included);
        const excludeSet = new Set(filteredGenres.excluded);
        
        output = output.filter((movie) => {
            const movieGenres = new Set(movie.movie_data.genres ?? [])

            if (filteredGenres.included === null) {
                return movieGenres.intersection(excludeSet).size === 0
            } else {
                return (
                    movieGenres.intersection(includeSet).size > 0 &&
                    movieGenres.intersection(excludeSet).size === 0
                )
            }
        })

        // filter on year range
        if (filteredYearRange) {
            output = output.filter(
                (movie) =>
                    movie.movie_data.year_released >= filteredYearRange[0] &&
                    movie.movie_data.year_released <= filteredYearRange[1]
            )
        }

        // exclude watchlist items (if watchlist present and exclude checkbox is selected)
        if (excludeWatchlist === true && userWatchlist !== null) {
            output = output.filter(
                (movie) => !userWatchlist.includes(movie.movie_data.movie_id)
            )
        }

        return output.slice(0, 50)
    }, [
        results,
        filteredGenres,
        filteredYearRange,
        excludeWatchlist,
        userWatchlist,
    ])

    return (
        <>
            {results && (
                <div id="download-container">
                    <Button
                        variant="outlined"
                        id="download-button"
                        onClick={() => {
                            exportCSV(displayedResults)
                            setListDownloaded(true)
                        }}
                    >
                        Download Recommendations
                    </Button>
                    {listDownloaded === true && (
                        <div className="import-prompt">
                            Import downloaded file{' '}
                            <a
                                target="_blank"
                                rel="noreferrer"
                                href="https://letterboxd.com/list/new/"
                            >
                                here
                            </a>{' '}
                            to create a Letterboxd list
                        </div>
                    )}
                </div>
            )}
            {results && (
                <ListFilters
                    results={results}
                    setFilteredGenres={setFilteredGenres}
                    setFilteredYearRange={setFilteredYearRange}
                    excludeWatchlist={excludeWatchlist}
                    setExcludeWatchlist={setExcludeWatchlist}
                />
            )}
            <div id="results">
                <ol id="recommendation-list">
                    {results &&
                        displayedResults.map((d) => (
                            <Result
                                key={d.movie_id}
                                textColor={colorScale(d.predicted_rating)}
                                {...d}
                            />
                        ))}
                    {results && displayedResults.length === 0 && (
                        <div className="no-item-message">
                            No Items Matching Filters
                        </div>
                    )}
                </ol>
            </div>
        </>
    )
}

export default React.memo(Results)
