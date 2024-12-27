import React, { useState } from 'react'
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
    const [listDownloaded, setListDownloaded] = useState(false);

    const [filteredGenres, setFilteredGenres] = useState(null);
    const [filteredYearRange, setFilteredYearRange] = useState(null);
    const [excludeWatchlist, setExcludeWatchlist] = useState(true);

    const displayedResults = !results
        ? []
        : results
              .filter((d) =>
                  filteredGenres
                      ? filteredGenres.some((genre) =>
                            d.movie_data.genres?.includes(genre)
                        )
                      : true
              )
              .filter((d) =>
                  filteredYearRange
                      ? d.movie_data.year_released >= filteredYearRange[0] &&
                        d.movie_data.year_released <= filteredYearRange[1]
                      : true
              )
              .filter(d => excludeWatchlist === false || !userWatchlist.includes(d.movie_data.movie_id))
              .slice(0, 50)

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
                        displayedResults.map((d, i) => (
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
