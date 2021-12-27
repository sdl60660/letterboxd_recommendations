import React, { useState } from 'react';
import { scaleLinear } from "d3-scale";
import {
    Button
} from '@mui/material'

import Result from "./Result";
import ListFilters from "./ListFilters";
import { exportCSV } from '../util/util';


const colorScale = scaleLinear().domain([1,5.5,9,10]).range(["red","#fde541","green","#1F3D0C"]);

const Results = ({ results }) => {
    const [ listDownloaded, setListDownloaded ] = useState(false);
    const [ filteredGenres, setFilteredGenres ] = useState(null);

    const displayedResults = results
        ?.filter(d => filteredGenres ? filteredGenres.some( genre => d.movie_data.genres?.includes(genre)) : true)
        .slice(0, 50) || [];

    return (
        <>
            { results && 
                <div id="download-container">
                    <Button variant="outlined" id="download-button" onClick={() => {
                        exportCSV(displayedResults);
                        setListDownloaded(true);
                    }}>Download Recommendations</Button>
                { listDownloaded === true &&
                    <div>Import downloaded file <a target="_blank" rel="noreferrer" href="https://letterboxd.com/list/new/">here</a> to create a Letterboxd list</div>
                }
                </div>
            }
            { results && <ListFilters results={results} setFilteredGenres={setFilteredGenres} /> }
            <div id="results">
                <ol id="recommendation-list">
                    { results && displayedResults.map((d, i) => (<Result key={d.movie_id} textColor={colorScale(d.predicted_rating)} {...d}/>)) }
                </ol>
            </div>
        </>
    )
}

export default Results;