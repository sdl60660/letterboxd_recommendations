import React, { useState } from 'react';
import { scaleLinear } from "d3-scale";
import {
    Button
} from '@mui/material'

import Result from "./Result";
import { exportCSV } from '../util/util';


const Results = ({ results }) => {
    const [ listDownloaded, setListDownloaded ] = useState(false);

    const colorScale = scaleLinear().domain([1,5.5,9,10]).range(["red","#fde541","green","#1F3D0C"]);

    return (
        <>
            { results && 
                <div id="download-container">
                    <Button variant="outlined" id="download-button" onClick={() => {
                        exportCSV(results);
                        setListDownloaded(true);
                    }}>Download Recommendations</Button>
                { listDownloaded === true &&
                    <div>Import downloaded file <a target="_blank" rel="noreferrer" href="https://letterboxd.com/list/new/">here</a> to create a Letterboxd list</div>
                }
                </div>
            }
            <div id="results">
                <ol id="recommendation-list">
                    { results && results.slice(0, 50).map((d, i) => (<Result key={d.movie_id} textColor={colorScale(d.predicted_rating)} {...d}/>)) }
                </ol>
            </div>
        </>
    )
}

export default Results;