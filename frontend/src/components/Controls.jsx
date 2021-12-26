import React, { useState, useEffect } from 'react';
import "../styles/Controls.scss";

import {
    FormGroup,
    FormControlLabel,
    Checkbox,
    FormControl,
    TextField,
    Button
} from "@mui/material";

import LabeledSlider from "./ui/LabeledSlider";
import { poll, getRecData } from "../util/util";
  

const Controls = ({ setQueryData }) => {
    const POLL_INTERVAL = 1000;

    const [username, setUsername] = useState("");
    const [modelStrength, setModelStrength] = useState(200000);
    const [popularityFilter, setPopularityFilter] = useState(-1);
    const [dataOptIn, setDataOptIn] = useState(false);

    const [runningModel, setRunningModel] = useState(false);


    const handleSubmit = async (e) => {
        e.preventDefault();

        console.log({
            username,
            modelStrength,
            popularityFilter,
            dataOptIn
        });

        setQueryData({
            username,
            modelStrength,
            popularityFilter,
            dataOptIn
        })

        setRunningModel(true);

        const response = await fetch(`/get_recs?username=${username}&popularity_filter=${popularityFilter}&training_data_size=${modelStrength}&data_opt_in=${dataOptIn}`, {
            method: 'GET'
        });
        const data = await response.json();

        console.log(data);
        
        // poll({
        //     fn: getRecData,
        //     data: {
        //         redisIDs: data,
        //         username
        //     },
        //     validate: validateData,
        //     interval: POLL_INTERVAL,
        //     maxAttempts: 300
        // })
        // .then((response) => {
        //     // console.log(response);
        //     recs = response.result;
        //     const movieIDList = recs.map((rec) => rec.movie_id);
        //     const selectMovieData = allMovieData.filter((movie) => movieIDList.includes(movie.movie_id))

        //     $progressList.insertAdjacentHTML('beforeend', '<div id="download-container"><button id="download-button" onclick="exportCSV()">Download Recommendations</button></div>');

        //     let divContent = '<ol id="recommendation-list">';

        //     recs.forEach((rec) => {
        //         const movieData = selectMovieData.find((d) => d.movie_id === rec.movie_id)
        //         const yearReleased = movieData.year_released == 0 ? 'N/A' : d3.format("d")(movieData.year_released)
        //         let imageURL = movieData.image_url === '' ? 'https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png' : `https://a.ltrbxd.com/resized/${movieData.image_url}`;
        //         if (!imageURL.endsWith('jpg') && !imageURL.endsWith('png')) {
        //             imageURL += '.jpg';
        //         }

        //         divContent += '<li><div class="movie-rec">';
        //         divContent += `<a class="grid-item poster-container" target="_blank" href="https://letterboxd.com/film/${movieData.movie_id}/"><img class="movie-poster" src="${imageURL}"></a>`
        //         divContent += `<a class="grid-item title-link" target="_blank" href="https://letterboxd.com/film/${movieData.movie_id}/">${movieData.movie_title} (${yearReleased})</a>`;
        //         divContent += `<div class="grid-item predicted-rating-score">${d3.format("0.2f")(rec.predicted_rating)}</div>`;
        //         divContent += '</div></li>';
        //     })

        //     divContent += '</ol>';
        //     $recResults.innerHTML = divContent;

        //     // Set predicted ratings to red-yellow-green color scale
        //     d3.selectAll('.predicted-rating-score').style('color', function(d) { return colorScale(d3.select(this).text()) });

        //     progressStep = 0;

        //     // Re-enable submit button
        //     setRunningModel(false);
        // })
        // .catch((err) => {
        //     // Replace task list with error message 
        //     let progressTasks = `<li id="server-busy-error"><div class="progress-container">${errorImg}</div><span class="progress-text">Sorry! Server is too busy with other requests right now. Try again later.</span></li>`;
        //     $progressList.innerHTML = progressTasks;

        //     // Re-enable submit button
        //     setRunningModel(false);
        // });
    }

    return (
        <div className="container" id="body-content-container">
            <form
                className="recommendation-form"
                noValidate
                onSubmit={handleSubmit}
            >
                {/*
                    <input
                        className="grid-item"
                        type="text"
                        name="username"
                        placeholder="Letterboxd Username"
                        id="username-input"
                        pattern="^[A-Za-z0-9_]*$"
                        oninvalid="setCustomValidity('Please provide a valid Letterboxd username (letters, numbers, and underscores only)')"
                        oninput="setCustomValidity('')"
                        required
                    />
                */}

                <FormControl>
                    <TextField
                        required
                        error={!/^[A-Za-z0-9_]*$/.test(username)}
                        value={username}
                        pattern="^[A-Za-z0-9_]*$"
                        onInput={(e) => setUsername(e.target.value)}
                        helperText={"Please provide a valid Letterboxd username (letters, numbers, and underscores only)"}
                        label="Letterboxd Username"
                        variant="standard"
                    />
                </FormControl>

                <FormGroup className={"form-slider"}>
                    <LabeledSlider
                        aria-label="Model strength slider. Increase value to get better results. Decrease to get faster results."
                        defaultValue={modelStrength}
                        value={modelStrength}
                        onChange={(e) => setModelStrength(e.target.value)}
                        step={100000}
                        min={100000}
                        max={700000}
                        valueLabelDisplay="off"
                        marks={true}
                        labels={["Faster Results", "Better Results"]}
                    />
                </FormGroup>

                <FormGroup className={"form-slider"}>
                    <LabeledSlider
                        aria-label="Poplularity filter slider. Increase value to only received less-watched movies."
                        defaultValue={popularityFilter}
                        value={popularityFilter}
                        onChange={(e) => setPopularityFilter(e.target.value)}
                        step={1}
                        min={-1}
                        max={7}
                        valueLabelDisplay="off"
                        marks={true}
                        labels={["All Movies", "Less-Reviewed Movies Only"]}
                    />
                </FormGroup>

                <FormGroup className={"data_opt_in_control"}>
                    <FormControlLabel
                        control={
                            <Checkbox
                                name={"data_opt_in"} 
                                value={dataOptIn}
                                checked={dataOptIn}
                                onChange={(e) => setDataOptIn(e.target.checked)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter") {
                                        e.preventDefault();
                                        setDataOptIn(!e.target.checked);
                                    }  
                                }}
                            />
                        }
                        label="Add your ratings to recommendations database?"
                    />
                </FormGroup>

                <FormGroup className={"submit-button"}>
                    <Button
                        variant="outlined"
                        type="submit"
                        disabled={!/^[A-Za-z0-9_]*$/.test(username) || username === "" || runningModel === true}
                        ariaDisabled={!/^[A-Za-z0-9_]*$/.test(username) || username === "" || runningModel === true}
                    >
                        Get Recommendations
                    </Button>
                </FormGroup>
            </form>
        </div>
    )
}




export default Controls;