
// Initialize global variables
const POLL_INTERVAL = 1000;
const $form = document.querySelector("#recommendation-form");
const $submitButton = document.querySelector("#form-submit-button");
const $recResults = document.querySelector("#results");
const $progressList = document.querySelector("#task-progress-list");

const checkmark = `<img class="checkmark" src="/static/images/checkbox.svg"></img>`;
const errorImg = `<img class="error" src="/static/images/error.png"></img>`;
const loadSpinner = `<div class="loader"></div>`;

const colorScale = d3.scaleLinear().domain([1,5.5,9,10]).range(["red","#fde541","green","#1F3D0C"]);

let recs = [];

let progressStep = 0;

// Determine if the user is browsing on mobile and adjust worldMapWidth if they are
const determinePhoneBrowsing = () => {
    if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
        return true;
    }
    else{
        return false;
    }
}

const exportCSV = () => {
    let rows = [
        ["LetterboxdURI"]
    ];
    console.log(recs);

    recs.forEach((row) => {
        rows.push([`https://letterboxd.com/film/${row['movie_id']}/`])
    })
    
    let csvContent = "data:text/csv;charset=utf-8," 
        + rows.map(e => e.join(",")).join("\n");

    const encodedUri = encodeURI(csvContent);
    let link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "letterboxd_recs.csv");
    document.body.appendChild(link); // Required for FF

    document.querySelector("#download-button").insertAdjacentHTML("afterend", '<div>Import downloaded file <a target="_blank" href="https://letterboxd.com/list/new/">here</a> to create a Letterboxd list</div>')

    link.click(); // This will download the data file as a CSV
}

const updateStatus = (elementID, statusInnerText, iconElement) => {
    document.querySelector(`#${elementID}`).innerHTML = `<div class="progress-container">${iconElement}</div><span class="progress-text">${statusInnerText}</span>`;
}

const poll = async ({ fn, data, validate, interval, maxAttempts }) => {
    let attempts = 0;
  
    const executePoll = async (resolve, reject) => {
        const result = await fn(data.redisIDs);
        attempts++;

        console.log(attempts, result);
  
        if (validate(result, data.username)) {
            return resolve(result);
        } else if (maxAttempts && attempts === maxAttempts) {
            return reject(new Error('Exceeded max attempts'));
        } else {
            setTimeout(executePoll, interval, resolve, reject);
        }
    };
  
    return new Promise(executePoll);
};

const getRecData = async (redisIDs) => {
    const paramString = new URLSearchParams(redisIDs).toString();

    const response = await fetch(`/results?${paramString}`, {
        method: 'GET'
    });
    const data = await response.json();
    return data;
};
  
const validateData = (data, username) => {
    if (data.statuses.redis_get_user_data_job_status === "finished" && progressStep === 0) {
        const numRatings = data.execution_data.num_user_ratings;

        if (data.execution_data.user_status === "user_not_found") {
            const gatherDataFinishedText = `Could not find Letterboxd user: ${username}. Will return generic recommendations.`;
            updateStatus("load-user-task-progress", gatherDataFinishedText, errorImg)
        }
        else {
            let gatherDataFinishedText = `Gathered ${numRatings} movie ratings from ${username}'s <a target="_blank" href="https://letterboxd.com/${username}/films/ratings/">profile</a>`

            if (data.execution_data.num_user_ratings < 50) {
                gatherDataFinishedText += ' (Rate more movies for more personalized results)';
            }

            updateStatus("load-user-task-progress", gatherDataFinishedText, checkmark)
        }
    
        updateStatus("build-model-task-progress", "Building recommendation model...", loadSpinner);
        
        progressStep = 1;
    }

    if (data.execution_data.build_model_stage === "running_model" && progressStep === 1) {
        updateStatus("build-model-task-progress", "Built recommendation model", checkmark);
        updateStatus("run-model-task-progress", `Generating recommendations for ${username}`, loadSpinner);

        progressStep = 2;
    }

    if (data.statuses.redis_build_model_job_status === "finished") {
        updateStatus("build-model-task-progress", "Finished building recommendation model", checkmark);
        updateStatus("run-model-task-progress", `Generated recommendations for ${username}`, checkmark);
    }
    
    return data.statuses.redis_build_model_job_status === "finished";
}

$form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = e.target.elements.username.value.toLowerCase();;
    const popularityFilter = e.target.elements.popularity_filter.value;
    const modelStrength = e.target.elements.model_strength.value;

    $submitButton.setAttribute("disabled", "disabled");

    const response = await fetch(`/get_recs?username=${username}&popularity_filter=${popularityFilter}&training_data_size=${modelStrength}`, {
        method: 'GET'
    });
    const data = await response.json();
    console.log(data);

    let progressTasks = '';
    progressTasks += `<li id="load-user-task-progress"><div class="progress-container"><div class="loader"></div></div><span class="progress-text">Gathering movie ratings from ${username}'s <a target="_blank" href="https://letterboxd.com/${username}/films/ratings/">profile</a></span></li>`;
    progressTasks += `<li id="build-model-task-progress"><div class="progress-container"><div class="waiting-loader"></div></div><span class="progress-text">Build recommendation model</span></li>`;
    progressTasks += `<li id="run-model-task-progress"><div class="progress-container"><div class="waiting-loader"></div></div><span class="progress-text">Generate recommendations for ${username}</span></li>`;
    $progressList.innerHTML = progressTasks;
    
    poll({
        fn: getRecData,
        data: {
            redisIDs: data,
            username
        },
        validate: validateData,
        interval: POLL_INTERVAL,
        maxAttempts: 250
    })
    .then((response) => {
        // console.log(response);
        recs = response.result;
        const movieIDList = recs.map((rec) => rec.movie_id);
        const selectMovieData = allMovieData.filter((movie) => movieIDList.includes(movie.movie_id))

        $progressList.insertAdjacentHTML('beforeend', '<div id="download-container"><button id="download-button" onclick="exportCSV()">Download Recommendations</button></div>');

        let divContent = '<ol id="recommendation-list">';

        recs.forEach((rec) => {
            const movieData = selectMovieData.find((d) => d.movie_id === rec.movie_id)
            const yearReleased = movieData.year_released == 0 ? 'N/A' : d3.format("d")(movieData.year_released)
            let imageURL = movieData.image_url === '' ? 'https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png' : `https://a.ltrbxd.com/resized/${movieData.image_url}`;
            if (!imageURL.endsWith('jpg') && !imageURL.endsWith('png')) {
                imageURL += '.jpg';
            }

            divContent += '<li><div class="movie-rec">';
            divContent += `<a class="grid-item poster-container" target="_blank" href="https://letterboxd.com/film/${movieData.movie_id}/"><img class="movie-poster" src="${imageURL}"></a>`
            divContent += `<a class="grid-item title-link" target="_blank" href="https://letterboxd.com/film/${movieData.movie_id}/">${movieData.movie_title} (${yearReleased})</a>`;
            divContent += `<div class="grid-item predicted-rating-score">${d3.format("0.2f")(rec.predicted_rating)}</div>`;
            divContent += '</div></li>';
        })

        divContent += '</ol>';
        $recResults.innerHTML = divContent;

        // Set predicted ratings to red-yellow-green color scale
        d3.selectAll('.predicted-rating-score').style('color', function(d) { return colorScale(d3.select(this).text()) });

        // Re-enable submit button
        $submitButton.removeAttribute("disabled");
        progressStep = 0;
    })
    .catch((err) => {
        // Replace task list with error message 
        let progressTasks = `<li id="server-busy-error"><div class="progress-container">${errorImg}</div><span class="progress-text">Sorry! Server is too busy with other requests right now. Try again later.</span></li>`;
        $progressList.innerHTML = progressTasks;

        // Re-enable submit button
        $submitButton.removeAttribute("disabled");
    });
});

var promises = [
    d3.csv('static/data/movie_data.csv')
];

let allMovieData = null;
Promise.all(promises).then(function(allData) {
    const phoneBrowsing = determinePhoneBrowsing();
    allMovieData = allData[0];
    console.log('Movie data loaded');
});