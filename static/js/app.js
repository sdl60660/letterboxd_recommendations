
// Initialize global variables


// Determine if the user is browsing on mobile and adjust worldMapWidth if they are
const determinePhoneBrowsing = () => {
    if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
        return true;
    }
    else{
        return false;
    }
}

const poll = async ({ fn, redisID, validate, interval, maxAttempts }) => {
    let attempts = 0;
  
    const executePoll = async (resolve, reject) => {
        const result = await fn(redisID);
        attempts++;

        console.log(attempts)
  
        if (validate(result)) {
            return resolve(result);
        } else if (maxAttempts && attempts === maxAttempts) {
            return reject(new Error('Exceeded max attempts'));
        } else {
            setTimeout(executePoll, interval, resolve, reject);
        }
    };
  
    return new Promise(executePoll);
};

const getRecData = async (redisID) => {
    const response = await fetch(`/results/${redisID}`, {
        method: 'GET'
    });
    const data = await response.json();
    return data;
};
  
const validateData = (data) => JSON.stringify(data) !== "{}";
const POLL_INTERVAL = 1000;

const $form = document.querySelector("#recommendation-form");
const $submitButton = document.querySelector("#form-submit-button");
const $recResults = document.querySelector("#results");
$form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = e.target.elements.username.value;
    $submitButton.setAttribute("disabled", "disabled");

    const response = await fetch(`/get_recs?username=${username}`, {   // assuming the backend is hosted on the same server
        method: 'GET'
    });
    const data = await response.json();
    console.log(data);
    
    poll({
        fn: getRecData,
        redisID: data.redis_job_id,
        validate: validateData,
        interval: POLL_INTERVAL,
    })
    .then((recs) => {
        console.log(recs);
        const movieIDList = recs.map((rec) => rec.movie_id);
        const selectMovieData = allMovieData.filter((movie) => movieIDList.includes(movie.movie_id))

        let divContent = '<ol>';

        recs.forEach((rec) => {
            const movieData = selectMovieData.find((d) => d.movie_id === rec.movie_id)
            const yearReleased = movieData.year_released == 0 ? 'N/A' : d3.format("d")(movieData.year_released)
            let imageURL = movieData.image_url === '' ? 'https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png' : `https://a.ltrbxd.com/resized/${movieData.image_url}`;
            if (!imageURL.endsWith('jpg') && !imageURL.endsWith('png')) {
                imageURL += '.jpg';
            }

            divContent += '<li><div class="movie-rec">';
            divContent += `<a target="_blank" href="https://letterboxd.com/film/${movieData.movie_id}/"><img class="movie-poster" src="${imageURL}">${movieData.movie_title} (${yearReleased})</a>: ${d3.format("0.2f")(rec.predicted_rating)}`
            divContent += '</div></li>';
        })

        divContent += '</ol>';
        $recResults.innerHTML = divContent;

        $submitButton.removeAttribute("disabled");
    })
    .catch(err => console.error(err));
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