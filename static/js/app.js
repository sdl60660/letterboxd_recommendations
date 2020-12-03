
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
        console.log(result)
        attempts++;
  
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
// const pollForNewUser = poll({
//     fn: getRecData,
//     redisID: redisID,
//     validate: validateData,
//     interval: POLL_INTERVAL,
// })
// .then(user => console.log(user))
// .catch(err => console.error(err));


const $form = document.querySelector("#recommendation-form");
const $recResults = document.querySelector("#results");
$form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = e.target.elements.username.value;

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
        let divContent = '<ol>';
        recs.forEach((rec) => {
            divContent += `<li>${rec.movie_id}: ${rec.predicted_rating}</li>`;
        })
        divContent += '</ol>'
        $recResults.innerHTML = divContent
    })
    .catch(err => console.error(err));
});

var promises = [
    // d3.json()
];

Promise.all(promises).then(function(allData) {
    const phoneBrowsing = determinePhoneBrowsing();
    console.log(phoneBrowsing, 'js is running');

});