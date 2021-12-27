

export const poll = async ({ fn, data, validate, interval, maxAttempts }) => {
    let attempts = 0;
  
    const executePoll = async (resolve, reject) => {
        const result = await fn(data.redisIDs);
        attempts++;

        console.log(attempts, result);
  
        if (validate(result, data.username, data.progressStep, data.setProgressStep, data.setRedisData)) {
            return resolve(result);
        } else if (maxAttempts && attempts === maxAttempts) {
            return reject(new Error('Exceeded max attempts'));
        } else {
            setTimeout(executePoll, interval, resolve, reject);
        }
    };
  
    return new Promise(executePoll);
};

export const getRecData = async (redisIDs) => {
    const paramString = new URLSearchParams(redisIDs).toString();

    const response = await fetch(`http://127.0.0.1:5453/results?${paramString}`, {
        method: 'GET'
    });
    const data = await response.json();
    return data;
};

export const exportCSV = (recs) => {
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

    // document.querySelector("#download-button").insertAdjacentHTML("afterend", '<div>Import downloaded file <a target="_blank" href="https://letterboxd.com/list/new/">here</a> to create a Letterboxd list</div>')

    link.click(); // This will download the data file as a CSV
}