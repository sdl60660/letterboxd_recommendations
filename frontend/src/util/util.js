

export const poll = async ({ fn, data, validate, interval, maxAttempts }) => {
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

export const getRecData = async (redisIDs) => {
    const paramString = new URLSearchParams(redisIDs).toString();

    const response = await fetch(`/results?${paramString}`, {
        method: 'GET'
    });
    const data = await response.json();
    return data;
};