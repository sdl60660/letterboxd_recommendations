import React from 'react';

import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

export const Checkmark = () => {
    return (
        // <img className="checkmark" src="images/checkbox.svg" alt="checkmark"></img>
        <CheckCircleIcon className="checkmark" />
    )
}

export const Error = () => {
    return (
        // <img className="checkmark" src="images/error.png" alt="error"></img>
        <ErrorOutlineIcon className="checkmark" />
    )
}

export const Loader = ({ running=false }) => {

    if (running === true) {
        return (
            <div className="loader" />
        )
    }
    else {
        return (
            <div className="waiting-loader" />
        )
    }
}