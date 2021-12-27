import React, { useState } from 'react';

export const Checkmark = () => {
    return (
        <img className="checkmark" src="images/checkbox.svg" alt="checkmark"></img>
    )
}

export const Error = () => {
    return (
        <img className="checkmark" src="images/error.png" alt="error"></img>
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