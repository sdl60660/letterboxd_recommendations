import React, { useState } from 'react';

export const Checkmark = () => {
    return (
        <img class="checkmark" src="images/checkbox.svg" alt="checkmark"></img>
    )
}

export const Error = () => {
    return (
        <img class="checkmark" src="images/error.png" alt="error"></img>
    )
}

export const Loader = ({ running=false }) => {

    if (running === true) {
        return (
            <div class="loader" />
        )
    }
    else {
        return (
            <div class="waiting-loader" />
        )
    }
}