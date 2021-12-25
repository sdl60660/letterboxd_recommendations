import React from 'react';

const Controls = (props) => {

    return (
        <div className="container" id="body-content-container">
            <form id="recommendation-form">
                <div className="grid-wrapper">
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

                    <label className="grid-item model-strength-label model-strength-label--left">Faster Results</label>
                    <input className="grid-item" type="range" id="model-strength-slider" name="model_strength"
                        value="200000" step="100000" min="100000" max="700000" />
                    <label className="grid-item model-strength-label model-strength-label--right">Better Results</label>

                    <label className="grid-item popularity-filter-label model-strength-label--left">All Movies</label>
                    <input className="grid-item" type="range" id="popularity-filter-slider" name="popularity_filter"
                        value="-1" step="1" min="-1" max="7" />
                    <label className="grid-item popularity-filter-label model-strength-label--right">Less-Reviewed Movies Only</label>

                    <label className="grid-item grid-label data-opt-in-label">Add your ratings to recommendations database?</label>
                    <label className="grid-item">
                        <input type="checkbox" name="data_opt_in" />
                    </label>

                    <button className="grid-item" id="form-submit-button">Get Recommendations</button>
                </div>
            </form>
        </div>
    )
}




export default Controls;