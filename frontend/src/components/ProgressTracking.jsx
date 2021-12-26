import React from 'react'

import { Checkmark, Error, Loader } from './ui/StatefulIcons'

const ProgressTracking = ({ queryData }) => {
    return (
        <div id="progress-tracking">
            <ul id="task-progress-list">
                {queryData && (
                    <>
                        <li id="load-user-task-progress">
                            <div class="progress-container">
                                <Loader running={true} />
                            </div>
                            <span class="progress-text">
                                Gathering movie ratings from {queryData.username}'s{' '}
                                <a
                                    target="_blank"
                                    rel="noreferrer"
                                    href={`https://letterboxd.com/${queryData.username}/films/ratings/`}
                                >
                                    profile
                                </a>
                            </span>
                        </li>

                        <li id="build-model-task-progress">
                            <div class="progress-container">
                                <Loader running={false} />
                            </div>
                            <span class="progress-text">
                                Build recommendation model
                            </span>
                        </li>

                        <li id="run-model-task-progress">
                            <div class="progress-container">
                                <Loader running={false} />
                            </div>
                            <span class="progress-text">
                                Generate recommendations for {queryData.username}
                            </span>
                        </li>
                    </>
                )}
            </ul>
        </div>
    )
}

export default ProgressTracking
