import React from 'react'
import { useState, useEffect } from 'react'
import '../styles/ProgressTracking.scss'

import { Checkmark, Error, Loader } from './ui/StatefulIcons'

const formatRatingGatherText = ({ queryData, redisData }) => {
    const userDataStatus = redisData?.statuses?.redis_get_user_data_job_status
    if (userDataStatus !== 'finished' && userDataStatus !== 'failed') {
        return (
            <>
                Gathering movie ratings from {queryData.username}'s{' '}
                <a
                    target="_blank"
                    rel="noreferrer"
                    href={`https://letterboxd.com/${queryData.username}/films/ratings/`}
                >
                    profile
                </a>
            </>
        )
    } else if (redisData?.execution_data?.user_status === 'user_not_found') {
        return `Could not find Letterboxd user: ${queryData.username}. Will return generic recommendations.`
    } else {
        const numRatings = redisData.execution_data.num_user_ratings
        const additionalRatingsPrompt =
            numRatings < 50
                ? ' (Rate more movies for more personalized results)'
                : ''

        return (
            <>
                Gathered {numRatings} movie ratings from {queryData.username}'s{' '}
                <a
                    target="_blank"
                    rel="noreferrer"
                    href={`https://letterboxd.com/${queryData.username}/films/ratings/`}
                >
                    profile
                </a>
                {additionalRatingsPrompt}
            </>
        )
    }
}

const renderIcon = ({ status, error = false }) => {
    if (error === true || status === 'failed') {
        return <Error />
    } else if (status === 'finished') {
        return <Checkmark />
    } else if (status === 'started') {
        return <Loader running={true} />
    } else {
        return <Loader running={false} />
    }
}

const ProgressTracking = ({ queryData, redisData }) => {
    const [stageProgress, setStageProgress] = useState({})

    useEffect(() => {
        if (redisData) {
            const statuses = redisData.statuses
            const executionData = redisData.execution_data

            setStageProgress({
                userData: statuses?.redis_get_user_data_job_status,
                buildModel:
                    executionData?.build_model_stage === 'running_model'
                        ? 'finished'
                        : statuses?.redis_build_model_job_status,
                runModel:
                    executionData?.build_model_stage !== 'running_model'
                        ? 'pending'
                        : statuses?.redis_build_model_job_status === 'finished'
                        ? 'finished'
                        : 'started',
            })
        }
    }, [redisData])

    return (
        <div id="progress-tracking">
            <ul id="task-progress-list">
                {queryData && !queryData.error && (
                    <>
                        <li id="load-user-task-progress">
                            <div className="progress-container">
                                {renderIcon({
                                    status: stageProgress.userData,
                                    error:
                                        redisData?.execution_data
                                            ?.user_status === 'user_not_found',
                                })}
                            </div>
                            <span className="progress-text">
                                {formatRatingGatherText({
                                    queryData,
                                    redisData,
                                })}
                            </span>
                        </li>

                        <li id="build-model-task-progress">
                            <div className="progress-container">
                                {renderIcon({
                                    status: stageProgress.buildModel,
                                })}
                            </div>
                            <span className="progress-text">
                                {stageProgress.buildModel === 'started'
                                    ? 'Building recommendation model...'
                                    : stageProgress.buildModel === 'finished'
                                    ? 'Built recommendation model'
                                    : 'Build recommendation model'}
                            </span>
                        </li>

                        <li id="run-model-task-progress">
                            <div className="progress-container">
                                {renderIcon({ status: stageProgress.runModel })}
                            </div>
                            <span className="progress-text">
                                {stageProgress.runModel === 'started'
                                    ? 'Generating'
                                    : stageProgress.runModel === 'finished'
                                    ? 'Generated'
                                    : 'Generate'}{' '}
                                recommendations for {queryData.username}
                            </span>
                        </li>
                    </>
                )}
                <>
                    {queryData && queryData.error && (
                        <li id="server-busy-error">
                            <div className="progress-container">
                                <Error />
                            </div>
                            <span className="progress-text">
                                Sorry! Server is too busy with other requests
                                right now. Try again later.
                            </span>
                        </li>
                    )}
                </>
            </ul>
        </div>
    )
}

export default ProgressTracking
