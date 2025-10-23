import React, { useState } from 'react'
import '../styles/Controls.scss'

import {
    FormGroup,
    FormControlLabel,
    Checkbox,
    FormControl,
    TextField,
    Button,
} from '@mui/material'

import LabeledSlider from './ui/LabeledSlider'
import { poll, getRecData } from '../util/util'

const validateData = (data, progressStep, setProgressStep, setRedisData) => {
    setRedisData(data)

    const userJobStatus = data.statuses.redis_get_user_data_job_status

    if (
        (userJobStatus === 'finished' || userJobStatus === 'failed') &&
        progressStep === 0
    ) {
        setProgressStep(1)
    }

    if (
        data.execution_data.build_model_stage === 'running_model' &&
        progressStep === 1
    ) {
        setProgressStep(2)
    }

    if (data.statuses.redis_build_model_job_status === 'finished') {
        setProgressStep(3)
    }

    return data.statuses.redis_build_model_job_status === 'finished'
}

const Controls = ({
    setQueryData,
    requestProgressStep,
    setRequestProgressStep,
    setRedisData,
    setResults = () => {},
    setUserWatchlist = () => {},
}) => {
    const POLL_INTERVAL = 1000

    const [username, setUsername] = useState('')
    const [modelStrength, setModelStrength] = useState(500000)
    // const [popularityFilter, setPopularityFilter] = useState(-1)
    const [dataOptIn, setDataOptIn] = useState(false)

    const [runningModel, setRunningModel] = useState(false)

    const handleSubmit = async (e) => {
        e.preventDefault()

        setQueryData({
            username,
            modelStrength,
            // popularityFilter,
            dataOptIn,
        })

        setRunningModel(true)

        const url =
            process.env.NODE_ENV === 'development'
                ? 'http://127.0.0.1:8000'
                : 'https://letterboxd-recommendations.herokuapp.com'

        const response = await fetch(
            `${url}/get_recs?username=${username}&training_data_size=${modelStrength}&data_opt_in=${dataOptIn}`,
            {
                method: 'GET',
            }
        )
        const data = await response.json()

        poll({
            fn: getRecData,
            data: {
                redisIDs: data,
                progressStep: requestProgressStep,
                setProgressStep: setRequestProgressStep,
                setRedisData,
            },
            validate: validateData,
            interval: POLL_INTERVAL,
            maxAttempts: 300,
        })
            .then((response) => {
                setUserWatchlist(response.execution_data?.user_watchlist)
                setResults(response.result)
            })
            .catch((error) => {
                // Replace task list with error message
                setQueryData({ error })
            })
            .finally(() => {
                // Re-enable submit button
                setRunningModel(false)
            })
    }

    return (
        <form
            className="recommendation-form"
            noValidate
            onSubmit={handleSubmit}
        >
            <FormControl>
                <TextField
                    required
                    error={!/^[A-Za-z0-9_]*$/.test(username)}
                    value={username}
                    pattern="^[A-Za-z0-9_]*$"
                    onInput={(e) => setUsername(e.target.value)}
                    helperText={'Please provide a valid Letterboxd username'}
                    label="Letterboxd Username"
                    variant="standard"
                />
            </FormControl>

            <FormGroup className={'form-slider'}>
                <LabeledSlider
                    aria-label="Model strength slider. Increase value to get better results. Decrease to get faster results."
                    defaultValue={modelStrength}
                    value={modelStrength}
                    onChange={(e) => setModelStrength(e.target.value)}
                    step={500000}
                    min={500000}
                    max={1500000}
                    valueLabelDisplay="off"
                    marks={true}
                    labels={['Faster Results', 'Better Results']}
                />
            </FormGroup>

            {/* <FormGroup className={'form-slider'}>
                <LabeledSlider
                    aria-label="Poplularity filter slider. Increase value to only received less-watched movies."
                    defaultValue={popularityFilter}
                    value={popularityFilter}
                    onChange={(e) => setPopularityFilter(e.target.value)}
                    step={1}
                    min={-1}
                    max={7}
                    valueLabelDisplay="off"
                    marks={true}
                    labels={['All Movies', 'Less-Reviewed Movies Only']}
                />
            </FormGroup> 
            */}

            <FormGroup className={'data_opt_in_control'}>
                <FormControlLabel
                    control={
                        <Checkbox
                            name={'data_opt_in'}
                            value={dataOptIn}
                            checked={dataOptIn}
                            onChange={(e) => setDataOptIn(e.target.checked)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                    e.preventDefault()
                                    setDataOptIn(!e.target.checked)
                                }
                            }}
                        />
                    }
                    label="Add your ratings to recommendations database?"
                />
            </FormGroup>

            <FormGroup className={'submit-button'}>
                <Button
                    variant="contained"
                    type="submit"
                    disabled={
                        !/^[A-Za-z0-9_]*$/.test(username) ||
                        username === '' ||
                        runningModel === true
                    }
                    aria-disabled={
                        !/^[A-Za-z0-9_]*$/.test(username) ||
                        username === '' ||
                        runningModel === true
                    }
                >
                    Get Recommendations
                </Button>
            </FormGroup>
        </form>
    )
}

export default Controls
