import React, { useState } from 'react'

import './styles/App.scss'

// import Header from './components/Header'
import Intro from './components/Intro'
import Controls from './components/Controls'
import ProgressTracking from './components/ProgressTracking'
import Results from './components/Results'
import Footer from './components/Footer'
// import DownNotice from "./components/DownNotice";
// import ContributionAsk from './components/ContributionAsk'

function App() {
    const [queryData, setQueryData] = useState(null)
    const [requestProgressStep, setRequestProgressStep] = useState(0)
    const [redisData, setRedisData] = useState(null)

    // const [userRatings, setUserRatings] = useState(null);
    const [results, setResults] = useState(null)

    const { user_watchlist: userWatchlist = [] } =
        redisData?.execution_data || {}

    return (
        <div className="App">
            {/* <Header /> */}
            {/* <DownNotice/> */}
            {/* <ContributionAsk /> */}
            <Intro />

            <div className="container" id="body-content-container">
                <Controls
                    setQueryData={setQueryData}
                    requestProgressStep={requestProgressStep}
                    setRequestProgressStep={setRequestProgressStep}
                    setRedisData={setRedisData}
                    setResults={setResults}
                />

                <ProgressTracking
                    queryData={queryData}
                    requestProgressStep={requestProgressStep}
                    setRequestProgressStep={setRequestProgressStep}
                    redisData={redisData}
                />

                <Results results={results} userWatchlist={userWatchlist} />
            </div>

            <Footer />
        </div>
    )
}

export default App
