import React, { useState } from 'react'

import './styles/App.scss'

// import Header from './components/Header'
import Intro from './components/Intro'
import Controls from './components/Controls'
import ProgressTracking from './components/ProgressTracking'
import Results from './components/Results'
import ChatInterface from './components/ChatInterface'
import Footer from './components/Footer'
// import DownNotice from "./components/DownNotice";
// import ContributionAsk from './components/ContributionAsk'

function App() {
    const [queryData, setQueryData] = useState(null)
    const [requestProgressStep, setRequestProgressStep] = useState(0)
    const [redisData, setRedisData] = useState(null)

    // const [userRatings, setUserRatings] = useState(null);
    const [results, setResults] = useState(null);
    const [userWatchlist, setUserWatchlist] = useState(null);

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
                    setUserWatchlist={setUserWatchlist}
                />

                <ProgressTracking
                    queryData={queryData}
                    requestProgressStep={requestProgressStep}
                    setRequestProgressStep={setRequestProgressStep}
                    redisData={redisData}
                />

                <Results results={results} userWatchlist={userWatchlist} />
                
                {queryData && queryData.username && (
                    <div className="chat-section">
                        <h2>Personal Recommendations Chat</h2>
                        <p>Chat with our AI assistant for more personalized recommendations</p>
                        <ChatInterface username={queryData.username} />
                    </div>
                )}
            </div>

            <Footer />
        </div>
    )
}

export default App
