import React from 'react'
import '../styles/DownNotice.scss'

const DownNotice = () => {
    return (
        <div className={'site-message down-notice'}>
            <p>
                NOTE: As of August 9, 2022, some site dependencies have changed
                and the site is currently not functioning as intended.
            </p>
            <p>
                I'm working on fixing this ASAP, but unfortunately the site will
                be down as long as you're seeing this message.
            </p>
        </div>
    )
}

export default DownNotice
