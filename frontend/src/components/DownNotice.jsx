import React from "react";
import "../styles/DownNotice.scss";

const DownNotice = () => {
  return (
    <div className={"site-message down-notice"}>
      <p>
        NOTE: As of January 28, 2026, some changes to the Letterboxd site are preventing this tool
        from accessing their data and functioning as intended.
      </p>
      <p>
        I'm working on fixing this ASAP, but unfortunately the site will be down as long as you're
        seeing this message.
      </p>
    </div>
  );
};

export default DownNotice;
