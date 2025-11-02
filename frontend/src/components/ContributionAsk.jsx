import React from "react";
import "../styles/ContributionAsk.scss";

const ContributionAsk = ({}) => {
  return (
    <div className={"site-message"}>
      <p>
        If you enjoy using this tool, please consider helping me cover some
        chunk of the cost to maintain it{" "}
        <a href="https://ko-fi.com/samlearner" target="_blank" rel="noreferrer">
          here
        </a>
        .
      </p>
    </div>
  );
};

export default ContributionAsk;
