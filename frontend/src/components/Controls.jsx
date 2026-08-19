import React, { useState } from "react";
import "../styles/Controls.scss";

import {
  FormGroup,
  FormControlLabel,
  Checkbox,
  FormControl,
  TextField,
  Button,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";

import { unzipSync, strFromU8 } from "fflate";

import LabeledSlider from "./ui/LabeledSlider";
import { poll, getRecData } from "../util/util";

// Export entries we send to the server (mirrors what the scraper gathers).
// profile.csv is only sent when opting in, since it's only used to attribute
// the database write and contains PII (email, bio) we otherwise don't need.
const EXPORT_FILES = ["ratings.csv", "watched.csv", "watchlist.csv", "likes/films.csv"];
const PROFILE_FILE = "profile.csv";

// Guard so a decompression bomb can't blow up the user's own browser tab
const MAX_UNZIP_BYTES = 60 * 1024 * 1024;

// Unzip in the browser and return the relevant CSVs as [{name, text}]; a lone
// CSV is sent as-is. The server only ever receives CSV text, never a ZIP.
const extractExportFiles = async (file, includeProfile) => {
  if (!file.name.toLowerCase().endsWith(".zip")) {
    return [{ name: file.name, text: await file.text() }];
  }

  const wanted = new Set(includeProfile ? [...EXPORT_FILES, PROFILE_FILE] : EXPORT_FILES);
  const entries = unzipSync(new Uint8Array(await file.arrayBuffer()), {
    filter: (f) => wanted.has(f.name) && f.originalSize < MAX_UNZIP_BYTES,
  });

  return Object.entries(entries).map(([name, bytes]) => ({
    name,
    text: strFromU8(bytes),
  }));
};

const validateData = (data, progressStep, setProgressStep, setRedisData) => {
  setRedisData(data);

  const userJobStatus = data.statuses.redis_get_user_data_job_status;

  if (
    (userJobStatus === "finished" || userJobStatus === "failed") &&
    progressStep === 0
  ) {
    setProgressStep(1);
  }

  if (
    data.execution_data.build_model_stage === "running_model" &&
    progressStep === 1
  ) {
    setProgressStep(2);
  }

  if (data.statuses.redis_build_model_job_status === "finished") {
    setProgressStep(3);
  }

  return data.statuses.redis_build_model_job_status === "finished";
};

const Controls = ({
  setQueryData,
  requestProgressStep,
  setRequestProgressStep,
  setRedisData,
  setResults = () => {},
  setUserWatchlist = () => {},
}) => {
  const POLL_INTERVAL = 1000;

  const [inputMode, setInputMode] = useState("username");
  const [username, setUsername] = useState("");
  const [file, setFile] = useState(null);
  const [modelStrength, setModelStrength] = useState(1000000);
  // const [popularityFilter, setPopularityFilter] = useState(-1)
  const [dataOptIn, setDataOptIn] = useState(false);

  const [runningModel, setRunningModel] = useState(false);

  const usernameValid = /^[A-Za-z0-9_]*$/.test(username) && username !== "";
  const canSubmit = !runningModel && (inputMode === "username" ? usernameValid : !!file);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit) return;

    setRunningModel(true);

    const url =
      process.env.NODE_ENV === "development"
        ? "http://127.0.0.1:8000"
        : "https://letterboxd-recommendations.herokuapp.com";

    let response;
    if (inputMode === "upload") {
      setQueryData({
        isUpload: true,
        filename: file.name,
        modelStrength,
        dataOptIn,
      });

      let files;
      try {
        files = await extractExportFiles(file, dataOptIn);
      } catch (e) {
        setQueryData({
          error:
            "Couldn't read that file. Please upload your Letterboxd export .zip or a CSV from it.",
        });
        setRunningModel(false);
        return;
      }

      response = await fetch(
        `${url}/get_recs_upload?training_data_size=${modelStrength}&data_opt_in=${dataOptIn}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ files }),
        }
      );
    } else {
      setQueryData({ username, modelStrength, dataOptIn });

      response = await fetch(
        `${url}/get_recs?username=${username}&training_data_size=${modelStrength}&data_opt_in=${dataOptIn}`,
        {
          method: "GET",
        }
      );
    }

    // A non-2xx response has no job ids; bail instead of polling garbage (which
    // would otherwise leave the request spinning forever).
    if (!response.ok) {
      setQueryData({
        error:
          response.status === 413
            ? "That file is too large to process. Try uploading just ratings.csv."
            : "Something went wrong processing your request. Please try again.",
      });
      setRunningModel(false);
      return;
    }

    const data = await response.json();

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
        setUserWatchlist(response.execution_data?.user_watchlist);
        setResults(response.result);
      })
      .catch((error) => {
        // Replace task list with error message
        setQueryData({ error });
      })
      .finally(() => {
        // Re-enable submit button
        setRunningModel(false);
      });
  };

  return (
    <form className="recommendation-form" noValidate onSubmit={handleSubmit}>
      <ToggleButtonGroup
        className="input-mode-toggle"
        value={inputMode}
        exclusive
        size="small"
        onChange={(e, value) => value && setInputMode(value)}
        aria-label="How to provide your Letterboxd data"
      >
        <ToggleButton value="username">Enter username</ToggleButton>
        <ToggleButton value="upload">Upload export</ToggleButton>
      </ToggleButtonGroup>

      {inputMode === "username" ? (
        <FormControl>
          <TextField
            required
            error={!/^[A-Za-z0-9_]*$/.test(username)}
            value={username}
            pattern="^[A-Za-z0-9_]*$"
            onInput={(e) => setUsername(e.target.value)}
            helperText={"Please provide a valid Letterboxd username"}
            label="Letterboxd Username"
            variant="standard"
          />
        </FormControl>
      ) : (
        <div className="upload-control">
          <Button variant="outlined" component="label" className="upload-button">
            {file ? file.name : "Choose export file (.zip or .csv)"}
            <input
              type="file"
              accept=".zip,.csv"
              hidden
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </Button>

          <div className="upload-instructions">
            <p className="upload-instructions__heading">How to get your Letterboxd export:</p>
            <ol>
              <li>
                On Letterboxd, open{" "}
                <a target="_blank" rel="noreferrer" href="https://letterboxd.com/settings/data/">
                  Settings → Data
                </a>{" "}
                and click <em>Export Your Data</em>.
              </li>
              <li>
                Upload the downloaded <code>.zip</code> here (or just the <code>ratings.csv</code>{" "}
                from inside it).
              </li>
            </ol>
            <p className="upload-instructions__note">
              Your file is used only to generate recommendations and isn't stored.
            </p>
          </div>
        </div>
      )}

      <FormGroup className={"form-slider"}>
        <LabeledSlider
          aria-label="Model strength slider. Increase value to get better results. Decrease to get faster results."
          defaultValue={modelStrength}
          value={modelStrength}
          onChange={(e) => setModelStrength(e.target.value)}
          step={2000000}
          min={1000000}
          max={5000000}
          valueLabelDisplay="off"
          marks={true}
          labels={["Faster Results", "Better Results"]}
        />
      </FormGroup>

      {/* <FormGroup className={"form-slider"}>
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
          labels={["All Movies", "Less-Reviewed Movies Only"]}
        />
      </FormGroup> */}

      <FormGroup className={"data_opt_in_control"}>
        <FormControlLabel
          control={
            <Checkbox
              name={"data_opt_in"}
              value={dataOptIn}
              checked={dataOptIn}
              onChange={(e) => setDataOptIn(e.target.checked)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  setDataOptIn(!e.target.checked);
                }
              }}
            />
          }
          label="Add your ratings to recommendations database?"
        />
      </FormGroup>

      <FormGroup className={"submit-button"}>
        <Button variant="contained" type="submit" disabled={!canSubmit} aria-disabled={!canSubmit}>
          Get Recommendations
        </Button>
      </FormGroup>
    </form>
  );
};

export default Controls;
