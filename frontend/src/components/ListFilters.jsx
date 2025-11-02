import React, { useState, useMemo } from "react";

import { useTheme } from "@mui/material/styles";
import Box from "@mui/material/Box";
import OutlinedInput from "@mui/material/OutlinedInput";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import Select from "@mui/material/Select";
import Chip from "@mui/material/Chip";
import ListItemText from "@mui/material/ListItemText";
import Checkbox from "@mui/material/Checkbox";
import Slider from "@mui/material/Slider";

import "../styles/ListFilters.scss";
import LabeledSlider from "./ui/LabeledSlider";

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 7.5 + ITEM_PADDING_TOP,
      width: 400,
      maxWidth: "90vw",
    },
  },
};

function getStyles(name, personName, theme) {
  return {
    fontWeight:
      personName.indexOf(name) === -1
        ? theme.typography.fontWeightRegular
        : theme.typography.fontWeightMedium,
  };
}

const ListFilters = ({
  results,
  setFilteredGenres,
  setFilteredYearRange,
  setFilteredPopularityRange,
  excludeWatchlist,
  setExcludeWatchlist,
}) => {
  const allGenres = useMemo(() => {
    return [
      ...new Set(
        results
          .map((d) => d.movie_data.genres)
          .flat()
          .filter((d) => d && d !== ""),
      ),
    ].sort();
  }, [results]);

  const allYears = useMemo(() => {
    return [
      Math.min(
        ...results
          .filter((d) => d.movie_data.year_released)
          .map((d) => d.movie_data.year_released),
      ),
      Math.max(
        ...results
          .filter((d) => d.movie_data.year_released)
          .map((d) => d.movie_data.year_released),
      ),
    ];
  }, [results]);

  const popularityRange = useMemo(() => {
    return [
      Math.min(
        ...results
          .filter((d) => d.movie_data.popularity)
          .map((d) => d.movie_data.popularity),
      ),
      Math.max(
        ...results
          .filter((d) => d.movie_data.popularity)
          .map((d) => d.movie_data.popularity),
      ),
    ];
  }, [results]);

  const [genres, setGenres] = useState({
    included: allGenres,
    excluded: ["Music"],
  });
  const [popularity, setPopularity] = useState(popularityRange);
  const [yearRange, setYearRange] = useState(allYears);

  const theme = useTheme();

  const handleGenreChange = (event, listType) => {
    const {
      target: { value },
    } = event;

    // On autofill we get a stringified value.
    const newGenreVal = typeof value === "string" ? value.split(",") : value;

    setGenres((curr) => {
      const output = { ...curr, [listType]: newGenreVal };

      if (listType === "include" && newGenreVal.length === allGenres.length) {
        setFilteredGenres({ ...output, include: null });
      } else {
        setFilteredGenres(output);
      }
      return output;
    });
  };

  const handlePopularityChange = (event, newValue) => {
    setPopularity(newValue);
    setFilteredPopularityRange(newValue);
  };

  const handleYearChange = (event, newValue) => {
    setYearRange(newValue);
    setFilteredYearRange(newValue);
  };

  return (
    <div className="list-filter-controls">
      <FormControl>
        <Box>
          <InputLabel id="year-filter-label" shrink={true}>
            Year Released
          </InputLabel>
          <Slider
            labelId="year-filter-label"
            id="year-filter"
            getAriaLabel={() => "Year Released filter"}
            value={yearRange}
            onChange={handleYearChange}
            valueLabelDisplay="auto"
            getAriaValueText={(value) => value}
            min={allYears[0]}
            max={allYears[1]}
            step={1}
          />
        </Box>
      </FormControl>

      {/* <FormControl>
                <Box>
                    <InputLabel id="popularity-filter-label" shrink={true}>
                        Popularity
                    </InputLabel>
                    <LabeledSlider
                        labelId="popularity-filter-label"
                        id="popularity-filter"
                        aria-label="Poplularity filter slider. Adjust value to only receive recommendations for more- or less-watched movies."
                        value={popularity}
                        onChange={handlePopularityChange}
                        valueLabelDisplay="off"
                        getAriaValueText={(value) => value}
                        min={popularityRange[0]}
                        max={popularityRange[1]}
                        marks={true}
                        labels={['Less', 'More']}
                        step={Math.round(popularityRange[1] - popularityRange[0]) / 10}
                    />
                </Box>
            </FormControl> */}

      <FormControl sx={{ m: 1, width: 400, maxWidth: "90vw" }}>
        <InputLabel id="included-genre-filter-label">
          Included genres
        </InputLabel>
        <Select
          labelId="included-genre-filter-label"
          id="included-genre-filter"
          multiple
          value={genres.included}
          onChange={(e) => handleGenreChange(e, "included")}
          getAriaLabel={() => "Included genre filter"}
          input={
            <OutlinedInput id="select-multiple-chip" label="Included genres" />
          }
          renderValue={(selected) =>
            selected.length === allGenres.length ? (
              <div className="default-all-display">All</div>
            ) : (
              <Box
                sx={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 0.5,
                }}
              >
                {selected.map((value) => (
                  <Chip key={value} label={value} />
                ))}
              </Box>
            )
          }
          MenuProps={MenuProps}
        >
          {allGenres.map((genre) => (
            <MenuItem
              key={genre}
              value={genre}
              style={getStyles(genre, genres.included, theme)}
            >
              <Checkbox checked={genres.included.indexOf(genre) > -1} />
              <ListItemText primary={genre} />
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControl sx={{ m: 1, width: 400, maxWidth: "90vw" }}>
        <InputLabel id="excluded-genre-filter-label">
          Excluded genres
        </InputLabel>
        <Select
          labelId="excluded-genre-filter-label"
          id="excluded-genre-filter"
          multiple
          value={genres.excluded}
          onChange={(e) => handleGenreChange(e, "excluded")}
          getAriaLabel={() => "Excluded genre filter"}
          input={
            <OutlinedInput id="select-multiple-chip" label="Excluded genres" />
          }
          renderValue={(selected) =>
            selected.length === allGenres.length ? (
              <div className="default-all-display">All</div>
            ) : (
              <Box
                sx={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 0.5,
                }}
              >
                {selected.map((value) => (
                  <Chip key={value} label={value} />
                ))}
              </Box>
            )
          }
          MenuProps={MenuProps}
        >
          {allGenres.map((genre) => (
            <MenuItem
              key={genre}
              value={genre}
              style={getStyles(genre, genres.excluded, theme)}
            >
              <Checkbox checked={genres.excluded.indexOf(genre) > -1} />
              <ListItemText primary={genre} />
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControlLabel
        control={
          <Checkbox
            defaultChecked
            name={"exclude_watchlist"}
            value={excludeWatchlist}
            checked={excludeWatchlist}
            onChange={(e) => setExcludeWatchlist(e.target.checked)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                setExcludeWatchlist(!e.target.checked);
              }
            }}
          />
        }
        label="Exclude movies already on watchlist"
      />
    </div>
  );
};

export default React.memo(ListFilters);
