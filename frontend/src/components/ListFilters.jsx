import React from "react";

import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import OutlinedInput from '@mui/material/OutlinedInput';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import Chip from '@mui/material/Chip';
import ListItemText from '@mui/material/ListItemText';
import Checkbox from '@mui/material/Checkbox';

import "../styles/ListFilters.scss";


const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 7.5 + ITEM_PADDING_TOP,
      width: 400,
      maxWidth: "90vw"
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

const ListFilters = ({ results, setFilteredGenres, setFilteredYearRange }) => {

    const allGenres = [...new Set(results.map(d => d.movie_data.genres).flat().filter(d => d && d !== ""))];
    const theme = useTheme();
    const [genres, setGenres] = React.useState(allGenres);

    const handleChange = (event) => {
        const {
        target: { value },
        } = event;

        // On autofill we get a stringified value.
        const newGenreVal = typeof value === 'string' ? value.split(',') : value;

        setGenres(newGenreVal);
        setFilteredGenres(newGenreVal);
    };

    return (
        <FormControl sx={{ m: 1, width: 400, maxWidth: "90vw" }}>
            <InputLabel id="genre-filter-label">Genres</InputLabel>
            <Select
                labelId="genre-filter-label"
                id="genre-filter"
                multiple
                value={genres}
                onChange={handleChange}
                input={<OutlinedInput id="select-multiple-chip" label="Genres" />}
                renderValue={(selected) => (
                    selected.length === allGenres.length ?
                    <div className="default-all-display">All</div> : 
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => (
                            <Chip key={value} label={value} />
                        ))}
                    </Box>
                )}
                MenuProps={MenuProps}
            >
            {allGenres.map((genre) => (
                <MenuItem
                    key={genre}
                    value={genre}
                    style={getStyles(genre, genres, theme)}
                >
                    <Checkbox checked={genres.indexOf(genre) > -1} />
                    <ListItemText primary={genre} />
                </MenuItem>
            ))}
            </Select>
      </FormControl>
    )
}

export default ListFilters;