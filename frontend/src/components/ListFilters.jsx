import React, { useState } from 'react'

import { useTheme } from '@mui/material/styles'
import Box from '@mui/material/Box'
import OutlinedInput from '@mui/material/OutlinedInput'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import FormControl from '@mui/material/FormControl'
import FormControlLabel from '@mui/material/FormControlLabel'
import Select from '@mui/material/Select'
import Chip from '@mui/material/Chip'
import ListItemText from '@mui/material/ListItemText'
import Checkbox from '@mui/material/Checkbox'
import Slider from '@mui/material/Slider'

import '../styles/ListFilters.scss'

const ITEM_HEIGHT = 48
const ITEM_PADDING_TOP = 8
const MenuProps = {
    PaperProps: {
        style: {
            maxHeight: ITEM_HEIGHT * 7.5 + ITEM_PADDING_TOP,
            width: 400,
            maxWidth: '90vw',
        },
    },
}

function getStyles(name, personName, theme) {
    return {
        fontWeight:
            personName.indexOf(name) === -1
                ? theme.typography.fontWeightRegular
                : theme.typography.fontWeightMedium,
    }
}

const ListFilters = ({
    results,
    setFilteredGenres,
    setFilteredYearRange,
    excludeWatchlist,
    setExcludeWatchlist,
}) => {
    const allGenres = [
        ...new Set(
            results
                .map((d) => d.movie_data.genres)
                .flat()
                .filter((d) => d && d !== '')
        ),
    ]
    const allYears = [
        Math.min(...results.map((d) => d.movie_data.year_released)),
        Math.max(...results.map((d) => d.movie_data.year_released)),
    ]

    const [genres, setGenres] = useState(allGenres)
    const [yearRange, setYearRange] = useState(allYears)

    const theme = useTheme()

    const handleGenreChange = (event) => {
        const {
            target: { value },
        } = event

        // On autofill we get a stringified value.
        const newGenreVal = typeof value === 'string' ? value.split(',') : value

        setGenres(newGenreVal)
        setFilteredGenres(
            newGenreVal.length === allGenres.length ? null : newGenreVal
        )
    }

    const handleYearChange = (event, newValue) => {
        setYearRange(newValue)
        setFilteredYearRange(newValue)
    }

    return (
        <div className="list-filter-controls">
            <FormControl>
                <Box sx={{ width: 400, maxWidth: '90vw' }}>
                    <InputLabel id="year-filter-label" shrink={true}>
                        Year Released
                    </InputLabel>
                    <Slider
                        labelId="year-filter-label"
                        id="year-filter"
                        getAriaLabel={() => 'Year Released filter'}
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

            <FormControl sx={{ m: 1, width: 400, maxWidth: '90vw' }}>
                <InputLabel id="genre-filter-label">Genres</InputLabel>
                <Select
                    labelId="genre-filter-label"
                    id="genre-filter"
                    multiple
                    value={genres}
                    onChange={handleGenreChange}
                    getAriaLabel={() => 'Genre filter'}
                    input={
                        <OutlinedInput
                            id="select-multiple-chip"
                            label="Genres"
                        />
                    }
                    renderValue={(selected) =>
                        selected.length === allGenres.length ? (
                            <div className="default-all-display">All</div>
                        ) : (
                            <Box
                                sx={{
                                    display: 'flex',
                                    flexWrap: 'wrap',
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
                            style={getStyles(genre, genres, theme)}
                        >
                            <Checkbox checked={genres.indexOf(genre) > -1} />
                            <ListItemText primary={genre} />
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>

            <FormControlLabel
                control={
                    <Checkbox
                        defaultChecked
                        name={'exclude_watchlist'}
                        value={excludeWatchlist}
                        checked={excludeWatchlist}
                        onChange={(e) => setExcludeWatchlist(e.target.checked)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                                e.preventDefault()
                                setExcludeWatchlist(!e.target.checked)
                            }
                        }}
                    />
                }
                label="Exclude movies already on watchlist"
            />
        </div>
    )
}

export default ListFilters
