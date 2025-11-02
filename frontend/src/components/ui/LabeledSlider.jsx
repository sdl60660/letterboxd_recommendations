import React from "react";
import "../../styles/Controls.scss";

import { Slider, Grid, FormLabel } from "@mui/material";

const LabeledSlider = (props) => {
  const {
    labels,
    defaultValue,
    marks = true,
    valueLabelDisplay = "off",
  } = props;

  return (
    <Grid container spacing={2} alignItems="center">
      <Grid item className="slider-end-label slider-end-label--left">
        <FormLabel>{labels[0]}</FormLabel>
      </Grid>
      <Grid item xs className="slider-wrapper">
        <Slider
          defaultValue={defaultValue}
          valueLabelDisplay={valueLabelDisplay}
          marks={marks}
          {...props}
        />
      </Grid>
      <Grid item className={"slider-end-label slider-end-label--right"}>
        <FormLabel>{labels[1]}</FormLabel>
      </Grid>
    </Grid>
  );
};

export default LabeledSlider;
