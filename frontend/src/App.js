import React, { useState } from "react";

import './styles/App.scss';

import Header from "./components/Header";
import Intro from "./components/Intro";
import Controls from "./components/Controls";
import ProgressTracking from "./components/ProgressTracking";
import Results from "./components/Results";
import Footer from "./components/Footer";

function App() {
  const [queryData, setQueryData] = useState(null);
  
  return (
    <div className="App">
      <Header />
      <Intro />
      <Controls setQueryData={setQueryData} />
      <ProgressTracking queryData={queryData} />
      <Results queryData={queryData} />
      <Footer />
    </div>
  );
}

export default App;
