import './styles/App.scss';

import Header from "./components/Header";
import Intro from "./components/Intro";
import Controls from "./components/Controls";
import Results from "./components/Results";
import Footer from "./components/Footer";

function App() {
  return (
    <div className="App">
      <Header />
      <Intro />
      <Controls />
      <Results />
      <Footer />
    </div>
  );
}

export default App;
