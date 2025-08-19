import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './Login';
import TextClassifier from './TextClassifier';
import Report from './Report';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/classify" element={<TextClassifier />} />
        <Route path="/report" element={<Report />} />
        <Route path="/" element={<Login />} />
      </Routes>
    </Router>
  );
}

export default App;