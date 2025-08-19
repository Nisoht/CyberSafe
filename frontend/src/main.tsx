import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Login from './components/Login';
import TextClassifier from './components/TextClassifier';
import Report from './components/Report';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/classify" element={<TextClassifier />} />
        <Route path="/report" element={<Report />} />
        <Route path="/" element={<Login />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);