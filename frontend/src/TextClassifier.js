import React, { useState } from 'react';
import axios from 'axios';

const TextClassifier = () => {
  const [text, setText] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    try {
      const token = localStorage.getItem('sb:token'); // Assumes Supabase auth token
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/predict`,
        { text },
        { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } }
      );
      setPredictions(response.data.predictions);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to classify text');
      setPredictions(null);
    }
  };

  return (
    <div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} />
      <button onClick={handleSubmit}>Classify</button>
      {predictions && (
        <div>
          <p>SVM: {predictions.svm}</p>
          <p>Naive Bayes: {predictions.naive_bayes}</p>
          <p>CNN: {predictions.cnn}</p>
          <p>LSTM: {predictions.lstm}</p>
          <p>BERT: {predictions.bert}</p>
          <p>RoBERTa: {predictions.roberta}</p>
        </div>
      )}
      {error && <p>Error: {error}</p>}
    </div>
  );
};

export default TextClassifier;