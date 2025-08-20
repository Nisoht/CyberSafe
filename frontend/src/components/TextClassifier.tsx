import { useState } from 'react';
import axios from 'axios';

const TextClassifier = () => {
  const [text, setText] = useState('');
  const [predictions, setPredictions] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClassify = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setPredictions(null);

    const token = localStorage.getItem('sb:token');
    if (!token) {
      setError('Please login first');
      return;
    }

    try {
      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}/api/predict`,
        { text },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        }
      );
      setPredictions(response.data.predictions);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to classify text');
    }
  };

  return (
    <div>
      <h2>Text Classifier</h2>
      <form onSubmit={handleClassify}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to classify"
          required
        />
        <button type="submit">Classify</button>
      </form>
      {error && <p>{error}</p>}
      {predictions && (
        <div>
          <h3>Predictions:</h3>
          <ul>
            {Object.entries(predictions).map(([model, result]) => (
              <li key={model}>{model}: {result as string}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default TextClassifier;