import { useState, useEffect } from 'react';
import axios from 'axios';

const Report = () => {
  const [reports, setReports] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReports = async () => {
      const token = localStorage.getItem('sb:token');
      if (!token) {
        setError('Please login first');
        return;
      }

      try {
        const response = await axios.get(`${import.meta.env.VITE_API_URL}/api/report`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setReports(response.data.reports);
      } catch (err: any) {
        setError(err.response?.data?.error || 'Failed to fetch reports');
      }
    };

    fetchReports();
  }, []);

  return (
    <div>
      <h2>Prediction History</h2>
      {error && <p>{error}</p>}
      {reports.length > 0 ? (
        <ul>
          {reports.map((report) => (
            <li key={report.id}>
              <p>Text: {report.text}</p>
              <p>Predictions: {JSON.stringify(report.prediction)}</p>
              <p>Date: {new Date(report.created_at).toLocaleString()}</p>
            </li>
          ))}
        </ul>
      ) : (
        <p>No predictions found</p>
      )}
    </div>
  );
};

export default Report;