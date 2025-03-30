import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Results.css';

function Results() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      const userId = JSON.parse(localStorage.getItem('user')).id;
      const response = await axios.get(`http://localhost:5000/api/videos/${userId}`);
      setVideos(response.data);
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
    setLoading(false);
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div className="results-container">
      <h2>Your Uploaded Videos</h2>
      <div className="videos-grid">
        {videos.map(video => (
          <div key={video._id} className="video-card">
            <h3>{video.originalName}</h3>
            <p><strong>Violation Type:</strong> {video.violationType.replace(/_/g, ' ')}</p>
            <p><strong>Status:</strong> {video.status}</p>
            <p><strong>Results:</strong> {video.results}</p>
            <p><strong>Uploaded:</strong> {new Date(video.createdAt).toLocaleDateString()}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Results;