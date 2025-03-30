import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Upload from './Upload';
import Results from './Results';
import './Home.css';

function Home() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');

  useEffect(() => {
    const userStr = localStorage.getItem('user');
    if (!userStr) {
      navigate('/login');
      return;
    }
    setUser(JSON.parse(userStr));
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  if (!user) return null;

  return (
    <div className="home-container">
      <nav className="home-nav">
        <h1>Traffic Violation Management System</h1>
        <div className="nav-buttons">
          <button onClick={() => setActiveTab('upload')} className={activeTab === 'upload' ? 'active' : ''}>
            Upload
          </button>
          <button onClick={() => setActiveTab('results')} className={activeTab === 'results' ? 'active' : ''}>
            Results
          </button>
          <button onClick={handleLogout} className="logout-button">
            Logout
          </button>
        </div>
      </nav>
      <div className="home-content">
        <div className="user-welcome">
          <h2>Welcome, {user.username}!</h2>
        </div>
        {activeTab === 'upload' ? <Upload /> : <Results />}
      </div>
    </div>
  );
}

export default Home;
