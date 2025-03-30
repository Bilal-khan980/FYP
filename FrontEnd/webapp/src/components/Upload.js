import React, { useState } from 'react';
import axios from 'axios';
import './Upload.css';

function Upload() {
  const [file, setFile] = useState(null);
  const [violationType, setViolationType] = useState('');
  const [uploading, setUploading] = useState(false);

  const violationTypes = [
    { value: 'overspeeding', label: 'Overspeeding' },
    { value: 'illegal_lane_change', label: 'Illegal Lane Change' },
    { value: 'driving_on_lane_line', label: 'Driving on Lane Line' },
    { value: 'damaged_brake_lights', label: 'Damaged Brake Lights' },
    { value: 'driving_htv_first_lane', label: 'Driving HTV in First Lane' },
    { value: 'illegal_license_plate', label: 'Illegal License Plate' }
  ];

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file || !violationType) {
      alert('Please select a file and violation type');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('violationType', violationType);
    formData.append('userId', JSON.parse(localStorage.getItem('user')).id);

    setUploading(true);
    try {
      await axios.post('http://localhost:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      alert('Upload successful!');
      setFile(null);
      setViolationType('');
    } catch (error) {
      console.error('Upload error:', error);
      alert('Upload failed');
    }
    setUploading(false);
  };

  return (
    <div className="upload-container">
      <h2>Upload Traffic Violation</h2>
      <form onSubmit={handleUpload} className="upload-form">
        <div className="form-group">
          <label>Select File (Video/Image):</label>
          <input
            type="file"
            accept="video/*,image/*"
            onChange={handleFileChange}
          />
        </div>
        <div className="form-group">
          <label>Violation Type:</label>
          <select
            value={violationType}
            onChange={(e) => setViolationType(e.target.value)}
            required
          >
            <option value="">Select violation type</option>
            {violationTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>
        <button type="submit" disabled={uploading}>
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </form>
    </div>
  );
}

export default Upload;
