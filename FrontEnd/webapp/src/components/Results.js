"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import { motion } from "framer-motion"
import { FileVideo, Clock, CheckCircle, AlertTriangle, Calendar, Tag, Info } from "lucide-react"
import "./Results.css"

function Results() {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchVideos()
  }, [])

  const fetchVideos = async () => {
    try {
      const userId = JSON.parse(localStorage.getItem("user")).id
      const response = await axios.get(`http://localhost:5000/api/videos/${userId}`)
      setVideos(response.data)
      setError(null)
    } catch (error) {
      console.error("Error fetching videos:", error)
      setError("Failed to load your videos. Please try again later.")
    }
    setLoading(false)
  }

  const formatViolationType = (type) => {
    return type
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ")
  }

  const getStatusIcon = (status) => {
    switch (status.toLowerCase()) {
      case "processing":
        return <Clock className="status-icon processing" />
      case "completed":
        return <CheckCircle className="status-icon completed" />
      case "failed":
        return <AlertTriangle className="status-icon failed" />
      default:
        return <Info className="status-icon" />
    }
  }

  const getStatusClass = (status) => {
    return `status-badge ${status.toLowerCase()}`
  }

  // Animation variants for staggered list
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  }

  if (loading) {
    return (
      <div className="results-loading">
        <div className="loading-spinner-large">
          <div className="spinner-large"></div>
        </div>
        <p>Loading your videos...</p>
      </div>
    )
  }

  return (
    <div className="results-container">
      <div className="results-header">
        <h2>Analysis Results</h2>
        <p>View the status and results of your uploaded traffic violation evidence</p>
      </div>

      {error && (
        <div className="error-message">
          <AlertTriangle size={20} />
          <span>{error}</span>
        </div>
      )}

      {videos.length === 0 ? (
        <div className="empty-state">
          <FileVideo size={48} className="empty-icon" />
          <h3>No videos found</h3>
          <p>You haven't uploaded any videos for analysis yet.</p>
          <button className="refresh-button" onClick={fetchVideos}>
            Refresh
          </button>
        </div>
      ) : (
        <motion.div className="videos-grid" variants={container} initial="hidden" animate="show">
          {videos.map((video) => (
            <motion.div key={video._id} className="video-card" variants={item}>
              <div className="video-card-header">
                <FileVideo className="video-icon" />
                <div className={getStatusClass(video.status)}>
                  {getStatusIcon(video.status)}
                  <span>{video.status}</span>
                </div>
              </div>

              <h3 className="video-title" title={video.originalName}>
                {video.originalName}
              </h3>

              <div className="video-details">
                <div className="detail-item">
                  <Tag size={16} />
                  <span>
                    <strong>Violation:</strong> {formatViolationType(video.violationType)}
                  </span>
                </div>

                <div className="detail-item">
                  <Calendar size={16} />
                  <span>
                    <strong>Uploaded:</strong>{" "}
                    {new Date(video.createdAt).toLocaleDateString(undefined, {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                    })}
                  </span>
                </div>
              </div>

              <div className="results-section">
                <h4>Analysis Results</h4>
                <p className="results-text">{video.results || "Results pending analysis completion."}</p>
              </div>

              <div className="card-actions">
                <button className="action-button">View Details</button>
              </div>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  )
}

export default Results

