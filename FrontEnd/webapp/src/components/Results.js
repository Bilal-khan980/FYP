"use client"

import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import axios from "axios"
import { motion } from "framer-motion"
import { FileVideo, Clock, CheckCircle, AlertTriangle, Calendar, Tag, Info } from "lucide-react"
import "./Results.css"

function Results() {
  const navigate = useNavigate()
  const [media, setMedia] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchMedia()
  }, [])

  const fetchMedia = async () => {
    try {
      const userId = JSON.parse(localStorage.getItem("user")).id
      const response = await axios.get(`http://localhost:5000/api/media/${userId}`)
      setMedia(response.data)
      setError(null)
    } catch (error) {
      console.error("Error fetching media:", error)
      setError("Failed to load your videos and images. Please try again later.")
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

  const itemAnimation = {
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
        <p>View the status and results of your uploaded traffic violation evidence (videos and images)</p>
      </div>

      {error && (
        <div className="error-message">
          <AlertTriangle size={20} />
          <span>{error}</span>
        </div>
      )}

      {media.length === 0 ? (
        <div className="empty-state">
          <FileVideo size={48} className="empty-icon" />
          <h3>No media found</h3>
          <p>You haven't uploaded any videos or images for analysis yet.</p>
          <button className="refresh-button" onClick={fetchMedia}>
            Refresh
          </button>
        </div>
      ) : (
        <motion.div className="videos-grid" variants={container} initial="hidden" animate="show">
          {media.map((item) => (
            <motion.div key={item._id} className="video-card" variants={itemAnimation}>
              <div className="video-card-header">
                {item.mediaType === 'video' ? (
                  <FileVideo className="video-icon" />
                ) : (
                  <img src="/image-icon.svg" alt="License plate" className="video-icon" />
                )}
                <div className={getStatusClass(item.status)}>
                  {getStatusIcon(item.status)}
                  <span>{item.status}</span>
                </div>
              </div>

              <h3 className="video-title" title={item.originalName}>
                {item.originalName}
              </h3>

              <div className="video-details">
                <div className="detail-item">
                  <Tag size={16} />
                  <span>
                    <strong>Violation:</strong> {formatViolationType(item.violationType)}
                  </span>
                </div>

                <div className="detail-item">
                  <Calendar size={16} />
                  <span>
                    <strong>Uploaded:</strong>{" "}
                    {new Date(item.createdAt).toLocaleDateString(undefined, {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                    })}
                  </span>
                </div>
              </div>

              <div className="results-section">
                <h4>Analysis Results</h4>
                <p className="results-text">{item.results || "Results pending analysis completion."}</p>
              </div>

              <div className="card-actions">
                <button
                  className="action-button"
                  onClick={() => navigate(`/${item.mediaType}/${item._id}`)}
                >
                  View Details
                </button>
              </div>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  )
}

export default Results

