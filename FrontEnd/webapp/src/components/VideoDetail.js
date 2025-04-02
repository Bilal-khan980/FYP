"use client"

import { useState, useEffect } from "react"
import { useParams, useNavigate } from "react-router-dom"
import axios from "axios"
import { ArrowLeft, Calendar, Tag, FileVideo, AlertTriangle, CheckCircle, XCircle } from "lucide-react"
import { motion } from "framer-motion"
import "./VideoDetail.css"

function VideoDetail() {
  const { videoId } = useParams()
  const navigate = useNavigate()
  const [video, setVideo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchVideoDetails()
  }, [videoId])

  const fetchVideoDetails = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/api/videos/detail/${videoId}`)
      setVideo(response.data)
      setError(null)
    } catch (error) {
      console.error("Error fetching video details:", error)
      setError("Failed to load video details. Please try again later.")
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
    if (status === "processed") return <CheckCircle size={20} />
    return <div className="status-dot"></div>
  }

  const getResultIcon = (results) => {
    if (results === "No violation") return <CheckCircle size={20} className="result-icon no-violation" />
    if (results === "HTV in first lane detected") return <XCircle size={20} className="result-icon violation" />
    return null
  }

  const getStatusClass = (status) => {
    return `status-badge ${status === "processed" ? "processed" : "pending"}`
  }

  const handleBack = () => {
    navigate(-1)
  }

  if (loading) {
    return (
      <div className="video-detail-loading">
        <div className="loading-spinner-large">
          <div className="spinner-large"></div>
        </div>
        <p>Loading video details...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="video-detail-error">
        <AlertTriangle size={48} />
        <h3>Error</h3>
        <p>{error}</p>
        <button className="back-button" onClick={handleBack}>
          <ArrowLeft size={16} />
          <span>Back to Results</span>
        </button>
      </div>
    )
  }

  if (!video) {
    return (
      <div className="video-detail-error">
        <AlertTriangle size={48} />
        <h3>Video Not Found</h3>
        <p>The requested video could not be found.</p>
        <button className="back-button" onClick={handleBack}>
          <ArrowLeft size={16} />
          <span>Back to Results</span>
        </button>
      </div>
    )
  }

  return (
    <div className="video-detail-container">
      <button className="back-button" onClick={handleBack}>
        <ArrowLeft size={16} />
        <span>Back to Results</span>
      </button>

      <div className="video-detail-header">
        <h2>{video.originalName}</h2>
        <div className={getStatusClass(video.status)}>
          {getStatusIcon(video.status)}
          <span>{video.status}</span>
        </div>
      </div>

      <div className="video-detail-content">
        <div className="video-detail-info">
          <div className="detail-item">
            <Tag size={16} />
            <span>
              <strong>Violation Type:</strong> {formatViolationType(video.violationType)}
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
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>

          <div className="detail-item">
            <FileVideo size={16} />
            <span>
              <strong>File:</strong> {video.filename}
            </span>
          </div>
        </div>

        <div className="video-detail-media">
          <div className="original-video">
            <h3>Original Video</h3>
            <div className="video-player">
              <video controls>
                <source src={`http://localhost:5000/uploads/${video.filename}`} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
          </div>

          {video.detectionProof && (
            <div className="detection-proof">
              <h3>Detection Result</h3>
              <div className="result-header">
                {getResultIcon(video.results)}
                <span className={video.results === "No violation" ? "no-violation" : "violation"}>
                  {video.results}
                </span>
              </div>
              <div className="proof-image">
                <img src={`http://localhost:5000/outputs/${video.detectionProof}`} alt="Detection Proof" />
              </div>
            </div>
          )}
        </div>

        <div className="analysis-results">
          <h3>Analysis Results</h3>
          <div className="results-content">
            {video.status === "processed" ? (
              <p className={video.results === "No violation" ? "no-violation-text" : "violation-text"}>
                {video.results}
              </p>
            ) : (
              <p className="pending-text">Analysis in progress. Please check back later.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default VideoDetail
