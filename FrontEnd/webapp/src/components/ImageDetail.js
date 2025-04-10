"use client"

import { useState, useEffect } from "react"
import { useParams, useNavigate } from "react-router-dom"
import axios from "axios"
import { ArrowLeft, Calendar, Tag, AlertTriangle, CheckCircle, XCircle } from "lucide-react"
import { motion } from "framer-motion"
import "./VideoDetail.css" // Reusing the same CSS

function ImageDetail() {
  const { imageId } = useParams()
  const navigate = useNavigate()
  const [image, setImage] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchImageDetails()
  }, [imageId])

  const fetchImageDetails = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/api/images/detail/${imageId}`)
      setImage(response.data)
      setError(null)
    } catch (error) {
      console.error("Error fetching image details:", error)
      setError("Failed to load image details. Please try again later.")
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

  const getResultIcon = (results, isLegal) => {
    if (isLegal === true) return <CheckCircle size={20} className="result-icon no-violation" />
    if (isLegal === false) return <XCircle size={20} className="result-icon violation" />
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
        <p>Loading image details...</p>
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

  if (!image) {
    return (
      <div className="video-detail-error">
        <AlertTriangle size={48} />
        <h3>Image Not Found</h3>
        <p>The requested image could not be found.</p>
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
        <h2>{image.originalName}</h2>
        <div className={getStatusClass(image.status)}>
          {getStatusIcon(image.status)}
          <span>{image.status}</span>
        </div>
      </div>

      <div className="video-detail-content">
        <div className="video-detail-info">
          <div className="detail-item">
            <Tag size={16} />
            <span>
              <strong>Violation Type:</strong> {formatViolationType(image.violationType)}
            </span>
          </div>

          <div className="detail-item">
            <Calendar size={16} />
            <span>
              <strong>Uploaded:</strong>{" "}
              {new Date(image.createdAt).toLocaleDateString(undefined, {
                year: "numeric",
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>
        </div>

        <div className="video-detail-media">
          {image.detectionProof ? (
            <div className="detection-proof">
              <h3>License Plate Detection Result</h3>
              <div className="result-header">
                {getResultIcon(image.results, image.isLegal)}
                <span className={image.isLegal ? "no-violation" : "violation"}>
                  {image.results}
                </span>
              </div>
              <div className="proof-image">
                <img src={`http://localhost:5000/outputs/${image.detectionProof}`} alt="License Plate Detection" />
              </div>
            </div>
          ) : (
            <div className="original-image">
              <h3>Original Image</h3>
              <div className="image-container">
                <img src={`http://localhost:5000/uploads/${image.filename}`} alt="Original" />
              </div>
            </div>
          )}
        </div>

        <div className="analysis-results">
          <h3>Analysis Results</h3>
          <div className="results-content">
            {image.status === "processed" ? (
              <p className={image.isLegal ? "no-violation-text" : "violation-text"}>
                {image.results}
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

export default ImageDetail
