"use client"

import { useState, useRef } from "react"
import axios from "axios"
import { UploadIcon, FileType, AlertCircle, CheckCircle } from "lucide-react"
import { motion } from "framer-motion"
import "./Upload.css"

function Upload() {
  const [file, setFile] = useState(null)
  const [fileName, setFileName] = useState("")
  const [violationType, setViolationType] = useState("")
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null) // 'success', 'error', or null
  const fileInputRef = useRef(null)

  const violationTypes = [
    { value: "overspeeding", label: "Overspeeding" },
    { value: "illegal_lane_change", label: "Illegal Lane Change" },
    { value: "driving_on_lane_line", label: "Driving on Lane Line" },
    { value: "damaged_brake_lights", label: "Damaged Brake Lights" },
    { value: "driving_htv_first_lane", label: "Driving HTV in First Lane" },
    { value: "illegal_license_plate", label: "Illegal License Plate" },
  ]

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setFileName(selectedFile.name)
    }
  }

  const handleUpload = async (e) => {
    e.preventDefault()
    if (!file || !violationType) {
      setUploadStatus("error")
      setTimeout(() => setUploadStatus(null), 3000)
      return
    }

    const formData = new FormData()
    formData.append("file", file)
    formData.append("violationType", violationType)
    formData.append("userId", JSON.parse(localStorage.getItem("user")).id)

    setUploading(true)
    try {
      await axios.post("http://localhost:5000/api/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      setUploadStatus("success")
      setTimeout(() => {
        setUploadStatus(null)
        setFile(null)
        setFileName("")
        setViolationType("")
      }, 3000)
    } catch (error) {
      console.error("Upload error:", error)
      setUploadStatus("error")
      setTimeout(() => setUploadStatus(null), 3000)
    }
    setUploading(false)
  }

  const triggerFileInput = () => {
    fileInputRef.current.click()
  }

  return (
    <div className="upload-container">
      <div className="upload-header">
        <h2>Upload Traffic Violation</h2>
        <p>Upload video or image evidence of traffic violations for AI analysis</p>
      </div>

      {uploadStatus === "success" && (
        <motion.div
          className="status-message success"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
        >
          <CheckCircle size={20} />
          <span>Upload successful! Your file has been submitted for analysis.</span>
        </motion.div>
      )}

      {uploadStatus === "error" && (
        <motion.div
          className="status-message error"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
        >
          <AlertCircle size={20} />
          <span>Upload failed. Please ensure you've selected a file and violation type.</span>
        </motion.div>
      )}

      <form onSubmit={handleUpload} className="upload-form">
        <div className="form-group file-upload-group">
          <label>Evidence File</label>
          <div className={`file-upload-area ${file ? "has-file" : ""}`} onClick={triggerFileInput}>
            <input
              type="file"
              ref={fileInputRef}
              accept="video/*,image/*"
              onChange={handleFileChange}
              className="hidden-file-input"
            />
            <div className="file-upload-content">
              {!file ? (
                <>
                  <div className="upload-icon-container">
                    <UploadIcon size={32} className="upload-icon" />
                  </div>
                  <div className="upload-text">
                    <span className="primary-text">Drag & drop or click to upload</span>
                    <span className="secondary-text">Supports images and videos</span>
                  </div>
                </>
              ) : (
                <div className="selected-file">
                  <FileType size={24} />
                  <span className="file-name">{fileName}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="form-group">
          <label>Violation Type</label>
          <div className="select-wrapper">
            <select
              value={violationType}
              onChange={(e) => setViolationType(e.target.value)}
              required
              className="styled-select"
            >
              <option value="">Select violation type</option>
              {violationTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <motion.button
          type="submit"
          disabled={uploading}
          className="upload-button"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {uploading ? (
            <div className="loading-spinner">
              <div className="spinner"></div>
              <span>Uploading...</span>
            </div>
          ) : (
            <>
              <UploadIcon size={18} />
              <span>Upload for Analysis</span>
            </>
          )}
        </motion.button>
      </form>
    </div>
  )
}

export default Upload

