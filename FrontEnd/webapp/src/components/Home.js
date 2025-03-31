"use client"

import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { LogOut, UploadIcon, List, Bell, Shield, Menu } from "lucide-react"
import Upload from "./Upload"
import Results from "./Results"
import "./Home.css"

function Home() {
  const navigate = useNavigate()
  const [user, setUser] = useState(null)
  const [activeTab, setActiveTab] = useState("upload")
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const userStr = localStorage.getItem("user")
    if (!userStr) {
      navigate("/login")
      return
    }
    setUser(JSON.parse(userStr))
  }, [navigate])

  const handleLogout = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("user")
    navigate("/login")
  }

  if (!user) return null

  return (
    <div className="home-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <Shield className="logo-icon" />
          <h2>TrafficAI</h2>
        </div>
        <div className="sidebar-menu">
          <button
            onClick={() => setActiveTab("upload")}
            className={`sidebar-button ${activeTab === "upload" ? "active" : ""}`}
          >
            <UploadIcon className="sidebar-icon" />
            <span>Upload</span>
          </button>
          <button
            onClick={() => setActiveTab("results")}
            className={`sidebar-button ${activeTab === "results" ? "active" : ""}`}
          >
            <List className="sidebar-icon" />
            <span>Results</span>
          </button>
        </div>
        <button onClick={handleLogout} className="logout-button">
          <LogOut className="sidebar-icon" />
          <span>Logout</span>
        </button>
      </div>

      <div className="main-content">
        <header className="top-header">
          <button className="mobile-menu-button" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
            <Menu />
          </button>
          <h1>Traffic Violation Detection</h1>
          <div className="header-actions">
            <button className="notification-button">
              <Bell />
              <span className="notification-badge">3</span>
            </button>
            <div className="user-profile">
              <div className="avatar">{user.username.charAt(0).toUpperCase()}</div>
              <span className="username">{user.username}</span>
            </div>
          </div>
        </header>

        {/* Mobile menu */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div
              className="mobile-menu"
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ duration: 0.3 }}
            >
              <div className="mobile-menu-header">
                <Shield className="logo-icon" />
                <h2>TrafficAI</h2>
                <button className="close-menu" onClick={() => setIsMobileMenuOpen(false)}>
                  ×
                </button>
              </div>
              <button
                onClick={() => {
                  setActiveTab("upload")
                  setIsMobileMenuOpen(false)
                }}
                className={`mobile-menu-button ${activeTab === "upload" ? "active" : ""}`}
              >
                <UploadIcon className="sidebar-icon" />
                <span>Upload</span>
              </button>
              <button
                onClick={() => {
                  setActiveTab("results")
                  setIsMobileMenuOpen(false)
                }}
                className={`mobile-menu-button ${activeTab === "results" ? "active" : ""}`}
              >
                <List className="sidebar-icon" />
                <span>Results</span>
              </button>
              <button onClick={handleLogout} className="mobile-menu-button logout">
                <LogOut className="sidebar-icon" />
                <span>Logout</span>
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="dashboard">
          <div className="welcome-card">
            <h2>Welcome, {user.username}!</h2>
            <p>Use our AI-powered system to detect traffic violations from uploaded footage.</p>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="content-container"
            >
              {activeTab === "upload" ? <Upload /> : <Results />}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

export default Home

