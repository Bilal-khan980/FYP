
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));
app.use('/outputs', express.static('outputs'));

// Create outputs directory if it doesn't exist
const outputsDir = path.join(__dirname, 'outputs');
if (!fs.existsSync(outputsDir)) {
  fs.mkdirSync(outputsDir, { recursive: true });
}

// MongoDB connection
mongoose.connect('mongodb+srv://Bilalkhan:Pakistan@cluster1.moct8fi.mongodb.net/BeAWarden', {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('Connected to MongoDB'))
.catch((err) => console.error('MongoDB connection error:', err));

// User Schema
const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true
  },
  email: {
    type: String,
    required: true,
    unique: true
  },
  password: {
    type: String,
    required: true
  }
}, { timestamps: true });

const User = mongoose.model('User', userSchema);

// Registration endpoint
app.post('/api/register', async (req, res) => {
  try {
    const { username, email, password } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(400).json({ message: 'User already exists' });
    }

    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create new user
    const newUser = new User({
      username,
      email,
      password: hashedPassword
    });

    await newUser.save();
    res.status(201).json({ message: 'User registered successfully' });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Login endpoint
app.post('/api/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Check password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id, email: user.email },
      'your_jwt_secret', // In production, use an environment variable
      { expiresIn: '1h' }
    );

    res.json({
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Configure multer for video upload
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: function(req, file, cb) {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 100000000 }, // 100MB limit
  fileFilter: function(req, file, cb) {
    const filetypes = /mp4|mov|avi|mkv|jpg|jpeg|png/;
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = filetypes.test(file.mimetype);
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb('Error: Videos Only!');
    }
  }
}).single('file');

// Video Schema
const videoSchema = new mongoose.Schema({
  filename: {
    type: String,
    required: true
  },
  originalName: {
    type: String,
    required: true
  },
  uploadedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  violationType: {
    type: String,
    enum: ['overspeeding', 'illegal_lane_change', 'driving_on_lane_line',
           'damaged_brake_lights', 'driving_htv_first_lane', 'illegal_license_plate'],
    required: true
  },
  status: {
    type: String,
    enum: ['pending', 'processed'],
    default: 'pending'
  },
  results: {
    type: String,
    default: 'pending'
  },
  detectionProof: {
    type: String,
    default: null
  }
}, { timestamps: true });

const Video = mongoose.model('Video', videoSchema);

// Add video upload endpoint
app.post('/api/upload', async (req, res) => {
  upload(req, res, async (err) => {
    if (err) {
      return res.status(400).json({ message: err });
    }
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    try {
      const video = new Video({
        filename: req.file.filename,
        originalName: req.file.originalname,
        uploadedBy: req.body.userId,
        violationType: req.body.violationType,
      });

      await video.save();

      // Process the video if it's an HTV first lane violation
      if (req.body.violationType === 'driving_htv_first_lane') {
        processHTVVideo(video._id, req.file.filename);
      } else {
        // For other violation types, just mark as processed with a dummy result
        // In a real system, you would process these with appropriate AI models
        setTimeout(async () => {
          await Video.findByIdAndUpdate(video._id, {
            status: 'processed',
            results: 'Analysis completed (demo)'
          });
          console.log('Processed video:', req.file.filename);
        }, 5000);
      }

      res.status(200).json({ message: 'Upload successful', video });
    } catch (error) {
      console.error('Upload error:', error);
      res.status(500).json({ message: 'Server error' });
    }
  });
});

// Function to process HTV videos with the FastAPI service
async function processHTVVideo(videoId, filename) {
  try {
    console.log(`Processing HTV video ${filename} with ID ${videoId}`);

    const videoPath = path.join(__dirname, 'uploads', filename);

    // Create form data for the FastAPI request
    const formData = new FormData();
    formData.append('file', fs.createReadStream(videoPath));

    // Send the video to the FastAPI service
    const response = await axios.post('http://localhost:8000/process-video/', formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    console.log('FastAPI response:', response.data);

    // Update the video record with the results
    if (response.data.violation_detected) {
      // If violation detected, download the image
      const imageUrl = `http://localhost:8000${response.data.image_url}`;
      const imageName = path.basename(response.data.image_url);
      const imagePath = path.join(__dirname, 'outputs', imageName);

      // Download the image
      const imageResponse = await axios({
        method: 'get',
        url: imageUrl,
        responseType: 'stream'
      });

      // Save the image to the outputs directory
      const writer = fs.createWriteStream(imagePath);
      imageResponse.data.pipe(writer);

      await new Promise((resolve, reject) => {
        writer.on('finish', resolve);
        writer.on('error', reject);
      });

      // Update the video record
      await Video.findByIdAndUpdate(videoId, {
        status: 'processed',
        results: response.data.message,
        detectionProof: imageName
      });

      console.log(`HTV violation detected for video ${filename}. Image saved at ${imagePath}`);
    } else {
      // No violation detected
      await Video.findByIdAndUpdate(videoId, {
        status: 'processed',
        results: response.data.message
      });

      console.log(`No HTV violation detected for video ${filename}`);
    }
  } catch (error) {
    console.error('Error processing HTV video:', error);

    // Update the video record with the error
    await Video.findByIdAndUpdate(videoId, {
      status: 'processed',
      results: 'Error processing video'
    });
  }
}

// Get user's videos endpoint
app.get('/api/videos/:userId', async (req, res) => {
  try {
    const videos = await Video.find({ uploadedBy: req.params.userId })
      .sort({ createdAt: -1 });
    res.json(videos);
  } catch (error) {
    console.error('Error fetching videos:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get video details endpoint
app.get('/api/videos/detail/:videoId', async (req, res) => {
  try {
    const video = await Video.findById(req.params.videoId);
    if (!video) {
      return res.status(404).json({ message: 'Video not found' });
    }
    res.json(video);
  } catch (error) {
    console.error('Error fetching video details:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
