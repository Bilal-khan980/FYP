
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

// Image Schema (similar to Video schema)
const imageSchema = new mongoose.Schema({
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
    enum: ['illegal_license_plate'], // Only for license plate detection
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
  },
  isLegal: {
    type: Boolean,
    default: null
  }
}, { timestamps: true });

const Image = mongoose.model('Image', imageSchema);

// Add upload endpoint for both videos and images
app.post('/api/upload', async (req, res) => {
  upload(req, res, async (err) => {
    if (err) {
      return res.status(400).json({ message: err });
    }
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    try {
      const isImage = req.file.mimetype.startsWith('image/');
      const isVideo = req.file.mimetype.startsWith('video/');
      const violationType = req.body.violationType;

      // Validate file type based on violation type
      if (violationType === 'illegal_license_plate' && !isImage) {
        return res.status(400).json({ message: 'Please upload an image file for license plate detection' });
      } else if (violationType !== 'illegal_license_plate' && !isVideo) {
        return res.status(400).json({ message: 'Please upload a video file for this violation type' });
      }

      // Handle image upload for license plate detection
      if (isImage && violationType === 'illegal_license_plate') {
        const image = new Image({
          filename: req.file.filename,
          originalName: req.file.originalname,
          uploadedBy: req.body.userId,
          violationType: violationType,
        });

        await image.save();

        // Process the image with license plate detection API
        processLicensePlateImage(image._id, req.file.filename);

        return res.status(200).json({ message: 'Image upload successful', image });
      }

      // Handle video upload
      const video = new Video({
        filename: req.file.filename,
        originalName: req.file.originalname,
        uploadedBy: req.body.userId,
        violationType: violationType,
      });

      await video.save();

      // Process the video based on violation type
      if (violationType === 'driving_htv_first_lane') {
        processHTVVideo(video._id, req.file.filename);
      } else if (violationType === 'illegal_license_plate') {
        processLicensePlateVideo(video._id, req.file.filename);
      } else if (violationType === 'driving_on_lane_line') {
        processDrivingOnLaneVideo(video._id, req.file.filename);
      } else if (violationType === 'illegal_lane_change') {
        processIllegalLaneChangeVideo(video._id, req.file.filename);
      } else if (violationType === 'overspeeding') {
        processOverspeedingVideo(video._id, req.file.filename);
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

      res.status(200).json({ message: 'Video upload successful', video });
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

// Function to process License Plate videos with the FastAPI service
async function processLicensePlateVideo(videoId, filename) {
  try {
    console.log(`Processing License Plate video ${filename} with ID ${videoId}`);

    const videoPath = path.join(__dirname, 'uploads', filename);

    // Create form data for the FastAPI request
    const formData = new FormData();
    formData.append('file', fs.createReadStream(videoPath));

    // Send the video to the License Plate Classification FastAPI service
    const response = await axios.post('http://localhost:8001/process-video/', formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    console.log('License Plate FastAPI response:', response.data);

    // Update the video record with the results
    if (response.data.violation_detected) {
      // If violation detected, download the image
      const imageUrl = `http://localhost:8001${response.data.image_url}`;
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

      console.log(`Illegal license plate detected for video ${filename}. Image saved at ${imagePath}`);
    } else {
      // No violation detected
      await Video.findByIdAndUpdate(videoId, {
        status: 'processed',
        results: response.data.message
      });

      console.log(`No illegal license plate detected for video ${filename}`);
    }

    // Clean up the uploaded video file
    try {
      fs.unlinkSync(videoPath);
      console.log(`Deleted uploaded video file: ${videoPath}`);
    } catch (unlinkError) {
      console.error(`Failed to delete uploaded video file: ${videoPath}`, unlinkError);
    }
  } catch (error) {
    console.error('Error processing license plate video:', error);
    await Video.findByIdAndUpdate(videoId, {
      status: 'processed',
      results: 'Error processing video'
    });
  }
}

// Function to process Overspeeding videos with the FastAPI service
async function processOverspeedingVideo(videoId, filename) {
  try {
    console.log(`Processing Overspeeding video ${filename} with ID ${videoId}`);

    const videoPath = path.join(__dirname, 'uploads', filename);

    // Make a copy of the video file to ensure we have it for viewing later
    const videoBackupPath = path.join(__dirname, 'uploads', `backup_${filename}`);
    fs.copyFileSync(videoPath, videoBackupPath);
    console.log(`Created backup of video file at: ${videoBackupPath}`);

    // Create form data for the FastAPI request
    const formData = new FormData();
    formData.append('file', fs.createReadStream(videoPath));

    // Send the video to the Overspeeding Detection FastAPI service
    const response = await axios.post('http://localhost:8005/process-video/', formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    console.log('Overspeeding FastAPI response:', response.data);

    // Update the video record with the results
    if (response.data.violation_detected) {
      // If violation detected, download the image
      const imageUrl = `http://localhost:8005${response.data.image_url}`;
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

      console.log(`Overspeeding violation detected for video ${filename}. Image saved at ${imagePath}`);
    } else {
      // No violation detected
      await Video.findByIdAndUpdate(videoId, {
        status: 'processed',
        results: response.data.message
      });

      console.log(`No overspeeding violation detected for video ${filename}`);
    }
  } catch (error) {
    console.error('Error processing overspeeding video:', error);

    // Update the video record with the error
    await Video.findByIdAndUpdate(videoId, {
      status: 'processed',
      results: 'Error processing video'
    });

    // Ensure the video file is restored in case of error
    try {
      const videoPath = path.join(__dirname, 'uploads', filename);
      const videoBackupPath = path.join(__dirname, 'uploads', `backup_${filename}`);

      if (!fs.existsSync(videoPath) && fs.existsSync(videoBackupPath)) {
        fs.copyFileSync(videoBackupPath, videoPath);
        console.log(`Restored video file from backup after error: ${videoPath}`);
      }

      // Clean up the backup file if it exists
      if (fs.existsSync(videoBackupPath)) {
        fs.unlinkSync(videoBackupPath);
        console.log(`Removed backup video file: ${videoBackupPath}`);
      }
    } catch (fileError) {
      console.error(`Error handling video files during error recovery: ${fileError}`);
    }
  }
}

// Function to process Illegal Lane Change videos with the FastAPI service
async function processIllegalLaneChangeVideo(videoId, filename) {
  try {
    console.log(`Processing Illegal Lane Change video ${filename} with ID ${videoId}`);

    const videoPath = path.join(__dirname, 'uploads', filename);

    // Make a copy of the video file to ensure we have it for viewing later
    const videoBackupPath = path.join(__dirname, 'uploads', `backup_${filename}`);
    fs.copyFileSync(videoPath, videoBackupPath);
    console.log(`Created backup of video file at: ${videoBackupPath}`);

    // Create form data for the FastAPI request
    const formData = new FormData();
    formData.append('file', fs.createReadStream(videoPath));

    // Send the video to the Illegal Lane Change Detection FastAPI service
    const response = await axios.post('http://localhost:8004/process-video/', formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    console.log('Illegal Lane Change FastAPI response:', response.data);

    // Update the video record with the results
    if (response.data.violation_detected) {
      // If violation detected, download the image
      const imageUrl = `http://localhost:8004${response.data.image_url}`;
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

      console.log(`Illegal lane change violation detected for video ${filename}. Image saved at ${imagePath}`);
    } else {
      // No violation detected
      await Video.findByIdAndUpdate(videoId, {
        status: 'processed',
        results: response.data.message
      });

      console.log(`No illegal lane change violation detected for video ${filename}`);
    }
  } catch (error) {
    console.error('Error processing illegal lane change video:', error);

    // Update the video record with the error
    await Video.findByIdAndUpdate(videoId, {
      status: 'processed',
      results: 'Error processing video'
    });

    // Ensure the video file is restored in case of error
    try {
      const videoPath = path.join(__dirname, 'uploads', filename);
      const videoBackupPath = path.join(__dirname, 'uploads', `backup_${filename}`);

      if (!fs.existsSync(videoPath) && fs.existsSync(videoBackupPath)) {
        fs.copyFileSync(videoBackupPath, videoPath);
        console.log(`Restored video file from backup after error: ${videoPath}`);
      }

      // Clean up the backup file if it exists
      if (fs.existsSync(videoBackupPath)) {
        fs.unlinkSync(videoBackupPath);
        console.log(`Removed backup video file: ${videoBackupPath}`);
      }
    } catch (fileError) {
      console.error(`Error handling video files during error recovery: ${fileError}`);
    }
  }
}

// Function to process Driving on Lane Line videos with the FastAPI service
async function processDrivingOnLaneVideo(videoId, filename) {
  try {
    console.log(`Processing Driving on Lane Line video ${filename} with ID ${videoId}`);

    const videoPath = path.join(__dirname, 'uploads', filename);

    // Make a copy of the video file to ensure we have it for viewing later
    const videoBackupPath = path.join(__dirname, 'uploads', `backup_${filename}`);
    fs.copyFileSync(videoPath, videoBackupPath);
    console.log(`Created backup of video file at: ${videoBackupPath}`);

    // Create form data for the FastAPI request
    const formData = new FormData();
    formData.append('file', fs.createReadStream(videoPath));

    // Send the video to the Driving on Lane Line Detection FastAPI service
    const response = await axios.post('http://localhost:8003/process-video/', formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    console.log('Driving on Lane Line FastAPI response:', response.data);

    // Update the video record with the results
    if (response.data.violation_detected) {
      // If violation detected, download the image
      const imageUrl = `http://localhost:8003${response.data.image_url}`;
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

      console.log(`Driving on lane line violation detected for video ${filename}. Image saved at ${imagePath}`);
    } else {
      // No violation detected
      await Video.findByIdAndUpdate(videoId, {
        status: 'processed',
        results: response.data.message
      });

      console.log(`No driving on lane line violation detected for video ${filename}`);
    }

    // Restore the backup video file to ensure it's available for viewing in the results
    try {
      // If the original file was deleted during processing, restore it from the backup
      if (!fs.existsSync(videoPath)) {
        fs.copyFileSync(videoBackupPath, videoPath);
        console.log(`Restored video file from backup for viewing: ${videoPath}`);
      }

      // Clean up the backup file if it exists
      if (fs.existsSync(videoBackupPath)) {
        fs.unlinkSync(videoBackupPath);
        console.log(`Removed backup video file: ${videoBackupPath}`);
      }
    } catch (fileError) {
      console.error(`Error handling video files: ${fileError}`);
    }
  } catch (error) {
    console.error('Error processing driving on lane line video:', error);

    // Update the video record with the error
    await Video.findByIdAndUpdate(videoId, {
      status: 'processed',
      results: 'Error processing video'
    });

    // Ensure the video file is restored in case of error
    try {
      if (!fs.existsSync(videoPath) && fs.existsSync(videoBackupPath)) {
        fs.copyFileSync(videoBackupPath, videoPath);
        console.log(`Restored video file from backup after error: ${videoPath}`);
      }

      // Clean up the backup file if it exists
      if (fs.existsSync(videoBackupPath)) {
        fs.unlinkSync(videoBackupPath);
        console.log(`Removed backup video file: ${videoBackupPath}`);
      }
    } catch (fileError) {
      console.error(`Error handling video files during error recovery: ${fileError}`);
    }
  }
}

// Function to process License Plate images with the FastAPI service
async function processLicensePlateImage(imageId, filename) {
  try {
    console.log(`Processing License Plate image ${filename} with ID ${imageId}`);

    const imagePath = path.join(__dirname, 'uploads', filename);

    // Create form data for the FastAPI request
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));

    // Send the image to the License Plate Classification FastAPI service
    const response = await axios.post('http://localhost:8002/detect-license-plate/', formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    console.log('License Plate Image API response:', response.data);

    // Check if there are any detections
    if (response.data.detections_count > 0) {
      // Get the processed image
      const imageUrl = `http://localhost:8002${response.data.image_url}`;
      const outputImageName = path.basename(response.data.image_url);
      const outputImagePath = path.join(__dirname, 'outputs', outputImageName);

      // Download the processed image
      const imageResponse = await axios({
        method: 'get',
        url: imageUrl,
        responseType: 'stream'
      });

      // Save the image to the outputs directory
      const writer = fs.createWriteStream(outputImagePath);
      imageResponse.data.pipe(writer);

      await new Promise((resolve, reject) => {
        writer.on('finish', resolve);
        writer.on('error', reject);
      });

      // Check if any illegal plates were detected
      const illegalPlates = response.data.detections.filter(detection => !detection.is_legal);
      const isLegal = illegalPlates.length === 0;
      const resultMessage = isLegal
        ? 'All license plates detected are legal.'
        : `${illegalPlates.length} illegal license plate(s) detected.`;

      // Update the image record
      await Image.findByIdAndUpdate(imageId, {
        status: 'processed',
        results: resultMessage,
        detectionProof: outputImageName,
        isLegal: isLegal
      });

      console.log(`License plate image processed: ${filename}. Result: ${resultMessage}`);
    } else {
      // No license plates detected
      await Image.findByIdAndUpdate(imageId, {
        status: 'processed',
        results: 'No license plates detected in the image.',
        isLegal: null
      });

      console.log(`No license plates detected in image ${filename}`);
    }

    // Clean up the uploaded image file
    try {
      fs.unlinkSync(imagePath);
      console.log(`Deleted uploaded image file: ${imagePath}`);
    } catch (unlinkError) {
      console.error(`Failed to delete uploaded image file: ${imagePath}`, unlinkError);
    }
  } catch (error) {
    console.error('Error processing license plate image:', error);
    await Image.findByIdAndUpdate(imageId, {
      status: 'processed',
      results: 'Error processing image'
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

// Get user's images endpoint
app.get('/api/images/:userId', async (req, res) => {
  try {
    const images = await Image.find({ uploadedBy: req.params.userId })
      .sort({ createdAt: -1 });
    res.json(images);
  } catch (error) {
    console.error('Error fetching images:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get all user's media (both videos and images)
app.get('/api/media/:userId', async (req, res) => {
  try {
    const videos = await Video.find({ uploadedBy: req.params.userId });
    const images = await Image.find({ uploadedBy: req.params.userId });

    // Combine videos and images into a single array with a type field
    const media = [
      ...videos.map(v => ({ ...v.toObject(), mediaType: 'video' })),
      ...images.map(i => ({ ...i.toObject(), mediaType: 'image' }))
    ];

    // Sort by creation date, newest first
    media.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    res.json(media);
  } catch (error) {
    console.error('Error fetching media:', error);
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

// Get image details endpoint
app.get('/api/images/detail/:imageId', async (req, res) => {
  try {
    const image = await Image.findById(req.params.imageId);
    if (!image) {
      return res.status(404).json({ message: 'Image not found' });
    }
    res.json(image);
  } catch (error) {
    console.error('Error fetching image details:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
