const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const bodyParser = require('body-parser');

// Initialize express
const app = express();

// Middleware
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect('mongodb://localhost/triptrek', { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB Connected'))
  .catch((err) => console.log(err));

// User Schema
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  data: { type: Object, default: {} }  // You can store user's travel plans or any other data here
});

const User = mongoose.model('User', userSchema);

// Handle login route
app.post('/login', async (req, res) => {
  const { email, password } = req.body;

  try {
    // Check if the user exists
    let user = await User.findOne({ email });
    
    // If user doesn't exist, create a new one
    if (!user) {
      const hashedPassword = await bcrypt.hash(password, 10);
      user = new User({ email, password: hashedPassword });
      await user.save();
      return res.json({ success: true, message: 'Account created! Redirecting to your dashboard...', userId: user._id });
    }

    // If user exists, check the password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ success: false, message: 'Invalid credentials' });
    }

    // Send back a response with user data or JWT token
    const token = jwt.sign({ userId: user._id }, 'your-jwt-secret', { expiresIn: '1h' });
    res.json({ success: true, message: 'Login successful! Redirecting to your dashboard...', token });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

// Server setup
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
