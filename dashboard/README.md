# Urban MCI Dashboard

A production-grade, real-time visualization dashboard for Urban Mass Casualty Incident (MCI) simulation.

## Overview

This React-based dashboard connects to the Flask API server (`app.py`) and provides:
- Real-time simulation visualization on a 2D map
- Interactive control panel for task selection and execution
- Live metrics and performance scoring
- Click-to-inspect details for patients, hospitals, and resources

## Prerequisites

1. **Backend Server Running** - The Flask API must be running:
   ```bash
   python app.py
   ```

2. **Node.js** - Required for building the React app (v14+ recommended)

## Quick Start

### Option 1: Development Mode

```bash
cd dashboard
npm install
npm start
```

The dashboard will open at `http://localhost:3000`.

### Option 2: Production Build

```bash
cd dashboard
npm install
npm run build
```

The built files will be in the `build/` directory.

## Environment Variables

The dashboard connects to the backend API. Configure the URL:

```bash
# Option 1: Environment variable
export REACT_APP_API_URL=http://localhost:7860

# Option 2: The default is http://localhost:7860
```

## Project Structure

```
dashboard/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── components/
│   │   ├── MapView.js      # Canvas-based simulation map
│   │   ├── MapView.css
│   │   ├── ControlPanel.js # Task selection & controls
│   │   ├── ControlPanel.css
│   │   ├── MetricsPanel.js # Real-time metrics display
│   │   └── MetricsPanel.css
│   ├── services/
│   │   └── api.js          # API service layer
│   ├── App.js              # Main app component
│   ├── App.css             # App layout styles
│   ├── index.js            # React entry point
│   └── index.css           # Global styles
├── package.json
└── README.md
```

## Features

### 🧭 Map View
- 2D canvas rendering of simulation area
- Color-coded patients by triage status (RED/YELLOW/GREEN/BLACK)
- Hospital locations with capacity rings
- Ambulance/SAR/Fire team icons
- Incident command center marker

### ⏱️ Real-Time Engine
- Polls `/state` endpoint every step
- Smooth position updates
- Click-to-inspect patients and hospitals

### 🎮 Control Panel
- Task selector (Easy/Medium/Hard)
- Reset, Step, and Auto Run buttons
- Connection status indicator

### 📊 Metrics Panel
- Elapsed time with golden hour indicator
- Performance score (0-100%)
- Patient status breakdown (trapped/triaged/transit/hospital/deceased)
- Resource availability (ambulances, hospital capacity)
- System alerts (collapse risk, road blockage, media)

## API Integration

The dashboard expects these endpoints from the backend:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment `{task: 1\|2\|3}` |
| `/step` | POST | Execute simulation step `{directives: []}` |
| `/state` | GET | Get current state |
| `/grade` | GET | Get performance score `{grade: 0-1, lives_saved, lives_lost}` |
| `/tasks` | GET | Get task list |
| `/health` | GET | Health check |

## Technologies Used

- **React 18** - UI framework
- **HTML5 Canvas** - Simulation map rendering
- **CSS Grid/Flexbox** - Responsive layout
- **Glassmorphism** - Modern dark theme styling

## Running with Docker

To run both backend and frontend together:

1. Build the React app: `cd dashboard && npm run build`
2. Copy the `build/` folder to serve from Flask, or
3. Run separately:
   - Terminal 1: `python app.py` (backend on port 7860)
   - Terminal 2: `npm start` (frontend on port 3000)

## Screenshots

The dashboard features:
- Dark theme with cyan/blue accents
- Real-time patient status bars
- Hospital capacity visualization
- Alert system for critical events
- Responsive layout for different screen sizes
