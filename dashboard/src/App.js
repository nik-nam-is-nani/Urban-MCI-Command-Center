import React, { useState, useEffect, useRef } from 'react';
import apiService from './services/api';
import MapView from './components/MapView';
import ControlPanel from './components/ControlPanel';
import MetricsPanel from './components/MetricsPanel';
import './App.css';

function App() {
  // State
  const [state, setState] = useState(null);
  const [grade, setGrade] = useState({ grade: 0, lives_saved: 0, lives_lost: 0 });
  const [tasks, setTasks] = useState([]);
  const [currentTask, setCurrentTask] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [autoRunIntervalMs, setAutoRunIntervalMs] = useState(1000);
  const [error, setError] = useState(null);
  const [selectedItem, setSelectedItem] = useState(null);
  
  const autoRunRef = useRef(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');

  // Initialize
  useEffect(() => {
    initialize();
    return () => {
      if (autoRunRef.current) {
        clearInterval(autoRunRef.current);
        autoRunRef.current = null;
      }
    };
  }, []);

  // Auto-run effect
  useEffect(() => {
    if (isAutoRunning) {
      autoRunRef.current = setInterval(async () => {
        try {
          const result = await apiService.step([]);
          console.log("[AUTO-RUN] Step result:", result);
          if (result.done) {
            stopAutoRun();
          } else {
            setState(result.state);
            setGrade(prev => ({ ...prev, grade: result.info.grade || 0 }));
          }
        } catch (err) {
          console.error("[AUTO-RUN] Error:", err);
          setError(err.message);
          stopAutoRun();
        }
      }, autoRunIntervalMs);
    }

    // Cleanup only interval handle on dependency change/unmount.
    // Do not force-set state here, otherwise auto-run gets toggled off instantly.
    return () => {
      if (autoRunRef.current) {
        clearInterval(autoRunRef.current);
        autoRunRef.current = null;
      }
    };
  }, [isAutoRunning, autoRunIntervalMs]);

  const initialize = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Check health first
      const healthInfo = await apiService.healthCheck();
      console.log("[INIT] Health check:", healthInfo);
      const isHealthy = healthInfo.status === 'healthy' || healthInfo.status === 'ok';
      setConnectionStatus(isHealthy ? 'connected' : 'disconnected');
      
      if (!isHealthy) {
        setError('Cannot connect to API. Make sure the Flask server is running.');
        setIsLoading(false);
        return;
      }

      // Get tasks and reset
      const [tasksData, resetData] = await Promise.all([
        apiService.getTasks(),
        apiService.reset(currentTask)
      ]);
      
      console.log("[INIT] Reset response:", resetData);
      
      setTasks(tasksData.tasks || []);
      setState(resetData.state);
      setCurrentTask(resetData.task);
      
      // Get initial grade
      const gradeData = await apiService.getGrade();
      setGrade(gradeData);
      
    } catch (err) {
      setError(err.message);
      setConnectionStatus('error');
    } finally {
      setIsLoading(false);
    }
  };

  const stopAutoRun = () => {
    if (autoRunRef.current) {
      clearInterval(autoRunRef.current);
      autoRunRef.current = null;
    }
    setIsAutoRunning(false);
  };

  const handleReset = async (task) => {
    try {
      stopAutoRun();
      setIsLoading(true);
      setError(null);
      setSelectedItem(null);
      
      console.log("[RESET] Starting reset for task:", task);
      const resetData = await apiService.reset(task);
      console.log("[RESET] Response:", resetData);
      setState(resetData.state);
      setCurrentTask(resetData.task);
      
      const gradeData = await apiService.getGrade();
      console.log("[RESET] Grade:", gradeData);
      setGrade(gradeData);
      
    } catch (err) {
      console.error("[RESET] Error:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStep = async (directives = []) => {
    try {
      const payload = Array.isArray(directives) ? directives : [];
      console.log(
        `[STEP] Calling step with ${payload.length} directive(s)` +
        (payload.length === 0 ? " (auto-agent)" : " (manual)")
      );
      const result = await apiService.step(payload);
      console.log("[STEP] Result:", result);
      setState(result.state);
      setGrade(prev => ({ 
        ...prev, 
        grade: result.info?.grade || 0,
        lives_saved: result.info?.lives_saved || 0,
        lives_lost: result.info?.lives_lost || 0
      }));
      
      if (result.done) {
        stopAutoRun();
      }
      
      return result;
    } catch (err) {
      console.error("[STEP] Error:", err);
      setError(err.message);
      throw err;
    }
  };

  const handleAutoRunToggle = () => {
    if (isAutoRunning) {
      stopAutoRun();
    } else {
      setIsAutoRunning(true);
    }
  };

  const handleItemClick = (item) => {
    setSelectedItem(item);
  };

  // Render loading state
  if (isLoading && !state) {
    return (
      <div className="app-loading">
        <div className="loading-spinner"></div>
        <p>Connecting to Simulation Engine...</p>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <h1 className="app-title">Urban MCI Command Center</h1>
          <span className="task-label">Task {currentTask}</span>
        </div>
        <div className="header-right">
          <div className={`connection-status ${connectionStatus}`}>
            <span className="status-dot"></span>
            <span className="status-text">
              {connectionStatus === 'connected' ? 'Connected' : 
               connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {/* Main content */}
      <main className="app-main">
        {/* Top panels */}
        <div className="top-panels">
          <ControlPanel
            tasks={tasks}
            currentTask={currentTask}
            onReset={handleReset}
            onStep={handleStep}
            onAutoRunToggle={handleAutoRunToggle}
            isAutoRunning={isAutoRunning}
            isLoading={isLoading}
            autoRunIntervalMs={autoRunIntervalMs}
            onAutoRunIntervalChange={setAutoRunIntervalMs}
          />
          <MetricsPanel
            state={state}
            grade={grade}
            currentTask={currentTask}
          />
        </div>

        {/* Map view */}
        <div className="map-container">
          <MapView
            state={state}
            onItemClick={handleItemClick}
            selectedItem={selectedItem}
          />
        </div>

        {/* Selected item detail panel */}
        {selectedItem && (
          <div className="detail-panel glass-card">
            <div className="detail-header">
              <h3>{selectedItem.type === 'victim' ? `Patient #${selectedItem.id}` : 
                   selectedItem.type === 'hospital' ? `Hospital: ${selectedItem.name}` :
                   selectedItem.type === 'team' ? `Team #${selectedItem.id}` : 'Unknown'}</h3>
              <button className="close-btn" onClick={() => setSelectedItem(null)}>×</button>
            </div>
            <div className="detail-content">
              {Object.entries(selectedItem).map(([key, value]) => (
                <div key={key} className="detail-row">
                  <span className="detail-label">{key}:</span>
                  <span className="detail-value">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
