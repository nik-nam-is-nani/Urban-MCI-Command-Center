import React from 'react';
import './MetricsPanel.css';

function MetricsPanel({ state, grade, currentTask }) {
  // Calculate metrics from state
  const summary = state?.summary || {};
  const step = state?.step || 0;
  const goldenHourRemaining = state?.golden_hour_remaining || 0;
  const secondaryCollapseRisk = state?.secondary_collapse_risk || 0;
  const roadBlocked = state?.road_blocked || false;
  const mediaPressure = state?.media_pressure || false;

  // Patient counts by status
  const trapped = summary.trapped || 0;
  const triaged = summary.triaged || 0;
  const inTransit = summary.in_transit || 0;
  const atHospital = summary.at_hospital || 0;
  const deceased = summary.deceased || 0;
  const total = summary.total_victims || 0;

  // Score calculation - use grade from backend
  const score = grade?.grade ?? 0;
  const livesSaved = grade?.lives_saved || 0;
  const livesLost = grade?.lives_lost || 0;

  // Calculate hospital utilization - handle correctly
  const hospitals = state?.hospitals || [];
  const availableBeds = hospitals.reduce((sum, h) => sum + (h.available_beds || 0), 0);
  // Estimate total capacity from available + current occupancy
  const totalCapacity = hospitals.reduce((sum, h) => sum + (h.total_capacity || 0), 0);
  const usedCapacity = totalCapacity - availableBeds;
  const utilization = totalCapacity > 0 ? (usedCapacity / totalCapacity) * 100 : 0;

  // Count free ambulances
  const teams = state?.teams || [];
  const freeAmbulances = teams.filter(t => t.type === 'AMBULANCE' && t.is_free).length;
  const totalAmbulances = teams.filter(t => t.type === 'AMBULANCE').length;

  return (
    <div className="metrics-panel glass-card">
      <div className="panel-header">
        <h3>Mission Metrics</h3>
      </div>
      
      <div className="panel-content">
        {/* Time & Score Row */}
        <div className="metrics-row">
          <div className="metric-card time-card">
            <div className="metric-label">Elapsed Time</div>
            <div className="metric-value">{step} / 120</div>
            <div className="metric-sub">
              {goldenHourRemaining > 0 ? (
                <span className="golden-hour">{goldenHourRemaining}m golden hour</span>
              ) : (
                <span className="overtime">OVERTIME</span>
              )}
            </div>
          </div>
          
          <div className="metric-card score-card">
            <div className="metric-label">Performance Score</div>
            <div className={`metric-value score ${score > 0.5 ? 'good' : score > 0.2 ? 'medium' : 'low'}`}>
              {(score * 100).toFixed(1)}%
            </div>
            <div className="metric-sub">
              <span className="saved">{livesSaved} saved</span>
              <span className="lost">{livesLost} lost</span>
            </div>
          </div>
        </div>

        {/* Patient Status Row */}
        <div className="metrics-section">
          <h4 className="section-title">Patient Status</h4>
          <div className="patient-bars">
            <div className="patient-bar-group">
              <div className="bar-row">
                <span className="bar-label">Trapped</span>
                <div className="bar-track">
                  <div 
                    className="bar-fill trapped" 
                    style={{ width: `${(trapped / Math.max(total, 1)) * 100}%` }}
                  />
                </div>
                <span className="bar-value">{trapped}</span>
              </div>
              <div className="bar-row">
                <span className="bar-label">Triaged</span>
                <div className="bar-track">
                  <div 
                    className="bar-fill triaged" 
                    style={{ width: `${(triaged / Math.max(total, 1)) * 100}%` }}
                  />
                </div>
                <span className="bar-value">{triaged}</span>
              </div>
              <div className="bar-row">
                <span className="bar-label">In Transit</span>
                <div className="bar-track">
                  <div 
                    className="bar-fill transit" 
                    style={{ width: `${(inTransit / Math.max(total, 1)) * 100}%` }}
                  />
                </div>
                <span className="bar-value">{inTransit}</span>
              </div>
              <div className="bar-row">
                <span className="bar-label">At Hospital</span>
                <div className="bar-track">
                  <div 
                    className="bar-fill hospital" 
                    style={{ width: `${(atHospital / Math.max(total, 1)) * 100}%` }}
                  />
                </div>
                <span className="bar-value">{atHospital}</span>
              </div>
              <div className="bar-row">
                <span className="bar-label">Deceased</span>
                <div className="bar-track">
                  <div 
                    className="bar-fill deceased" 
                    style={{ width: `${(deceased / Math.max(total, 1)) * 100}%` }}
                  />
                </div>
                <span className="bar-value">{deceased}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Resources Row */}
        <div className="metrics-row resources-row">
          <div className="metric-card resource-card">
            <div className="metric-label">Ambulances</div>
            <div className="resource-display">
              <span className="resource-available">{freeAmbulances}</span>
              <span className="resource-separator">/</span>
              <span className="resource-total">{totalAmbulances}</span>
            </div>
            <div className="resource-sub">Available</div>
          </div>
          
          <div className="metric-card resource-card">
            <div className="metric-label">Hospital Beds</div>
            <div className="resource-display">
              <span className="resource-available">{availableBeds}</span>
              <span className="resource-separator">free</span>
            </div>
            <div className="capacity-bar">
              <div 
                className="capacity-fill" 
                style={{ width: `${Math.max(0, Math.min(100, utilization))}%` }}
              />
            </div>
            <div className="resource-sub">{totalCapacity - availableBeds} in use / {totalCapacity} total</div>
          </div>
        </div>

        {/* Alerts Section */}
        <div className="metrics-section alerts-section">
          <h4 className="section-title">System Alerts</h4>
          <div className="alerts-grid">
            {secondaryCollapseRisk > 0.5 && (
              <div className="alert-card critical">
                <span className="alert-icon">!</span>
                <span className="alert-text">Collapse Risk: {(secondaryCollapseRisk * 100).toFixed(0)}%</span>
              </div>
            )}
            {roadBlocked && (
              <div className="alert-card warning">
                <span className="alert-icon">!</span>
                <span className="alert-text">Road Blocked</span>
              </div>
            )}
            {mediaPressure && (
              <div className="alert-card info">
                <span className="alert-icon">!</span>
                <span className="alert-text">Media Present</span>
              </div>
            )}
            {utilization > 80 && (
              <div className="alert-card warning">
                <span className="alert-icon">!</span>
                <span className="alert-text">Hospitals Near Capacity</span>
              </div>
            )}
            {secondaryCollapseRisk <= 0.5 && !roadBlocked && !mediaPressure && utilization <= 80 && (
              <div className="alert-card normal">
                <span className="alert-icon">OK</span>
                <span className="alert-text">All Clear</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default MetricsPanel;