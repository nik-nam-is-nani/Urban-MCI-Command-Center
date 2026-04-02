import React, { useState } from 'react';
import './ControlPanel.css';

function ControlPanel({
  tasks,
  currentTask,
  onReset,
  onStep,
  onAutoRunToggle,
  isAutoRunning,
  isLoading,
  autoRunIntervalMs,
  onAutoRunIntervalChange,
}) {
  const [directiveType, setDirectiveType] = useState('triage');
  const [victimId, setVictimId] = useState('');
  const [teamId, setTeamId] = useState('');
  const [hospitalId, setHospitalId] = useState('');
  const [triageTag, setTriageTag] = useState('1');
  const [customDirectives, setCustomDirectives] = useState([]);
  const [keepQueue, setKeepQueue] = useState(false);
  const [directiveError, setDirectiveError] = useState('');

  const parseNonNegativeInt = (value) => {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed < 0) {
      return null;
    }
    return parsed;
  };

  const addDirective = () => {
    let directive = null;

    if (directiveType === 'triage') {
      const parsedVictim = parseNonNegativeInt(victimId);
      const parsedTag = parseNonNegativeInt(triageTag);
      if (parsedVictim === null || parsedTag === null || parsedTag > 3) {
        setDirectiveError('Triage needs valid victim_id and tag (0-3).');
        return;
      }
      directive = { type: 'triage', victim_id: parsedVictim, tag: parsedTag };
    } else if (directiveType === 'dispatch') {
      const parsedTeam = parseNonNegativeInt(teamId);
      const parsedVictim = parseNonNegativeInt(victimId);
      const parsedHospital = parseNonNegativeInt(hospitalId);
      if (parsedTeam === null || parsedVictim === null || parsedHospital === null) {
        setDirectiveError('Dispatch needs team_id, victim_id, and hospital_id.');
        return;
      }
      directive = {
        type: 'dispatch',
        team_id: parsedTeam,
        victim_id: parsedVictim,
        hospital_id: parsedHospital,
      };
    } else if (directiveType === 'assign_sar' || directiveType === 'assign_fire') {
      const parsedTeam = parseNonNegativeInt(teamId);
      const parsedVictim = parseNonNegativeInt(victimId);
      if (parsedTeam === null || parsedVictim === null) {
        setDirectiveError('This action needs team_id and victim_id.');
        return;
      }
      directive = { type: directiveType, team_id: parsedTeam, victim_id: parsedVictim };
    }

    if (!directive) {
      setDirectiveError('Unable to build directive.');
      return;
    }

    setDirectiveError('');
    setCustomDirectives((prev) => [...prev, directive]);
  };

  const removeDirective = (index) => {
    setCustomDirectives((prev) => prev.filter((_, i) => i !== index));
  };

  const clearQueue = () => {
    setCustomDirectives([]);
    setDirectiveError('');
  };

  const formatDirective = (directive) => {
    if (directive.type === 'triage') {
      return `triage victim=${directive.victim_id} tag=${directive.tag}`;
    }
    if (directive.type === 'dispatch') {
      return `dispatch team=${directive.team_id} victim=${directive.victim_id} hospital=${directive.hospital_id}`;
    }
    return `${directive.type} team=${directive.team_id} victim=${directive.victim_id}`;
  };

  const handleStepClick = async () => {
    try {
      const payload = customDirectives.length > 0 ? customDirectives : [];
      const result = await onStep(payload);
      if (result && payload.length > 0 && !keepQueue) {
        setCustomDirectives([]);
      }
    } catch {
      // App-level error banner already handles request failures.
    }
  };

  const updateAutoRunInterval = (value) => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return;
    }
    const clamped = Math.max(200, Math.min(10000, parsed));
    onAutoRunIntervalChange(clamped);
  };

  const needsTeamId = directiveType === 'dispatch' || directiveType === 'assign_sar' || directiveType === 'assign_fire';
  const needsVictimId = directiveType !== '';
  const needsHospitalId = directiveType === 'dispatch';
  const needsTag = directiveType === 'triage';

  return (
    <div className="control-panel glass-card">
      <div className="panel-header">
        <h3>Control Center</h3>
      </div>

      <div className="panel-content">
        <div className="control-section">
          <label className="section-label">Mission Level</label>
          <div className="task-buttons">
            {tasks.map((task) => (
              <button
                key={task.id}
                className={`task-btn ${currentTask === task.id ? 'active' : ''}`}
                onClick={() => onReset(task.id)}
                disabled={isLoading}
              >
                <span className="task-number">{task.id}</span>
                <span className="task-name">
                  {task.id === 1 ? 'Easy' : task.id === 2 ? 'Medium' : 'Hard'}
                </span>
                <span className="task-victims">{task.victims} victims</span>
              </button>
            ))}
          </div>
        </div>

        <div className="control-section">
          <label className="section-label">Actions</label>
          <div className="action-buttons">
            <button
              className="action-btn reset"
              onClick={() => onReset(currentTask)}
              disabled={isLoading}
            >
              Reset
            </button>

            <button
              className="action-btn step"
              onClick={handleStepClick}
              disabled={isLoading || isAutoRunning}
            >
              {customDirectives.length > 0 ? `Step (${customDirectives.length})` : 'Step'}
            </button>

            <button
              className={`action-btn auto ${isAutoRunning ? 'running' : ''}`}
              onClick={onAutoRunToggle}
              disabled={isLoading}
            >
              {isAutoRunning ? 'Stop' : 'Auto Run'}
            </button>
          </div>
        </div>

        <div className="control-section">
          <label className="section-label">Custom Inputs</label>
          <div className="custom-grid">
            <div className="custom-row">
              <label className="field-label">Directive Type</label>
              <select
                className="custom-input"
                value={directiveType}
                onChange={(e) => setDirectiveType(e.target.value)}
                disabled={isLoading || isAutoRunning}
              >
                <option value="triage">triage</option>
                <option value="dispatch">dispatch</option>
                <option value="assign_sar">assign_sar</option>
                <option value="assign_fire">assign_fire</option>
              </select>
            </div>

            {needsTeamId && (
              <div className="custom-row">
                <label className="field-label">team_id</label>
                <input
                  className="custom-input"
                  type="number"
                  min="0"
                  value={teamId}
                  onChange={(e) => setTeamId(e.target.value)}
                  disabled={isLoading || isAutoRunning}
                />
              </div>
            )}

            {needsVictimId && (
              <div className="custom-row">
                <label className="field-label">victim_id</label>
                <input
                  className="custom-input"
                  type="number"
                  min="0"
                  value={victimId}
                  onChange={(e) => setVictimId(e.target.value)}
                  disabled={isLoading || isAutoRunning}
                />
              </div>
            )}

            {needsHospitalId && (
              <div className="custom-row">
                <label className="field-label">hospital_id</label>
                <input
                  className="custom-input"
                  type="number"
                  min="0"
                  value={hospitalId}
                  onChange={(e) => setHospitalId(e.target.value)}
                  disabled={isLoading || isAutoRunning}
                />
              </div>
            )}

            {needsTag && (
              <div className="custom-row">
                <label className="field-label">tag</label>
                <select
                  className="custom-input"
                  value={triageTag}
                  onChange={(e) => setTriageTag(e.target.value)}
                  disabled={isLoading || isAutoRunning}
                >
                  <option value="1">1 RED</option>
                  <option value="2">2 YELLOW</option>
                  <option value="3">3 GREEN</option>
                  <option value="0">0 BLACK</option>
                </select>
              </div>
            )}

            <div className="custom-buttons">
              <button
                className="mini-btn"
                onClick={addDirective}
                disabled={isLoading || isAutoRunning}
              >
                Add Directive
              </button>
              <button
                className="mini-btn clear"
                onClick={clearQueue}
                disabled={isLoading || isAutoRunning || customDirectives.length === 0}
              >
                Clear Queue
              </button>
            </div>
          </div>

          <label className="check-row">
            <input
              type="checkbox"
              checked={keepQueue}
              onChange={(e) => setKeepQueue(e.target.checked)}
              disabled={isLoading || isAutoRunning}
            />
            Keep queue after step
          </label>

          {directiveError && <div className="input-error">{directiveError}</div>}

          <div className="queue-box">
            {customDirectives.length === 0 ? (
              <div className="queue-empty">No custom directives queued. Step uses auto-agent.</div>
            ) : (
              customDirectives.map((directive, index) => (
                <div key={`${directive.type}-${index}`} className="queue-item">
                  <span className="queue-text">{index + 1}. {formatDirective(directive)}</span>
                  <button
                    className="queue-remove"
                    onClick={() => removeDirective(index)}
                    disabled={isLoading || isAutoRunning}
                  >
                    x
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="custom-row">
            <label className="field-label">Auto-run interval (ms)</label>
            <input
              className="custom-input"
              type="number"
              min="200"
              max="10000"
              step="100"
              value={autoRunIntervalMs}
              onChange={(e) => updateAutoRunInterval(e.target.value)}
              disabled={isLoading}
            />
          </div>
        </div>

        <div className="control-section">
          <label className="section-label">System Status</label>
          <div className="status-indicators">
            <div className="status-item">
              <span className="status-label">Simulation</span>
              <span className={`status-value ${isLoading ? 'loading' : 'ready'}`}>
                {isLoading ? 'Processing...' : 'Ready'}
              </span>
            </div>
            <div className="status-item">
              <span className="status-label">Auto Run</span>
              <span className={`status-value ${isAutoRunning ? 'active' : 'inactive'}`}>
                {isAutoRunning ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ControlPanel;
