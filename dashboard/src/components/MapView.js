import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import './MapView.css';

const MAP_WIDTH = 760;
const MAP_HEIGHT = 460;
const WORLD_SIZE = 200;
const MAP_PADDING = 24;

const TAG_COLORS = {
  RED: '#ef4444',
  YELLOW: '#f59e0b',
  GREEN: '#22c55e',
  BLACK: '#6b7280',
  UNTRIAGED: '#94a3b8',
};

const STATUS_COLORS = {
  TRAPPED: '#94a3b8',
  TRIAGED: '#3b82f6',
  IN_TRANSIT: '#f59e0b',
  AT_HOSPITAL: '#22c55e',
  DECEASED: '#6b7280',
};

const TEAM_COLORS = {
  AMBULANCE: '#10b981',
  SEARCH_RESCUE: '#8b5cf6',
  FIRE: '#f97316',
};

const TEAM_LINK_COLORS = {
  AMBULANCE: 'rgba(16, 185, 129, 0.55)',
  SEARCH_RESCUE: 'rgba(139, 92, 246, 0.55)',
  FIRE: 'rgba(249, 115, 22, 0.55)',
};

function getPatientColor(status, tag) {
  if (status === 'DECEASED') {
    return STATUS_COLORS.DECEASED;
  }

  if (tag === 'RED') {
    return TAG_COLORS.RED;
  }
  if (tag === 'YELLOW') {
    return TAG_COLORS.YELLOW;
  }
  if (tag === 'GREEN') {
    return TAG_COLORS.GREEN;
  }
  if (tag === 'BLACK') {
    return TAG_COLORS.BLACK;
  }

  if (status === 'TRIAGED' || status === 'IN_TRANSIT' || status === 'AT_HOSPITAL') {
    return STATUS_COLORS[status] || TAG_COLORS.UNTRIAGED;
  }

  return TAG_COLORS.UNTRIAGED;
}

function MapView({ state, onItemClick, selectedItem }) {
  const canvasRef = useRef(null);
  const hitTargetsRef = useRef([]);
  const [hoveredItem, setHoveredItem] = useState(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [showLabels, setShowLabels] = useState(true);
  const [showAssignments, setShowAssignments] = useState(true);
  const [showDeceased, setShowDeceased] = useState(true);

  const victimsById = useMemo(() => {
    const map = new Map();
    (state?.victims || []).forEach((victim) => map.set(victim.id, victim));
    return map;
  }, [state]);

  const quickStats = useMemo(() => {
    const victims = state?.victims || [];
    let red = 0;
    let yellow = 0;
    let green = 0;
    let black = 0;
    let untriaged = 0;

    victims.forEach((victim) => {
      if (!victim.assigned_tag) {
        untriaged += 1;
        return;
      }
      if (victim.assigned_tag === 'RED') {
        red += 1;
      } else if (victim.assigned_tag === 'YELLOW') {
        yellow += 1;
      } else if (victim.assigned_tag === 'GREEN') {
        green += 1;
      } else if (victim.assigned_tag === 'BLACK') {
        black += 1;
      }
    });

    return { red, yellow, green, black, untriaged };
  }, [state]);

  const toScreen = useCallback((location) => {
    const [x, y] = location;
    const usableWidth = MAP_WIDTH - MAP_PADDING * 2;
    const usableHeight = MAP_HEIGHT - MAP_PADDING * 2;
    return {
      x: MAP_PADDING + (x / WORLD_SIZE) * usableWidth,
      y: MAP_PADDING + (y / WORLD_SIZE) * usableHeight,
    };
  }, []);

  const isSelected = useCallback(
    (type, id) => {
      if (!selectedItem || selectedItem.type !== type) {
        return false;
      }
      return selectedItem.id === id;
    },
    [selectedItem]
  );

  const registerHitTarget = useCallback((target) => {
    hitTargetsRef.current.push(target);
  }, []);

  const drawBackground = useCallback((ctx) => {
    const gradient = ctx.createLinearGradient(0, 0, 0, MAP_HEIGHT);
    gradient.addColorStop(0, '#0f172a');
    gradient.addColorStop(1, '#0b1220');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, MAP_WIDTH, MAP_HEIGHT);
  }, []);

  const drawGrid = useCallback((ctx) => {
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i += 1) {
      const x = MAP_PADDING + ((MAP_WIDTH - MAP_PADDING * 2) * i) / 10;
      const y = MAP_PADDING + ((MAP_HEIGHT - MAP_PADDING * 2) * i) / 10;

      ctx.beginPath();
      ctx.moveTo(x, MAP_PADDING);
      ctx.lineTo(x, MAP_HEIGHT - MAP_PADDING);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(MAP_PADDING, y);
      ctx.lineTo(MAP_WIDTH - MAP_PADDING, y);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(
      MAP_PADDING,
      MAP_PADDING,
      MAP_WIDTH - MAP_PADDING * 2,
      MAP_HEIGHT - MAP_PADDING * 2
    );

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Map Range: x=0..200, y=0..200', MAP_PADDING, 16);
  }, []);

  const drawAssignments = useCallback(
    (ctx) => {
      if (!showAssignments || !state?.teams) {
        return;
      }

      state.teams.forEach((team) => {
        const victimId = team.transport_victim ?? team.assigned_victim;
        if (victimId == null) {
          return;
        }

        const victim = victimsById.get(victimId);
        if (!victim || (!showDeceased && victim.status === 'DECEASED')) {
          return;
        }

        const from = toScreen(team.location);
        const to = toScreen(victim.location);
        const color = TEAM_LINK_COLORS[team.type] || 'rgba(148, 163, 184, 0.55)';

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
        ctx.restore();
      });
    },
    [showAssignments, showDeceased, state, toScreen, victimsById]
  );

  const drawHospitals = useCallback(
    (ctx) => {
      (state?.hospitals || []).forEach((hospital) => {
        const position = toScreen(hospital.location);
        const radius = 17;
        const utilization =
          hospital.total_capacity > 0
            ? (hospital.total_capacity - hospital.available_beds) / hospital.total_capacity
            : 0;

        ctx.fillStyle = 'rgba(59, 130, 246, 0.16)';
        ctx.beginPath();
        ctx.arc(position.x, position.y, radius + 7, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#2563eb';
        ctx.beginPath();
        ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle =
          utilization > 0.8 ? '#ef4444' : utilization > 0.5 ? '#f59e0b' : '#22c55e';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(
          position.x,
          position.y,
          radius + 3,
          -Math.PI / 2,
          -Math.PI / 2 + Math.PI * 2 * Math.min(1, utilization)
        );
        ctx.stroke();

        if (isSelected('hospital', hospital.id)) {
          ctx.strokeStyle = '#f8fafc';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(position.x, position.y, radius + 10, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (showLabels) {
          ctx.fillStyle = '#e2e8f0';
          ctx.font = '11px system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(`H${hospital.id}`, position.x, position.y + 32);
        }

        registerHitTarget({
          type: 'hospital',
          data: hospital,
          x: position.x,
          y: position.y,
          radius: radius + 8,
        });
      });
    },
    [state, toScreen, isSelected, showLabels, registerHitTarget]
  );

  const drawVictims = useCallback(
    (ctx) => {
      (state?.victims || []).forEach((victim) => {
        if (!showDeceased && victim.status === 'DECEASED') {
          return;
        }

        const position = toScreen(victim.location);
        const radius = victim.status === 'DECEASED' ? 4 : 5.8;
        const color = getPatientColor(victim.status, victim.assigned_tag);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
        ctx.fill();

        if (victim.assigned_tag === 'RED') {
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.55)';
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.arc(position.x, position.y, radius + 3, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (isSelected('victim', victim.id)) {
          ctx.strokeStyle = '#f8fafc';
          ctx.lineWidth = 1.7;
          ctx.beginPath();
          ctx.arc(position.x, position.y, radius + 5, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (showLabels && (victim.assigned_tag === 'RED' || victim.status === 'DECEASED')) {
          ctx.fillStyle = '#cbd5e1';
          ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
          ctx.textAlign = 'left';
          ctx.fillText(`V${victim.id}`, position.x + 6, position.y - 6);
        }

        registerHitTarget({
          type: 'victim',
          data: victim,
          x: position.x,
          y: position.y,
          radius: radius + 4,
        });
      });
    },
    [state, showDeceased, toScreen, isSelected, showLabels, registerHitTarget]
  );

  const drawTeams = useCallback(
    (ctx) => {
      (state?.teams || []).forEach((team) => {
        const position = toScreen(team.location);
        const baseColor = TEAM_COLORS[team.type] || '#e2e8f0';
        const color = team.is_free ? baseColor : '#f8fafc';

        ctx.save();
        ctx.translate(position.x, position.y);

        if (team.type === 'AMBULANCE') {
          ctx.fillStyle = color;
          ctx.fillRect(-7, -5, 14, 10);
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(-2, -4, 4, 8);
          ctx.fillRect(-4, -2, 8, 4);
        } else if (team.type === 'SEARCH_RESCUE') {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(0, 0, 6.8, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = '#0f172a';
          ctx.font = 'bold 8px system-ui, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('S', 0, 0);
        } else {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.moveTo(0, -7);
          ctx.lineTo(6, 6);
          ctx.lineTo(-6, 6);
          ctx.closePath();
          ctx.fill();
        }

        ctx.restore();

        if (isSelected('team', team.id)) {
          ctx.strokeStyle = '#f8fafc';
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.arc(position.x, position.y, 11, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (showLabels) {
          ctx.fillStyle = '#cbd5e1';
          ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
          ctx.textAlign = 'center';
          ctx.fillText(`T${team.id}`, position.x, position.y + 16);
        }

        registerHitTarget({
          type: 'team',
          data: team,
          x: position.x,
          y: position.y,
          radius: 12,
        });
      });
    },
    [state, toScreen, isSelected, showLabels, registerHitTarget]
  );

  const drawIncidentCenter = useCallback((ctx) => {
    const position = toScreen([100, 100]);
    ctx.fillStyle = 'rgba(6, 182, 212, 0.22)';
    ctx.beginPath();
    ctx.arc(position.x, position.y, 16, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(position.x, position.y, 22, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#67e8f9';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('INCIDENT CORE', position.x, position.y + 32);
  }, [toScreen]);

  const drawRoadBlockage = useCallback((ctx) => {
    if (!state?.road_blocked) {
      return;
    }

    ctx.fillStyle = 'rgba(245, 158, 11, 0.2)';
    ctx.fillRect(MAP_WIDTH - 54, MAP_PADDING, 36, MAP_HEIGHT - MAP_PADDING * 2);

    ctx.fillStyle = '#f59e0b';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('ROAD', MAP_WIDTH - 36, MAP_HEIGHT / 2 - 8);
    ctx.fillText('BLOCK', MAP_WIDTH - 36, MAP_HEIGHT / 2 + 6);
  }, [state]);

  useEffect(() => {
    if (!state || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    hitTargetsRef.current = [];

    drawBackground(ctx);
    drawGrid(ctx);
    drawAssignments(ctx);
    drawHospitals(ctx);
    drawVictims(ctx);
    drawTeams(ctx);
    drawIncidentCenter(ctx);
    drawRoadBlockage(ctx);
  }, [
    state,
    drawBackground,
    drawGrid,
    drawAssignments,
    drawHospitals,
    drawVictims,
    drawTeams,
    drawIncidentCenter,
    drawRoadBlockage,
  ]);

  const findHitTarget = useCallback((mouseX, mouseY) => {
    const targets = hitTargetsRef.current;
    for (let i = targets.length - 1; i >= 0; i -= 1) {
      const target = targets[i];
      const dx = mouseX - target.x;
      const dy = mouseY - target.y;
      if (Math.sqrt(dx * dx + dy * dy) <= target.radius) {
        return target;
      }
    }
    return null;
  }, []);

  const getMousePositionOnCanvas = useCallback((event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      canvasX: (event.clientX - rect.left) * scaleX,
      canvasY: (event.clientY - rect.top) * scaleY,
      uiX: event.clientX - rect.left,
      uiY: event.clientY - rect.top,
    };
  }, []);

  const handleMouseMove = useCallback(
    (event) => {
      if (!state || !canvasRef.current) {
        return;
      }

      const { canvasX, canvasY, uiX, uiY } = getMousePositionOnCanvas(event);
      const target = findHitTarget(canvasX, canvasY);

      if (!target) {
        setHoveredItem(null);
        canvasRef.current.style.cursor = 'default';
        return;
      }

      setHoverPosition({ x: uiX + 14, y: uiY + 12 });
      setHoveredItem({ ...target.data, entityType: target.type });
      canvasRef.current.style.cursor = 'pointer';
    },
    [state, findHitTarget, getMousePositionOnCanvas]
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredItem(null);
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'default';
    }
  }, []);

  const handleCanvasClick = useCallback(
    (event) => {
      if (!state || !canvasRef.current) {
        return;
      }

      const { canvasX, canvasY } = getMousePositionOnCanvas(event);
      const target = findHitTarget(canvasX, canvasY);
      if (!target) {
        return;
      }

      onItemClick({ ...target.data, type: target.type });
    },
    [state, onItemClick, findHitTarget, getMousePositionOnCanvas]
  );

  const renderHoverTooltip = () => {
    if (!hoveredItem) {
      return null;
    }

    if (hoveredItem.entityType === 'victim') {
      return (
        <div className="hover-tooltip-body">
          <div className="tooltip-title">Patient #{hoveredItem.id}</div>
          <div>Status: {hoveredItem.status}</div>
          <div>Tag: {hoveredItem.assigned_tag || 'UNTRIAGED'}</div>
          <div>Injury Time: {hoveredItem.minutes_since_injury.toFixed(1)} min</div>
          <div>
            Coord: ({hoveredItem.location[0].toFixed(1)}, {hoveredItem.location[1].toFixed(1)})
          </div>
        </div>
      );
    }

    if (hoveredItem.entityType === 'hospital') {
      return (
        <div className="hover-tooltip-body">
          <div className="tooltip-title">{hoveredItem.name}</div>
          <div>Trauma Level: {hoveredItem.trauma_level}</div>
          <div>
            Beds: {hoveredItem.available_beds}/{hoveredItem.total_capacity} free
          </div>
          <div>Travel: {hoveredItem.travel_time_minutes} min</div>
          <div>
            Coord: ({hoveredItem.location[0].toFixed(1)}, {hoveredItem.location[1].toFixed(1)})
          </div>
        </div>
      );
    }

    return (
      <div className="hover-tooltip-body">
        <div className="tooltip-title">Team #{hoveredItem.id}</div>
        <div>Type: {hoveredItem.type}</div>
        <div>Status: {hoveredItem.is_free ? 'FREE' : 'BUSY'}</div>
        <div>Assigned Victim: {hoveredItem.assigned_victim ?? 'none'}</div>
        <div>Transport Victim: {hoveredItem.transport_victim ?? 'none'}</div>
        <div>Free At Step: {hoveredItem.free_at_step}</div>
      </div>
    );
  };

  return (
    <div className="map-view glass-card">
      <div className="map-header">
        <h3>Simulation View</h3>
        <div className="map-controls">
          <button
            type="button"
            className={`map-toggle ${showLabels ? 'active' : ''}`}
            onClick={() => setShowLabels((value) => !value)}
          >
            Labels
          </button>
          <button
            type="button"
            className={`map-toggle ${showAssignments ? 'active' : ''}`}
            onClick={() => setShowAssignments((value) => !value)}
          >
            Assignments
          </button>
          <button
            type="button"
            className={`map-toggle ${showDeceased ? 'active' : ''}`}
            onClick={() => setShowDeceased((value) => !value)}
          >
            Deceased
          </button>
        </div>
        <div className="map-legend">
          <span className="legend-item">
            <span className="legend-dot" style={{ background: TAG_COLORS.RED }} />
            Critical
          </span>
          <span className="legend-item">
            <span className="legend-dot" style={{ background: TAG_COLORS.YELLOW }} />
            Serious
          </span>
          <span className="legend-item">
            <span className="legend-dot" style={{ background: TAG_COLORS.GREEN }} />
            Minor
          </span>
          <span className="legend-item">
            <span className="legend-dot" style={{ background: TAG_COLORS.BLACK }} />
            Deceased
          </span>
          <span className="legend-item">
            <span className="legend-dot" style={{ background: TAG_COLORS.UNTRIAGED }} />
            Untriaged
          </span>
        </div>
      </div>

      <div className="canvas-container">
        <div className="map-overlay">
          <div className="overlay-title">
            Step {state?.step || 0} | Golden Hour Left: {state?.golden_hour_remaining || 0}m
          </div>
          <div className="overlay-row">
            <span>RED {quickStats.red}</span>
            <span>YELLOW {quickStats.yellow}</span>
            <span>GREEN {quickStats.green}</span>
            <span>BLACK {quickStats.black}</span>
            <span>UNTRIAGED {quickStats.untriaged}</span>
          </div>
        </div>

        <canvas
          ref={canvasRef}
          width={MAP_WIDTH}
          height={MAP_HEIGHT}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleCanvasClick}
        />

        {hoveredItem && (
          <div
            className="hover-tooltip"
            style={{
              left: hoverPosition.x,
              top: hoverPosition.y,
            }}
          >
            {renderHoverTooltip()}
          </div>
        )}
      </div>
    </div>
  );
}

export default MapView;
