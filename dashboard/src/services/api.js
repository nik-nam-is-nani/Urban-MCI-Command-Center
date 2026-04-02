/**
 * API Service Layer for Urban MCI Dashboard
 * Uses proxy configured in package.json for CORS-free requests
 */

const api = {
  /**
   * Reset the environment with a specific task
   * @param {number} task - Task number (1, 2, or 3)
   * @returns {Promise<Object>} - Initial state
   */
  async reset(task = 1) {
    const response = await fetch('/reset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ task }),
    });
    
    if (!response.ok) {
      throw new Error(`Reset failed: ${response.status}`);
    }
    
    return response.json();
  },

  /**
   * Execute a step in the simulation
   * @param {Array} directives - List of action directives
   * @returns {Promise<Object>} - Step result with state, reward, done, info
   */
  async step(directives = []) {
    const response = await fetch('/step', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ directives }),
    });
    
    if (!response.ok) {
      throw new Error(`Step failed: ${response.status}`);
    }
    
    return response.json();
  },

  /**
   * Get current simulation state
   * @returns {Promise<Object>} - Current state
   */
  async getState() {
    const response = await fetch('/state');
    
    if (!response.ok) {
      throw new Error(`Get state failed: ${response.status}`);
    }
    
    return response.json();
  },

  /**
   * Get current performance grade
   * @returns {Promise<Object>} - Grade info
   */
  async getGrade() {
    const response = await fetch('/grade');
    
    if (!response.ok) {
      throw new Error(`Get grade failed: ${response.status}`);
    }
    
    return response.json();
  },

  /**
   * Get list of available tasks
   * @returns {Promise<Object>} - Tasks info
   */
  async getTasks() {
    const response = await fetch('/tasks');
    
    if (!response.ok) {
      throw new Error(`Get tasks failed: ${response.status}`);
    }
    
    return response.json();
  },

  /**
   * Health check
   * @returns {Promise<Object>} - Health info
   */
  async healthCheck() {
    try {
      const response = await fetch('/health');
      if (!response.ok) return { status: 'unhealthy', agent_connected: false };
      return response.json();
    } catch {
      return { status: 'unhealthy', agent_connected: false };
    }
  }
};

export default api;