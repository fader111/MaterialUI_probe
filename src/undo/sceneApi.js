// src/undo/sceneApi.js
export const sceneApi = {
  executeCommand: (cmd) => {
    if (window.commandManager && typeof window.commandManager.execute === 'function') {
      window.commandManager.execute(cmd);
    } else {
      console.error('window.commandManager is not available or does not have an execute method.');
    }
  },
  applyTransform: (id, transform) => {
    // This stub does nothing but log for now. Real implementation should update mesh directly if possible.
    console.log('[sceneApi] applyTransform called for', id, transform);
    // Optionally, you could trigger a custom event or update a global state here.
    // For now, rely on setOrthoData to update React state and re-render.
  }
};
      