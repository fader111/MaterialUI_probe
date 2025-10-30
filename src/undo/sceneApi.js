// src/undo/sceneApi.js
export const sceneApi = {
  executeCommand: (cmd) => {
    if (window.commandManager && typeof window.commandManager.execute === 'function') {
      window.commandManager.execute(cmd);
    } else {
      console.error('window.commandManager is not available or does not have an execute method.');
    }
  }
};
      