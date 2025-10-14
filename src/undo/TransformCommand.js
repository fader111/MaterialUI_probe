// TransformCommand: stores before/after transforms for one or many objects
// before/after are maps: { <id>: { position: {x,y,z}, rotation: {x,y,z}, scale: {x,y,z} } }


export default class TransformCommand {
  constructor(objectIds = [], before = {}, after = {}, sceneApi, setOrthoData, stage) {
    this.objectIds = Array.isArray(objectIds) ? objectIds : [objectIds]
    this.before = before || {}
    this.after = after || {}
    this.sceneApi = sceneApi
    this.setOrthoData = setOrthoData
    this.stage = stage
  }

  do() {
    // Update mesh (optional, for immediate effect)
    if (this.sceneApi && typeof this.sceneApi.applyTransform === 'function') {
      for (const id of this.objectIds) {
        const t = this.after[id]
        if (t) this.sceneApi.applyTransform(id, t)
      }
      if (this.after.__spline) {
        this.sceneApi.updateSpline && this.sceneApi.updateSpline(this.after.__spline)
      }
    }
    // Update React state (source of truth)
    if (this.setOrthoData && typeof this.stage !== 'undefined') {
      this.setOrthoData(prev => {
        if (!prev || !prev.Staging) return prev;
        const newOrthoData = { ...prev, Staging: [...prev.Staging] };
        const newStage = { ...newOrthoData.Staging[this.stage], RelativeToothTransforms: { ...newOrthoData.Staging[this.stage].RelativeToothTransforms } };
        for (const id of this.objectIds) {
          const t = this.after[id];
          if (t) {
            newStage.RelativeToothTransforms[id] = {
              ...newStage.RelativeToothTransforms[id],
              translation: { x: t.position[0], y: t.position[1], z: t.position[2] },
              rotation: t.quaternion
                ? { x: t.quaternion[0], y: t.quaternion[1], z: t.quaternion[2], w: t.quaternion[3] }
                : newStage.RelativeToothTransforms[id].rotation
            };
          }
        }
        newOrthoData.Staging[this.stage] = newStage;
        return newOrthoData;
      });
    }
  }

  undo() {
    if (this.sceneApi && typeof this.sceneApi.applyTransform === 'function') {
      for (const id of this.objectIds) {
        const t = this.before[id]
        if (t) this.sceneApi.applyTransform(id, t)
      }
      if (this.before.__spline) {
        this.sceneApi.updateSpline && this.sceneApi.updateSpline(this.before.__spline)
      }
    }
    if (this.setOrthoData && typeof this.stage !== 'undefined') {
      this.setOrthoData(prev => {
        if (!prev || !prev.Staging) return prev;
        const newOrthoData = { ...prev, Staging: [...prev.Staging] };
        const newStage = { ...newOrthoData.Staging[this.stage], RelativeToothTransforms: { ...newOrthoData.Staging[this.stage].RelativeToothTransforms } };
        for (const id of this.objectIds) {
          const t = this.before[id];
          if (t) {
            newStage.RelativeToothTransforms[id] = {
              ...newStage.RelativeToothTransforms[id],
              translation: { x: t.position[0], y: t.position[1], z: t.position[2] },
              rotation: t.quaternion
                ? { x: t.quaternion[0], y: t.quaternion[1], z: t.quaternion[2], w: t.quaternion[3] }
                : newStage.RelativeToothTransforms[id].rotation
            };
          }
        }
        newOrthoData.Staging[this.stage] = newStage;
        return newOrthoData;
      });
    }
  }

  canMergeWith(other) {
    if (!(other instanceof TransformCommand)) return false
    // same set of ids (simple stringify approach)
    return JSON.stringify(this.objectIds) === JSON.stringify(other.objectIds)
  }

  mergeWith(other) {
    // merge after snapshots
    this.after = { ...this.after, ...other.after }
  }
}
