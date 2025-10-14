// Thin adapter around your existing global transforms and three.js scene.
// You should replace getMeshById, globalTransforms, and updateArcSplineFromState with
// the functions/objects you already have in your app.

// Minimal stubs provided to avoid runtime errors; replace as-needed.

// eslint-disable-next-line no-unused-vars
const globalTransforms = window.__globalTeethTransforms || {}

function getMeshById(id) {
  // The consuming app should implement this.
  // Example: return scene.getObjectByName(id) or a map lookup.
  if (window.getMeshById) return window.getMeshById(id)
  return null
}

function updateArcSplineFromState(splineState) {
  if (window.updateArcSplineFromState) return window.updateArcSplineFromState(splineState)
}

export const sceneApi = {
  applyTransform(id, tf) {
    const mesh = getMeshById(id)
    // tf.position and tf.quaternion are arrays (from TransformCommand)
    if (mesh) {
      if (tf.position) {
        if (Array.isArray(tf.position)) {
          mesh.position.set(tf.position[0], tf.position[1], tf.position[2])
        } else {
          mesh.position.set(tf.position.x, tf.position.y, tf.position.z)
        }
      }
      if (tf.quaternion) {
        if (Array.isArray(tf.quaternion)) {
          mesh.quaternion.set(tf.quaternion[0], tf.quaternion[1], tf.quaternion[2], tf.quaternion[3])
        } else {
          mesh.quaternion.set(tf.quaternion.x, tf.quaternion.y, tf.quaternion.z, tf.quaternion.w)
        }
      }
      if (tf.scale) {
        if (Array.isArray(tf.scale)) {
          mesh.scale.set(tf.scale[0], tf.scale[1], tf.scale[2])
        } else {
          mesh.scale.set(tf.scale.x, tf.scale.y, tf.scale.z)
        }
      }
      // if mesh.matrixAutoUpdate is false you might need mesh.updateMatrix()
    }
    // Always update globalTransforms for consistency
    globalTransforms[id] = tf
  },
  updateSpline(splineState) {
    updateArcSplineFromState(splineState)
  }
}
