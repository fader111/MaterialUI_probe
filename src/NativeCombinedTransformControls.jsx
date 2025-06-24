import { useEffect } from 'react'
import { useThree, extend } from '@react-three/fiber'
import { TransformControls as ThreeTransformControls } from 'three/examples/jsm/controls/TransformControls'
import * as THREE from 'three'
extend({ ThreeTransformControls })

/**
 * NativeCombinedTransformControls
 * Uses original Three.js TransformControls to show both translation and rotation gizmos at once.
 * Usage: <NativeCombinedTransformControls object={meshRef.current} enabled={true} onObjectChange={fn} />
 */
export default function NativeCombinedTransformControls({ object, enabled = true, onObjectChange }) {
  const { camera, gl, scene } = useThree()

  useEffect(() => {
    if (!object || !(object instanceof THREE.Object3D) || !object.parent) return
    // Translation control
    const translateCtrl = new ThreeTransformControls(camera, gl.domElement)
    translateCtrl.setMode('translate')
    translateCtrl.attach(object)
    scene.add(translateCtrl)
    // Rotation control
    const rotateCtrl = new ThreeTransformControls(camera, gl.domElement)
    rotateCtrl.setMode('rotate')
    rotateCtrl.attach(object)
    scene.add(rotateCtrl)
    // Enable/disable
    translateCtrl.enabled = enabled
    rotateCtrl.enabled = enabled
    // Listen for changes
    if (onObjectChange) {
      translateCtrl.addEventListener('objectChange', onObjectChange)
      rotateCtrl.addEventListener('objectChange', onObjectChange)
    }
    // Animation frame update
    const update = () => {
      translateCtrl.update()
      rotateCtrl.update()
    }
    gl.setAnimationLoop(update)
    // Clean up
    return () => {
      if (onObjectChange) {
        translateCtrl.removeEventListener('objectChange', onObjectChange)
        rotateCtrl.removeEventListener('objectChange', onObjectChange)
      }
      scene.remove(translateCtrl)
      scene.remove(rotateCtrl)
      translateCtrl.dispose()
      rotateCtrl.dispose()
      gl.setAnimationLoop(null)
    }
  }, [object, enabled, camera, gl, scene, onObjectChange])

  return null // Controls are managed directly in the scene
}
