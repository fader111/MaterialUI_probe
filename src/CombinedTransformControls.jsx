import React, { useRef, useEffect } from 'react'
import { TransformControls } from '@react-three/drei'

/**
 * CombinedTransformControls
 * Shows both translation (arrows) and rotation (orbits) gizmos for a mesh at the same time.
 * Usage: <CombinedTransformControls object={meshRef.current} onObjectChange={fn} enabled={true} />
 */
export default function CombinedTransformControls({ object, enabled = true, onObjectChange }) {
  const translateRef = useRef()
  const rotateRef = useRef()

  // Sync enabled/disabled state
  useEffect(() => {
    if (translateRef.current) translateRef.current.enabled = enabled
    if (rotateRef.current) rotateRef.current.enabled = enabled
  }, [enabled])

  // Forward onObjectChange from either control
  useEffect(() => {
    if (translateRef.current) {
      translateRef.current.addEventListener('objectChange', onObjectChange)
    }
    if (rotateRef.current) {
      rotateRef.current.addEventListener('objectChange', onObjectChange)
    }
    return () => {
      if (translateRef.current) {
        translateRef.current.removeEventListener('objectChange', onObjectChange)
      }
      if (rotateRef.current) {
        rotateRef.current.removeEventListener('objectChange', onObjectChange)
      }
    }
  }, [onObjectChange])

  if (!object) return null

  return (
    <>
      <TransformControls
        ref={translateRef}
        object={object}
        mode="translate"
        enabled={enabled}
        showX
        showY
        showZ
        size={2.5} // Make arrows and orbits thicker/larger
        axisThickness={8} // Custom prop for Drei v10+ (if available)
        lineWidth={8} // For some Drei/three.js versions
      />
      <TransformControls
        ref={rotateRef}
        object={object}
        mode="rotate"
        enabled={enabled}
        showX
        showY
        showZ
        size={2.5}
        axisThickness={8}
        lineWidth={8}
      />
    </>
  )
}
