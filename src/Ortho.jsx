import React, { useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { TrackballControls, TransformControls } from '@react-three/drei'
import * as THREE from 'three'

function CameraFollowingLight({ camera }) {
  const lightRef = useRef()
  useFrame(() => {
    if (camera && lightRef.current) {
      // Get camera direction
      const dir = camera.getWorldDirection(new THREE.Vector3())
      // Offset the light a bit in front of the camera
      const camPos = camera.position
      lightRef.current.position.set(
        camPos.x + dir.x * 5,
        camPos.y + dir.y * 5,
        camPos.z + dir.z * 5
      )
      // Optionally, point the light at the origin
      lightRef.current.target.position.set(0, 0, 0)
      lightRef.current.target.updateMatrixWorld()
    }
  })
  return <directionalLight ref={lightRef} intensity={0.7} />
}

const Cube = React.forwardRef(function Cube(
  { color, position, setPosition, active, setActive, name, rotate, angle },
  ref
) {
  const meshRef = ref || useRef()
  // Update position in state when moved
  const handleObjectChange = () => {
    if (meshRef.current) {
      const p = meshRef.current.position
      setPosition([p.x, p.y, p.z])
    }
  }
  useFrame(() => {
    if (meshRef.current && !active && name === 'red') {
      // Remove angulation/slider effect
      if (rotate) {
        meshRef.current.rotation.y += 0.05
      }
    }
  })
  return (
    <>
      <mesh
        ref={meshRef}
        position={position}
        onPointerDown={(e) => {
          e.stopPropagation()
          setActive(name)
        }}
        userData={{ id: name }}
        castShadow
        receiveShadow
      >
        <boxGeometry args={ [1, 1, 1]} />
        <meshStandardMaterial color={color} />
      </mesh>
      {active && meshRef.current && (
        <TransformControls
          object={meshRef.current}
          mode="rotate"
          onObjectChange={handleObjectChange}
        />
      )}
    </>
  )
})

export default function Ortho({ rotate, angle }) {
  const [camera, setCamera] = useState(null)
  const [activeCube, setActiveCube] = useState(null)
  const [cube1Pos, setCube1Pos] = useState([-1, 0, 0])
  const [cube2Pos, setCube2Pos] = useState([1, 0, 0])
  const cube1Ref = useRef()
  const cube2Ref = useRef()

  // Deselect cubes on background click
  const handlePointerMissed = (e) => {
    setActiveCube(null)
  }

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas
        camera={{ fov: 10, position: [0, 0, 20] }}
        onCreated={({ camera }) => setCamera(camera)}
        onPointerMissed={handlePointerMissed}
        shadows
      >
        <ambientLight intensity={0.7} />
        {camera && <CameraFollowingLight camera={camera} />}
        <Cube
          ref={cube1Ref}
          color="red"
          position={cube1Pos}
          setPosition={setCube1Pos}
          active={activeCube === 'cube1'}
          setActive={() => setActiveCube('cube1')}
          name="red"
          rotate={rotate}
          angle={angle}
        />
        <Cube
          ref={cube2Ref}
          color="blue"
          position={cube2Pos}
          setPosition={setCube2Pos}
          active={activeCube === 'cube2'}
          setActive={() => setActiveCube('cube2')}
          name="blue"
        />
        <TrackballControls
          rotateSpeed={4}
          minDistance={1}
          maxDistance={200}
          enabled={!activeCube}
        />
      </Canvas>
    </div>
  )
}
