import React, { useRef, useState, useEffect, useMemo, useCallback, Suspense} from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { TrackballControls } from '@react-three/drei'
import * as THREE from 'three'
import CombinedTransformControls from './CombinedTransformControls'
// import NativeCombinedTransformControls from './NativeCombinedTransformControls'
import { ToothPlacement } from './ToothPlacement';
import Overlay from './Overlay';

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
  return <directionalLight ref={lightRef} color="white" intensity={2} />
}

export default function Ortho(props) {
  const [stage, setStage] = useState(0);
  let [T2Stage, setT2Stage] = useState(0);
  const [camera, setCamera] = useState(null)
  const controlsRef = useRef(null);
  let [ showMode, setShowMode ] = useState(2); // <-- now controlled here
  const toothPlacementRef = useRef(null);
  const [orthoData, setOrthoData] = useState([]); // ensure orthoData is managed here
  const [controlsEnabled, setControlsEnabled] = useState(true);
  const [meshVersion, setMeshVersion] = useState(Date.now());
  // New state for left panel
  const [shortRoots, setShortRoots] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(false);
  // Track the loaded filename for ToothPlacement
  const [baseCaseFilename, setBaseCaseFilename] = useState(() => {
    // Initialize from localStorage if available
    try {
      const stored = localStorage.getItem('baseCaseFilename');
      return stored ? stored : null;
    } catch (e) {
      return null;
    }
  });

  // Initial data fetch  
  useEffect(() => {
    const fetchOrthoData = async () => {
      try {
        const response = await fetch('/orthoData.json');
        if (!response.ok) {
          throw new Error('Error fetching case data');
        }
        const data = await response.json();
        setOrthoData(data);
        // console.log('Initial ortho data loaded:', orthoData); // тут пусто обычно!!!
      } catch (error) {
        console.error(error);
      }
    };
    fetchOrthoData();
  }, []); // Only run on mount

  // Handler to reload orthoData after file upload/processing
  const handleFileLoaded = useCallback(async (filename) => {
    try {
      const timestamp = Date.now();
      const response = await fetch(`/orthoData.json?_=${timestamp}`); // cache-busting
      if (!response.ok) throw new Error('Failed to reload orthoData');
      const data = await response.json();
      setMeshVersion(timestamp);
      setOrthoData(data);
      const cleanFilename = filename.replace(/\.oas$/i, '');
      setBaseCaseFilename(cleanFilename);
      try {
        localStorage.setItem('baseCaseFilename', cleanFilename);
      } catch (e) {
        // Ignore localStorage errors
      }
      if (props.onFileLoaded) {
        await props.onFileLoaded(filename);
      }
    } catch (err) {
      console.error('Failed to reload orthoData:', err);
    }
  }, [props.onFileLoaded]);

  // Handle T2Stage updates
  useEffect(() => {
    setT2Stage(orthoData.Staging && orthoData.Staging.length > 0 ? orthoData.T2Stage -1 : 0)
  }, [orthoData, setT2Stage]);

  // задание начальной позиции камеры
  useEffect(() => {
    if (controlsRef.current && camera) {
      camera.up = new THREE.Vector3(0, 1, 1); // hack
      camera.position.set(0, -350, 0); // без хака блочит работу контролов если камера в y плоскости. баг в библиотеке TrackballControls
    }
  }, [camera, controlsRef])

  // выбор режима вида при нажатии кнопок выбора вида в overlay
  useEffect(() => {
  // useMemo(() => {
    if (!camera) {
      // console.log('DEBUG: camera is missing');
      return;
    }
    if (!controlsRef.current) {
      // console.log('DEBUG: controlsRef.current is missing');
      return;
    }
    const controls = controlsRef.current;
    // console.log('DEBUG: showMode', showMode, 'camera', camera, 'controls', controlsRef.current);
    // console.log('DEBUG: camera position before', camera.position.toArray());
    // Set camera position and up vector
    if (showMode === 0) {
      camera.up.set(0, -1, 1);
      camera.position.set(0, 0, -350);
    } else if (showMode === 1) {
      camera.up.set(0, 1, 1);
      camera.position.set(0, 0, 350);
    } else if (showMode === 2) {
      camera.up.set(0, 0, 1);
      camera.position.set(0, -350, 0);
    } else if (showMode === 3) {
      camera.up.set(0, 0, 1);
      camera.position.set(-350, 0, 0);
    } else if (showMode === 4) {
      camera.up.set(0, 0, 1);
      camera.position.set(350, 0, 0);
    } else if (showMode === 5) {
      camera.up.set(0, 0, 1);
      camera.position.set(0, 350, 0);
    }
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    controls.update();
    // console.log('DEBUG: camera position after', camera.position.toArray());
    // console.log('DEBUG: camera up', camera.up.toArray());
  }, [showMode, camera, controlsRef]);

  // Handler for right panel view buttons
  const handleViewSelect = useCallback((viewKey) => {
    // console.log('DEBUG: handleViewSelect called with', viewKey);
    // Map viewKey to showMode index (should match ToothPlacement logic)
    const viewMap = {
      bottom: 0,
      upper: 1,
      front: 2,
      right: 3,
      left: 4,
      rear: 5
    };
    const mode = viewMap[viewKey] ?? 2;
    // console.log('DEBUG: setShowMode called with', mode);
    setShowMode(mode);
  }, []);

  // Handler for left panel
  const handleShortRootsToggle = useCallback(() => setShortRoots(v => !v), []);
  const handleLandmarksToggle = useCallback(() => setShowLandmarks(v => !v), []);
  
  // Handler for T2 prediction (refactored to update orthoData.Staging)
  const handlePredictT2 = useCallback(async () => {
    try {
      const base_case_id = baseCaseFilename || '00000000';
      const template_case_id = '103931_8.4'; // TODO: make dynamic if needed
      const response = await fetch('http://localhost:8000/predict-t2/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_case_id, template_case_id })
      });
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      const prediction = await response.json();
      setOrthoData(prev => {
        if (!prev || !prev.Staging) return prev;
        const newOrthoData = { ...prev, Staging: [...prev.Staging] };
        const stageIdx = newOrthoData.Staging.length - 1; // or use current stage
        const newStage = { ...newOrthoData.Staging[stageIdx], RelativeToothTransforms: { ...newOrthoData.Staging[stageIdx].RelativeToothTransforms } };
        for (const toothID in prediction) {
          newStage.RelativeToothTransforms[toothID] = prediction[toothID];
        }
        newOrthoData.Staging[stageIdx] = newStage;
        return newOrthoData;
      });
    } catch (err) {
      // Optionally handle error UI here
      console.error(err);
    }
  }, [baseCaseFilename]);

  // Handler for Init Predict (now updates orthoData.Staging like handlePredictT2)
  const handlePredictInit = useCallback(async () => {
    if (!baseCaseFilename) {
      return;
    }
    try {
      const resp = await fetch('http://localhost:8000/predict-init/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_case_id: baseCaseFilename.replace(/\.oas$/, '') })
      });
      if (!resp.ok) throw new Error('Server error');
      const prediction = await resp.json();
      setOrthoData(prev => {
        if (!prev || !prev.Staging) return prev;
        const newOrthoData = { ...prev, Staging: [...prev.Staging] };
        const stageIdx = newOrthoData.Staging.length - 1; // or use current stage
        const newStage = { ...newOrthoData.Staging[stageIdx], RelativeToothTransforms: { ...newOrthoData.Staging[stageIdx].RelativeToothTransforms } };
        for (const toothID in prediction) {
          newStage.RelativeToothTransforms[toothID] = prediction[toothID];
        }
        newOrthoData.Staging[stageIdx] = newStage;
        return newOrthoData;
      });
    } catch (err) {
      console.error('Init Predict error:', err);
    }
  }, [baseCaseFilename]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Overlay
        onViewSelect={(viewKey) => {
          // console.log('DEBUG: Overlay onViewSelect', viewKey, typeof viewKey);
          if (typeof viewKey !== 'undefined') {
            handleViewSelect(viewKey);
          } else {
            // console.log('DEBUG: Overlay onViewSelect called with undefined!');
          }
        }}
        onShortRootsToggle={handleShortRootsToggle}
        shortRoots={shortRoots}
        onLandmarksToggle={handleLandmarksToggle}
        showLandmarks={showLandmarks}
        onPredictT2={handlePredictT2}
        onPredictInit={handlePredictInit}
        stage={stage}
        maxStage={T2Stage}
        onStageChange={setStage}
        onFileLoaded={handleFileLoaded}
        baseCaseFilename={baseCaseFilename}
      >
        <Canvas
          camera={{ fov: 10, position: [0, 0, 20] }}
          onCreated={({ camera }) => setCamera(camera)}
          // shadows
        >
          <Suspense fallback={<div>Loading...</div>} />
          <ambientLight intensity={0.3} />
          {camera && <CameraFollowingLight camera={camera} />}
          <axesHelper args={[5]} />
          <ToothPlacement
            ref={toothPlacementRef}
            orthoData={orthoData}
            setOrthoData={setOrthoData}
            stage={stage}
            showMode={showMode}
            onShowModeChange={setShowMode}
            trackballControlsRef={controlsRef} 
            setControlsEnabled={setControlsEnabled}
            meshVersion={meshVersion}
            useShortRoots={shortRoots}
            showLandmarks={showLandmarks}
            baseCaseFilename={baseCaseFilename}
          />
          <TrackballControls
            ref={controlsRef}
            rotateSpeed={4}
            minDistance={100}
            maxDistance={900}
            onChange={() => {
              // console.log('activeCube during TrackballControls:', activeCube);
            }}
          />
        </Canvas>
      </Overlay>
    </div>
  )
}
