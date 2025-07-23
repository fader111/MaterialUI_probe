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
  const { orthoData, setOrthoData, isFileLoaded, loading, onFileLoaded, baseCaseFilename } = props;
  const [stage, setStage] = useState(0);

  // Reset stage to 0 when a new case with lower stage amount is loaded (baseCaseFilename changes)
  useEffect(() => {
    setStage(0);
  }, [baseCaseFilename]);
  
  let [T2Stage, setT2Stage] = useState(0);
  const [camera, setCamera] = useState(null)
  const controlsRef = useRef(null);
  let [ showMode, setShowMode ] = useState(2); // <-- now controlled here
  const toothPlacementRef = useRef(null);
  const [controlsEnabled, setControlsEnabled] = useState(true);
  const [meshVersion, setMeshVersion] = useState(Date.now());
  // New state for left panel
  const [shortRoots, setShortRoots] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(false);

  // Use orthoData, setOrthoData, isFileLoaded, loading, baseCaseFilename from props?

  // Handle T2Stage updates
  useEffect(() => {
    setT2Stage(orthoData && orthoData.Staging && orthoData.Staging.length > 0 ? orthoData.T2Stage -1 : 0)
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
    if (!camera) {
      return;
    }
    if (!controlsRef.current) {
      return;
    }
    const controls = controlsRef.current;
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
  }, [showMode, camera, controlsRef]);

  // Handler for right panel view buttons
  const handleViewSelect = useCallback((viewKey) => {
    const viewMap = {
      bottom: 0,
      upper: 1,
      front: 2,
      right: 3,
      left: 4,
      rear: 5
    };
    const mode = viewMap[viewKey] ?? 2;
    setShowMode(mode);
  }, []);

  // Handler for left panel
  const handleShortRootsToggle = useCallback(() => setShortRoots(v => !v), []);
  const handleLandmarksToggle = useCallback(() => setShowLandmarks(v => !v), []);
  
  // Handler for T2 prediction (refactored to update orthoData.Staging)
  const handlePredictT2 = useCallback(async () => {
    console.log("handlePredictT2 called");
    try {
      const base_case_id = baseCaseFilename || '00000000';
      // const template_case_id = '00000000'; // TODO: make dynamic if needed
      const template_case_id = '120076_1'; // TODO: make dynamic if needed
      // const template_case_id = '103931_8.4'; // TODO: make dynamic if needed
      // const template_case_id = '120737_1'; // cs rotated on 90 - 12 teeth per jaw!!!
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
        const stageT2Idx = newOrthoData.Staging.length - 1;
        const newStage = { ...newOrthoData.Staging[stageT2Idx], RelativeToothTransformsHead: { ...newOrthoData.Staging[stageT2Idx].RelativeToothTransformsHead } };
        for (const toothID in prediction) {
          newStage.RelativeToothTransformsHead[toothID] = prediction[toothID];
        }
        newOrthoData.Staging[stageT2Idx] = newStage;
        return newOrthoData;
      });
    } catch (err) {
      console.error(err);
    }
  }, [baseCaseFilename, setOrthoData]);

  // Handler for Init Predict (now updates orthoData.Staging like handlePredictT2)
  const handlePredictInit = useCallback(async () => {
    console.log("handlePredictInit called");
    if (!baseCaseFilename) {
      console.error('handlePredictInit called without baseCaseFilename!');
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
        const stageT2Idx = newOrthoData.Staging.length - 1;
        const newStage = { ...newOrthoData.Staging[stageT2Idx], RelativeToothTransformsHead: { ...newOrthoData.Staging[stageT2Idx].RelativeToothTransformsHead } };
        for (const toothID in prediction) {
          newStage.RelativeToothTransformsHead[toothID] = prediction[toothID];
        }
        newOrthoData.Staging[stageT2Idx] = newStage;
        return newOrthoData;
      });
    } catch (err) {
      console.error('Init Predict error:', err);
    }
  }, [baseCaseFilename, setOrthoData]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Overlay
        onViewSelect={(viewKey) => {
          if (typeof viewKey !== 'undefined') {
            handleViewSelect(viewKey);
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
        onFileLoaded={onFileLoaded}
        baseCaseFilename={baseCaseFilename}
      >
        {loading ? (
          <div style={{ textAlign: 'center', marginTop: '20%' }}>Loading...</div>
        ) : isFileLoaded ? (
          <Canvas
            camera={{ fov: 10, position: [0, 0, 20] }}
            onCreated={({ camera }) => setCamera(camera)}
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
            />
          </Canvas>
        ) : (
          <div style={{ textAlign: 'center', marginTop: '20%' }}>Please load a file to start</div>
        )}
      </Overlay>
    </div>
  );
}
