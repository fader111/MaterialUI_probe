import React from 'react';
import { useLoader, useFrame } from '@react-three/fiber';
import { TransformControls, Html, Text } from '@react-three/drei';
// import { Html, Text } from '@react-three/drei';
import { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { TextureLoader } from 'three/src/loaders/TextureLoader';
import CombinedTransformControls from "./CombinedTransformControls";

export function Tooth(props) {  
  const { toothID, url, stagingData, onTransform, landmarks, trackballControlsRef, isClicked, onToothClick, useShortRoots = false, showLandmarks = true } = props;

  const [hovered, hover] = useState(false);
  const toothRef = useRef(); 

  // Robust STL loading with error handling
  const [crown, setCrown] = useState(null);
  const [root, setRoot] = useState(null);
  const [loadError, setLoadError] = useState(false);
  const [texture, setTexture] = useState(null);

  // STL loading 
  const caseId = props.baseCaseFilename || props.meshVersion;
  useEffect(() => {
    let isMounted = true;
    setLoadError(false);
    // Load crown STL
    new STLLoader().load(
      `/crowns/${toothID}.stl?case=${caseId}`,
      geometry => { if (isMounted) setCrown(geometry); },
      undefined,
      err => { if (isMounted) setLoadError(true); }
    );
    // Load root STL
    new STLLoader().load(
      useShortRoots
        ? `/shortRoots/${toothID}.stl?case=${caseId}`
        : `/roots/${toothID}.stl?case=${caseId}`,
      geometry => { if (isMounted) setRoot(geometry); },
      undefined,
      err => { if (isMounted) setLoadError(true); }
    );
    // Load texture
    new TextureLoader().load(
      `/textures/teeth.png`,
      tex => { if (isMounted) setTexture(tex); },
      undefined,
      err => { if (isMounted) setTexture(null); }
    );
    return () => { isMounted = false; };
  }, [toothID, caseId, useShortRoots]);

  const position = stagingData.position;
  const quaternion = stagingData.quaternion;

  const combinedGeometries = useMemo(() => {
    if (!crown || !root) return { crownGeometry: null, rootGeometry: null };
    const crownGeometry = new THREE.BufferGeometry();
    const rootGeometry = new THREE.BufferGeometry();
    crownGeometry.setAttribute('position', new THREE.BufferAttribute(crown.attributes.position.array, 3));
    crownGeometry.setAttribute('normal', new THREE.BufferAttribute(crown.attributes.normal.array, 3));
    rootGeometry.setAttribute('position', new THREE.BufferAttribute(root.attributes.position.array, 3));
    rootGeometry.setAttribute('normal', new THREE.BufferAttribute(root.attributes.normal.array, 3));
    return { crownGeometry, rootGeometry };
  }, [crown, root]);

  const crownMaterial = texture ? new THREE.MeshStandardMaterial({
    map: texture,
    color: getColor({ clicked: isClicked, hovered }),
    transparent: true,
    opacity: 1.0
  }) : new THREE.MeshStandardMaterial({
    color: getColor({ clicked: isClicked, hovered }),
    transparent: true,
    opacity: 1.0
  });

  const rootMaterial = new THREE.MeshStandardMaterial({
    color: 0x999999, // grey color
    transparent: true,
    opacity: 0.9,
  });

  function getColor({ clicked, hovered }) {
    if (clicked && hovered) return 0x90caf9;
    else if (clicked && !hovered) return 0xadd8e6;
    else if (!clicked && hovered) return 0xcccccc;
    else return "white";
  }

  function LandMark({ lmType, color }) {
    const lmData = landmarks[lmType];
    // If it's a line landmark (object with start/end), draw a line
    if (lmData && lmData.start && lmData.end) {
      const points = [lmData.start, lmData.end];
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
      // const lineMaterial = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 });
      const lineMaterial = new THREE.LineBasicMaterial({ color });
      return (
        <>
          <line geometry={lineGeometry} material={lineMaterial} />
          <mesh position={lmData.start}>
            <sphereGeometry args={[0.13]} />
            <meshStandardMaterial color={color} />
          </mesh>
          <mesh position={lmData.end}>
            <sphereGeometry args={[0.13]} />
            <meshStandardMaterial color={color} />
          </mesh>
        </>
      );
    }
    // Otherwise, draw a sphere for point landmark
    if (lmData) {
      return (
        <mesh position={lmData}>
          <sphereGeometry args={[0.2]} />
          <meshStandardMaterial color={color} />
        </mesh>
      );
    }
    return null;
  }

  const [meshCenter, setMeshCenter] = useState(new THREE.Vector3());

  useEffect(() => {
    if (combinedGeometries && combinedGeometries.crownGeometry) {
      const center = new THREE.Vector3();
      if (combinedGeometries.crownGeometry.computeBoundingBox) {
        combinedGeometries.crownGeometry.computeBoundingBox();
        if (combinedGeometries.crownGeometry.boundingBox) {
          combinedGeometries.crownGeometry.boundingBox.getCenter(center);
          setMeshCenter(center);
        }
      }
    }
  }, [combinedGeometries]);

  const MainLine = ({ start, end }) => {
    const points = [start, end];
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const lineMaterial = new THREE.LineBasicMaterial({ 
      color: 0x9090EE, // light blue
      transparent: true,
      opacity: 0.6,
      linewidth: 1
    });
    return <line geometry={lineGeometry} material={lineMaterial} />;
  };

  // Log missing tooth info to console
  useEffect(() => {
    if (loadError) {
      console.warn(`Missing STL for tooth ${toothID}`);
    }
  }, [loadError, toothID]);

  // render meshContent 
  const meshContent = loadError ? (
    <group ref={toothRef} position={position} quaternion={quaternion}>
      {/* <Html style={{ color: 'red', background: 'rgba(255,0,0,0.1)', padding: '4px', borderRadius: '4px', fontWeight: 'bold' }}>
        Missing STL for tooth <span style={{color:'black'}}>{toothID}</span>
      </Html> */}
    </group>
  ) : (
    <group
      ref={toothRef}
      position={position}
      quaternion={quaternion}
      onClick={(event) => {
        event.stopPropagation();
        if (!isClicked) {
          onToothClick(toothID);
        }
      }}
      onPointerOver={(event) => (event.stopPropagation(), hover(true))}
      onPointerOut={() => hover(false)}
    >
      {combinedGeometries.crownGeometry && <mesh geometry={combinedGeometries.crownGeometry} material={crownMaterial} />}
      {combinedGeometries.rootGeometry && <mesh geometry={combinedGeometries.rootGeometry} material={rootMaterial} />}
      <ToothNumberLabel toothID={toothID} />
      {showLandmarks && (
        <>
          <LandMark lmType="BCPoint" color="darkorange" />
          <LandMark lmType="FEGJPoint" color="brown" />
          <LandMark lmType="MRAPoint" color="darkblue" />
          <LandMark lmType="MDWLine" color="darkred"       />
        </>
      )}
      {useShortRoots && showLandmarks && landmarks?.MRAPoint && meshCenter && (
        <MainLine 
          start={meshCenter}
          end={landmarks.MRAPoint}
        />
      )}
    </group>
  );

  const [isDragging, setIsDragging] = useState(false);
  const [initialTransform, setInitialTransform] = useState(null);
  
  useEffect(() => {
    if (isClicked && toothRef.current) {
      setInitialTransform({
        position: toothRef.current.position.clone(),
        quaternion: toothRef.current.quaternion.clone()
      });
    }
  }, [isClicked]);

  const handleTransformStart = useCallback(() => {
    if (trackballControlsRef?.current) {
      trackballControlsRef.current.enabled = false;
      setIsDragging(true);
    }
  }, [trackballControlsRef]);
  // }, []);

  const handleTransformEnd = useCallback(() => {
    if (trackballControlsRef?.current) {
      trackballControlsRef.current.enabled = true;
      setIsDragging(false);
      // console.log("end drag")
    }
    if (toothRef.current && initialTransform) {
      const newTransforms = {
        translation: toothRef.current.position.clone(),
        rotation: toothRef.current.quaternion.clone()
      };
      onTransform(toothID, newTransforms);
      console.log("end drag 2 part")
    }
  }, [trackballControlsRef, initialTransform]);
  // }, []);



  const handleObjectChange = useCallback(() => {
    if (toothRef.current && initialTransform) {
      const newTransforms = {
        translation: toothRef.current.position.clone(),
        rotation: toothRef.current.quaternion.clone()
      };
      onTransform(toothID, newTransforms);
      console.log("from handleObjectChange")
    }
  }, [toothID, onTransform, initialTransform, isDragging]);
  // }, []);

  const TransformHint = () => (
    <>
      {!isDragging && (
        <Html
          style={{
            position: 'absolute',
            top: '20%',      // Moved higher up
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(209, 201, 201, 0.37)',
            padding: '8px',
            borderRadius: '4px',
            color: 'white',
            fontSize: '14px',
            fontFamily: 'Arial',
            pointerEvents: 'none',
            userSelect: 'none',
            whiteSpace: 'nowrap',
            zIndex: 1000
          }}
          prepend
          portal
        >
          Press T for translation, R for rotation
        </Html>
      )}
    </>
  );

  const [transformMode, setTransformMode] = useState('rotate');

  // Add keyboard event handler
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (!isClicked) return;
      
      if (event.key.toLowerCase() === 't' || event.key.toLowerCase() === 'е') {
        setTransformMode('translate');
      } else if (event.key.toLowerCase() === 'r' || event.key.toLowerCase() === 'к') {
        setTransformMode('rotate');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isClicked]);
  
  return (
    <>
      {meshContent}
      {isClicked && toothRef.current && (
        <>
          <TransformControls
            enabled={true}
            mode={transformMode}
            size={0.7}
            object={toothRef.current}
            space="local"
            // lineWidth={9} // Это не катит - надо форкать и менять контрол или использовать THREE вариант
            onMouseDown={handleTransformStart}
            onMouseUp={handleTransformEnd}
            // onObjectChange={handleObjectChange}
            onPointerDown={(e) => e.stopPropagation()}
            onPointerUp={(e) => e.stopPropagation()}
            onPointerMove={(e) => e.stopPropagation()}
          />
          <TransformHint />
        </>
      )}
    </>
  );
}

// 3D label for tooth number
function ToothNumberLabel({ toothID }) {
  // Support supernumerary teeth: if id > 50 and id-40 is a valid tooth, treat as supernumerary
  const idNum = parseInt(toothID);
  let displayID = toothID;
  let isSupernumerary = false;
  if (idNum > 50 && idNum - 40 > 0 && idNum - 40 < 50) {
    // displayID = `${idNum - 40}`;
    isSupernumerary = true;
  }
  const isUpper = idNum < 30 || (isSupernumerary && idNum - 40 < 30);
  const isAnterior = (isSupernumerary ? (idNum - 40) : idNum) % 10 <= 3;
  const position = isAnterior ? [0, -4, 5] : [0, -6, 3];
  const rotation = isUpper ? [Math.PI / 2, 0, Math.PI] : [Math.PI / 2, 0, 0];
  return (
    <Text
      position={position}
      rotation={rotation}
      fontSize={1.0}
      color={isSupernumerary ? "#b22222" : "black"}
      anchorX="center"
      anchorY="middle"
      outlineWidth={0.04}
      outlineColor={isSupernumerary ? "#ffcccc" : "white"}
    >
      {displayID}
    </Text>
  );
}
