import React from 'react';
import { useLoader, useFrame } from '@react-three/fiber';
import { TransformControls, Html } from '@react-three/drei';
import { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { TextureLoader } from 'three/src/loaders/TextureLoader'

export function Tooth(props) {  
  const { toothID, url, stagingData, onTransform, landmarks, trackballControlsRef, isClicked, onToothClick, useShortRoots = false, showLandmarks = true } = props;

  const [hovered, hover] = useState(false);
  const toothRef = useRef(); 

  // Fallback error state for STL loading
  const [loadError, setLoadError] = useState(false);

  // STL loading with error handling
  const crown = useLoader(STLLoader, `/crowns/${toothID}.stl?ts=${props.meshVersion}`, loader => `${toothID}-crown-${props.meshVersion}`);
  const root = useLoader(STLLoader, useShortRoots ? 
    `/shortRoots/${toothID}.stl?ts=${props.meshVersion}` : 
    `/roots/${toothID}.stl?ts=${props.meshVersion}`,
    loader => `${toothID}-root-${props.meshVersion}`
  );
  const texture = useLoader(TextureLoader, `/textures/teeth.png`);

  const position = stagingData.position;
  const quaternion = stagingData.quaternion;

  const combinedGeometries = useMemo(() => {
    const crownGeometry = new THREE.BufferGeometry();
    const rootGeometry = new THREE.BufferGeometry();
    crownGeometry.setAttribute('position', new THREE.BufferAttribute(crown.attributes.position.array, 3));
    crownGeometry.setAttribute('normal', new THREE.BufferAttribute(crown.attributes.normal.array, 3));
    rootGeometry.setAttribute('position', new THREE.BufferAttribute(root.attributes.position.array, 3));
    rootGeometry.setAttribute('normal', new THREE.BufferAttribute(root.attributes.normal.array, 3));
    return { crownGeometry, rootGeometry };
  }, [crown, root]);

  const crownMaterial = new THREE.MeshStandardMaterial({
    map: texture,
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
    const lmPoint = landmarks[lmType];
    return (
      <mesh position={lmPoint}>
        <sphereGeometry args={[0.2]} />
        <meshStandardMaterial color={color} />
      </mesh>
    );
  }

  const [meshCenter, setMeshCenter] = useState(new THREE.Vector3());

  useEffect(() => {
    if (combinedGeometries) {
      const center = new THREE.Vector3();
      combinedGeometries.crownGeometry.computeBoundingBox();
      combinedGeometries.crownGeometry.boundingBox.getCenter(center);
      setMeshCenter(center);
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

  // render meshContent 
  const meshContent = (
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
      <mesh geometry={combinedGeometries.crownGeometry} material={crownMaterial} />
      <mesh geometry={combinedGeometries.rootGeometry} material={rootMaterial} />
      {showLandmarks && (
        <>
          <LandMark lmType="BCPoint" color="darkorange" />
          <LandMark lmType="FEGJPoint" color="brown" />
          <LandMark lmType="MRAPoint" color="darkblue" />
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
  // }, [toothID, onTransform, initialTransform, isDragging]);
  }, []);

  const TransformHint = () => (
    <>
      {!isDragging && (
        <Html
          style={{
            position: 'absolute',
            top: '-280px',      // Moved higher up
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
