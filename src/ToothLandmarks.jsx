import React from 'react';
import * as THREE from 'three';
import { Text } from '@react-three/drei';

export function LandMark({ lmData, color }) {
  if (lmData && lmData.start && lmData.end) {
    const points = [lmData.start, lmData.end];
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
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

export function LandmarksGroup({ landmarks }) {
  return (
    <>
      <LandMark lmData={landmarks.BCPoint} color="darkorange" />
      <LandMark lmData={landmarks.FEGJPoint} color="brown" />
      <LandMark lmData={landmarks.MRAPoint} color="darkblue" />
      <LandMark lmData={landmarks.MDWLine} color="darkred" />
    </>
  );
}

export function MainLine({ start, end }) {
  const points = [start, end];
  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
  const lineMaterial = new THREE.LineBasicMaterial({ 
    color: 0x9090EE, // light blue
    transparent: true,
    opacity: 0.6,
    linewidth: 1
  });
  return <line geometry={lineGeometry} material={lineMaterial} />;
}

// 3D label for tooth number
export function ToothNumberLabel({ toothID }) {
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
