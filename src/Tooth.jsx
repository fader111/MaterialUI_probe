import React from "react";
import TransformCommand from "./undo/TransformCommand";
import { sceneApi } from "./undo/sceneApi";
const getCommandManager = () => window.commandManager;
import { useLoader, useFrame } from "@react-three/fiber";
import { TransformControls, Html, Text } from "@react-three/drei";
import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { useToothTransform } from "./useToothTransform";
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { MeshBVH } from "three-mesh-bvh";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";
import { TextureLoader } from "three/src/loaders/TextureLoader";
import {
  LandMark,
  LandmarksGroup,
  MainLine,
  ToothNumberLabel,
} from "./ToothLandmarks.jsx";
import CombinedTransformControls from "./CombinedTransformControls";

// Custom hook for loading tooth mesh and texture
function useToothMesh(toothID, caseId, useShortRoots) {
  const [crown, setCrown] = useState(null);
  const [root, setRoot] = useState(null);
  const [loadError, setLoadError] = useState(false);
  const [texture, setTexture] = useState(null);

  useEffect(() => {
    let isMounted = true;
    setLoadError(false);
    // Load crown STL
    new STLLoader().load(
      `/crowns/${toothID}.stl?case=${caseId}`,
      (geometry) => {
        if (isMounted) setCrown(geometry);
      },
      undefined,
      (err) => {
        if (isMounted) setLoadError(true);
      }
    );
    // Load root STL
    new STLLoader().load(
      useShortRoots
        ? `/shortRoots/${toothID}.stl?case=${caseId}`
        : `/roots/${toothID}.stl?case=${caseId}`,
      (geometry) => {
        if (isMounted) setRoot(geometry);
      },
      undefined,
      (err) => {
        if (isMounted) setLoadError(true);
      }
    );
    // Load texture
    new TextureLoader().load(
      `/textures/teeth.png`,
      (tex) => {
        if (isMounted) setTexture(tex);
      },
      undefined,
      (err) => {
        if (isMounted) setTexture(null);
      }
    );
    return () => {
      isMounted = false;
    };
  }, [toothID, caseId, useShortRoots]);

  return { crown, root, loadError, texture };
}

export function Tooth(props) {
  const {
    toothID,
    url,
    stagingData,
    onTransform,
    landmarks,
    trackballControlsRef,
    isClicked,
    onToothClick,
    useShortRoots = false,
    showLandmarks = true,
  } = props;

  const [hovered, hover] = useState(false);
  // Removed old toothRef declaration; now provided by useToothTransform

  // Use custom hook for mesh loading
  const caseId = props.baseCaseFilename || props.meshVersion;
  const { crown, root, loadError, texture } = useToothMesh(
    toothID,
    caseId,
    useShortRoots
  );

  const position = stagingData.position;
  const quaternion = stagingData.quaternion;

  // Merge crown and root into a single BufferGeometry for BVH
  // Prepare separate BufferGeometries for rendering and merged for BVH
  const { crownGeometry, rootGeometry, mergedGeometry } = useMemo(() => {
    if (!crown || !root)
      return { crownGeometry: null, rootGeometry: null, mergedGeometry: null };
    const crownGeometry = new THREE.BufferGeometry();
    const rootGeometry = new THREE.BufferGeometry();
    crownGeometry.setAttribute("position", new THREE.BufferAttribute(crown.attributes.position.array, 3));
    crownGeometry.setAttribute("normal", new THREE.BufferAttribute(crown.attributes.normal.array, 3));
    rootGeometry.setAttribute("position", new THREE.BufferAttribute(root.attributes.position.array, 3));
    rootGeometry.setAttribute("normal", new THREE.BufferAttribute(root.attributes.normal.array, 3));
    const mergedGeometry = mergeGeometries([crownGeometry, rootGeometry]);
    return { crownGeometry, rootGeometry, mergedGeometry };
  }, [crown, root]);

  // Build BVH for collision detection
  const bvh = useMemo(() => {
    if (!mergedGeometry) return null;
    return new MeshBVH(mergedGeometry);
  }, [mergedGeometry]);

  const crownMaterial = texture
    ? new THREE.MeshStandardMaterial({
        map: texture,
        color: getColor({ clicked: isClicked, hovered }),
        transparent: true,
        opacity: 1.0,
      })
    : new THREE.MeshStandardMaterial({
        color: getColor({ clicked: isClicked, hovered }),
        transparent: true,
        opacity: 1.0,
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

  const [meshCenter, setMeshCenter] = useState(new THREE.Vector3());

  useEffect(() => {
    if (mergedGeometry) {
      const center = new THREE.Vector3();
      if (mergedGeometry.computeBoundingBox) {
        mergedGeometry.computeBoundingBox();
        if (mergedGeometry.boundingBox) {
          mergedGeometry.boundingBox.getCenter(center);
          setMeshCenter(center);
        }
      }
    }
  }, [mergedGeometry]);

  // Log missing tooth info to console
  useEffect(() => {
    if (loadError) {
      console.warn(`Missing STL for tooth ${toothID}`);
    }
  }, [loadError, toothID]);

  // Transform/undo logic extracted to custom hook
  const {
    isDragging,
    handleTransformStart,
    handleTransformEnd,
    handleObjectChange,
    toothRef,
  } = useToothTransform({
    toothID,
    trackballControlsRef,
    onTransform: (id, transform) => {
      // Always pass plain objects, not THREE.Vector3/Quaternion
      const plainTransform = {
        position: transform.position
          ? {
              x: transform.position.x,
              y: transform.position.y,
              z: transform.position.z,
            }
          : null,
        quaternion: transform.quaternion
          ? {
              x: transform.quaternion.x,
              y: transform.quaternion.y,
              z: transform.quaternion.z,
              w: transform.quaternion.w,
            }
          : null,
      };
      if (onTransform) onTransform(id, plainTransform);
    },
    isClicked,
    setOrthoData: props.setOrthoData,
    stage: props.stage,
  });

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
      {crownGeometry && (
        <mesh
          geometry={crownGeometry}
          material={crownMaterial}
          visible={true} // false for debug!!!!!
        />
      )}
      {rootGeometry && (
        <mesh 
          geometry={rootGeometry} 
          material={rootMaterial} 
          visible={true} // false for debug!!!!!
          />
      )}
      {/* BVH is built from mergedGeometry, debug mesh with texture */}
      {mergedGeometry && (
        <mesh
          geometry={mergedGeometry}
          material={
            texture
              ? new THREE.MeshStandardMaterial({
                  map: texture,
                  color: 0xffffff,
                  transparent: true,
                  opacity: 1.0,
                })
              : new THREE.MeshStandardMaterial({
                  color: 0xffffff,
                  transparent: true,
                  opacity: 1.0,
                })
          }
          visible={false} // true for debug!!!!!
        />
      )}
      <ToothNumberLabel toothID={toothID} />
      {showLandmarks && <LandmarksGroup landmarks={landmarks} />}
      {useShortRoots && showLandmarks && landmarks?.MRAPoint && meshCenter && (
        <MainLine start={meshCenter} end={landmarks.MRAPoint} />
      )}
    </group>
  );
  const [transformMode, setTransformMode] = useState("rotate");

  // Add keyboard event handler
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (!isClicked) return;

      if (event.key.toLowerCase() === "t" || event.key.toLowerCase() === "е") {
        setTransformMode("translate");
      } else if (
        event.key.toLowerCase() === "r" ||
        event.key.toLowerCase() === "к"
      ) {
        setTransformMode("rotate");
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
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
            onObjectChange={handleObjectChange}
            onPointerDown={(e) => e.stopPropagation()}
            onPointerUp={(e) => e.stopPropagation()}
            onPointerMove={(e) => e.stopPropagation()}
          />
        </>
      )}
      {/* <TransformHint /> */}
      {isClicked && (
        <Html
          position={[0, 0, 20]}
          // fullscreen
          style={{
            position: "absolute",
            top: "50%", // Move higher up
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(209, 201, 201, 0.37)",
            padding: "8px",
            borderRadius: "4px",
            color: "white",
            fontSize: "14px",
            fontFamily: "Arial",
            pointerEvents: "none",
            userSelect: "none",
            whiteSpace: "nowrap",
            zIndex: 1000,
          }}
          prepend
          portal
        >
          Press T for translation, R for rotation
        </Html>
      )}
    </>
  );
}
