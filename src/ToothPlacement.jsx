import React, { useState, useEffect, useMemo, useCallback, forwardRef, useImperativeHandle } from 'react';
import * as THREE from 'three';
import { Tooth } from './Tooth';
// import { Tooth } from './ToothRotated';
import { rt, transform, toVec3, calcLinearStaging, calcMAPSStaging } from "./misc";
import { SpaseCollisionResolver } from './SpaseCollisionResolver';
import { sceneApi } from './undo/sceneApi';
import TransformCommand from './undo/TransformCommand';
import { Html } from '@react-three/drei';
import { MeshBVH, MeshBVHHelper } from "three-mesh-bvh";

export const ToothPlacement = forwardRef((props, ref) => {
    // --- stagingDataSelector and stagingData must be defined before use ---
    const {
        trackballControlsRef, 
        setControlsEnabled, 
        orthoData,
        setOrthoData,
        stage,
        // onStagingDataUpdate, // Not in the Ortho!! remove?? 
        stagingPatterns, 
        stagingType = "Case", 
        showMode: showModeProp = 2, // allow controlled showMode
        useShortRoots = false,
        showLandmarks = true,
    } = props;

    // --- stagingDataSelector and stagingData ---
    // (copy from below, move up)
    // const caseStagingData = orthoData?.Staging || null;
    // ...existing code for caseStagingData, stagesNum, stagingDataT1, stagingDataT2, jsonT1Vec3, etc...
    const caseStagingData = orthoData?.Staging || null;
    const stagesNum = caseStagingData ? caseStagingData.length : 0;
    const stagingDataT1 = caseStagingData && stagesNum > 0 ? caseStagingData[0] : null;
    const stagingDataT2 = caseStagingData && stagesNum > 0 ? caseStagingData[stagesNum - 1] : null;
    const { jsonT1Vec3, jsonT2Vec3, jsonStageVec3 } = useMemo(() => {
        let stageVec3 = {};
        let t1Vec3 = {};
        let t2Vec3 = {};
        if (caseStagingData && stagesNum > 0) {
            for (const toothID in stagingDataT2.RelativeToothTransforms) {
                const toothRt = rt(caseStagingData[stage]?.RelativeToothTransforms?.[toothID]);
                const toothRtT1 = rt(stagingDataT1.RelativeToothTransforms[toothID]);
                const toothRtT2 = rt(stagingDataT2.RelativeToothTransforms[toothID]);
                stageVec3[toothID] = {
                    position: toothRt.translation,
                    quaternion: toothRt.quaternion
                };
                t1Vec3[toothID] = {
                    position: toothRtT1.translation,
                    quaternion: toothRtT1.quaternion
                };
                t2Vec3[toothID] = {
                    position: toothRtT2.translation,
                    quaternion: toothRtT2.quaternion
                };
            }
        }
        return { jsonT1Vec3: t1Vec3, jsonT2Vec3: t2Vec3, jsonStageVec3: stageVec3 };
    }, [caseStagingData, stagingDataT1, stage]);
    const linearStagingData = useMemo(() => calcLinearStaging(jsonT1Vec3, jsonT2Vec3, stagesNum), [stagesNum]);
    const MAPSStagingData = useMemo(() => calcMAPSStaging(jsonT1Vec3, jsonT2Vec3, stagesNum, stagingPatterns, null), [stagesNum, jsonT1Vec3, stagingPatterns]);
    const stagingDataSelector = useMemo(() => ({
        Case: jsonStageVec3,
        Linear: linearStagingData[stage] || {},
        MAPS: MAPSStagingData[stage] || {}
    }), [jsonStageVec3, linearStagingData, MAPSStagingData, stage]);
    let stagingData = stagingDataSelector[stagingType] || {};

    // --- validToothIDs and currentStageData ---
    const currentStageData = stagingData;
    const validToothIDs = Object.keys(currentStageData).filter(toothID => /^\d+$/.test(toothID));
    
    // --- force update after all teeth have registered in sceneApi ---
    const registered = sceneApi.getRegisteredObjects();
    const [, forceUpdate] = useState(0);
    // // ...existing code...
    // useEffect(() => {
    //     if (
    //         validToothIDs.length > 0 &&
    //         registered &&
    //         registered.length >= validToothIDs.length
    //     ) {
    //         forceUpdate(n => n + 1);
    //     }
    //     // eslint-disable-next-line react-hooks/exhaustive-deps
    // }, [validToothIDs.length, registered && registered.length]);
    
    // console.log("ToothPlacement props.meshVersion:", props.meshVersion);
    const [landmarksT1, setLandmarksT1] = useState(null);
    const [clickedToothId, setClickedToothId] = useState(null);

    const handleToothClick = (toothId) => {
        setClickedToothId(toothId === clickedToothId ? null : toothId);
    };

    const handleCanvasClick = useCallback((event) => {
        setClickedToothId(null);
        if (event.target.nodeName === 'CANVAS') {
        }
    }, []);

    // const jsonMandibularData = orthoData?.mandibularRelativeTransform || null;
    // const jsonMaxillaData = orthoData?.maxillaRelativeTransform || null;
    // ...existing code...

    // const handleToothTransformControl_ = useCallback((toothId, transforms) => {
    //     console.log("handleToothTransformControl called for toothId:", toothId, "with transforms:", transforms);
    //     if (orthoData?.Staging && orthoData.Staging[stage]) {
    //         // Use rt from misc to convert transforms to translation/quaternion
    //         const { translation, quaternion } = rt(transforms);
    //         // Determine jawRT and its inverse for this tooth
    //         const jawRT = (parseInt(toothId) < 30) ? rt(orthoData?.mandibularRelativeTransform || null) : rt(orthoData?.maxillaRelativeTransform || null);
    //         // Convert world transform to jaw coordinate system
    //         // Inverse jawRT: T_jaw = jawRT^-1 * T_world
    //         const invJawQuat = jawRT.quaternion.clone().invert();
    //         const invJawTrans = jawRT.translation.clone().negate().applyQuaternion(invJawQuat);
    //         // Transform translation
    //         const jawTranslation = translation.clone().sub(jawRT.translation).applyQuaternion(invJawQuat);
    //         const translationObj = { x: jawTranslation.x, y: jawTranslation.y, z: jawTranslation.z };
    //         // Transform rotation
    //         const jawQuaternion = invJawQuat.clone().multiply(quaternion.clone());
    //         const rotationObj = {
    //             x: jawQuaternion.x,
    //             y: jawQuaternion.y,
    //             z: jawQuaternion.z,
    //             w: jawQuaternion.w
    //         };
    //         setOrthoData(prev => {
    //             if (!prev || !prev.Staging) return prev;
    //             const newOrthoData = { ...prev, Staging: [...prev.Staging] };
    //             const newStage = { ...newOrthoData.Staging[stage], RelativeToothTransforms: { ...newOrthoData.Staging[stage].RelativeToothTransforms } };
    //             newStage.RelativeToothTransforms[toothId] = {
    //                 ...newStage.RelativeToothTransforms[toothId],
    //                 translation: translationObj,
    //                 rotation: rotationObj
    //             };
    //             newOrthoData.Staging[stage] = newStage;
    //             return newOrthoData;
    //         });
    //     }
    // // }, [orthoData, stage, setOrthoData]);
    // }, []);

        // handke 
    const handleToothTransformControl = useCallback((toothId, transforms) => {
        // console.log("handleToothTransformControl called for toothId:", toothId, "with transforms:", transforms);
        if (orthoData?.Staging && orthoData.Staging[stage]) {
            // Defensive checks
            const localTranslation = transforms?.position;
            const rotation = transforms?.quaternion;
            let localRotation = null;
            if (rotation && typeof rotation.x === 'number' && typeof rotation.y === 'number' && typeof rotation.z === 'number' && typeof rotation.w === 'number') {
                localRotation = new THREE.Quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
            } else {
                // fallback to identity quaternion
                localRotation = new THREE.Quaternion();
            }
            // Ensure translation is always a plain object
            const translationObj = (localTranslation && localTranslation instanceof THREE.Vector3)
                ? { x: localTranslation.x, y: localTranslation.y, z: localTranslation.z }
                : (localTranslation || { x: 0, y: 0, z: 0 });
            const localTransforms = {
                translation: translationObj,
                rotation: {
                    x: localRotation.x,
                    y: localRotation.y,
                    z: localRotation.z,
                    w: localRotation.w
                }
            };
            setOrthoData(prev => {
                if (!prev || !prev.Staging) return prev;
                const newOrthoData = { ...prev, Staging: [...prev.Staging] };
                const newStage = { ...newOrthoData.Staging[stage], RelativeToothTransforms: { ...newOrthoData.Staging[stage].RelativeToothTransforms } };
                newStage.RelativeToothTransforms[toothId] = {
                    ...newStage.RelativeToothTransforms[toothId],
                    translation: localTransforms.translation,
                    rotation: localTransforms.rotation
                };
                newOrthoData.Staging[stage] = newStage;
                return newOrthoData;
            });
        }
    // }, [orthoData, stage, mandibulaRt, maxillaRt, setOrthoData]);
    }, [orthoData, stage, setOrthoData]);

    // Local state for showMode if not controlled
    const [showMode, setShowMode] = useState(showModeProp);
    // Sync with parent if controlled
    React.useEffect(() => { setShowMode(showModeProp); }, [showModeProp]);

    // Only R3F objects inside group
    // Use computed stagingData only
    // console.log("rerenderStageData", rerenderStageData);
    // console.log("currentStage Data", stage, currentStageData);

    // Filter only valid tooth IDs
    // console.log("currentStageData", currentStageData);

    // Compute ordered teeth and centers for visualization and collision logic
    
    let ordered = [];
    let centers = [];
    let meshes = [];
    if (registered && registered.length > 0) {
        // Use maxilla for visualization (or make jaw a prop/state if needed)
        const jaw = 'maxilla';
        const filtered = registered.filter(o => {
            const id = parseInt(o.id, 10);
            const quadrant = Math.floor(id / 10);
            const isMaxillary = quadrant === 1 || quadrant === 2;
            const isMandibular = quadrant === 3 || quadrant === 4;
            if (jaw === 'mandible') return isMandibular;
            if (jaw === 'maxilla' ) return isMaxillary;
            return true;
        });
        let sorted = [];
        if (jaw === 'maxilla') {
            const q1 = filtered.filter(o => Math.floor(parseInt(o.id,10)/10) === 1)
                .sort((a,b) => parseInt(b.id,10) - parseInt(a.id,10));
            const q2 = filtered.filter(o => Math.floor(parseInt(o.id,10)/10) === 2)
                .sort((a,b) => parseInt(a.id,10) - parseInt(b.id,10));
            sorted = [...q1, ...q2];
        } else if (jaw === 'mandible') {
            const q3 = filtered.filter(o => Math.floor(parseInt(o.id,10)/10) === 3)
                .sort((a,b) => parseInt(a.id,10) - parseInt(b.id,10));
            const q4 = filtered.filter(o => Math.floor(parseInt(o.id,10)/10) === 4)
                .sort((a,b) => parseInt(a.id,10) - parseInt(b.id,10));
            sorted = [...q3, ...q4];
        } else {
            sorted = filtered.sort((a, b) => parseInt(a.id, 10) - parseInt(b.id, 10));
        }
        ordered = sorted;
        centers = ordered.map(o => new THREE.Vector3(o.center.x, o.center.y, o.center.z));
        meshes = ordered.map(o => o.mesh).filter(m => !!m);
    }

    // Local function to resolve collisions, reused by ref and UI button
    const resolveCollisionsLocal = (jaw = 'both') => {
        try {
            if (!ordered || ordered.length < 2) {
                console.warn('Not enough teeth to resolve collisions');
                return;
            }
            const result = SpaseCollisionResolver(meshes, centers);
            const optimizedCenters = result.centers || centers;
            // Применить новые позиции к orthoData
            setOrthoData(prev => {
                if (!prev || !prev.Staging || !prev.Staging[stage]) return prev;
                const newOrthoData = { ...prev, Staging: [...prev.Staging] };
                const newStage = { ...newOrthoData.Staging[stage], RelativeToothTransforms: { ...newOrthoData.Staging[stage].RelativeToothTransforms } };
                ordered.forEach((o, i) => {
                    const toothId = o.id;
                    const prevTransform = newStage.RelativeToothTransforms[toothId] || {};
                    newStage.RelativeToothTransforms[toothId] = {
                        ...prevTransform,
                        translation: { x: optimizedCenters[i].x, y: optimizedCenters[i].y, z: optimizedCenters[i].z }
                        // rotation: prevTransform.rotation // оставляем прежний quaternion
                    };
                });
                newOrthoData.Staging[stage] = newStage;
                return newOrthoData;
            });
        } catch (err) {
            console.error('resolveCollisions failed', err);
        }
    };

    // Expose API to parent via ref: resolveCollisions(jaw)
    useImperativeHandle(ref, () => ({
        resolveCollisions: resolveCollisionsLocal
    }));

    return (
        <>
        <group onClick={handleCanvasClick}>
            {/* Debug visualization: lines between centers, color for colliding pairs, and bounding boxes */}
            {centers && centers.length > 1 && (
                <>
                    {centers.map((c, i) => {
                        if (i === centers.length - 1) return null;
                        const c2 = centers[i + 1];
                        const dist = c.distanceTo(c2);
                        const color = dist < 10 ? 'red' : 'green';
                        return (
                            <line key={`debug-line-${i}`}
                                geometry={new THREE.BufferGeometry().setFromPoints([c, c2])}
                            >
                                <lineBasicMaterial attach="material" color={color} linewidth={2} />
                            </line>
                        );
                    })}
                    {centers.map((c, i) => (
                        <mesh key={`debug-center-${i}`} position={c}>
                            <sphereGeometry args={[0.5, 8, 8]} />
                            <meshBasicMaterial color="yellow" />
                        </mesh>
                    ))}                    
                </>
            )}
            {validToothIDs.map((toothID) => (
                (showMode === 0 && toothID < 30) ||
                (showMode === 1 && toothID > 30) ||
                (showMode === 2) ||
                (showMode === 3) ||
                (showMode === 4) ||
                (showMode === 5) 
                ? (
                    <Tooth
                        key={toothID}
                        toothID={toothID}
                        onTransform={handleToothTransformControl}
                        trackballControlsRef={trackballControlsRef}
                        setControlsEnabled={setControlsEnabled}
                        stage={stage}
                        setOrthoData={setOrthoData}
                        stagingData={currentStageData[toothID]}
                        landmarks={landmarksT1 ? landmarksT1[toothID] : {}}
                        url={`/meshes/${toothID}.stl?ts=${props.meshVersion}`}
                        meshVersion={props.meshVersion}
                        baseCaseFilename={props.baseCaseFilename}
                        isClicked={toothID === clickedToothId}
                        onToothClick={handleToothClick}
                        useShortRoots={useShortRoots}
                        showLandmarks={showLandmarks}
                    />
                ) : null
            ))}
        </group>
        <Html fullscreen>
            <div style={{ position: 'absolute', top: 112, left: 14, zIndex: 2000 }}>
                <button
                    onClick={() => resolveCollisionsLocal('maxilla')}
                    style={{ padding: '8px 12px', borderRadius: 6, border: 'none', background: '#1976d2', color: 'white', cursor: 'pointer', boxShadow: '0 2px 6px rgba(0,0,0,0.2)'}}
                >
                    Resolve Upper Jaw
                </button>
            </div>
        </Html>
        </>
    );
});
