import React, { useState, useEffect, useMemo, useCallback, forwardRef } from 'react';
import * as THREE from 'three';
import { Tooth } from './Tooth';
// import { Tooth } from './ToothRotated';
import { rt, transform, toVec3, calcLinearStaging, calcMAPSStaging } from "./misc";

export const ToothPlacement = forwardRef((props, ref) => {
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
    const caseStagingData = orthoData?.Staging || null;
    const mandibularRT = rt(orthoData?.mandibularRelativeTransform || null);
    const maxillaRT = rt(orthoData?.maxillaRelativeTransform || null);
    // console.log("ToothPlacement rerender, stage:", stage, "stagingType:", stagingType);
    // console.log("ToothPlacement rerender, stage:", stage, "stagingData", caseStagingData);
    // const mandibularOcclusalToJawTransform = orthoData?.mandibularOcclusalToJawTransform || null;
    // const maxillaOcclusalToJawTransform = orthoData?.maxillaOcclusalToJawTransform || null;
    
    const stagesNum = caseStagingData ? caseStagingData.length : 0;
    const stagingDataT1 = caseStagingData && stagesNum > 0 ? caseStagingData[0] : null;
    const stagingDataT2 = caseStagingData && stagesNum > 0 ? caseStagingData[stagesNum - 1] : null;

    const { jsonT1Vec3, jsonT2Vec3, jsonStageVec3 } = useMemo(() => {
        // console.log("call useMemo for jsonVec3");
        let stageVec3 = {};
        let t1Vec3 = {};
        let t2Vec3 = {};

        if (caseStagingData && stagesNum > 0) {
            for (const toothID in stagingDataT2.RelativeToothTransforms) {
                
                // const jawRT = (parseInt(toothID) < 30 ) ? mandibularRT : maxillaRT;
                // const toothRt = transform(jawRT, rt(caseStagingData[stage]?.RelativeToothTransforms?.[toothID]));
                // const toothRtT1 = transform(jawRT, rt(stagingDataT1.RelativeToothTransforms[toothID]));
                // const toothRtT2 = transform(jawRT, rt(stagingDataT2.RelativeToothTransforms[toothID]));

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

    useEffect(() => {
        if (caseStagingData && stagesNum > 0 ) {
            const landmarksT1_ = {};
            for (const toothID in stagingDataT1.RelativeToothTransforms) {
                let lmTypes = {};
                const toothLandmarks = stagingDataT1.Landmarks[toothID] || {};
                for (const lmType in toothLandmarks) {
                    if (lmType.endsWith('Point')) {
                        // Single point landmark
                        lmTypes[lmType] = toVec3(toothLandmarks[lmType]);
                    } else if (lmType.endsWith('Line')) {
                        // Line landmark: store as {start, end}
                        const line = toothLandmarks[lmType];
                        lmTypes[lmType] = {
                            start: toVec3(line.start),
                            end: toVec3(line.end)
                        };
                    }
                }
                landmarksT1_[toothID] = lmTypes;
            }
            // console.log("LandmarksT1_", landmarksT1_["11"]);
            setLandmarksT1(landmarksT1_);
        }
    // }, [caseStagingData, stagingDataT1, mandibulaRt, maxillaRt, stagesNum, orthoData]);
    }, [caseStagingData, stagingDataT1, stagesNum]);

    const linearStagingData = useMemo(() => {
        // console.log("call LinearStagingData from UseMemo");
        return calcLinearStaging(jsonT1Vec3, jsonT2Vec3, stagesNum);
    // }, [jsonT1Vec3, jsonT2Vec3, stagesNum]); // jsonT1Vec3, jsonT2Vec3, меняются что вызывает срабатывание 
    }, [stagesNum]);

    const MAPSStagingData = useMemo(() => {
        // console.log("call MAPSStagingData from UseMemo");
        // console.log("useMemoStagingPatterns", stagingPatterns, stagingPatternsTrigger)
        // if (!landmarksT1) return {};
        // const patterns = { 0: "Expand", 1: "Procline", 2: "Distalize" }; // old one - changed to context stagingPatterns
        return calcMAPSStaging(jsonT1Vec3, jsonT2Vec3, stagesNum, stagingPatterns, landmarksT1);
    // }, [jsonT1Vec3, jsonT2Vec3, stagesNum, landmarksT1]);
    // }, [stagesNum, stagingPatternsTrigger, jsonT1Vec3, stagingPatterns]);
    }, [stagesNum, jsonT1Vec3, stagingPatterns]);
    // }, []);
    
    const stagingDataSelector = useMemo(() => ({
        Case: jsonStageVec3,
        Linear: linearStagingData[stage] || {}, // not in use
        MAPS: MAPSStagingData[stage] || {}
    }), [jsonStageVec3, linearStagingData, MAPSStagingData, stage]);

    let stagingData = stagingDataSelector[stagingType] || {};

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
        console.log("handleToothTransformControl called for toothId:", toothId, "with transforms:", transforms);
        if (orthoData?.Staging && orthoData.Staging[stage]) {
            const localTranslation = transforms.translation;
            const localRotation = new THREE.Quaternion(
                transforms.rotation.x,
                transforms.rotation.y,
                transforms.rotation.z,
                transforms.rotation.w
            );
            // Ensure translation is always a plain object !!! Refactor that!!!!
            const translationObj = (localTranslation instanceof THREE.Vector3)
                ? { x: localTranslation.x, y: localTranslation.y, z: localTranslation.z }
                : localTranslation;
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
    const currentStageData = stagingData;
    // console.log("rerenderStageData", rerenderStageData);
    // console.log("currentStage Data", stage, currentStageData);

    // Filter only valid tooth IDs
    const validToothIDs = Object.keys(currentStageData).filter(toothID => /^\d+$/.test(toothID));
    // console.log("currentStageData", currentStageData);

    return (
        <group onClick={handleCanvasClick}>
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
    );
});
