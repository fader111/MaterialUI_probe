import React, { useState, useEffect, useMemo, useCallback, forwardRef } from 'react';
import * as THREE from 'three';
import { Tooth } from './Tooth';
// import { Tooth } from './ToothRotated';
import { rt, toVec3, calcLinearStaging, calcMAPSStaging } from "./misc";

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
    // const mandibularOcclusalToJawTransform = orthoData?.mandibularOcclusalToJawTransform || null;
    // const maxillaOcclusalToJawTransform = orthoData?.maxillaOcclusalToJawTransform || null;
    
    const stagesNum = caseStagingData ? caseStagingData.length : 0;
    const stagingDataT1 = caseStagingData && stagesNum > 0 ? caseStagingData[0] : null;
    const stagingDataT2 = caseStagingData && stagesNum > 0 ? caseStagingData[stagesNum - 1] : null;

    // Guard: don't render until all required data is loaded
    if (!caseStagingData || !stagingDataT1 || !stagingDataT2 || !stagingDataT1.RelativeToothTransformsHead['11'].translation) {
        return null;
    }
    // console.log("stagingDataT1.RelativeToothTransformsHead 11 translation:", stagingDataT1.RelativeToothTransformsHead['11'].translation);
    
    const { jsonT1Vec3, jsonT2Vec3, jsonStageVec3 } = useMemo(() => {
        let stageVec3 = {};
        let t1Vec3 = {};
        let t2Vec3 = {};

        if (caseStagingData && stagesNum > 0 && stagingDataT1 && stagingDataT2 && stagingDataT1.RelativeToothTransformsHead && stagingDataT2.RelativeToothTransformsHead) {
            // for (const toothID in stagingDataT2.RelativeToothTransforms) {
            for (const toothID in stagingDataT2.RelativeToothTransformsHead) {
                const toothRt = rt(caseStagingData[stage]?.RelativeToothTransformsHead?.[toothID]);
                // const toothRt = rt(caseStagingData[stage]?.RelativeToothTransforms?.[toothID]);
                const toothRtT1 = rt(stagingDataT1.RelativeToothTransformsHead[toothID]);
                const toothRtT2 = rt(stagingDataT2.RelativeToothTransformsHead[toothID]);
                // const jawTranstation = toothID > 30 ? mandibulaRt.translation : maxillaRt.translation;
                // const jawRotation = toothID > 30 ? mandibulaRt.quaternion : maxillaRt.quaternion;

                stageVec3[toothID] = {
                    position: toothRt.translation,
                        // .clone()
                        // .applyQuaternion(jawRotation)
                        // .add(jawTranstation),
                    // quaternion: jawRotation
                        // .clone()
                        // .multiply(toothRt.quaternion)
                    quaternion : toothRt.quaternion
                };
                t1Vec3[toothID] = {
                    position: toothRtT1.translation,
                        // .clone()
                        // .applyQuaternion(jawRotation)
                        // .add(jawTranstation),
                    // quaternion: jawRotation
                        // .clone()
                        // .multiply(toothRtT1.quaternion)
                    quaternion : toothRtT1.quaternion
                };
                t2Vec3[toothID] = {
                    position: toothRtT2.translation,
                        // .clone()
                        // .applyQuaternion(jawRotation)
                        // .add(jawTranstation),
                    // quaternion: jawRotation
                        // .clone()
                        // .multiply(toothRtT2.quaternion)
                    quaternion : toothRtT2.quaternion
                };
            }
        }
        return { jsonT1Vec3: t1Vec3, jsonT2Vec3: t2Vec3, jsonStageVec3: stageVec3 };
    // }, [stagingDataT1, stagingDataT2, stage, mandibulaRt, maxillaRt]); // Removed orthoData to avoid redundant recalculations
    }, [caseStagingData, stagingDataT1, stage]);

    // Update landmarks
    // useEffect(() => {
    //     // Reset landmarks when orthoData changes
    //     setLandmarksT1(null);
    // }, [caseKey]); // Changed dependency to caseKey for better granularity

    useEffect(() => {
        if (caseStagingData && stagesNum > 0 ) {
            const landmarksT1_ = {};
            for (const toothID in stagingDataT1.RelativeToothTransformsHead) {
                // const toothRt0 = rt(stagingDataT1.RelativeToothTransformsHead[toothID]);
                // const jawTranstation1 = toothID > 30 ? mandibulaRt.translation : maxillaRt.translation;
                // const jawRotation = toothID > 30 ? mandibulaRt.quaternion : maxillaRt.quaternion;
                // const position0 = toothRt0.translation.add(jawTranstation1);
                // const quaternion0 = toothRt0.quaternion.multiply(jawRotation);
                let lmTypes = {};
                for (const lmType in (stagingDataT1.Landmarks[toothID] || {})) {
                    const lmPoint = toVec3(stagingDataT1.Landmarks[toothID][lmType])
                    lmTypes[lmType] = lmPoint;
                }
                landmarksT1_[toothID] = lmTypes;
            }
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

    // Add state to trigger rerender with new T2 prediction
    // const [rerenderStageData, setRerenderStagingData] = useState(null);

    const handleToothTransform = useCallback((toothId, transforms) => {
        if (orthoData?.Staging && orthoData.Staging[stage]) {
            // const jawTranslation = toothId > 30 ? mandibulaRt.translation : maxillaRt.translation;
            // const jawRotation = toothId > 30 ? mandibulaRt.quaternion : maxillaRt.quaternion;
            const localTranslation = transforms.translation;
            const localRotation = new THREE.Quaternion(
                transforms.rotation.x,
                transforms.rotation.y,
                transforms.rotation.z,
                transforms.rotation.w
            );
            const localTransforms = {
                translation: localTranslation,
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
                        onTransform={handleToothTransform}
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
