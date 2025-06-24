import React, { useState, useEffect, useMemo, useCallback, useImperativeHandle, forwardRef } from 'react';
import * as THREE from 'three';
import { Tooth } from './Tooth';
// import { Tooth } from './ToothRotated';
import { rt, toVec3, calcLinearStaging, calcMAPSStaging } from "./misc";

export const ToothPlacement = forwardRef((props, ref) => {
    const {
        trackballControlsRef, 
        setControlsEnabled, 
        orthoData,
        stage,
        onStagingDataUpdate, // Add this prop
        stagingPatterns, 
        stagingPatternsTrigger, 
        stagingType = "Case", 
        showMode: showModeProp = 2, // allow controlled showMode
        onShowModeChange // callback for parent to control showMode
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

    // const [_stagingPatterns, setStagingPatterns] = useState(stagingPatterns);
    // const updateStagingPatterns = (newPatterns) => {
    //     setStagingPatterns({ ...newPatterns });
    //   };
    // updateStagingPatterns({..._stagingPatterns});

    // console.log("call toothplacement - stagingPatterns:", stagingPatterns)
    // Set default values when jsonStagingData is null or empty
    const jsonMandibularData = orthoData?.mandibularRelativeTransform || null;
    const jsonMaxillaData = orthoData?.maxillaRelativeTransform || null;
    const jsonStagingData = orthoData?.Staging || null;
    const mandibularOcclusalToJawTransform = orthoData?.mandibularOcclusalToJawTransform || null;
    const maxillaOcclusalToJawTransform = orthoData?.maxillaOcclusalToJawTransform || null;
    
    const stagesNum = jsonStagingData ? jsonStagingData.length : 0;
    const stagingDataT1 = jsonStagingData && stagesNum > 0 ? jsonStagingData[0] : null;
    const stagingDataT2 = jsonStagingData && stagesNum > 0 ? jsonStagingData[stagesNum - 1] : null;

    const mandibulaRt = useMemo(() => { // really need to useMemo here?
        return jsonMandibularData ? rt(jsonMandibularData) : { translation: new THREE.Vector3(), quaternion: new THREE.Quaternion() };
    }, [jsonMandibularData]);

    const maxillaRt = useMemo(() => {
        return jsonMaxillaData ? rt(jsonMaxillaData) : { translation: new THREE.Vector3(), quaternion: new THREE.Quaternion() };
    }, [jsonMaxillaData]);
    // }, []);

    // console.log("mandibulaRt", mandibulaRt, "maxillaRt", maxillaRt);
    const { jsonT1Vec3, jsonT2Vec3, jsonStageVec3 } = useMemo(() => {
        // console.log("call usememo 25")
        let stageVec3 = {};
        let t1Vec3 = {};
        let t2Vec3 = {};

        // console.log("from toothplace useMemo", " stagesNum", stagesNum, "stage", stage);
        // console.log("stagingDataT2.RelativeToothTransforms[11]", stagingDataT2.RelativeToothTransforms[11]);
        if (jsonStagingData && stagesNum > 0) {
            // console.log("mandibularOcclusalToJawTransform", mandibularOcclusalToJawTransform); 
            // console.log("maxillaOcclusalToJawTransform", maxillaOcclusalToJawTransform);
            // console.log("mandibulaRt", mandibulaRt);
            // console.log("maxillaRt", maxillaRt);
            for (const toothID in stagingDataT2.RelativeToothTransforms) {
                const toothRt = rt(jsonStagingData[stage].RelativeToothTransforms[toothID]);
                const toothRtT1 = rt(stagingDataT1.RelativeToothTransforms[toothID]);
                const toothRtT2 = rt(stagingDataT2.RelativeToothTransforms[toothID]);
                // console.log(toothID, "toothRtT2", toothRtT2);
                const jawTranstation = toothID > 30 ? mandibulaRt.translation : maxillaRt.translation;
                const jawRotation = toothID > 30 ? mandibulaRt.quaternion : maxillaRt.quaternion;
                const occlusalTransform = toothID > 30 ? rt(mandibularOcclusalToJawTransform) : rt(maxillaOcclusalToJawTransform);
                
                stageVec3[toothID] = {
                    // position: toothRt.translation,
                    position: toothRt.translation.add(jawTranstation),//.sub(occlusalTransform.translation),
                    // position: toothRt.translation,//.sub(occlusalTransform.translation),
                    // quaternion: toothRt.quaternion//.multiply(jawRotation)
                    quaternion: toothRt.quaternion//.premultiply(occlusalTransform.quaternion.clone().invert())
                    // quaternion: occlusalTransform.quaternion.clone().invert().multiply(toothRt.quaternion)
                };
                t1Vec3[toothID] = {
                    // position: toothRtT1.translation.add(occlusalTransform.translation),
                    // position: toothRtT1.translation,//.add(jawTranstation),
                    position: toothRtT1.translation,//.sub(occlusalTransform.translation),
                    // quaternion: toothRtT1.quaternion.multiply(occlusalTransform.quaternion)
                    quaternion: toothRtT1.quaternion//.multiply(jawRotation)
                };
                t2Vec3[toothID] = {
                    // position: toothRtT2.translation,//.add(jawTranstation),
                    position: toothRtT2.translation,//.sub(occlusalTransform.translation),
                    // position: toothRtT2.translation.add(occlusalTransform.translation),
                    quaternion: toothRtT2.quaternion//.multiply(jawRotation)
                    // quaternion: toothRtT2.quaternion.multiply(occlusalTransform.quaternion)
                };
            }
        }
        return { jsonT1Vec3: t1Vec3, jsonT2Vec3: t2Vec3, jsonStageVec3: stageVec3 };
    // }, [jsonStagingData, stage, mandibulaRt, maxillaRt, stagesNum, stagingDataT1, stagingDataT2]);
    }, [orthoData, stagingDataT1, stagingDataT2, stage]);
    
    // Update landmarks
    useEffect(() => {
        if (jsonStagingData && stagesNum > 0) {
            // console.log("TP use effect 55")
            const landmarksT1_ = {};
            for (const toothID in stagingDataT1.RelativeToothTransforms) {
                const toothRt0 = rt(stagingDataT1.RelativeToothTransforms[toothID]);
                const jawTranstation1 = toothID > 30 ? mandibulaRt.translation : maxillaRt.translation;
                const jawRotation = toothID > 30 ? mandibulaRt.quaternion : maxillaRt.quaternion;
                const position0 = toothRt0.translation.add(jawTranstation1);
                const quaternion0 = toothRt0.quaternion.multiply(jawRotation);
                let lmTypes = {};
                for (const lmType in stagingDataT1.Landmarks[toothID]) {
                    const lmPoint = toVec3(stagingDataT1.Landmarks[toothID][lmType])
                        //.sub(position0) //old version used before generating ortho json data locally
                        //.applyQuaternion(quaternion0.clone().invert());
                    lmTypes[lmType] = lmPoint;
                }
                landmarksT1_[toothID] = lmTypes;
            }
            setLandmarksT1(landmarksT1_);
        }
    }, [jsonStagingData, stagingDataT1, mandibulaRt, maxillaRt, stagesNum]);

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

    const stagingData = stagingDataSelector[stagingType] || {};

    const handleToothTransform = useCallback((toothId, transforms) => {
        if (jsonStagingData && jsonStagingData[stage]) {
            const jawTranslation = toothId > 30 ? mandibulaRt.translation : maxillaRt.translation;
            const jawRotation = toothId > 30 ? mandibulaRt.quaternion : maxillaRt.quaternion;

            // Convert from global back to local space
            const localTranslation = transforms.translation//.clone().sub(jawTranslation);
            // Create Quaternion from rotation object
            const localRotation = new THREE.Quaternion(
                transforms.rotation.x,
                transforms.rotation.y,
                transforms.rotation.z,
                transforms.rotation.w
            );
            // localRotation.premultiply(jawRotation.clone().invert());

            const localTransforms = {
                translation: localTranslation,
                rotation: {
                    x: localRotation.x,
                    y: localRotation.y,
                    z: localRotation.z,
                    w: localRotation.w
                }
            };

            const updatedStagingData = [...jsonStagingData];
            updatedStagingData[stage] = {
                ...updatedStagingData[stage],
                RelativeToothTransforms: {
                    ...updatedStagingData[stage].RelativeToothTransforms,
                    [toothId]: {
                        ...updatedStagingData[stage].RelativeToothTransforms[toothId],
                        translation: localTransforms.translation,
                        rotation: localTransforms.rotation
                    }
                }
            };
            console.log("updatedStagingData", stage, updatedStagingData[stage])
            onStagingDataUpdate(updatedStagingData);
        }
    }, [jsonStagingData, stage, onStagingDataUpdate, mandibulaRt, maxillaRt]);

    // Handler to call AI prediction endpoint (moved from Overlay)
    const handlePredictT2 = async (setLoadingPrediction, setPredictionError) => {
        console.log('handlePredictT2 called in ToothPlacement');
        setLoadingPrediction(true);
        setPredictionError(null);
        try {
            const base_case_id = orthoData?.base_case_id || '00000000';
            const template_case_id = orthoData?.template_case_id || '00000000';
            console.log('Sending fetch to /predict-t2/', { base_case_id, template_case_id });
            const response = await fetch('http://localhost:8000/predict-t2/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ base_case_id, template_case_id })
            });
            console.log('Fetch response:', response);
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            const data = await response.json();
            console.log('Prediction data received:', data);
            if (jsonStagingData && jsonStagingData.length > 0) {
                const updatedStagingData = [...jsonStagingData];
                const t2StageIdx = updatedStagingData.length - 1;
                const t2Stage = { ...updatedStagingData[t2StageIdx] };
                const toothIDs = Object.keys(t2Stage.RelativeToothTransforms);
                const newTransforms = {};
                toothIDs.forEach((toothID, idx) => {
                    newTransforms[toothID] = data.prediction[idx];
                });
                t2Stage.RelativeToothTransforms = newTransforms;
                updatedStagingData[t2StageIdx] = t2Stage;
                console.log('Updated staging data after prediction:', updatedStagingData);
                if (onStagingDataUpdate) onStagingDataUpdate(updatedStagingData);
            }
        } catch (err) {
            console.error('Error in handlePredictT2:', err);
            setPredictionError(err.message);
        } finally {
            setLoadingPrediction(false);
        }
    };
    useImperativeHandle(ref, () => ({ handlePredictT2 }));

    // Local state for showMode if not controlled
    const [showMode, setShowMode] = useState(showModeProp);
    // Sync with parent if controlled
    React.useEffect(() => { setShowMode(showModeProp); }, [showModeProp]);

    // Only R3F objects inside group
    return (
        <group onClick={handleCanvasClick}>
            {Object.keys(stagingData).map((toothID) => (
                (showMode === 0 && toothID < 30) ||
                (showMode === 1 && toothID > 30) ||
                (showMode === 2) ||
                // (showMode === 3 && toothID == 41) 
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
                        stagingData={stagingData[toothID]}
                        landmarks={landmarksT1 ? landmarksT1[toothID] : {}}
                        url={`/meshes/${toothID}.stl?ts=${props.meshVersion}`}
                        meshVersion={props.meshVersion}
                        isClicked={toothID === clickedToothId}
                        onToothClick={handleToothClick}
                    />
                ) : null
            ))}
        </group>
    );
});
