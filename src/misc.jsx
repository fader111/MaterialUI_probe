// import { Stage } from '@react-three/drei';
import * as THREE from 'three'
// import { useThree } from '@react-three/fiber'
// import { AsciiEffect } from 'three/examples/jsm/Addons.js';

// Rigid Transforms as a Translation + Quaternion
export function rt(obj_) {
  const obj_trans = new THREE.Vector3(
    parseFloat(obj_.translation.x),
    parseFloat(obj_.translation.y),
    parseFloat(obj_.translation.z)
  );
  const obj_quat = new THREE.Quaternion(
    parseFloat(obj_.rotation.x),
    parseFloat(obj_.rotation.y),
    parseFloat(obj_.rotation.z),
    parseFloat(obj_.rotation.w)
  );
  return { translation: obj_trans, quaternion: obj_quat }
};

export function toVec3(obj_) {
  return new THREE.Vector3(parseFloat(obj_.x), parseFloat(obj_.y), parseFloat(obj_.z));
}

/**
 * naiv implementation of staging
 * Take T1 and T2 
 * return linear staging between them
 */
export function calcLinearStaging(jsonT1Vec3, jsonT2Vec3, stagesNum) {
  const outputStaging = {}

  // calculate tooth position for each stage
  for (let stage = 0; stage < stagesNum; stage++) {
    let stageData = {};
    for (const toothID in jsonT2Vec3) {
      const t1Position = jsonT1Vec3[toothID].position.clone();
      const t2Position = jsonT2Vec3[toothID].position.clone();
      const t1Quaternion = jsonT1Vec3[toothID].quaternion.clone();
      const t2Quaternion = jsonT2Vec3[toothID].quaternion.clone();

      // Interpolate position and quaternion
      const t = stage / (stagesNum - 1);
      let position = t1Position.lerp(t2Position, t);
      let quaternion = t1Quaternion.slerp(t2Quaternion, t);

      stageData[toothID] = {
        position: position,
        quaternion: quaternion,
      };
    }
    outputStaging[stage] = stageData;
  }
  return outputStaging;
}

const range = (start, end) => {
  if (typeof start !== 'number' || typeof end !== 'number')
    throw new Error('Both start and end must be numbers')
  if (start > end)
    return []
  return Array.from({ length: end - start + 1 }, (_, i) => start + i);
};

const getQuadrant = (toothID) => { return Math.floor(Number(toothID) / 10) }

/**
* Helper functions to apply Expand. If currStage is between start and endStage, 
* scalar factor adds. Then the vector wich gives after this shift 
* shall be used as start vector for lerp. but the factor must differs 
* position - Vec3 for each teeth.
*/
const applyExpand = (
  position,
  factor,
  currStage,
  toothID,
  teethAssignedIDs = ["34", "35", "36", "44", "45", "46", "14", "15", "16", "24", "25", "26"],
  startStage = 0,
  endStage = 30
) => {
  let expandDir = new THREE.Vector3(0, 0, 0);
  // console.log("currStage", currStage)

  const quadrant = getQuadrant(toothID)
  const expandVal = 5 // Value defines the total expand
  // console.log(toothID, quadrant, expandDir, typeof toothID)
  if (teethAssignedIDs.includes(toothID) && currStage > startStage && currStage < endStage) {
    const xMovement = (quadrant == 3 || quadrant == 2) ? expandVal : (quadrant == 1 || quadrant == 4) ? -expandVal : 0;
    expandDir.add(new THREE.Vector3(xMovement, 0, 0));
  }
  // console.log(currStage, factor)
  return position.clone().add(expandDir.multiplyScalar(factor));
};

/**
 * Apply Procline pattern
 * rotate anteriors around MRAPoint
 * 
 */
const applyProcline = (
  position,
  rotation,
  stage0ToothPosition,
  stage0ToothQuaternion,
  landmarks,
  factor,
  currStage,
  toothID,
  teethAssignedIDs = ["33", "32", "31", "41", "42", "43", "13", "12", "11", "21", "22", "23"],
  startStage = 0,
  endStage = 32
) => {
  if (teethAssignedIDs.includes(toothID) && currStage > startStage && currStage < endStage) {
    const localPivotPoint = landmarks[toothID]["MRAPoint"].clone();

    // Transform the local pivot point to world coordinates
    const worldPivotPoint = localPivotPoint.clone().applyQuaternion(rotation.clone()).add(position.clone());
    // let returnVector = new THREE.Vector3(-5.495, -17.591, 19.052); //worldPivotPoint toth11 in stage1
    const returnVector = localPivotPoint.clone().applyQuaternion(stage0ToothQuaternion.clone()).add(stage0ToothPosition.clone());
    // console.log("worldPivotPoint:", toothID, currStage, worldPivotPoint);
    // console.log("stage0ToothPosition:", toothID, currStage, returnVector);

    // Translate the mesh to have the pivot point at the origin
    const translatedPosition = position.clone().sub(worldPivotPoint.clone());
    const quadrant = getQuadrant(toothID)
    // console.log(toothID, quadrant)
    const yVector = new THREE.Vector3(0, (quadrant == 2 || quadrant == 4) ? -1 : 1, 0); // разбежка по y - менять всё на движ от арки
    // Create rotation quaternion
    const rotationQuaternionX = new THREE.Quaternion()
      .setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2000);
    const rotationQuaternionY = new THREE.Quaternion()
      .setFromAxisAngle(yVector, Math.PI / 8000);

    // Apply the rotation to the translated position
    const rotatedPosition = translatedPosition.clone()
      .applyQuaternion(rotationQuaternionX.clone())
      .applyQuaternion(rotationQuaternionY.clone());

    // Translate position back from the pivot point allways to initial vector
    const finalPosition = rotatedPosition.clone().add(returnVector.clone());

    // Combine existing quaternion with the rotation
    const updatedQuaternion = rotation.clone()
      .multiply(rotationQuaternionX.clone())
      .multiply(rotationQuaternionY.clone());

    return {
      position: finalPosition,
      quaternion: updatedQuaternion
    };
  }

  return {
    position: position.clone(),
    quaternion: rotation.clone()
  };
};

const applyDistalize = (
  position,
  quaternion,
  t2Poistion,
  t2Quaternion,
  factor,
  currStage,
  toothID,
  teethAssignedIDs = ["17", "16", "15", "14", "24", "25", "26", "27"],
  startStage = 0,
  endStage = 54
) => {
  // let distalizeDir = new THREE.Vector3(0, 0, 0);
  if (teethAssignedIDs.includes(toothID) && currStage > startStage && currStage < endStage) {
    const lerpPosition = position.clone().lerp(t2Poistion.clone(), factor);
    const lerpQuaternion = quaternion.clone().slerp(t2Quaternion.clone(), factor)
    const toothSecondNumber = toothID - getQuadrant(toothID) * 10

    let toothSpeed;
    let toothStartStage;

    if (toothSecondNumber == 7) { toothSpeed = factor * 5, toothStartStage = startStage }
    else if (toothSecondNumber == 6) { toothSpeed = factor * 4, toothStartStage = startStage + 4 }
    else if (toothSecondNumber == 5) { toothSpeed = factor * 4, toothStartStage = startStage + 8 }
    else if (toothSecondNumber == 4) { toothSpeed = factor * 4, toothStartStage = 27 } // хардкор переделывать 
    else { toothSpeed = factor, toothStartStage = startStage }

    // для зуба задействованного в паттерне и который надо двигать
    if (currStage > toothStartStage) {
      const position_ = position.clone().lerp(t2Poistion.clone(), toothSpeed)
      const quaternion_ = quaternion.clone().slerp(t2Quaternion.clone(), toothSpeed)
      return { position: position_, quaternion: quaternion_ }
    }

    // для зуба задействованного в паттерне и который либо пока стоит, либо уже приехал
    const positionHold = position.clone().sub(lerpPosition.clone().sub(position.clone()))
    const quaternionHold = quaternion.clone().slerp(lerpQuaternion.clone(), toothSpeed)
    return { position: positionHold, quaternion: quaternionHold }  // зуб стоит на месте ждет своей очереди
  }
  // для зуба не задействованного в паттерне
  return { position: position.clone(), quaternion: quaternion.clone() }
};

export function calcMAPSStaging(jsonT1Vec3, jsonT2Vec3, stagesNum, patterns, landmarks) {
  const outputStaging = {};
  // console.log("calcMAPSStaging call")

  // Initialize the positions and quaternions for each tooth
  let stage0Position = {};
  let stage0Quaternion = {};
  let prevStagePositions = {};
  let prevStageQuaternions = {};

  for (const toothID in jsonT1Vec3) {
    stage0Position[toothID] = jsonT1Vec3[toothID].position.clone();
    stage0Quaternion[toothID] = jsonT1Vec3[toothID].quaternion.clone();
    prevStagePositions[toothID] = jsonT1Vec3[toothID].position.clone();
    prevStageQuaternions[toothID] = jsonT1Vec3[toothID].quaternion.clone();
  }

  // Compute the position for each stage
  for (let stage = 0; stage < stagesNum; stage++) {
    let stageData = {};

    for (const toothID in jsonT2Vec3) {
      const t2Position = jsonT2Vec3[toothID].position.clone();
      const t2Quaternion = jsonT2Vec3[toothID].quaternion.clone();

      // Interpolate position and quaternion from the previous stage to the target
      let stageToothPosition;
      let stageToothQuaternion;
      let t = 0
      if (stage === 0) {
        stageToothPosition = prevStagePositions[toothID].clone();
        stageToothQuaternion = prevStageQuaternions[toothID].clone();
      } else {
        // Calculate the interpolation factor based on the current stage
        t = 1 / (stagesNum - stage); // Decrease the step incrementally

        stageToothPosition = prevStagePositions[toothID].lerp(t2Position, t);
        stageToothQuaternion = prevStageQuaternions[toothID].slerp(t2Quaternion, t);

        // Apply patterns 
        for (let pattern in patterns) {
          if (patterns[pattern] === "Expand") {
            // console.log(pattern)
            stageToothPosition = applyExpand(stageToothPosition.clone(), t, stage, toothID);
          }
          if (patterns[pattern] === "Procline") {
            ({ position: stageToothPosition, quaternion: stageToothQuaternion } =
              applyProcline(stageToothPosition.clone(),
                stageToothQuaternion.clone(),
                stage0Position[toothID].clone(),
                stage0Quaternion[toothID].clone(),
                landmarks,
                t, stage, toothID))
          }
          if (patterns[pattern] === "Distalize") {
            ({ position: stageToothPosition, quaternion: stageToothQuaternion } =
              applyDistalize(stageToothPosition.clone(),
                stageToothQuaternion.clone(),
                t2Position.clone(),
                t2Quaternion.clone(),
                t, stage, toothID))
          }
          // Add other pattern conditions here
        }
      }

      stageData[toothID] = {
        position: stageToothPosition.clone(),
        quaternion: stageToothQuaternion.clone()
      };

      // Update previous stage position and quaternion for the next stage
      prevStagePositions[toothID] = stageToothPosition.clone();
      prevStageQuaternions[toothID] = stageToothQuaternion.clone();
      // if (toothID == "43") console.log(stage, prevStageQuaternions[toothID]); 
      // if (toothID == "43") console.log(stage, prevStagePositions[toothID]); 
    }
    outputStaging[stage] = stageData;
  }
  return outputStaging;
}
