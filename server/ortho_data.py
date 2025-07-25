import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Any
from backend.ormco import JawType, LandmarkID

def getToothRelativeTransform(tooth, stage):
    # rel_transform = tooth.relativeTransform(stage)
    # if tooth is not None and hasattr(tooth, "relativeTransform"):
    try:
        translation = tooth.relativeTransform(stage).translation
        rotation = tooth.relativeTransform(stage).rotation
        return {
            "translation": {
                "x": translation.x, "y": translation.y, "z": translation.z},
            "rotation": {
                "x": rotation.im.x, "y": rotation.im.y, "z": rotation.im.z,
                "w": rotation.re}
        }
    except:
        print(f"toothID {tooth} has no relativeTransform for stage {stage}")
        print(f"tooth has attr relativeTransform {hasattr(tooth, 'relativeTransform')}")
        return {
            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        }

def getToothRelativeTransformHead(tooth, stage, mandibular_rt, maxillary_rt) -> Dict[str, Dict[str, Any]]:
    tooth_id = tooth.getClinicalID()
    # jaw_rt = mandibular_rt if isLower(tooth_id) else maxillary_rt # TODO рефакторить используя isLower 
    jaw_rt = mandibular_rt if tooth_id > 30 else maxillary_rt # TODO рефакторить используя isLower 
    tooth_rt = getToothRelativeTransform(tooth, stage)
    t = tooth_rt["translation"]
    q = tooth_rt["rotation"]
    jt = jaw_rt["translation"]
    jq = jaw_rt["rotation"]
    t_vec = np.array([t["x"], t["y"], t["z"]])
    q_quat = [q["x"], q["y"], q["z"], q["w"]]
    jt_vec = np.array([jt["x"], jt["y"], jt["z"]])
    jq_quat = [jq["x"], jq["y"], jq["z"], jq["w"]]
    t_rot = R.from_quat(jq_quat).apply(t_vec)
    final_translation = t_rot + jt_vec
    final_quat = R.from_quat(jq_quat) * R.from_quat(q_quat)
    final_quat_xyzw = final_quat.as_quat()
    return {
        "translation": {
            "x": float(final_translation[0]),
            "y": float(final_translation[1]),
            "z": float(final_translation[2])
        },
        "rotation": {
            "x": float(final_quat_xyzw[0]),
            "y": float(final_quat_xyzw[1]),
            "z": float(final_quat_xyzw[2]),
            "w": float(final_quat_xyzw[3])
        }
    }

class OrthoData:
    def __init__(self, ortho_case):
        self.ortho_case = ortho_case
        self.caseID = self.ortho_case.caseID
        self.tp = self.ortho_case.get_treatment_plan()
        self.t2 = max([self.tp.GetJaw(jaw).GetT2() for jaw in JawType])
        self.mandibular_rt = self.getJawRelativeTransform(JawType.Mandible)
        self.maxillary_rt = self.getJawRelativeTransform(JawType.Maxilla)
        self.ortho_data = {
            "CaseID": str(self.caseID),
            "T2Stage": str(self.t2),
            "mandibularRelativeTransform": self.mandibular_rt,
            "maxillaRelativeTransform": self.maxillary_rt,
            "Staging": self.getStagingData()
        }

    def getPoints(self, point):
        return {
            "x": point.x,
            "y": point.y,
            "z": point.z
        }

    def getJawRelativeTransform(self, jawType):
        jaw = self.tp.GetJaw(jawType)
        rt = jaw.relativeTransform(0)
        translation = rt.translation
        rotation = rt.rotation
        return {
            "translation": {"x": translation.x, "y": translation.y, "z": translation.z},
            "rotation": {"x": rotation.im.x, "y": rotation.im.y, "z": rotation.im.z, "w": rotation.re}
        }
    
    def getStagingData(self):
        staging_data = []
        for stage_number, stage in enumerate(range(0, self.t2)):
            stage_data = {}
            stage_data["Stage"] = stage_number
            relativeToothTransforms = {}
            relativeToothTransformsHead = {}
            for jawType in JawType:
                jaw = self.tp.GetJaw(jawType)
                for tooth in jaw.getTeeth():
                    tooth_id = tooth.getClinicalID()
                    relativeToothTransforms[str(tooth_id)] = getToothRelativeTransform(tooth, stage)
                    relativeToothTransformsHead[str(tooth_id)] = getToothRelativeTransformHead(tooth, stage, self.mandibular_rt, self.maxillary_rt)
            stage_data["RelativeToothTransformsHead"] = relativeToothTransformsHead
            lm_dict = {"BCPoint": LandmarkID.BCPoint,
                    "FEGJPoint": LandmarkID.FEGJPoint,
                    "MRAPoint": LandmarkID.MeanRootApex}
            landmarks = {}
            for jawType in JawType:
                jaw = self.tp.GetJaw(jawType)
                for tooth in jaw.getTeeth():
                    tooth_landmarks = {}
                    for lm in lm_dict:
                        tooth_landmarks[lm] = self.getPoints(
                            tooth.getLandmarks().getPoint(lm_dict[lm]))
                    landmarks[str(tooth.getClinicalID())] = tooth_landmarks
            stage_data["Landmarks"] = landmarks
            staging_data.append(stage_data)
        return staging_data
