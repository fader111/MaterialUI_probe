import os, sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'Cython'))
# from ormco import *
from autosetup_ml.utils import *
# import multiprocessing

parser = argparse.ArgumentParser('export_teeth')
parser.add_argument('--filename', default='server/00000000.oas', help='Path to .oas file')
# parser.add_argument('--filename', default='server/Asuncion_Domingo_Nicolau_909346_P.oas', help='Path to .oas file')
args = vars(parser.parse_args())
if not os.path.isfile(args['filename']): parser.exit(1, 'File not found')

print(f"oas file name {args['filename']}")
doc = PyOrthoDoc()
doc.OpenCase(args['filename'])
case = doc.getCase()
tp = doc.GetActiveTreatmentPlan()
caseID = case.GetCaseID()

mandibularOcclussalCS = tp.getJawOcclusalCS(JawType.Mandible)
maxillaOcclussalCS = tp.getJawOcclusalCS(JawType.Maxilla)
t2 = max([tp.GetJaw(jaw).GetT2() for jaw in JawType])

# exportDir = os.path.join(os.path.dirname(args['filename']), "teeth")
exportDirMeshes = "public/meshes"
exportDirShortRoots = "public/shortRoots"
exportDirLongRoots = "public/roots"
exportDirCrowns = "public/crowns"
# os.makedirs(exportDirMeshes, exist_ok=True)
os.makedirs(exportDirShortRoots, exist_ok=True)
os.makedirs(exportDirLongRoots, exist_ok=True)
os.makedirs(exportDirCrowns, exist_ok=True)

def toLH(obj):
    if isinstance(obj, PyVector):
        return (obj[0], obj[1], -obj[2])
    elif isinstance(obj, PyQuaternion):
        return (obj.re, -obj.im[0], -obj.im[1], obj.im[2])

def toRH(obj):
    if isinstance(obj, PyVector):
        return (obj[0], obj[1], -obj[2])
    elif isinstance(obj, PyQuaternion):
        return (obj.im[1], obj.im[2], obj.re, obj.im[0] )

def getJawRelativeTransform(jawType):
    jaw = tp.GetJaw(jawType)
    rt = jaw.relativeTransform(0)
    translation = rt.translation
    rotation = rt.rotation
    return {
        "translation": {"x": translation.x, "y": translation.y, "z": translation.z},
        "rotation": {"x": rotation.im.x, "y": rotation.im.y, "z": rotation.im.z, "w": rotation.re}
    }

def getOcclusalToJawTransform(jawType):
    occlusalToJawTransform = tp.getJawOcclusalCS(jawType)
    if jawType == JawType.Mandible:
        translation = toLH(occlusalToJawTransform.translation)
        rotation = toLH(occlusalToJawTransform.rotation)
    else:
        translation = toRH(occlusalToJawTransform.translation)
        rotation = toRH(occlusalToJawTransform.rotation)
    # print(f"occlussalCS {occlussalCS}")
    # print(f"occlusalToJawTransform {occlusalToJawTransform}")
    # print(f"rotation {rotation}")
    return {
        "translation": {"x": translation[0], "y": translation[1], "z": translation[2]},
        "rotation": {"x": rotation[0], "y": rotation[1], "z": rotation[2], "w": rotation[3]}
    }

def getPoints(point):
    return {
        "x": point.x,
        "y": point.y,
        "z": point.z
    }

def getToothRelativeTransform(tooth, stage):
    translation = tooth.relativeTransform(stage).translation
    rotation = tooth.relativeTransform(stage).rotation
    return {
        "translation": {
            "x": translation.x, "y": translation.y, "z": translation.z},
        "rotation": {
            "x": rotation.im.x, "y": rotation.im.y, "z": rotation.im.z,
            "w": rotation.re}
    }

def getStagingData():
    staging_data = []
    for stage_number, stage in enumerate(range(0, t2)):
        # из стейджинга понадобятся MandibularTransform, MaxillaryTransform, RelativeToothTransforms, Landmarks
        stage_data = {}
        stage_data["Stage"] = stage_number
        stage_data["MandibularTransform"] = getJawRelativeTransform(JawType.Mandible)
        stage_data["MaxillaryTransform"] = getJawRelativeTransform(JawType.Maxilla)

        # collect RelativeToothTransforms
        relativeToothTransforms = {}
        for jawType in JawType:
            jaw = tp.GetJaw(jawType)
            for tooth in jaw.getTeeth():
                relativeToothTransforms[str(
                    tooth.getClinicalID())] = getToothRelativeTransform(tooth, stage)
        stage_data["RelativeToothTransforms"] = relativeToothTransforms

        # collect Landmarks
        lm_dict = {"BCPoint": LandmarkID.BCPoint,
                   "FEGJPoint": LandmarkID.FEGJPoint,
                   "MRAPoint": LandmarkID.MeanRootApex,
                   #   "MDWLine":LandmarkID.MDWLine, # TODO fix
                   }

        landmarks = {}
        for jawType in JawType:
            jaw = tp.GetJaw(jawType)
            for tooth in jaw.getTeeth():
                tooth_landmarks = {}
                for lm in lm_dict:
                    tooth_landmarks[lm] = getPoints(
                        tooth.getLandmarks().getPoint(lm_dict[lm]))
                landmarks[str(tooth.getClinicalID())] = tooth_landmarks

        stage_data["Landmarks"] = landmarks

        staging_data.append(stage_data)

    return staging_data

def getOcclusalCSdata(occlussalCS):
    translation = occlussalCS.translation
    rotation = occlussalCS.rotation
    return {
        "translation": {"x": translation.x, "y": translation.y, "z": translation.z},
        "rotation": {"x": rotation.im.x, "y": rotation.im.y, "z": rotation.im.z, "w": rotation.re}
    }

def getTeethBoltonData():
    tooth_data = {}
    for tooth in tp.getTeeth():
        tooth_info = {
            # "toothClinicalID": tooth.getClinicalID(),
            "toothRelativeTransform": getToothRelativeTransform(tooth, t2)
        }
        if tooth.getBoltonLine():
            tooth_info["toothBoltonStartPoint"] = getPoints(
                tooth.getBoltonLine().startPoint)
            tooth_info["toothBoltonEndPoint"] = getPoints(
                tooth.getBoltonLine().endPoint)

        tooth_data[tooth.getClinicalID()] = tooth_info

    return tooth_data

# Create final ortho_data structure
ortho_data = {
    "CaseID": str(caseID),
    "T2Stage": str(t2),
    "mandibularRelativeTransform": getJawRelativeTransform(JawType.Mandible),
    "maxillaRelativeTransform": getJawRelativeTransform(JawType.Maxilla),
    "mandibularOcclusalToJawTransform": getOcclusalToJawTransform(JawType.Mandible),
    "maxillaOcclusalToJawTransform": getOcclusalToJawTransform(JawType.Maxilla),
    "mandibularOcclussalCS": getOcclusalCSdata(mandibularOcclussalCS),
    "maxillaOcclussalCS": getOcclusalCSdata(maxillaOcclussalCS),
    "Bolton": getTeethBoltonData(),
    "Staging": getStagingData()
}

def export_surface_type(exportDir, method_name):
    """for multiprocessing not implemented because of mp restrictions """
    # Re-imports and re-initialization for subprocess safety
    from autosetup_ml.utils import exportMesh
    from ormco import JawType
    doc = PyOrthoDoc()
    doc.OpenCase(args['filename'])
    tp = doc.GetActiveTreatmentPlan()
    for jawType in JawType:
        jaw = tp.GetJaw(jawType)
        for tooth in jaw.getTeeth():
            clID = str(tooth.getClinicalID())
            mesh_data = getattr(tooth, method_name)()
            exportMesh(os.path.join(exportDir, clID + '.stl'), mesh_data)

def export_json_data(export_file, ortho_data):
    """Export ortho_data to a JSON file."""
    with open(export_file, "w") as json_file:
        json.dump(ortho_data, json_file, indent=4)
    # print(f"Ortho data exported to {export_file}")

if __name__ == "__main__":
    from autosetup_ml.utils import exportMesh
    from ormco import JawType

    # Clean export directories before exporting new files
    for folder in [exportDirMeshes, exportDirShortRoots, exportDirLongRoots, exportDirCrowns]:
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Save to JSON file
    with open(f"public/orthoData.json", "w") as json_file:
        json.dump(ortho_data, json_file, indent=4)

    for jawType in JawType:
        jaw = tp.GetJaw(jawType)
        for tooth in jaw.getTeeth():
            clID = str(tooth.getClinicalID())
            # exportMesh(os.path.join(exportDirMeshes, clID + '.stl'), tooth.getToothSurface()) old one
            exportMesh(os.path.join(exportDirShortRoots, clID + '.stl'), tooth.getShortRootData())
            exportMesh(os.path.join(exportDirLongRoots, clID + '.stl'), tooth.getRootData())
            exportMesh(os.path.join(exportDirCrowns, clID + '.stl'), tooth.getCrownData())

    # json_proc = multiprocessing.Process(target=export_json_data, args=("public/orthoData.json", ortho_data))
    # json_proc.start()
    # json_proc.join()

    # jobs = []
    # for exportDir, method_name in surface_types:
    #     p = multiprocessing.Process(target=export_surface_type, args=(exportDir, method_name))
    #     jobs.append(p)
    #     p.start()

    # for p in jobs:
    #     p.join()