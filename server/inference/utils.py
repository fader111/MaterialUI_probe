import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# print(f"dirname  {os.path.dirname(__file__)}")
from omegaconf import OmegaConf
from pathlib import Path
import glob
import torch
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import math
import pyvista as pv
# from config import *
from scipy.linalg import svd
from numpy.typing import NDArray
# from backend.ormco import *
from typing import *
from datetime import datetime
import tarfile
import sqlite3
import tempfile

HorizontalRes = 10 # Костыли 
VerticalRes = 4 # Костыли
ZAngleMinDegrees = 0.0
ZAngleMaxDegrees = 85.0
HeightOriginShiftPosterior = 0.3
HeightOriginShiftAnterior = 0.5
WidthOriginShift = 0.25
YShiftAnterior = -0.3 
YShiftPosterior = 0
HeightOriginShiftAbsAnterior = 0.5
HeightOriginShiftAbsPosterior = 0

up_teeth_nums16 = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22,
                    23, 24, 25, 26, 27, 28]  # Jaw_id = 2 верхняя / по 16 зубов
dw_teeth_nums16 = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42,
                    43, 44, 45, 46, 47, 48]  # Jaw_id = 1 нижняя  / 16 зубов
up_teeth_nums14 = [17, 16, 15, 14, 13, 12, 11, 21, 22,
                    23, 24, 25, 26, 27]  
dw_teeth_nums14 = [37, 36, 35, 34, 33, 32, 31, 41, 42,
                    43, 44, 45, 46, 47] 
stub_missing_tooth_landmarks = \
                {
                    "MDWLine": {
                        "start": {"x": "0","y": "0","z": "0"},
                        "end": {"x": "0","y": "0","z": "0"}
                    },
                    "BCPoint": {"x": "0","y": "0","z": "0"},
                    "MRAPoint": {"x": "0","y": "0","z": "0"},
                    "FEGJPoint": {"x": "0","y": "0","z": "0"},
                    "BRLine": {
                        "start": {"x": "0","y": "0","z": "0"},
                        "end": {"x": "0","y": "0","z": "0"}
                    }
                }
stub_missing_tooth_rt = \
                {
                    "translation": {"x": "0","y": "0","z": "0"},
                    "rotation": {"x": "0","y": "0","z": "0","w": "1"}
                }

def register_resolvers() -> None:
    """Register custom OmegaConf resolvers for project configuration"""
    
    def get_project_root() -> str:
        """Get the project root directory"""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    
    # Register multiplication resolver
    OmegaConf.register_new_resolver(
        "multiply",
        # lambda x, y: int(x) * int(y)
        lambda *args: int(eval('*'.join(map(str, args))))
    )
    
    # Register project root resolver
    OmegaConf.register_new_resolver(
        "project_root",
        get_project_root
    )

def get_dataset_fr_file(fname, dataset_dir):
    ds_fname = os.path.join(dataset_dir, fname)
    dataset = torch.load(ds_fname)
    print(f"data Succesfully loaded from {ds_fname}")
    return dataset

def load_json_data(json_path):
    with open(json_path, 'rb') as json_file: # 'rb' for support cirillic symbols from json
        try:
           data = json.load(json_file)
        except:
            return None
    return data

def quaternion(transform):
    return [
    transform["rotation"]["x"],
    transform["rotation"]["y"],
    transform["rotation"]["z"],
    transform["rotation"]["w"],
    ]

def translate(transform):
    return np.array([
    transform["translation"]["x"],
    transform["translation"]["y"],
    transform["translation"]["z"]
    ])

def get_transform_matrix(transform):
    quat = quaternion(transform)
    translation = translate(transform)
    # Преобразуем кватернион в матрицу поворота
    rotation_matrix = R.from_quat(quat).as_matrix() 
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def quaternion_from_ormco(rt):
    return [
        rt.rotation.im.x,
        rt.rotation.im.y,
        rt.rotation.im.z,
        rt.rotation.re
    ]

def translate_from_ormco(rt):
    return np.array([
        rt.translation.x,
        rt.translation.y,
        rt.translation.z
    ])

def rigid_transform_from_ormco(rt):
    """
    Convert an ORMCO rigid transform to a 4x4 transformation matrix.
    
    Args:
        rt (ormco.PyRigidTransform): The ORMCO rigid transform object.

    Returns:
        numpy.ndarray: A 4x4 transformation matrix.
    """
    quat = quaternion_from_ormco(rt)
    translation = translate_from_ormco(rt)
    # Преобразуем кватернион в матрицу поворота
    rotation_matrix = R.from_quat(quat).as_matrix() 
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def get_transform_matrix_from_three_rt(rt)-> NDArray[float]:
    """
    Convert a Three.js-like rigid transform to a 4x4 transformation matrix.
    
    Args:
        rt : The THREE.JS rigid transform object.

    Returns:
        numpy.ndarray: A 4x4 transformation matrix.
    """
    def quaternion_from_three(rt):
        return [
            rt["rotation"]["x"],
            rt["rotation"]["y"],
            rt["rotation"]["z"],
            rt["rotation"]["w"]
        ]
    
    def translate_from_three(rt):
        return np.array([
            rt["translation"]["x"],
            rt["translation"]["y"],
            rt["translation"]["z"]
        ])
    
    # Преобразуем кватернион в матрицу поворота
    rotation_matrix = R.from_quat(quaternion_from_three(rt)).as_matrix() 
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translate_from_three(rt)
    return transform_matrix

def get_transform_matrix_from_ormco(rt) -> NDArray[float]:
    quat = quaternion_from_ormco(rt)
    translation = translate_from_ormco(rt)
    # Преобразуем кватернион в матрицу поворота
    rotation_matrix = R.from_quat(quat).as_matrix() 
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def calc_transform_matrix_fr_points(A, B): # Kabsch for 2 sets of landmarks
    # Ensure the input arrays are numpy arrays
    A = np.array(A)
    B = np.array(B)
    
    # Center the points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    
    # Compute the covariance matrix
    H = AA.T @ BB
    
    # Compute the Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = Vt.T @ U.T
    
    # Handle the reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute the translation
    t = centroid_B - R @ centroid_A
    
    # Create the 4x4 transformation matrix
    tr_matrix = np.eye(4)
    tr_matrix[:3, :3] = R
    tr_matrix[:3, 3] = t
    
    return tr_matrix

def calc_point_grid(tooth_clID, is_posterior, mesh=None, vertices=None, faces=None)->tuple[NDArray[float]]:
    ''' works both with meshes and it's vertices and faces '''
    if not mesh:
        mesh = pv.PolyData(vertices, faces)
    # Create source to ray trace
    origins, dirs = createToothOriginsDirs(mesh, is_posterior)
    # Perform ray trace
    intersections = mesh.multi_ray_trace(origins, dirs)
    return intersections[0]

def degreesToRadians(degrees):
    return degrees * math.pi / 180.0

def createToothOriginsDirs(mesh:pv.PolyData, is_posterior:bool) -> tuple[NDArray[float]]:
    '''делает точки испускания лучей и вектора направления испукания лучей'''
    out = []
    width = mesh.bounds[1] - mesh.bounds[0]
    height = mesh.bounds[5]
    # print(f"width, height {width} {height}")
    heightOriginShift = HeightOriginShiftPosterior if is_posterior else HeightOriginShiftAnterior
    HeightOriginShiftAbs = HeightOriginShiftAbsPosterior if is_posterior else HeightOriginShiftAbsAnterior
    
    for j in range(VerticalRes):
        zAngle = degreesToRadians(
            ZAngleMinDegrees + j * (ZAngleMaxDegrees - ZAngleMinDegrees) / (VerticalRes - 1))
        
        YShift = YShiftPosterior if is_posterior else YShiftAnterior
        z0 = heightOriginShift * height * math.sin(zAngle) + HeightOriginShiftAbs

        for i in range(HorizontalRes):
            xyAngle = 2.0 * math.pi * i / HorizontalRes
            # x0 = WidthOriginShift * width * math.cos(xyAngle) # original version
            # x0 below makes lower x values closer to each other 
            # it needs for anteriors with thin width on zero plate
            x0 = WidthOriginShift * width * math.cos(xyAngle) * (j+1) / VerticalRes
            origin = [x0, YShift, z0]
            # print (f"origin \n{origin}")
            dir = [math.cos(xyAngle) * math.cos(zAngle), math.sin(xyAngle) * math.cos(zAngle), math.sin(zAngle)]
            out.append([origin, dir])
    
    out_np = np.array(out, dtype=np.float32)
    return out_np[:, 0, :], out_np[:, 1, :]

def apply_rigid_transform(point_cloud:NDArray, transform:NDArray)->NDArray:
    """
    Apply a rigid transformation to a point cloud.
    
    Args:
        point_cloud (numpy.ndarray): An (N, 3) array representing the point cloud.
        transform (numpy.ndarray): A (4, 4) rigid transformation matrix.

    Returns:
        numpy.ndarray: Transformed point cloud as an (N, 3) array.
    """
    # Add a column of ones to the point cloud to make it homogeneous (N, 4)
    ones = np.ones((point_cloud.shape[0], 1))
    homogeneous_points = np.hstack((point_cloud, ones))  # Shape: (N, 4)
    
    # Apply the transformation matrix
    transformed_points = homogeneous_points @ transform.T  # Shape: (N, 4)
    
    # Convert back to Cartesian coordinates by dropping the homogeneous component
    return transformed_points[:, :3]

def get_pv_mesh_from_raw_mesh(raw_mesh):
    vertices = raw_mesh.vertices()
    faces = raw_mesh.faces() # array.array

    faces = [i for face in faces for i in face] # get flat faces

    # generate vertices appropriate for creating mesh using faces indices
    complete_vertices = np.array([vertices[i] for i in faces], dtype=np.float32)

    # specific format faces for pyvista
    pv_faces = np.array([[3, i, i+1, i+2] for i in range(0, len(faces), 3)]).flatten()

    return pv.PolyData(complete_vertices, pv_faces)

def calculate_tooth_axis_from_layers(tooth_points, num_layers=4, points_per_layer=10):
    """
    Calculate tooth central axis using layer centroids
    
    Args:
        tooth_points: numpy array of shape [500, 3] containing tooth surface points
        num_layers: number of layers (10)
        points_per_layer: points per layer (50)
    Returns:
        direction_vector: principal axis of the tooth
        centroid: center point of the tooth
    """
    # Reshape points to separate layers
    layers = tooth_points.reshape(num_layers, points_per_layer, 3)
    
    # Calculate centroids for each layer
    layer_centroids = np.mean(layers, axis=1)  # Shape: [10, 3]
    
    # Calculate overall centroid
    tooth_centroid = np.mean(layer_centroids, axis=0)
    
    # Fit line through layer centroids using SVD
    x = layer_centroids - tooth_centroid
    U, S, Vh = svd(x)
    
    # The direction vector is the first right singular vector
    direction_vector = Vh[0]
    
    # Ensure the vector points from root to crown (assuming z increases towards crown)
    if direction_vector[2] < 0:
        direction_vector = -direction_vector
        
    return direction_vector, tooth_centroid

def get_tooth_point_grid(mesh, is_posterior):
    # Create source to ray trace
    origins, dirs = createToothOriginsDirs(mesh, is_posterior)

    # Perform ray trace
    intersections = mesh.multi_ray_trace(origins, dirs)
    return origins, dirs, intersections[0]

def draw_shapes_pv(plt: pv.Plotter, 
                orthoCase, 
                sets_to_plot: List[str], 
                predictions: NDArray, 
                rt_points_t1: NDArray, 
                rt_points_t2: NDArray
                ):
    """Draw the shapes of case"""
    tp = orthoCase.get_treatment_plan()
    t2 = orthoCase.get_T2_stage()
    pred_centroids = []
    
    predictions = predictions.reshape(28, VerticalRes * HorizontalRes, 3)
    rt_points_t1 = rt_points_t1.reshape(28, VerticalRes * HorizontalRes, 3)
    rt_points_t2 = rt_points_t2.reshape(28, VerticalRes * HorizontalRes, 3)
    
    for tooth_idx, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
        tooth = tp.getToothByID(tooth_id) 
        if not tooth: 
            # print(f"no tooth {tooth_id}")
            continue
        raw_mesh = tooth.getToothSurface() 
        pv_mesh = get_pv_mesh_from_raw_mesh(raw_mesh)
        
        tooth_points_t1 = rt_points_t1[tooth_idx]  # Get points for the current tooth

        tooth_rt_t1 = tooth.relativeTransform(0)
        tooth_rt_t2 = tooth.relativeTransform(t2)

        tr_matrix_t1 = get_transform_matrix_from_ormco(tooth_rt_t1)
        tr_matrix_t2 = get_transform_matrix_from_ormco(tooth_rt_t2)
        
        tooth_points_pred = predictions[tooth_idx]
        predicted_matrix = calc_transform_matrix_fr_points(tooth_points_t1, tooth_points_pred)
        pv_center_point_pred = pv.Sphere(radius=0.3, center=(0, 0, 0)).transform(predicted_matrix @ tr_matrix_t1)

        pv_mesh_pred = pv_mesh.copy().transform(predicted_matrix @ tr_matrix_t1)
        
        pv_mesh_t1 = pv_mesh.copy().transform(tr_matrix_t1)
        pv_mesh_t2 = pv_mesh.copy().transform(tr_matrix_t2)

        pred_centroids.append(pv_center_point_pred.center)
        # plt.add_mesh(center_point_t2, color='green', opacity=1.0)
        plt.add_mesh(pv_center_point_pred, color='blue', opacity=1.0)
       
        if "T1" in sets_to_plot: plt.add_mesh(pv_mesh_t1, show_edges=False, opacity=0.2, color="r", lighting=True, label="T1")
        if "T2" in sets_to_plot: plt.add_mesh(pv_mesh_t2, show_edges=False, opacity=0.2, color="g", lighting=True, label="T2")
        if "Pred" in sets_to_plot: plt.add_mesh(pv_mesh_pred, show_edges=False, opacity=0.4, color="b", lighting=True, label="Pred")

def draw_landmarks_shapes_pv(plt: pv.Plotter, 
                orthoCase, 
                sets_to_plot: list, 
                predictions: np.ndarray, 
                rt_points_t1: np.ndarray, 
                rt_points_t2: np.ndarray,
                jaws_to_plot = ["upper", "lower"],
                show_centriods = True
                ):
    """
    Draw the shapes of case using landmarks for upper, lower, or all teeth.
    jaws_to_plot: list of 'upper', 'lower', or both. Default is both.
    """
    tp = orthoCase.get_treatment_plan()
    t2 = orthoCase.get_T2_stage()
    pred_centroids = []
    teeth_to_plot = dw_teeth_nums14 + up_teeth_nums14
    
    predictions = predictions.reshape(28, 5, 3)
    rt_points_t1 = rt_points_t1.reshape(28, 5, 3)
    rt_points_t2 = rt_points_t2.reshape(28, 5, 3)
    
    if not "upper" in jaws_to_plot: # only lower jaw
        predictions = predictions[:14,:,:]  # teeth 0-14    
        rt_points_t1 = rt_points_t1[:14,:,:]
        rt_points_t2 = rt_points_t2[:14,:,:]
        teeth_to_plot = dw_teeth_nums14  # only lower teeth

    if not "lower" in jaws_to_plot: # only upper jaw
        predictions = predictions[14:,:,:]
        rt_points_t1 = rt_points_t1[14:,:,:]
        rt_points_t2 = rt_points_t2[14:,:,:]
        teeth_to_plot = up_teeth_nums14
    
    for tooth_idx, tooth_id in enumerate(teeth_to_plot):
        tooth = tp.getToothByID(tooth_id) 
        if not tooth: 
            # print(f"no tooth {tooth_id}")
            continue
        raw_mesh = tooth.getToothSurface() 
        pv_mesh = get_pv_mesh_from_raw_mesh(raw_mesh)
        
        tooth_points_t1 = rt_points_t1[tooth_idx]  # Get points for the current tooth

        tooth_rt_t1 = tooth.relativeTransform(0)
        tooth_rt_t2 = tooth.relativeTransform(t2)

        tr_matrix_t1 = get_transform_matrix_from_ormco(tooth_rt_t1)
        tr_matrix_t2 = get_transform_matrix_from_ormco(tooth_rt_t2)
        
        tooth_points_pred = predictions[tooth_idx]
        predicted_matrix = calc_transform_matrix_fr_points(tooth_points_t1, tooth_points_pred)
        pv_center_point_pred = pv.Sphere(radius=0.3, center=(0, 0, 0)).transform(predicted_matrix @ tr_matrix_t1)

        pv_mesh_pred = pv_mesh.copy().transform(predicted_matrix @ tr_matrix_t1)
        
        pv_mesh_t1 = pv_mesh.copy().transform(tr_matrix_t1)
        pv_mesh_t2 = pv_mesh.copy().transform(tr_matrix_t2)

        pred_centroids.append(pv_center_point_pred.center)
        if show_centriods:
            # plt.add_mesh(center_point_t2, color='green', opacity=1.0)
            plt.add_mesh(pv_center_point_pred, color='blue', opacity=1.0)
       
        if "T1" in sets_to_plot: plt.add_mesh(pv_mesh_t1, show_edges=False, opacity=0.2, color="r", lighting=True, label="T1")
        if "T2" in sets_to_plot: plt.add_mesh(pv_mesh_t2, show_edges=False, opacity=0.2, color="g", lighting=True, label="T2")
        if "Pred" in sets_to_plot: plt.add_mesh(pv_mesh_pred, show_edges=False, opacity=0.4, color="b", lighting=True, label="Pred")


def case_point_grids(orthoCase) -> Tuple[NDArray, NDArray]:
    """Return the point grids for the prediction"""
    tp = orthoCase.get_treatment_plan()
    t2 = orthoCase.get_T2_stage()
    rt_points_t1_list, rt_points_t2_list = [], []
    for tooth_idx, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
        tooth = tp.getToothByID(tooth_id) # TODO проверить зуб на nullptr 
        if tooth:
            tooth_id = tooth.getClinicalID()
            raw_mesh = tooth.getToothSurface() 
            pv_mesh = get_pv_mesh_from_raw_mesh(raw_mesh)
            grid_0 = calc_point_grid(tooth_id, isPosterior(tooth_id), mesh=pv_mesh)
            
            tooth_rt_t1 = tooth.relativeTransform(0)
            tooth_rt_t2 = tooth.relativeTransform(t2)

            tr_matrix_t1 = get_transform_matrix_from_ormco(tooth_rt_t1)
            tr_matrix_t2 = get_transform_matrix_from_ormco(tooth_rt_t2)
            rt_points_t1 = apply_rigid_transform(grid_0, tr_matrix_t1)
            rt_points_t2 = apply_rigid_transform(grid_0, tr_matrix_t2)

            # Ensure rt_points_t1 and rt_points_t2 are reshaped to (VerticalRes * HorizontalRes, 3)
            if rt_points_t1.shape != (VerticalRes * HorizontalRes, 3):
                print("rt_points_t1.shape", rt_points_t1.shape, "tooth_id", tooth_id)
                rt_points_t1 = np.zeros((VerticalRes * HorizontalRes, 3))  # Fallback to zeros if shape is incorrect
                # rt_points_t1 = np.resize(rt_points_t1, (VerticalRes * HorizontalRes, 3))
            if rt_points_t2.shape != (VerticalRes * HorizontalRes, 3):
                print("rt_points_t2.shape", rt_points_t2.shape, "tooth_id", tooth_id)
                rt_points_t2 = np.zeros((VerticalRes * HorizontalRes, 3))  # Fallback to zeros if shape is incorrect
                # rt_points_t2 = np.resize(rt_points_t2, (VerticalRes * HorizontalRes, 3))
                
            
            rt_points_t1_list.append(rt_points_t1)  # Collect rt_points_t1
            rt_points_t2_list.append(rt_points_t2)  # Collect rt_points_t2
        else:
            print(f"no tooth {tooth_id}")
            rt_points_t1_list.append(np.zeros((VerticalRes * HorizontalRes,3)))  # Collect rt_points_t1
            rt_points_t2_list.append(np.zeros((VerticalRes * HorizontalRes,3)))  # Collect rt_points_t2

    rt_points_t1 = np.array(rt_points_t1_list)  # Convert list to NumPy array
    rt_points_t2 = np.array(rt_points_t2_list)  # Convert list to NumPy array

    return rt_points_t1, rt_points_t2

def case_landmark_grids_orgins(ortho_data):
    """
    Extracts landmark grids for both jaws IN ZERO COORDS
    Returns:
        rt_points_t1: np.ndarray of shape (28, 5, 3) for T1
        rt_points_t2: np.ndarray of shape (28, 5, 3) for T2
    """
    num_teeth = 28
    points_per_tooth = 5
    coordinates_per_point = 3

    combined_vector = np.zeros((num_teeth, points_per_tooth, coordinates_per_point), dtype=np.float32)

    for tooth_index, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
        landmarks = ortho_data["Staging"][0]["Landmarks"].get(str(tooth_id), stub_missing_tooth_landmarks)
        # grid_from_landmarks logic (5 points: 2 from MDWLine, 1 each from BCPoint, MeanRootApex, FEGJPoint)
        grid = []
        MDWLine = landmarks["MDWLine"]
        grid.append([MDWLine["start"]["x"], MDWLine["start"]["y"], MDWLine["start"]["z"]])
        grid.append([MDWLine["end"]["x"], MDWLine["end"]["y"], MDWLine["end"]["z"]])
        for lm in ["BCPoint", "MRAPoint", "FEGJPoint"]:
            point = landmarks[lm]
            grid.append([point["x"], point["y"], point["z"]])
        grid_0 = np.array(grid, dtype=np.float32)
        combined_vector[tooth_index] = grid_0

    return combined_vector

def case_landmark_grids(ortho_data):
    """
    Extracts landmark grids for both jaws from an OrthoCase object.
    If buccal ridge points are present, use them for inference, 
        if not(zeros there) use MDWLine points instead.
    Returns:
        rt_points_t1: np.ndarray of shape (28, 5, 3) for T1
        rt_points_t2: np.ndarray of shape (28, 5, 3) for T2
    """
    num_teeth = 28
    points_per_tooth = 5
    coordinates_per_point = 3

    combined_vector_t1 = np.zeros((num_teeth, points_per_tooth, coordinates_per_point), dtype=np.float32)
    combined_vector_t2 = np.zeros((num_teeth, points_per_tooth, coordinates_per_point), dtype=np.float32)
    # tp = orthoCase.get_treatment_plan()
    # t2 = orthoCase.get_T2_stage()

    # tooth_index = 0
    # for jawType in JawType:
        # jaw = tp.GetJaw(jawType)
        # teeth_nums = dw_teeth_nums14 if jawType == JawType.Mandible else up_teeth_nums14

    for tooth_index, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
        
        def has_zero(line):
            """Проверяем, есть ли нули в координатах линии"""
            for point in ["start", "end"]:
                for coord in ["x", "y", "z"]:
                    if float(line[point][coord]) == 0.0:
                        return True
            return False
        
        # tooth = jaw.getToothByID(tooth_id)  
        landmarks = ortho_data["Staging"][0]["Landmarks"].get(str(tooth_id), stub_missing_tooth_landmarks)
        # grid_from_landmarks logic (5 points: 2 from MDWLine, 1 each from BCPoint, MeanRootApex, FEGJPoint)
        grid = []
        FirstLine = landmarks["BRLine"] if not has_zero(landmarks["BRLine"]) else landmarks["MDWLine"]
        # print(f"use {"BRLine" if not has_zero(landmarks["BRLine"]) else "MDWLine"} for tooth {tooth_id}")

        grid.append([FirstLine["start"]["x"], FirstLine["start"]["y"], FirstLine["start"]["z"]])
        grid.append([FirstLine["end"]["x"], FirstLine["end"]["y"], FirstLine["end"]["z"]])
        for lm in ["BCPoint", "MRAPoint", "FEGJPoint"]:
            point = landmarks[lm]
            grid.append([point["x"], point["y"], point["z"]])
        grid_0 = np.array(grid, dtype=np.float32)

        # t2 = int(ortho_data["T2Stage"]) - 1
        t2 = int(ortho_data["T2Stage"])
        tooth_rt_t1 = ortho_data["Staging"][0]["RelativeToothTransforms"].get(str(tooth_id), stub_missing_tooth_rt)
        tooth_rt_t2 = ortho_data["Staging"][-1]["RelativeToothTransforms"].get(str(tooth_id), stub_missing_tooth_rt)
        tr_matrix_t1 = get_transform_matrix(tooth_rt_t1)
        tr_matrix_t2 = get_transform_matrix(tooth_rt_t2)
        rt_points_t1 = apply_rigid_transform(grid_0, tr_matrix_t1)
        rt_points_t2 = apply_rigid_transform(grid_0, tr_matrix_t2)

        combined_vector_t1[tooth_index] = rt_points_t1
        combined_vector_t2[tooth_index] = rt_points_t2

    return combined_vector_t1, combined_vector_t2

def extract_tar_gz(tar_gz_file_path, output_dir):
    """Extract a .tar.gz file to the specified output directory."""
    with tarfile.open(tar_gz_file_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)


def find_db_file(output_dir):
    """Find the .db file within the specified directory."""
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.db'):
                return os.path.join(root, file)
    return None

def get_prescription_preferences_tables_content(oas_file_path: str) -> dict:
    """Extract and merge data from TREATMENT_PRESCRIPTION and CLINICAL_PREFERENCES tables."""
    out_dict = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        extract_tar_gz(oas_file_path, tmpdirname)
        oas_file_path = find_db_file(tmpdirname)

        if oas_file_path:
            try:
                conn = sqlite3.connect(oas_file_path)
                cursor = conn.cursor()

                # Combined query to retrieve data from both tables
                cursor.execute("""
                    SELECT tp.PrescriptionStr AS content
                    FROM TREATMENT_PRESCRIPTION tp
                    JOIN REVISIONS r ON tp.ID = r.ID
                    WHERE r.RevisionStatus = 'APPROVED'
                    UNION ALL
                    SELECT cp.ClinicalPreferencesStr AS content
                    FROM CLINICAL_PREFERENCES cp
                    JOIN REVISIONS r ON cp.ID = r.ID
                    WHERE r.RevisionStatus = 'APPROVED'
                """)

                results = cursor.fetchall()

                for row in results:
                    content_dict = json.loads(row[0])
                    out_dict.update(content_dict)

                conn.close()
            except sqlite3.Error as e:
                print(f"SQLite error: {e}")
                return None
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return None
        else:
            print("No SQLite DB file found.")
            return None
    return out_dict

def get_transform_from_matrix(matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Extract a transform dictionary (with quaternion and translation) from a 4x4 transformation matrix.
    Args:
        matrix (np.ndarray): 4x4 transformation matrix
    Returns:
        dict: { 'rotation': {x, y, z, w}, 'translation': {x, y, z} }
    """
    matrix = np.asarray(matrix)
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix"
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
    return {
        'rotation': {
            'x': float(quat[0]),
            'y': float(quat[1]),
            'z': float(quat[2]),
            'w': float(quat[3]),
        },
        'translation': {
            'x': float(translation[0]),
            'y': float(translation[1]),
            'z': float(translation[2]),
        }
    }

# def fromOrmco(obj):
#     if isinstance(obj, PyPoint):
#         return Vec3(obj.x, obj.y, obj.z)
#     elif isinstance(obj, PyLine):
#         return [obj.startPoint, obj.endPoint]
#     elif isinstance(obj, PyVector):
#         return Vec3(obj.x, obj.y, obj.z)
#     elif isinstance(obj, PyQuaternion):
#         return Quat(obj.re, obj.im.x, obj.im.y, obj.im.z)
#     elif isinstance(obj, PyRigidTransform):
#         vec = fromOrmco(obj.translation)
#         quat = fromOrmco(obj.rotation)
#         return RigidTransform(vec, quat)
    
# def eval_batch(oas_folder=r"E:\awsCollectedData1\anonim", 
#                html_export=False,
#                plot=True, 
#                num_files_to_plot =5, 
#                sets_to_plot = ["T2", "Pred",],
#                eval_fn = None,
#                model_type = "tooth_level_transformer_cond_decoder",
#                ):
#     oas_files = glob.glob(os.path.join(oas_folder, "*.oas"))

#     case_ids = [int(os.path.basename(f).split('.')[0]) for f in oas_files]
#     # case_ids_ = [999817, 975662, 128365]

#     case_id = case_ids[0] # if one case to plot

#     # loss logging
#     losses = {}
#     time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
#     for case_id in case_ids[:num_files_to_plot]:
#         print(f"Case {case_id}")
#         orthoCase = OrthoCase(os.path.join(oas_folder, f"{case_id}.oas"))
#         plt = pv.Plotter()
#         case_path = os.path.join(oas_folder, f"{case_id}.oas")
#         plt.add_text(f"Case {case_id}", position="lower_right", color="blue", font_size=12, 
#                     viewport=True, render=False)

#         rt_points_t1, rt_points_t2 = case_point_grids(orthoCase)
#         print("type eval fn ", type(eval_fn))
#         # predictions, loss = eval_fn(rt_points_t1, rt_points_t2, case_path=case_path)
#         predictions, loss = eval_fn(rt_points_t1, rt_points_t2)
#         print("loss - ", loss)
#         losses[case_id] = f'{loss:.3}'
#         predictions = predictions.cpu().detach().numpy()

#         draw_shapes_pv(plt, 
#                     orthoCase, 
#                     sets_to_plot, 
#                     predictions, 
#                     rt_points_t1, 
#                     rt_points_t2)

#         plt.add_legend(labels=sets_to_plot)
#         # Define the learning rate and model parameters
#         learning_rate = 0.001

#         # Create the subfolder name for html export
#         subfolder_name = f"lr_{learning_rate}_model_{model_type}_{time_stamp}"
#         subfolder_path = os.path.join("..", "html", subfolder_name)

#         # Update the html_path to include the subfolder
#         html_path = os.path.join(subfolder_path, f"case_{case_id}_{loss:.3}.html")
#         # html_path = f"../html/case_{case_id}_{loss:.3}.html"
#         plt.view_xz()
#         if html_export:
#             os.makedirs(subfolder_path, exist_ok=True)
#             plt.export_html(html_path)
#         if plot: 
#             plt.show()
#     return losses