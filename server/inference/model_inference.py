# takes base and template cases and make # predictions for the base case
# base case goes thru the Initial Autoencoder then ArchFormRegressor applies 
# the template to the base case prediction result
import sys
import os
import torch
from typing import List, Dict, Any
# from server.inference.models import ArchFormRegressor, InitAutoencoder
# from autosetup_ml.utils import *
from .utils import *
from .models import ArchFormRegressor, InitAutoencoder
# from server.ortho_data import getToothRelativeTransform, getToothRelativeTransformHead
import torch.nn as nn
import json
import contextlib
import time

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        # Save original file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        # Redirect stdout/stderr to devnull
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            # Restore original file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

def show_2_cloud_points_in_pv(points1, points2=None, title="Points"):
    """
    Side method for debugging.
    Displays points in a PyVista plotter.
    :param points: numpy array of shape (N_teeth, N_points, 3) representing the points.
    :param title: title for the plot.
    Each tooth's points are shown in a different random color.
    """
    import pyvista as pv
    import numpy as np

    plotter = pv.Plotter()
    color1 = [1.0, 0.0, 0.0]  # Red for first cloud (float RGB)
    color2 = [0.0, 0.0, 1.0]  # Blue for second cloud (float RGB)

    # Combine all points for each cloud into a single array for each color
    all_points1 = points1.reshape(-1, 3)
    point_cloud1 = pv.PolyData(all_points1)
    plotter.add_mesh(point_cloud1, color=color1, point_size=10, render_points_as_spheres=True)
    
    if points2 is not None:
        all_points2 = points2.reshape(-1, 3)
        point_cloud2 = pv.PolyData(all_points2)
        plotter.add_mesh(point_cloud2, color=color2, point_size=10, render_points_as_spheres=True)

    plotter.add_text(title, position='upper_edge', font_size=10, color='grey')
    plotter.show()

class OrthoCaseDataLoader:
    def __init__(self, file_path="public/orthoData.json"):
        # print(f"os.path.abspath(__file__) {os.path.abspath(__file__)}")
        # print(f"os.path.dirname(os.path.abspath(__file__)) {os.path.dirname(os.path.abspath(__file__))}")
        abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", file_path)
        with open(abs_file_path, "r", encoding="utf-8") as f:
            self.ortho_data = json.load(f)

    def get_landmarks(self):
        return case_landmark_grids(self.ortho_data)

    def get_landmarks_orgins(self):
        return case_landmark_grids_orgins(self.ortho_data)


class AutoencoderInference:
    def __init__(self, checkpoint_path, num_teeth=28, num_points=5, coord_dim=3):
        self.model = InitAutoencoder(num_teeth=num_teeth, num_points=num_points, coord_dim=coord_dim).to("cpu")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval().cpu()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.coord_dim = coord_dim
    def predict(self, rt_points_t1, rt_points_t2):
        x_tensor = torch.from_numpy(rt_points_t1).float().unsqueeze(0)
        y_tensor = torch.from_numpy(rt_points_t2).float().unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(x_tensor)
            loss = torch.nn.L1Loss()(predictions, y_tensor)
        return predictions.squeeze(0).cpu().detach().numpy(), loss.item()

class ArchRegressorInference:
    def __init__(self, 
                 checkpoint_path, 
                 num_teeth=28, 
                 num_points=5, 
                 coord_dim=3, 
                 hidden_dim=512, 
                 num_layers=4):
        self.model = ArchFormRegressor(num_teeth=num_teeth, num_points=num_points, coord_dim=coord_dim, hidden_dim=hidden_dim, num_layers=num_layers).to("cpu")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval().cpu()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.coord_dim = coord_dim
    def predict(self, ae_pred, template, targets=None):
        ae_pred_tensor = torch.from_numpy(ae_pred).float().unsqueeze(0)
        template_tensor = torch.from_numpy(template).float().unsqueeze(0)
        if targets is not None:
            targets_tensor = torch.from_numpy(targets).float().unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(ae_pred_tensor, template_tensor)
            if targets is not None:
                loss_item = torch.nn.L1Loss()(predictions, targets_tensor).item()
            else:
                loss_item = 0.0
        return predictions.squeeze(0).cpu().detach().numpy(), loss_item

class OrthoInferencePipeline:
    def __init__(self, ae_ckpt, reg_ckpt=None):
        self.ae = AutoencoderInference(ae_ckpt)
        self.reg = ArchRegressorInference(reg_ckpt) if reg_ckpt else None

    def _compose_transforms_from_points(self, base_loader: OrthoCaseDataLoader, prediction_points, base_case_points_t1) -> Dict[str, Dict[str, Any]]:
        transforms_dict = {}
        for tooth_idx, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
            tooth_points_t1 = base_case_points_t1[tooth_idx]
            tooth_points_pred = prediction_points[tooth_idx]
            # tooth = base_loader.get_tooth_by_cl_id(tooth_id)
            # toothRT0 = getToothRelativeTransform(tooth, stage=0)
            toothRT0 = base_loader.ortho_data["Staging"][0]["RelativeToothTransforms"].get(str(tooth_id), None)
            if toothRT0 is None:
                print(f"[ERROR] Skipping tooth {tooth_id} not presented.")
                continue
            t1_matrix = get_transform_matrix(toothRT0)
            pred_matrix = calc_transform_matrix_fr_points(tooth_points_t1, tooth_points_pred)
            total_matrix = pred_matrix @ t1_matrix
            tooth_transform = get_transform_from_matrix(total_matrix)
            transforms_dict[str(tooth_id)] = tooth_transform
        # print(f"transforms_dict 11 {transforms_dict["11"]}")
        return transforms_dict
        
    def _compose_transforms_to_jaw(self, transforms_dict: Dict[str, Dict[str, Any]], mandible_rt, maxillary_rt) -> Dict[str, Dict[str, Any]]:
        mandible_jaw_translation = translate(mandible_rt)
        mandible_jaw_quaternion = quaternion(mandible_rt)
        maxilla_jaw_translation = translate(maxillary_rt)
        maxilla_jaw_quaternion = quaternion(maxillary_rt)

        for tooth_id, tf in transforms_dict.items():
            try:
                tid = int(tooth_id)
            except Exception:
                continue
            # Determine if the tooth is in the mandible or maxilla
            jaw_translation = mandible_jaw_translation if tid > 30 else maxilla_jaw_translation
            jaw_quaternion = mandible_jaw_quaternion if tid > 30 else maxilla_jaw_quaternion

            # Ensure all values are float before creating numpy arrays
            t_vec = np.array([
                float(tf["translation"]["x"]),
                float(tf["translation"]["y"]),
                float(tf["translation"]["z"])
            ])
            q_quat = [
                float(tf["rotation"]["x"]),
                float(tf["rotation"]["y"]),
                float(tf["rotation"]["z"]),
                float(tf["rotation"]["w"])
            ]
            jt_vec = np.array([
                float(jaw_translation[0]),
                float(jaw_translation[1]),
                float(jaw_translation[2])
            ])
            jq_quat = [
                float(jaw_quaternion[0]),
                float(jaw_quaternion[1]),
                float(jaw_quaternion[2]),
                float(jaw_quaternion[3])
            ]
            t_rot = R.from_quat(jq_quat).apply(t_vec)
            final_translation = t_rot + jt_vec
            final_quat = R.from_quat(jq_quat) * R.from_quat(q_quat)
            final_quat_xyzw = final_quat.as_quat()
            tf["translation"] = {
                "x": float(final_translation[0]),
                "y": float(final_translation[1]),
                "z": float(final_translation[2])
            }
            tf["rotation"] = {
                "x": float(final_quat_xyzw[0]),
                "y": float(final_quat_xyzw[1]),
                "z": float(final_quat_xyzw[2]),
                "w": float(final_quat_xyzw[3])
            }
        return transforms_dict
    
    def apply_transform_to_point_cloud(self, points, transforms: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Applies a rigid transformation to a point cloud.
        :param points: numpy array of shape (N_teeth, N_points, 3) representing the point cloud.
        :param transforms: dictionary with 'translation' and 'rotation' keys.
        :return: transformed points in the same shape as input.
        """
        # transforms = {}
        # zero_transform = {
            # "translation": {"x": 0.0, "y": 0.0, "z": 1.0},
            # "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0} # STUB!!! remove!!!!
        # }
        
        points_ = points.copy()
        for idx, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
            
            # print(str(tooth_id), json.dumps(transforms[str(tooth_id)])) # STUB !! remove!!!

            # transforms[str(tooth_id)] = zero_transform
            
            if str(tooth_id) not in transforms:
                print(f"[ERROR] Tooth {tooth_id} not found in transforms dictionary.")
                continue
            
            tooth_points = points_[idx]
            tr_matrix = get_transform_matrix_from_three_rt(transforms[str(tooth_id)])
            transformed_points = apply_rigid_transform(tooth_points, tr_matrix)
            points_[idx] = transformed_points
        return points_
    
    def point_cloud_to_jaw(self, points, mandibular_jaw_rt, maxillary_jaw_rt):
        """
        Transforms a point cloud to the jaw coordinate system.
        :param points: numpy array of shape (teeth _in_case, N, 3) representing the point cloud.
        :param *_jaw_rt: relative transform of the jaws.
        :return: transformed points in the jaw coordinate system.
        """
        points_ = points.copy()
        mandibular_matrix = get_transform_matrix(mandibular_jaw_rt)#.copy()
        maxillary_matrix = get_transform_matrix(maxillary_jaw_rt)#.copy()

        for tooth_idx in range(points.shape[0]):  # 0..27
            tooth_points = points_[tooth_idx]
            jaw_matrix = mandibular_matrix if tooth_idx > 30 else maxillary_matrix
            transformed_points = apply_rigid_transform(tooth_points, jaw_matrix)
            points_[tooth_idx] = transformed_points
        return points_
    
    def run_t2_predict(self, template_case_path, template_transforms={}) -> Dict[str, Dict[str, Any]]:
        base_loader = OrthoCaseDataLoader()
        # Get jaw transforms from ortho_data
        base_mandible_jaw_rt = base_loader.ortho_data.get('mandibularRelativeTransform', None)
        # print(f"base_mandible_jaw_rt {base_mandible_jaw_rt}")
        base_maxilla_jaw_rt = base_loader.ortho_data.get('maxillaRelativeTransform', None)
        base_case_points_t1, base_case_points_t2 = base_loader.get_landmarks()
        base_case_points_origins = base_loader.get_landmarks_orgins() # wo translations
        base_case_points_t1_ = base_case_points_t1.copy() # orig points for compose transforms
        
        # Apply jaw transformations to base point clouds
        base_case_points_t1 = self.point_cloud_to_jaw(base_case_points_t1, base_mandible_jaw_rt, base_maxilla_jaw_rt)
        base_case_points_t2 = self.point_cloud_to_jaw(base_case_points_t2, base_mandible_jaw_rt, base_maxilla_jaw_rt)
        
        # вводим новый режим: если есть на входе template_transforms, то это значит что используем корретированные 
        # трансформ контролом новые положения зубов базового кейса в качестве шаблона.
        # теперь нам нужны новые точки шаблона в T2 т.к. только точки можно отправить в предикт. 
        # у нас есть их трансформы - это трансформы относительно локальных (нулевых координат), к лендмаркам базового кейса
        if template_transforms:
            # для режима коррекции темплейта с фронта
            template_points_t2_front = self.apply_transform_to_point_cloud(base_case_points_origins, template_transforms)
            # show_2_cloud_points_in_pv(base_case_points_t1, template_points_t2_front, title="t1_ (red) Transformed T2 (blue)")
            
            # template_diff = template_points_t2_front - base_case_points_t1 # for diff mode
            template_input = template_points_t2_front
            template_points_t2 = None # stub
        else:
            template_loader = OrthoCaseDataLoader(template_case_path)
            template_points_t1, template_points_t2 = template_loader.get_landmarks()
            template_mandible_jaw_rt = template_loader.ortho_data.get('mandibularRelativeTransform', None)
            template_maxilla_jaw_rt = template_loader.ortho_data.get('maxillaRelativeTransform', None)
        
            # No needs???. They're already in jaw (Apply jaw transformations to template point clouds)
            template_points_t1 = self.point_cloud_to_jaw(template_points_t1, template_mandible_jaw_rt, template_maxilla_jaw_rt)
            template_points_t2 = self.point_cloud_to_jaw(template_points_t2, template_mandible_jaw_rt, template_maxilla_jaw_rt)
            
            # template_points_t1 = self.point_cloud_to_jaw(template_points_t1, base_mandible_jaw_rt, base_maxilla_jaw_rt)
            # template_points_t2 = self.point_cloud_to_jaw(template_points_t2, base_mandible_jaw_rt, base_maxilla_jaw_rt)
            
            # template_input = template_points_t2 - template_points_t1 # for diff mode
            template_input = template_points_t2

        init_prediction_points, _ = self.ae.predict(base_case_points_t1, base_case_points_t2)
        
        predictions, _ = self.reg.predict(init_prediction_points, template_input, template_points_t2) if self.reg else (init_prediction_points, 0)
        # show_2_cloud_points_in_pv(predictions, template_points_t2, title="Regressor Predictions (red) and Template T2 (blue)")
        # Compose transforms
        transforms_dict = self._compose_transforms_from_points(base_loader, predictions, base_case_points_t1_) 

        # transform to Head coordinates no needs anymore due to work in head cs for now.
        # transforms_dict = self._compose_transforms_to_jaw(transforms_dict, base_mandible_jaw_rt, base_maxilla_jaw_rt)
        print(f"T2 inference done from {'manual T2' if template_transforms else 'extern file'}")
        return transforms_dict

    def run_init_predict(self) -> Dict[str, Dict[str, Any]]:
        # base_loader = OrthoCaseDataLoader(base_case_path)
        # base_mandible_jaw_rt = base_loader.ortho_case.tp.GetJaw(JawType.Mandible).relativeTransform(0)
        # base_maxilla_jaw_rt = base_loader.ortho_case.tp.GetJaw(JawType.Maxilla).relativeTransform(0)
        base_loader = OrthoCaseDataLoader()
        # Get jaw transforms from ortho_data
        base_mandible_jaw_rt = base_loader.ortho_data.get('mandibularRelativeTransform', None)
        # print(f"base_mandible_jaw_rt {base_mandible_jaw_rt}")
        base_maxilla_jaw_rt = base_loader.ortho_data.get('maxillaRelativeTransform', None)

        base_case_points_t1, base_case_points_t2 = base_loader.get_landmarks()
        init_prediction_points, loss = self.ae.predict(base_case_points_t1, base_case_points_t2)

        transforms_dict = self._compose_transforms_from_points(base_loader, init_prediction_points, base_case_points_t1)

        # transform to Head coordinates !!!!!!!!!!!!!!!!!!!!! выключил вчера. проверять как это работатет !!!!!!!!!!!!!!!!!!!!!!!!!!
        transforms_dict = self._compose_transforms_to_jaw(transforms_dict, base_mandible_jaw_rt, base_maxilla_jaw_rt)
        print(f"Init inference done (class pipeline)")
        return transforms_dict

if __name__ == "__main__":
    # Example usage with new SOLID pipeline
    base_case_path = os.path.join("server", "00000000.oas")
    template_case_path = os.path.join("server", "00000000.oas")
    ae_ckpt = "server/inference/init_ae/best_model.pth"
    reg_ckpt = "server/inference/arch_regressor/best_model.pth"
    pipeline = OrthoInferencePipeline(ae_ckpt, reg_ckpt)
    transforms = pipeline.run_t2_predict(base_case_path, template_case_path)
    # transforms = pipeline.run_init_predict(base_case_path)
    print("Transforms for tooth 37:")
    print(json.dumps(transforms['37'], indent=2, ensure_ascii=False))
    # Optionally print all transforms or debug info
    # print(json.dumps(transforms, indent=2, ensure_ascii=False))
