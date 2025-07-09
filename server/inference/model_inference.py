# takes base and template cases and make # predictions for the base case
# base case goes thru the Initial Autoencoder then ArchFormRegressor applies 
# the template to the base case prediction result
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Cython'))
import torch
from typing import List, Dict, Any
# from server.inference.models import ArchFormRegressor, InitAutoencoder
from autosetup_ml.utils import *
import torch
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
        
class InitAutoencoder(nn.Module):
    def __init__(self, num_teeth: int = 28, num_points: int = 5, coord_dim: int = 3):
        super().__init__()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.dense_dim = num_teeth * num_points * coord_dim
        self.dense_dim2 = round(self.dense_dim//1.5) #1.5
        self.dense_dim3 = round(self.dense_dim2//2)
        self.dim_code = round(self.dense_dim3//2)
        # self.dim_code = self.dense_dim // 6
        
        # Encoder
        self.encoder = nn.Sequential(
            # nn.Linear(self.dense_dim, self.dense_dim),
            # nn.ELU(),
            nn.Linear(self.dense_dim, self.dense_dim2),
            nn.ELU(),
            nn.Linear(self.dense_dim2, self.dense_dim3),
            nn.ELU(),
            nn.Linear(self.dense_dim3, self.dim_code)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.dim_code, self.dense_dim3),
            nn.ELU(),
            nn.Linear(self.dense_dim3, self.dense_dim2),
            nn.ELU(),
            nn.Linear(self.dense_dim2, self.dense_dim)
            # nn.ELU(),
            # nn.Linear(self.dense_dim, self.dense_dim)
        )

    def forward(self, x):
        # Pass input through encoder
        x = x.view(x.size(0), -1) # Flatten the input
        encoded = self.encoder(x)
        # Pass encoded representation through decoder
        decoded = self.decoder(encoded)
        # Reshape back to original dimensions
        decoded = decoded.view(-1, self.num_teeth, self.num_points, self.coord_dim)
        return decoded
 
class ArchFormRegressor(nn.Module):
    def __init__(self, num_teeth: int = 28, 
                 num_points: int = 5, 
                 coord_dim: int = 3, 
                 hidden_dim: int = 512,
                 num_layers: int = 3):
        super().__init__()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.input_dim = num_teeth * num_points * coord_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.input_dim
        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_dim * 2, hidden_dim))
        layers.append(nn.ELU())
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ELU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, ae_pred, template):
        # ae_pred, template: [batch, num_teeth, num_points, coord_dim]
        ae_pred = ae_pred.view(ae_pred.size(0), -1)
        template = template.view(template.size(0), -1)
        x = torch.cat([ae_pred, template], dim=1)
        out = self.net(x)
        return out.view(-1, self.num_teeth, self.num_points, self.coord_dim)

def init_ae_eval_fn(rt_points_t1, 
                    rt_points_t2, 
                    checkpoint_path, 
                    num_teeth=28, 
                    points_per_tooth=5, 
                    coordinates_per_point=3
                    ):
    """
    Evaluate InitAutoencoder on a single case.
    Args:
        rt_points_t1: np.ndarray, shape [num_teeth, points_per_tooth, 3]
        rt_points_t2: np.ndarray, shape [num_teeth, points_per_tooth, 3]
        checkpoint_path: str, path to model checkpoint
        num_teeth, points_per_tooth, coordinates_per_point: model params
    Returns:
        predictions: torch.Tensor, shape [1, num_teeth, points_per_tooth, 3]
        loss: float
    """
    model = InitAutoencoder(num_teeth=num_teeth,
                            num_points=points_per_tooth,
                            coord_dim=coordinates_per_point
                            ).to("cpu")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval().cpu()
    x_tensor = torch.from_numpy(rt_points_t1).float().unsqueeze(0)
    y_tensor = torch.from_numpy(rt_points_t2).float().unsqueeze(0)
    with torch.no_grad():
        predictions = model(x_tensor)
        loss = torch.nn.L1Loss()(predictions, y_tensor)
    return predictions, loss.item()

def regressor_eval_fn(ae_pred, 
                      template, 
                      targets, 
                      checkpoint_path, 
                      num_teeth=28, 
                      points_per_tooth=5, 
                      coordinates_per_point=3, 
                      hidden_dim=512, 
                      num_layers=4
                      ):
    """
    Evaluate ArchFormRegressor on a single case.
    Args:
        ae_pred: np.ndarray, shape [num_teeth, points_per_tooth, 3]
        template: np.ndarray, shape [num_teeth, points_per_tooth, 3]
        targets: np.ndarray, shape [num_teeth, points_per_tooth, 3]
        checkpoint_path: str, path to model checkpoint
        ...model params...
    Returns:
        predictions: torch.Tensor, shape [1, num_teeth, points_per_tooth, 3]
        loss: float
    """
    model = ArchFormRegressor(num_teeth=num_teeth, 
                              num_points=points_per_tooth, 
                              coord_dim=coordinates_per_point,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers
                              ).to("cpu")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval().cpu()
    ae_pred_tensor = torch.from_numpy(ae_pred).float().unsqueeze(0)
    template_tensor = torch.from_numpy(template).float().unsqueeze(0)
    targets_tensor = torch.from_numpy(targets).float().unsqueeze(0)
    with torch.no_grad():
        predictions = model(ae_pred_tensor, template_tensor)
        loss = torch.nn.L1Loss()(predictions, targets_tensor)
    return predictions, loss.item()

def getToothRelativeTransform(tooth, stage):
    try:
        rel_transform = tooth.relativeTransform(stage)
        if rel_transform is None:
            print(f"[ERROR] Null pointer: relativeTransform({stage}) is None for tooth {getattr(tooth, 'cl_id', 'unknown')}")
            return None
        translation = rel_transform.translation
        rotation = rel_transform.rotation
        return {
            "translation": {
                "x": translation.x, "y": translation.y, "z": translation.z},
            "rotation": {
                "x": rotation.im.x, "y": rotation.im.y, "z": rotation.im.z,
                "w": rotation.re}
        }
    except Exception as e:
        print(f"[ERROR] Exception in getToothRelativeTransform for tooth {getattr(tooth, 'cl_id', 'unknown')}: {e}")
        return None

def get_pediction( 
                    base_case_points_t1: NDArray,
                    base_case_points_t2: NDArray,
                    template_points_t1: NDArray,
                    template_points_t2: NDArray,
                    init_ae_checkpoint_path: str = "server/inference/init_ae/best_model.pth",
                    regressor_checkpoint_path: str = "server/inference/arch_regressor/best_model.pth"
)-> Tuple[NDArray, NDArray, float]:  
    
    # Init AE prediction
    init_predictions, _ = init_ae_eval_fn(base_case_points_t1, base_case_points_t2, checkpoint_path=init_ae_checkpoint_path)
    init_predictions = init_predictions.cpu().detach().numpy()
    
    # Regressor prediction using template from another case
    template_input = template_points_t2 - template_points_t1
    # Invert x and y coordinates for each point in each tooth
    template_input = template_input.copy()
    template_input[..., 0] *= -1  # invert x
    template_input[..., 1] *= -1  # invert y
    
    regress_predictions, regress_loss = regressor_eval_fn(init_predictions, 
                                                            template_input, 
                                                            template_points_t2,
                                                            checkpoint_path=regressor_checkpoint_path,
                                                            hidden_dim=512, 
                                                            num_layers=4
                                                            )
    regress_predictions = regress_predictions.squeeze(0).cpu().detach().numpy()
    return regress_predictions, base_case_points_t1, regress_loss
    # return base_case_points_t2, base_case_points_t1, regress_loss # DEBUG!!!!!!!!!!!!!!!

def get_predicted_transforms(
        base_case_id = "00000000",
        template_case_id = "00000000",
        # base_oas_file_path = "server/00000000.oas",
        oas_folder="server",
        init_ae_checkpoint_path = "server/inference/init_ae/best_model.pth",
        regressor_checkpoint_path = "server/inference/arch_regressor/best_model.pth"
    ) -> Tuple[List[Dict[str, Any]]]:
    # base_case_id = "00000000"
    # print(f"!!! inference  base case id {base_case_id}, template case id {template_case_id}")
    # Get base case points
    base_oas_file_path = os.path.join(oas_folder, f"{base_case_id}.oas")
    with suppress_stdout_stderr(): 
        base_orthoCase = OrthoCase(base_oas_file_path)
    base_case_points_t1, base_case_points_t2 = case_landmark_grids(base_orthoCase)     
    # add blocking to wait 2 seconds here
    # time.sleep(2)
    # Get template case points
    template_oas_file_path = os.path.join(oas_folder, f"{template_case_id}.oas")
    with suppress_stdout_stderr():
        template_orthoCase = OrthoCase(template_oas_file_path)
    # print(f"!!! inference2  {template_orthoCase}")
    template_points_t1, template_points_t2 = case_landmark_grids(template_orthoCase)     
    
    # Get base case arch form for reference
    # prefs_base = get_prescription_preferences_tables_content(base_oas_file_path)
    # base_archform_str = prefs_base.get("clinicalPreferences", {}).get("finalToothPosition", {}).get("archForm")
    
    # Get template arch form
    # prefs_templ = get_prescription_preferences_tables_content(template_oas_file_path)
    # template_archform_str = prefs_templ.get("clinicalPreferences", {}).get("finalToothPosition", {}).get("archForm")
    

    predictions, base_case_points_t1, loss = get_pediction(
        base_case_points_t1 = base_case_points_t1,
        base_case_points_t2 = base_case_points_t2,
        template_points_t1 = template_points_t1,
        template_points_t2 = template_points_t2,
        init_ae_checkpoint_path = init_ae_checkpoint_path,
        regressor_checkpoint_path = regressor_checkpoint_path)

    transforms_dict = {}

    for tooth_idx, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
        tooth_points_t1 = base_case_points_t1[tooth_idx]
        tooth_points_pred = predictions[tooth_idx]

        # Get T1 transform matrix
        tooth = base_orthoCase.get_tooth_by_cl_id(tooth_id)
        # if tooth is None:
        #     print(f"Tooth with ID {tooth_id} not found in base case.")
        #     continue
        toothRT0 = getToothRelativeTransform(tooth, 0)
        if toothRT0 is None: # if tooth not present, 
            print(f"[ERROR] Skipping tooth {tooth_id} not presented.")
            continue
        t1_matrix = get_transform_matrix(toothRT0)
        # Get predicted transform (from T1 to predicted)
        pred_matrix = calc_transform_matrix_fr_points(tooth_points_t1, tooth_points_pred)
        total_matrix = pred_matrix @ t1_matrix
        # Extract transform
        tooth_transform = get_transform_from_matrix(total_matrix)
        transforms_dict[str(tooth_id)] = tooth_transform
    print(f"T2 inference done ")
    return transforms_dict
    # return {"prediction": transforms_dict, "loss": loss}

class OrthoCaseLoader:
    def __init__(self, file_path):
        with suppress_stdout_stderr():
            self.ortho_case = OrthoCase(file_path)
    def get_landmarks(self):
        return case_landmark_grids(self.ortho_case)
    def get_tooth_by_cl_id(self, tooth_id):
        return self.ortho_case.get_tooth_by_cl_id(tooth_id)

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
        return predictions.cpu().detach().numpy(), loss.item()

class ArchRegressorInference:
    def __init__(self, checkpoint_path, num_teeth=28, num_points=5, coord_dim=3, hidden_dim=512, num_layers=4):
        self.model = ArchFormRegressor(num_teeth=num_teeth, num_points=num_points, coord_dim=coord_dim, hidden_dim=hidden_dim, num_layers=num_layers).to("cpu")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval().cpu()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.coord_dim = coord_dim
    def predict(self, ae_pred, template, targets):
        ae_pred_tensor = torch.from_numpy(ae_pred).float().unsqueeze(0)
        template_tensor = torch.from_numpy(template).float().unsqueeze(0)
        targets_tensor = torch.from_numpy(targets).float().unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(ae_pred_tensor, template_tensor)
            loss = torch.nn.L1Loss()(predictions, targets_tensor)
        return predictions.squeeze(0).cpu().detach().numpy(), loss.item()

class OrthoInferencePipeline:
    def __init__(self, ae_ckpt, reg_ckpt):
        self.ae = AutoencoderInference(ae_ckpt)
        self.reg = ArchRegressorInference(reg_ckpt)
    def run(self, base_case_path, template_case_path):
        base_loader = OrthoCaseLoader(base_case_path)
        template_loader = OrthoCaseLoader(template_case_path)
        base_case_points_t1, base_case_points_t2 = base_loader.get_landmarks()
        template_points_t1, template_points_t2 = template_loader.get_landmarks()
        # Prepare template input (invert x and y) wich is strange behaviour!!!! check frontend!!!
        template_input = template_points_t2 - template_points_t1
        template_input = template_input.copy()
        template_input[..., 0] *= -1
        template_input[..., 1] *= -1
        # AE prediction
        init_predictions, _ = self.ae.predict(base_case_points_t1, base_case_points_t2)
        # Regressor prediction
        predictions, _ = self.reg.predict(init_predictions, template_input, template_points_t2)
        # Compose transforms
        transforms_dict = {}
        for tooth_idx, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
            tooth_points_t1 = base_case_points_t1[tooth_idx]
            tooth_points_pred = predictions[tooth_idx]
            tooth = base_loader.get_tooth_by_cl_id(tooth_id)
            toothRT0 = getToothRelativeTransform(tooth, 0)
            if toothRT0 is None:
                print(f"[ERROR] Skipping tooth {tooth_id} not presented.")
                continue
            t1_matrix = get_transform_matrix(toothRT0)
            pred_matrix = calc_transform_matrix_fr_points(tooth_points_t1, tooth_points_pred)
            total_matrix = pred_matrix @ t1_matrix
            tooth_transform = get_transform_from_matrix(total_matrix)
            transforms_dict[str(tooth_id)] = tooth_transform
        print(f"T2 inference done (class pipeline)")
        return transforms_dict

if __name__ == "__main__":
    # Example usage with new SOLID pipeline
    base_case_path = os.path.join("server", "00000000.oas")
    template_case_path = os.path.join("server", "00000000.oas")
    ae_ckpt = "server/inference/init_ae/best_model.pth"
    reg_ckpt = "server/inference/arch_regressor/best_model.pth"
    pipeline = OrthoInferencePipeline(ae_ckpt, reg_ckpt)
    transforms = pipeline.run(base_case_path, template_case_path)
    print("Transforms for tooth 37:")
    print(json.dumps(transforms['37'], indent=2, ensure_ascii=False))
    # Optionally print all transforms or debug info
    # print(json.dumps(transforms, indent=2, ensure_ascii=False))
