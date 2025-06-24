# takes base and template cases and make # predictions for the base case
# base case goes thru the Initial Autoencoder then ArchFormRegressor applies 
# the template to the base case prediction result
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Cython'))
import torch
from typing import List, Dict, Any
from server.inference.models import ArchFormRegressor, InitAutoencoder
from autosetup_ml.utils import *

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

def get_pediction(oas_folder: str = "server",
                    base_case_id: str = "00000000",
                    template_case_id: str = "00000000",
                    base_oas_file_path: str = "server/00000000.oas",
                    init_ae_checkpoint_path: str = "server/inference/init_ae/best_model.pth",
                    regressor_checkpoint_path: str = "server/inference/arch_regressor/best_model.pth"
)-> Tuple[NDArray, NDArray, float]:  
    
    # Get base case points
    base_oas_file_path = os.path.join(oas_folder, f"{base_case_id}.oas")
    base_orthoCase = OrthoCase(base_oas_file_path)
    base_case_points_t1, base_case_points_t2 = case_landmark_grids(base_orthoCase)     

    # Get template case points
    template_oas_file_path = os.path.join(oas_folder, f"{template_case_id}.oas")
    template_orthoCase = OrthoCase(template_oas_file_path)
    template_points_t1, template_points_t2 = case_landmark_grids(template_orthoCase)     
    
    # Get base case arch form for reference
    prefs_base = get_prescription_preferences_tables_content(base_oas_file_path)
    base_archform_str = prefs_base.get("clinicalPreferences", {}).get("finalToothPosition", {}).get("archForm")
    
    # Get template arch form
    prefs_templ = get_prescription_preferences_tables_content(template_oas_file_path)
    template_archform_str = prefs_templ.get("clinicalPreferences", {}).get("finalToothPosition", {}).get("archForm")
    
    # Init AE prediction
    init_predictions, _ = init_ae_eval_fn(base_case_points_t1, base_case_points_t2, checkpoint_path=init_ae_checkpoint_path)
    init_predictions = init_predictions.cpu().detach().numpy()
    
    # Regressor prediction using template from another case
    template_input = template_points_t2 - template_points_t1
    regress_predictions, regress_loss = regressor_eval_fn(init_predictions, 
                                                            template_input, 
                                                            template_points_t2,
                                                            checkpoint_path=regressor_checkpoint_path,
                                                            hidden_dim=512, 
                                                            num_layers=4
                                                            )
    regress_predictions = regress_predictions.squeeze(0).cpu().detach().numpy()
    return regress_predictions, base_case_points_t1, regress_loss

def get_transforms(oas_folder="server",
        base_case_id = "00000000",
        template_case_id = "00000000",
        base_oas_file_path = "server/00000000.oas",
        init_ae_checkpoint_path = "server/inference/init_ae/best_model.pth",
        regressor_checkpoint_path = "server/inference/arch_regressor/best_model.pth"
    ) -> Tuple[List[Dict[str, Any]]]:
    
    predictions, base_case_points_t1, loss = get_pediction(
    oas_folder=oas_folder,
    base_case_id = base_case_id,
    template_case_id = template_case_id,
    base_oas_file_path = base_oas_file_path,
    init_ae_checkpoint_path = init_ae_checkpoint_path,
    regressor_checkpoint_path = regressor_checkpoint_path)

    # teeth_transfom_matrices = np.zeros((28, 4, 4), dtype=np.float32)
    teeth_transforms = []

    for tooth_idx, tooth_id in enumerate(dw_teeth_nums14 + up_teeth_nums14):
        
        tooth_points_t1 = base_case_points_t1[tooth_idx] 
        tooth_points_pred = predictions[tooth_idx]  
        
        tansform_matrix = calc_transform_matrix_fr_points(tooth_points_t1, tooth_points_pred)
        tooth_transform = get_transform_from_matrix(tansform_matrix)
        teeth_transforms.extend([tooth_transform])    
    return teeth_transforms, loss
    

if __name__ == "__main__":
    # Example usage
    teeth_transforms, loss = get_transforms(oas_folder="server",
        base_case_id = "00000000",
        template_case_id = "00000000",
        base_oas_file_path = "server/00000000.oas",
        init_ae_checkpoint_path = "server/inference/init_ae/best_model.pth",
        regressor_checkpoint_path = "server/inference/arch_regressor/best_model.pth"
    )
    # print("Transforms:", teeth_transforms)cls
    # print("Predictions shape:", predictions.shape)
    # print("base_case_points_t1 shape:", base_case_points_t1.shape)
    # print("Predictions ", predictions)
    # print("Loss:", loss)
    