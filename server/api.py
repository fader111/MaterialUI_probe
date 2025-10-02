from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import glob
# from subprocess import run, PIPE
import subprocess
import multiprocessing
import sys
# from autosetup_ml.utils import *
from server.inference.model_inference import OrthoInferencePipeline
from fastapi import Body
# from backend.ormco import JawType, LandmarkID
import json
# from server.ortho_data import OrthoData
from typing import List, Dict, Any
import numpy as np
# from starlette.concurrency import run_in_threadpool
# from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from scipy.spatial.transform import Rotation as R

app = FastAPI()

EXE_PATH = r"E:\WebGLServer\orthoplatform\Build\windows-msbuild-cl\Bin\OASDatabase\Release\OASDatabase.exe"

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OAS_DIR = os.path.join(os.path.dirname(__file__), "./")
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "../public/")
MESH_DIR = os.path.join(os.path.dirname(__file__), "../public/meshes/")
ROOTS_DIR = os.path.join(os.path.dirname(__file__), "../public/roots/")
SHORTROOTS_DIR = os.path.join(os.path.dirname(__file__), "../public/shortRoots/")

# Top-level function for multiprocessing
def run_and_store(idx, job, ret_dict):
    result = subprocess.run(job, capture_output=True, text=True)
    ret_dict[idx] = {
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    }

# Top-level function for multiprocessing
def run_job(job):
    return subprocess.run(job, capture_output=True, text=True)

class ExportTeethRequest(BaseModel):
    filename: str

@app.post("/oas-files/upload") # copy oas to server folder 
def upload_oas_file(file: UploadFile = File(...)):
    dest = os.path.join(OAS_DIR, file.filename)
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.post("/export-teeth-and-data/")
def export_teeth(request: ExportTeethRequest):
    filename = request.filename
    oas_path = os.path.join(OAS_DIR, filename)
    print(f"oas_path {oas_path}")
    if not os.path.isfile(oas_path):
        raise HTTPException(status_code=404, detail="OAS file not found")
    orthoDataFilePath = os.path.join(PUBLIC_DIR, "orthoData.json")
    jobs = [
        [
            EXE_PATH,
            "ExportTeethSurfaces",
            oas_path,
            PUBLIC_DIR
        ],
        [
            EXE_PATH,
            "ExportStagingData",
            oas_path,
            orthoDataFilePath
        ]
    ]

    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(run_job, jobs)

    errors = [r.stderr for r in results if r.returncode != 0]
    if errors:
        return JSONResponse(status_code=500, content={"error": "\n".join(errors)})
    return {"status": "ok", "output": [r.stdout for r in results]}

@app.post("/get_case_data/")
async def get_case_data(base_case_id: str = Body(..., embed=True)):
    """
    Returns case data including staging and tooth transforms.
    Accepts: { base_case_id: str }
    Reads orthoData.json and returns its contents (does not generate it).
    """
    orthoDataFilePath = os.path.join(PUBLIC_DIR, "orthoData.json")
    if not os.path.isfile(orthoDataFilePath):
        raise HTTPException(status_code=404, detail="orthoData.json not found")
    try:
        with open(orthoDataFilePath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"ERROR: Failed to load orthoData.json: {e}")
        raise HTTPException(status_code=500, detail="Failed to load case data")

@app.post("/predict-t2/")
def predict_t2(
    base_case_id: str = Body(...),
    template_case_id: str = Body(...),
    # template_transforms: Optional[Dict[str, Any]] = Body(None, embed=True)
    template_transforms = Body(None, embed=True)
):
    # print(f"template_transforms {json.dumps(template_transforms)}")
    # base_case_path = os.path.join("server", f"{base_case_id}.oas")
    template_case_path = os.path.join("server", f"{template_case_id}_orthoData.json")
    ae_ckpt = "server/inference/init_ae/best_model.pth"
    reg_ckpt = "server/inference/arch_regressor/best_model.pth"
    # reg_ckpt = "server/inference/arch_regressor/best_model_1500.pth"
    # reg_ckpt = "server/inference/arch_regressor/best_model_template_diff.pth" # for difference mode
    pipeline = OrthoInferencePipeline(ae_ckpt, reg_ckpt)
    result = pipeline.run_t2_predict(template_case_path, template_transforms)
    return result

@app.post("/predict-init/")
def predict_init(
    base_case_id: str = Body(..., embed=True)
):
    base_case_path = os.path.join("server", f"{base_case_id}.oas")
    ae_ckpt = "server/inference/init_ae/best_model.pth"
    pipeline = OrthoInferencePipeline(ae_ckpt)
    result = pipeline.run_init_predict()
    return result
