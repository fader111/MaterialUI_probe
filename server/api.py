from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import glob
from subprocess import run, PIPE
import sys
from server.inference.model_inference import OrthoInferencePipeline
from fastapi import Body

from autosetup_ml.utils import *
from typing import List, Dict, Any
import numpy as np
from starlette.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OAS_DIR = os.path.join(os.path.dirname(__file__), "./")
MESH_DIR = os.path.join(os.path.dirname(__file__), "../public/meshes/")
ROOTS_DIR = os.path.join(os.path.dirname(__file__), "../public/roots/")
SHORTROOTS_DIR = os.path.join(os.path.dirname(__file__), "../public/shortRoots/")

ortho_case_cache = {
    "file_path": None,
    "ortho_case": None
}

def get_cached_ortho_case(file_path="backend/oas/00000000.oas"):
    if ortho_case_cache["ortho_case"] is None or ortho_case_cache["file_path"] != file_path:
        ortho_case_cache["file_path"] = file_path
        ortho_case_cache["ortho_case"] = OrthoCase(file_path)
    return ortho_case_cache["ortho_case"]

@app.get("/oas-files/")
def list_oas_files():
    files = [os.path.basename(f) for f in glob.glob(os.path.join(OAS_DIR, "*.oas"))]
    return {"oas_files": files}

@app.get("/oas-files/{filename}")
def get_oas_file(filename: str):
    path = os.path.join(OAS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

@app.post("/oas-files/upload")
def upload_oas_file(file: UploadFile = File(...)):
    dest = os.path.join(OAS_DIR, file.filename)
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.get("/meshes/{tooth_id}.stl")
def get_mesh(tooth_id: str):
    path = os.path.join(MESH_DIR, f"{tooth_id}.stl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Mesh not found")
    return FileResponse(path, media_type="application/sla")

@app.get("/roots/{tooth_id}.stl")
def get_root(tooth_id: str):
    path = os.path.join(ROOTS_DIR, f"{tooth_id}.stl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Root not found")
    return FileResponse(path, media_type="application/sla")

@app.get("/shortRoots/{tooth_id}.stl")
def get_short_root(tooth_id: str):
    path = os.path.join(SHORTROOTS_DIR, f"{tooth_id}.stl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Short root not found")
    return FileResponse(path, media_type="application/sla")

class ExportTeethRequest(BaseModel):
    filename: str

@app.post("/export-teeth/")
def export_teeth(request: ExportTeethRequest):
    filename = request.filename
    oas_path = os.path.join(OAS_DIR, filename)
    if not os.path.isfile(oas_path):
        raise HTTPException(status_code=404, detail="OAS file not found")
    # Run the export teeth and json data script
    result = run([
        sys.executable, os.path.join(os.path.dirname(__file__), "export_teeth_and_json_data.py"),
        "--filename", oas_path
    ], stdout=PIPE, stderr=PIPE, text=True)
    if result.returncode != 0:
        return JSONResponse(status_code=500, content={"error": result.stderr})
    return {"status": "ok", "output": result.stdout}

@app.post("/predict-t2/")
def predict_t2(
    base_case_id: str = Body(...),
    template_case_id: str = Body(...)
):
    base_case_path = os.path.join("server", f"{base_case_id}.oas")
    template_case_path = os.path.join("server", f"{template_case_id}.oas")
    ae_ckpt = "server/inference/init_ae/best_model.pth"
    reg_ckpt = "server/inference/arch_regressor/best_model.pth"
    pipeline = OrthoInferencePipeline(ae_ckpt, reg_ckpt)
    result = pipeline.run_t2_predict(base_case_path, template_case_path)
    return result

@app.post("/predict-init/")
def predict_init(
    base_case_id: str = Body(..., embed=True)
):
    base_case_path = os.path.join("server", f"{base_case_id}.oas")
    ae_ckpt = "server/inference/init_ae/best_model.pth"
    pipeline = OrthoInferencePipeline(ae_ckpt)
    result = pipeline.run_init_predict(base_case_path)
    return result

@app.post("/get_teeth_meshes/")
async def get_teeth_meshes(payload: dict = Body(...)):
    """
    Returns mesh data for multiple teeth in a single batch request.
    Accepts: { tooth_ids: [int], file_path: str (optional) }
    """
    tooth_ids = payload["tooth_ids"]
    file_path = payload.get("file_path", "backend/oas/00000000.oas")
    ortho_case = get_cached_ortho_case(file_path)
    def process_one(tooth_id):
        crown_vertices, crown_faces = ortho_case.get_crown_vertices_faces(int(tooth_id))
        crown_vertices, crown_faces = ortho_case.convert_expanded_mesh_to_standard(crown_vertices, crown_faces)
        root_vertices, root_faces = ortho_case.get_root_vertices_faces(int(tooth_id))
        root_vertices, root_faces = ortho_case.convert_expanded_mesh_to_standard(root_vertices, root_faces)
        short_root_vertices, short_root_faces = ortho_case.get_short_root_vertices_faces(int(tooth_id))
        short_root_vertices, short_root_faces = ortho_case.convert_expanded_mesh_to_standard(short_root_vertices, short_root_faces)
        return {
            "crown": {"vertices": crown_vertices, "faces": crown_faces},
            "root": {"vertices": root_vertices, "faces": root_faces},
            "short_root": {"vertices": short_root_vertices, "faces": short_root_faces}
        }

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_one, tooth_ids))
    return dict(zip(tooth_ids, results))
