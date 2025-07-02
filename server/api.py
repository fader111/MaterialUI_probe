from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import glob
from subprocess import run, PIPE
import sys
from server.inference.model_inference import get_pediction, get_predicted_transforms
from fastapi import Body

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
    # pred, loss = get_transforms(
    pred = get_predicted_transforms(
        base_case_id=base_case_id,
        template_case_id=template_case_id
    )
    # Convert pred (numpy or torch tensor) to list for JSON serialization
    # print(f"DEBUG: base_case_id={base_case_id}, template_case_id={template_case_id}")
    # print(f"DEBUG: pred={pred}, loss={loss}")
    # print(f"DEBUG: pred={pred}, loss={loss}")
    # return {"prediction": pred, "loss": loss}
    return pred

