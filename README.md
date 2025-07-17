# Dental Ortho Visualization & Template-based clinical features prediction App

This application is a full-stack dental visualization and prediction tool built with React, React Three Fiber, and a Python FastAPI backend. It allows users to load orthodontic case files, visualize 3D teeth and roots, interact with the scene, and run ML-based predictions for tooth movement and staging.

---

## Frontend (React + Vite)

### Component Hierarchy & Data Flow

- **App.jsx**
  - Root component. Manages global state: `orthoData`, loading status, and selected case filename. Handles file loading and passes all state down to `Ortho`.
- **Ortho.jsx**
  - Main 3D scene and UI logic. Manages stage selection, camera, controls, and view modes. Handles prediction requests and updates `orthoData`.
  - Passes all relevant props to `Overlay` (UI controls) and `ToothPlacement` (3D scene).
- **Overlay.jsx**
  - UI panel with buttons for file operations, view selection, toggling roots/landmarks, and running predictions. Calls handlers from `Ortho` via props.
- **ToothPlacement.jsx**
  - Renders all teeth for the current stage. Receives `orthoData`, `stage`, and visualization options. Computes and passes transformation data to each `Tooth`.
  - Handles per-tooth interaction and transformation updates.
- **Tooth.jsx**
  - Renders a single tooth (crown and root) as a 3D mesh. Applies transformation, handles STL loading, and displays landmarks. Supports selection and manipulation.

### Data & Interaction Flow
1. **File Load**: User loads a case file via Overlay. `App` fetches case data from backend and updates `orthoData`.
2. **Visualization**: `Ortho` receives new `orthoData`, resets stage, and passes data to `ToothPlacement`.
3. **3D Scene**: `ToothPlacement` maps over all teeth for the current stage, passing transformation and mesh info to each `Tooth`.
4. **User Actions**: User can select teeth, toggle roots/landmarks, change view, or run predictions. All actions update state in `Ortho` and propagate down.
5. **Prediction**: When user runs a prediction, `Ortho` sends a request to the backend, receives new transforms, and updates `orthoData`.

### Key Features
- 3D visualization of teeth, roots, and landmarks
- Stage slider and view controls
- Prediction of tooth movement (T2, Init Predict)
- STL mesh loading with cache-busting
- Landmark and root toggling
- Responsive UI with overlay panels

---

## Backend (Python FastAPI)

### Structure & Responsibilities
- **api.py**: Main FastAPI app. Handles all HTTP endpoints for file management, mesh serving, prediction, and data export.
- **inference/model_inference.py**: ML pipeline for running predictions (autoencoder, regressor, etc). Handles T2 and Init predictions, composes transforms.
- **export_teeth_.py**: Utilities for exporting teeth and case data.
- **utils.py**: Math, geometry, and data utilities for transforms, mesh processing, and landmark extraction.

### Main Endpoints
- `/get_case_data/`: Loads and returns all data for a given case (staging, transforms, landmarks, etc).
- `/predict-t2/`, `/predict-init/`: Run ML predictions for the current case and return new tooth transforms.
- `/meshes/{tooth_id}.stl`, `/roots/{tooth_id}.stl`, `/shortRoots/{tooth_id}.stl`: Serve STL meshes for crowns and roots.
- `/oas-files/`, `/oas-files/upload`: List and upload case files.

### Backend-Frontend Interaction
- The frontend fetches case data and predictions from the backend as JSON.
- Meshes are loaded on demand as STL files via HTTP.
- Prediction endpoints return new transformation data, which is merged into the current case state on the frontend.
- All data is kept in sync via React state and effect hooks.

---

## Typical Workflow
1. User loads a case file (OAS) via the UI.
2. App fetches and displays the 3D scene for the case.
3. User can explore, manipulate, and annotate the scene.
4. User runs predictions (T2, Init) to see ML-generated tooth movements for better T2 using Template as a source of desired clinical features.
5. All changes are visualized in real time, and the user can export results as needed.

---

## Technologies Used
- **Frontend**: React, React Three Fiber, Three.js, Vite, Material UI
- **Backend**: Python, FastAPI, PyTorch (for ML), Numpy, Scipy
- **3D Data**: STL meshes, JSON transforms, landmark data

---

## Build & Run Instructions

### Prerequisites
- Node.js (v18+ recommended)
- Python 3.9+
- pip (Python package manager)

### Frontend (React + Vite)

1. Install dependencies:
   ```sh
   npm install
   ```
2. Start the development server:
   ```sh
   npm run dev
   ```
   The app will be available at http://localhost:5173

### Backend (FastAPI)

1. Create and activate a Python virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/Mac:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```sh
   uvicorn server.api:app --reload
   ```
   The backend will be available at http://localhost:8000

### Usage
- Open the frontend in your browser (http://localhost:5173)
- Load or upload a case file (OAS) via the UI
- Interact with the 3D scene and run predictions

---

For more details, see the code in `src/` (frontend) and `server/` (backend).
