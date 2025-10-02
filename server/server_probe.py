"""probe with oasdatabase in multiple processes - takes about 5 seconds """
# E:\WebGLServer\orthoplatform\Build\windows-msbuild-cl\Bin\OASDatabase\Release\OASDatabase.exe ExportStagingData D:\projectsd\MaterialsUI_probe\server\100310.oas E:\outSurfs\json.json
# E:\WebGLServer\orthoplatform\Build\windows-msbuild-cl\Bin\OASDatabase\Release\OASDatabase.exe ExportTeethSurfaces D:\projectsd\MaterialsUI_probe\server\100310.oas E:\outSurfs

import multiprocessing
import subprocess
import os
import time

base_case_id = "100310"
base_case_id = "00000000"

EXE_PATH = r"E:\WebGLServer\orthoplatform\Build\windows-msbuild-cl\Bin\OASDatabase\Release\OASDatabase.exe"
OAS_DIR = os.path.join(os.path.dirname(__file__), "./")
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "../public/")
MESH_DIR = os.path.join(os.path.dirname(__file__), "../public/meshes/")
ROOTS_DIR = os.path.join(os.path.dirname(__file__), "../public/roots/")
SHORTROOTS_DIR = os.path.join(os.path.dirname(__file__), "../public/shortRoots/")

def run_oasdatabase(command_args):
    """Run OASDatabase.exe with the given arguments."""
    if not os.path.isfile(command_args[0]):
        print(f"Executable not found: {command_args[0]}")
        return
    result = subprocess.run(
        command_args, 
        capture_output=True, 
        text=True,
        # stdout=subprocess.DEVNULL,   # Suppress stdout
        # stderr=subprocess.DEVNULL    # Suppress stderr (optional)
        )
    # print(f"Command: {' '.join(command_args)}")
    # print(f"Return code: {result.returncode}")
    # print(f"Output: {result.stdout}")
    # print(f"Error: {result.stderr}")

if __name__ == "__main__":
    # Example parameter sets (customize as needed)
    testDirDataExport = r"E:\outSurfs"
    oas_path = os.path.join(OAS_DIR, f"{base_case_id}.oas")
    orthoDataFilePath = os.path.join(testDirDataExport, "orthoData.json")
    jobs = [
        [
            EXE_PATH,
            "ExportStagingData",
            oas_path,
            orthoDataFilePath
        ],
        [
            EXE_PATH,
            "ExportTeethSurfaces",
            oas_path,
            testDirDataExport
        ],
        # Add more jobs as needed
    ]

    start_time = time.time()
    processes = []
    for job in jobs:
        p = multiprocessing.Process(target=run_oasdatabase, args=(job,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    end_time = time.time()

    print(f"Total execution time multiprocessing for case {base_case_id} : {end_time - start_time:.2f} seconds")

    # Sequential execution for comparison
    seq_start_time = time.time()
    for job in jobs:
        run_oasdatabase(job)
    seq_end_time = time.time()
    print(f"Total execution time sequential for case {base_case_id} : {seq_end_time - seq_start_time:.2f} seconds")


