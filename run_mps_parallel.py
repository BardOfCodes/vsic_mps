import argparse
import json
import os
import _pickle as cPickle

from run_mps import (DATA_DIR, SPLIT_FILE, MODE, 
                      SCALE_FACTOR, RESOLUTION, MATLAB_DIR)
from run_mps import (trimesh_process_mesh, 
                      params_to_geolipi, 
                      inner_loop)

BASE_CSV_PATH = os.path.join(MATLAB_DIR, "temp_{0}.csv")
BASE_MAT_PATH = os.path.join(MATLAB_DIR, "temp_{0}.mat")

def main(args):
    proc_id = args.proc_id
    args.csv_path = args.csv_path.format(proc_id)
    args.mat_file = args.mat_file.format(proc_id)
    args.predictions_file = args.predictions_file.format(proc_id)

    # Load the train mesh files.
    if args.prim_mode == "sq":
        mat_file = "run_mps.m"
        mat_func = "run_mps"
    elif args.prim_mode == "cuboid":
        mat_file = "run_mps_cu.m"
        mat_func = "run_mps_cu"
    # Convert to sdf using mesh2sdf.
    with open(args.split_file, "r") as f:
        split = json.load(f)
    all_models = split[args.mode]

    n_jobs = args.n_jobs
    frac = len(all_models) // n_jobs
    start = proc_id * frac
    end = (proc_id + 1) * frac
    if proc_id == n_jobs - 1:
        end = len(all_models)
    cur_models = all_models[start:end]


    all_exprs, stats = inner_loop(args, mat_func, cur_models)
        # break
    # save it as pkl file
    cPickle.dump(all_exprs, open(args.predictions_file, 'wb'))

    if args.timing_file is None:
        timing_file = args.predictions_file.replace(".pkl", "_timing.json")
    else:
        timing_file = args.timing_file
    with open(timing_file, "w") as f:
        json.dump(stats, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--proc_id", type=int, default=0)
    parser.add_argument("--scale_factor", type=float, default=SCALE_FACTOR)
    parser.add_argument("--res", type=int, default=RESOLUTION)
    parser.add_argument("--prim_mode", type=str, default="sq")
    parser.add_argument("--split_file", type=str, default=SPLIT_FILE)
    parser.add_argument("--mode", type=str, default=MODE)
    parser.add_argument("--csv_path", type=str, default=BASE_CSV_PATH)
    parser.add_argument("--mat_file", type=str, default=BASE_MAT_PATH)
    parser.add_argument("--predictions_file", default='final_expressions_{0}.pkl', type=str)
    parser.add_argument("--timing_file", type=str, default=None)
    args = parser.parse_args()

    main(args)