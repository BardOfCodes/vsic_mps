import argparse
import trimesh
import json
import os
import numpy as np
import time
import csv
import mesh2sdf
import scipy
import geolipi.symbolic as gls
import _pickle as cPickle


# DATA_DIR = "/media/aditya/OS/data/compat"
# SPLIT_FILE = "/home/aditya/projects/challenge/vsic/metadata/vsic_split.json"
DATA_DIR = "/users/aganesh8/data/aganesh8/data/compat"
SPLIT_FILE = "/users/aganesh8/data/aganesh8/projects/challenge/vsic/metadata/vsic_split.json"

MODE = "test"
SCALE_FACTOR = 1.75
RESOLUTION = 100
file_path = os.path.dirname(os.path.abspath(__file__))
MATLAB_DIR = os.path.join(file_path, "MATLAB")
BASE_CSV_PATH = os.path.join(MATLAB_DIR, "temp.csv")
BASE_MAT_PATH = os.path.join(MATLAB_DIR, "temp.mat")


def trimesh_process_mesh(model_file):
    data_file = open(model_file, "r")
    gltf_data = json.load(data_file)
    gltf_data.pop('images')
    gltf_str = json.dumps(gltf_data)
    gltf_f = trimesh.util.wrap_as_stream(gltf_str)
    obj = trimesh.load(
        gltf_f,
        file_type=".gltf",
        force="mesh",
    )
    obj.merge_vertices(merge_tex=True, merge_norm=True)

    return obj

def main(args):
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

    all_exprs, stats = inner_loop(args, mat_func, all_models)
        # break
    # save it as pkl file
    cPickle.dump(all_exprs, open(args.predictions_file, 'wb'))

    if args.timing_file is None:
        timing_file = args.predictions_file.replace(".pkl", "_timing.json")
    else:
        timing_file = args.timing_file
    with open(timing_file, "w") as f:
        json.dump(stats, f)

def inner_loop(args, mat_func, all_models):
    all_exprs = []
    stats = []
    for ind, cur_model in enumerate(all_models):
        if ind % 100 == 0:
            print(f"Processing {ind}/{len(all_models)}: {cur_model}")
        model_file = os.path.join(DATA_DIR, 'models', f"{cur_model}.gltf")
        processed_mesh = trimesh_process_mesh(model_file)
        # Convert to Manifold
        vertices = np.asarray(processed_mesh.vertices)
        faces = np.asarray(processed_mesh.faces)
        vertices = vertices * args.scale_factor
        level = 2/args.res
        sdf, mesh = mesh2sdf.compute(
            vertices, faces, args.res, fix=True, level=level, return_mesh=True)

        grid_config = np.array(
            [[args.res], 
            [-1], [1], 
            [-1], [1], 
            [-1], [1]]
        )
        writevoxel = np.reshape(np.swapaxes(sdf, 0, 2), (args.res**3, 1))
        writevoxel = np.append(grid_config, writevoxel).reshape(-1, 1)
        with open(args.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(writevoxel)
        print('SDF saved to ' + args.csv_path + '.')
        # Now run the mps algorithm.
        # change to matlab dir
        cur_dir = os.getcwd()
        os.chdir(MATLAB_DIR)
        param_1 = args.csv_path
        param_2 = args.mat_file
        cmd = f"matlab -r \"try; {mat_func}('{param_1}', '{param_2}'); catch; end; quit;\""
        start = time.time()
        os.system(cmd)
        end = time.time()
        total_time = end - start
        os.chdir(cur_dir)
        # Now load the mat and generate expression.
        params = scipy.io.loadmat(args.mat_file)['x']
        mps_time = scipy.io.loadmat(args.mat_file)['mps_time'].item()
        print(f"Time taken for {cur_model} with matlab loading is {total_time}, and only MPS is {mps_time}")

        stats.append({"model": cur_model, "time": total_time, "mps_time": mps_time})

        expression = params_to_geolipi(params, mode=args.prim_mode)
        expression = expression.cpu().sympy()
        # expression_str = str(expression)
        all_exprs.append(expression)
    return all_exprs, stats


def params_to_geolipi(params, mode="sq"):

    n_primitives = params.shape[0]
    shapes = []
    for i in range(n_primitives):
        cur_param = params[i]

        epsilon_1 = cur_param[0:1]
        epsilon_2 = cur_param[1:2]
        skew_param = cur_param[2:5]
        cuboid_params = cur_param[2:5]
        cuboid_params = tuple(cuboid_params.tolist())
        # Rotation is tricky!
        rotation_params = -cur_param[5:8][::-1]

        translate_params = cur_param[8:11]
        epsilon_1 = tuple(epsilon_1.tolist())
        epsilon_2 = tuple(epsilon_2.tolist())
        skew_param = tuple(skew_param.tolist())
        translate_params = tuple(translate_params.tolist())
        rotation_params = tuple(rotation_params.tolist())

        if mode == "sq":
            expression = gls.InexactSuperQuadrics3D(skew_param, epsilon_1, epsilon_2)
        elif mode == "cuboid":
            expression = gls.Cuboid3D(cuboid_params)

        expression = gls.EulerRotate3D(expression, rotation_params)
        expression = gls.Translate3D(expression, translate_params)
        shapes.append(expression)

    expression = gls.Union(*shapes)
    return expression





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=float, default=SCALE_FACTOR)
    parser.add_argument("--res", type=int, default=RESOLUTION)
    parser.add_argument("--prim_mode", type=str, default="sq")
    parser.add_argument("--split_file", type=str, default=SPLIT_FILE)
    parser.add_argument("--mode", type=str, default=MODE)
    parser.add_argument("--csv_path", type=str, default=BASE_CSV_PATH)
    parser.add_argument("--mat_file", type=str, default=BASE_MAT_PATH)
    parser.add_argument("--predictions_file", default='final_expressions.pkl', type=str)
    parser.add_argument("--timing_file", type=str, default=None)
    args = parser.parse_args()

    main(args)