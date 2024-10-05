import numpy as np
import torch
import pandas as pd
import os
import time
import argparse

import toml 
import json
from omegaconf import OmegaConf
from multiprocessing import Pool,Process
import subprocess
import platform

from src.trainer.seq import SequentialTrainer
from src.trainer.stat import StaticTrainer

def parse_args(parser):
    # Trainer setup
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test',  action='store_true')
    parser.add_argument("--trainer",   type=str, default="poisson")
    parser.add_argument("--use_data_driven", action='store_true', help="use data driven trainer")
    parser.add_argument("--seed",      type=int, default=42,  help="random seed")
    parser.add_argument("--use_variance_test", action='store_true', help="use variance test")
    parser.add_argument("--test_index", type=int, default=0, help="test index for variance test")
    parser.add_argument("--device", type=str, default="cpu", help="device: cpu or cuda:id")
    parser.add_argument("--fast_residual", action='store_true', help="use fast residual")
    parser.add_argument("--use_coord_feat", action='store_true', help="use coordinate features")
    
    # Datasetup
    parser.add_argument("--n_samples", type=int, default=256, help="number of samples")
    parser.add_argument("--n_c", type=int, default=1, help="number of wave speed")
    parser.add_argument("--c0_lim", nargs="*", type=float, default=[1.0,2.5], help="Gaussian Sampling: range of wave speed")
    parser.add_argument("--v_lim",  nargs="*", type=float, default=[1.0,2.5], help="Gaussian Sampling: range of wave speed")
    parser.add_argument("--batch_size",type=int, default=256, help="size of the batches")
    parser.add_argument("--window_size", type=int, default=4, help="window size for long term trainer")
    parser.add_argument("--use_fem_start", action='store_true', help="For LongTermTrainer use fem as starting frames, else the starting frames will be u0,u0,u0,u0")
    parser.add_argument("--n_detector", type=float, default=0.1, help="rate of label for physical data driven trainer")
    ## Mesh Generation
    parser.add_argument("--use_analytical_dataset",  action='store_true', help="use analytical dataset otherwise use random dataset")
    parser.add_argument("--inverse_dataset", type=str, default="gaussian", choices=["analytical", "gaussian", "linear_layer"], help="inverse problem dataset")
    parser.add_argument("--element", type=str, default="tri", 
                        choices=["tri","quad"])
    parser.add_argument("--shape",   type=str, default="rectangle", 
                        choices=["rectangle", "circle", "ellipse", "L_shape"])
    parser.add_argument("--xlims",  nargs='*', type=float, default=None, help="x boundary for rectangle, ")
    parser.add_argument("--ylims",  nargs='*', type=float, default=None, help="y boundary for rectangle, ")
    parser.add_argument("--radius", nargs='*', type=float, default=None, help="radius for circle and ellipse")
    parser.add_argument("--center", nargs='*', type=float, default=None, help="center for circle and ellipse")
    parser.add_argument("-dv", "--chara_length", type=float, default=0.1, help="characteristic length of the mesh")
    parser.add_argument("--grid",   type=int, default=32, help="number of grid points for each axis for quad mesh")
    parser.add_argument("--use_free_boundary", action="store_true", help="use free boundary condition other wise it's zero boundary")
    parser.add_argument("--verbose", action="store_true", help="verbose for mesh generation")
    parser.add_argument("--no_use_dense", action="store_true", help="How to convert a mesh to a graph")
    
    # Training setup
    parser.add_argument("--epoch",     type=int, default=100, help="number of epochs of training")
    parser.add_argument("--discount_factor", type=float, default=1, help="discount factors for long term trainer")
    parser.add_argument("--validation_ratio", type=float, default=0.2, help="validation ratio for training")
    parser.add_argument("--use_tensorboard", action='store_true', help="use tensorboard")
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--loss_scale", type=float, default=1, help="loss scale to multiply the loss")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam","lbfgs","combine"])
    parser.add_argument("--max_iter", type=int, default=50000, help="lbfgs: max_iter")
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument("--eval_every_eps", type=int, default=2, help="evaluate every n epochs")
    parser.add_argument("--record_c_every_eps", type=int, default=5, help="record c every n epochs for wave equation")
    parser.add_argument("--scheduler", type=str, help="the scheduler", default= "step", choices=["step", "cos", "exp", None])
    parser.add_argument("--scheduler_step_size", type=int, default=2000, help="scheduler step size")
    parser.add_argument("--scheduler_gamma", type=float, default=0.9, help="scheduler gamma")
    parser.add_argument("--n_boost", type=int, default=0, help="number of boosting that to do the multi-stage training")
    parser.add_argument("--use_boost_scale", action="store_true", help="whether to use scaling in the boosting")
    parser.add_argument("--use_label_inplace", action="store_true", help="use label inplace for data driven trainer")
    ## for PINN
    parser.add_argument("--pinn_batch_size", type=int, default=16*4096, help="batch size for PINN")
    parser.add_argument("--pde_sample_ratio", type=float, default=1.0, help="sampling ratio for PINN pde loss")
    parser.add_argument("--bd_sample_ratio", type=float, default=1.0, help="sampmling ratio for PINN boundary loss")
    parser.add_argument("--use_pinn_pretrain", action="store_true", help="use data pretrain for PINN")
    parser.add_argument("--pinn_n_points", type=int, default=1000, help="number of points for PINN")
    parser.add_argument("--lambda_pde", type=float, default=0, help="penalty parameter for PDE loss")
    parser.add_argument("--lambda_sb", type=float, default=0, help="penalty parameter for BC loss")  
    parser.add_argument("--lambda_init", type=float, default=0, help="penalty parameter for IC loss")  

    # Path setup
    parser.add_argument("--ckpt", action= 'store_true', help="load model")
    parser.add_argument("--ckpt_path", type=str, default="model/test.pt", help="path to the checkpoint")
    parser.add_argument("--loss_path", type=str, default="loss/test.png", help="path to the loss")
    parser.add_argument("--result_path", type=str, default="result/test.mp4", help="path to the result")
    parser.add_argument("--database_path", type=str, default=".database/database.csv", help="path to the database")
    ## Plotting 
    parser.add_argument("--no_plot", action='store_true', help="not to plot the mesh after the test")
    parser.add_argument("--fix_colorbar", action='store_true', help="fix colorbar for heat and wave")
    
    # PDE Equation setup
    parser.add_argument("--K",  type=int, default=4, help="frequency for analytical dataset")
    parser.add_argument("--dt", type=float, default=0.01, help="time step size")
    parser.add_argument("--t",  type=float, default=0.1, help="total time")
    parser.add_argument("--a",  type=float, default=1.0, help="wave speed for heat equation")
    parser.add_argument("--c",  type=float, default=1.0, help="wave speed for wave equation")
    parser.add_argument("--predict_steps", type=int, default=10, help="number of predict time steps to time the train steps for heat and wave")
    parser.add_argument("--epsilon", type=float, default= 22, help="epsilon for AC equation")
    parser.add_argument("--kurtosis", action='store_true', help="use kurtosis for AC equation optimization")
    parser.add_argument("--boundary_value", type=float, default=0, help="boundary value for wave and heat equation")
    parser.add_argument("--boundary_value_range", type=float, nargs='*', default=None, help="boundary value pair for wave and heat equation")
    parser.add_argument("--boundary_value_seed", type=int, default=42, help="random seed for boundary value")
    parser.add_argument("--dirch_boundary_type", type=str, default="const", choices=["const", "random", "linear"], help="boundary type for dirichlet boundary condition")
    
    # Implement Model Architectures
    parser.add_argument("--gnn", 
                        # choices=["gcn", "gat", "sage", "sgc", "gin", "sign", "mlp", "chebnet", "graphunet", "appnp"], default="gcn",
                        help="gnn type, if you want to combine models, use '+' to connect them")
    parser.add_argument("--encoder", default="identity", choices=["identity","mlp", "freq", "cnn","rnn","gru","lstm","lem"])
    parser.add_argument("--decoder", default="identity", choices=["identity","mlp", "freq"])
    parser.add_argument("--n_hidden", type=int, default = 64,
                        help="number of hidden neurons")
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--dropout_in', type=float, default=0.0, help='input dropout rate')
    parser.add_argument('--dropout',    type=float, default=0.0, help='dropout rate')
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--num_hops",   type=int, default=3, help="number of hops for SIGN and SGC")
    parser.add_argument("--num_order",  type=int, default=2, help="number of orders for ChebNet")
    parser.add_argument("--alpha",      type=float, default=0.1, help="alpha for APPNP")
    parser.add_argument("--pool_ratio", type=float, default=0.5, help="pooling ratio for GraphUNet")
    parser.add_argument("--encoder_n_layers", type=int, default=2, help="number of layers for encoder")
    parser.add_argument("--encoder_frequency", type=int, default=1, help="used for freq, the frequency of the input")
    parser.add_argument("--encoder_use_bn", action='store_true', help="for mlp and freq, whether to use batch normalization")
    parser.add_argument("--encoder_use_res", action='store_true', help="for mlp and freq, whether to use resnet connection")
    parser.add_argument("--decoder_n_layers", type=int, default=2, help="number of layers for decoder")
    parser.add_argument("--decoder_frequency", type=int, default=1, help="used for FrequencyMLPDecoder, the frequency of the output")
    parser.add_argument("--decoder_use_bn", action='store_true', help="for mlp and freq, whether to use batch normalization")
    parser.add_argument("--decoder_use_res", action='store_true', help="for mlp and freq, whether to use resnet connection")
    parser.add_argument("--decoder_kernel_size", type=int, default=3, help="used for CNN decoder")
    parser.add_argument("--num_heads", type=int, default=4, help="number of heads for GAT")
    parser.add_argument("--use_input_norm", action='store_true', help="use input normalization")
    parser.add_argument("--lambda_u", type=float, default=0.5, help="penalty parameter for PINN")
    
    args = parser.parse_args()

    assert args.train or args.test, "Please specify --train or --test"

    return args

class FileParser:
    def __init__(self, filename):
        if filename.endswith(".toml"):
            with open(filename) as f:
                self.kwargs = OmegaConf.load(f)
        elif filename.endswith(".json"):
            with open(filename) as f:
                self.kwargs = OmegaConf.load(f)
        else:
            raise NotImplementedError(f"File type {filename} not supported, currently only toml and json are supported.")
        
    def add_argument(self, *args, **kwargs):

        for arg in args:
            if arg.startswith("--"):
                arg = arg[2:]
            if arg not in self.kwargs:
                if "action" in kwargs:
                    self.kwargs[arg] = False
                else:
                    self.kwargs[arg] = kwargs.get("default", None)
   
    def parse_args(self):
        return argparse.Namespace(**self.kwargs)

def parse_cmd():
    parser = argparse.ArgumentParser()
    return [parse_args(parser)], True

def parse_files():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None, help="config file path")
    parser.add_argument("-f", "--folder", type=str, default=None, help="folder path")
    parser.add_argument("--debug", action="store_true", help="debug mode, to dispalce multiprocessing")
    parser.add_argument("--num_works_per_device", type=int, default=10, help="number of works per device")
    parser.add_argument("--visible_devices", nargs='*', type=int, default=None, help="visible devices")
    args = parser.parse_args()
    assert args.config or args.folder, "Please specify --config or --folder"
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_devices))
    if args.config:
        args.arg_files = [args.config]
    else:
        args.arg_files = []
        for root, dirs, files in os.walk(args.folder):
            for name in files:
                if name.endswith(".toml") or name.endswith(".json"):
                    args.arg_files.append(os.path.join(root, name))
    return args

def run_arg(arg):
    # make sure all paths are exist
    basepath = os.path.dirname(os.path.abspath(__file__))
    for _path in ["ckpt_path", "loss_path", "result_path", "database_path"]:
        if os.path.isabs(arg.path[_path]):
            continue
        _abspath = os.path.join(basepath, arg.path[_path])
        _dirpath = os.path.dirname(_abspath)
        # make sure the path directory exist
        if not os.path.exists(_dirpath):
            os.makedirs(_dirpath)
        # turn the relative path to abs path 
        arg.path[_path] = _abspath
    arg.datarow = vars(arg).copy()
    arg.datarow['nbytes'] = -1
    arg.datarow['nparams'] = -1
    arg.datarow['p2r edges'] = -1
    arg.datarow['r2r edges'] = -1
    arg.datarow['r2p edges'] = -1
    arg.datarow['training time'] = np.nan
    arg.datarow['inference time'] = np.nan
    arg.datarow['time']    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    arg.datarow['relative error (poseidon_metric)'] = np.nan
    Trainer = {
        "seq": SequentialTrainer,
        "stat": StaticTrainer,
    }[arg.setup["trainer_name"]]
    t = Trainer(arg)
    if arg.setup["train"]:
        if arg.setup["ckpt"]:
            t.load_ckpt()
        t.fit()
    if arg.setup["test"]:
        t.load_ckpt()
        if arg.setup["use_variance_test"]:
            t.variance_test()
        else:
            t.test()


    if os.path.exists(arg.path["database_path"]):
        database = pd.read_csv(arg.path["database_path"])
    else:
        database = pd.DataFrame(columns=arg.datarow.keys())
    database.loc[len(database)] = arg.datarow
    database.to_csv(arg.path["database_path"], index=False)

    return t

def run_arg_file_popen_handle(arg_file):
    command = f"python main.py -c {arg_file}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode == 0:
        print(f"Job {arg_file}: {out.decode('utf-8').strip()}")
    else:
        print(f"Job {arg_file} error: {err.decode('utf-8').strip()}")

def run_arg_files(arg_files, is_debug, num_works_per_device=3):
    if len(arg_files) == 1:
        run_arg(FileParser(arg_files[0]).parse_args())
        #run_arg(parse_args(FileParser(arg_files[0])))
    elif is_debug:
        for arg_file in arg_files:
            print("\n")
            print(arg_file, end="\n\n\n")
            run_arg(parse_args(FileParser(arg_file)))
    elif platform.system() == "Windows":
        processes = []
        for arg_file in arg_files:
            arg = parse_args(FileParser(arg_file))
            p = Process(target=run_arg, args=(arg,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif platform.system() == "Linux":
        num_devices = torch.cuda.device_count()
        processes = {"cpu":[]}
        for i in range(num_devices):
            processes[f"cuda:{i}"] = []
        for arg_file in arg_files:
            arg = parse_args(FileParser(arg_file))
            p = Process(target=run_arg_file_popen_handle, args=(arg_file,))
            if arg.device.startswith("cuda"):
                device_id = int(arg.device[-1])
                processes[f"cuda:{device_id}"].append(p)
            else:
                processes["cpu"].append(p)
        
        max_jobs = max([len(v) for k,v in processes.items()])
        max_runs = (max_jobs + num_works_per_device - 1)  // num_works_per_device
        for i in range(max_runs):
            for k, v in processes.items():
                for p in v[i*num_works_per_device:(i+1)*num_works_per_device]:
                    p.start()
            for k, v in processes.items():
                for p in v[i*num_works_per_device:(i+1)*num_works_per_device]:
                    p.join()
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported")


if __name__ == '__main__':
    config = parse_files()
    run_arg_files(config.arg_files, config.debug, config.num_works_per_device)