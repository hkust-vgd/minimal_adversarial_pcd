import os
import argparse
from datetime import datetime
import datasets
from attack import Attack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="minimual adversarial attack on point cloud"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use [default: GPU 0]"
    )

    # model
    parser.add_argument(
        "--model_name",
        default="pointnet_cls",
        help="default: pointnet_cls",
    )
    parser.add_argument(
        "--model_code_path", default="./pointnet/models/pointnet_cls.py"
    )
    parser.add_argument(
        "--ckpt_path",
        default="./scanobnn/obj_bg/model.ckpt",
        help="write your model path",
    )

    # dataset
    parser.add_argument("--dataset_type", default="scanobnn")
    parser.add_argument(
        "--data_path",
        default="./scanobnn/test_objectdataset.h5",
        help="Location of test file",
    )
    parser.add_argument(
        "--class_names", default="./scanobnn/shape_names.txt"
    )
    parser.add_argument(
        "--num_class",
        type=int,
        default=15,
        help="Number of classes to classify. [modelent40:40, scanobnn:15]",
    )
    parser.add_argument(
        "--num_point",
        type=int,
        default=1024,
        help="Point Number [256/512/1024/2048] [default: 1024]",
    )

    parser.add_argument(
        "--with_bg",
        default=True,
        help="Whether to have background or not [default: True]",
    )
    parser.add_argument(
        "--normalized",
        default=True,
        help="Whether to normalize data or not [default: False]",
    )
    parser.add_argument(
        "--center_data",
        default=True,
        help="Whether to explicitly center the data [default: False]",
    )

    # log, result
    parser.add_argument("--save_dir", default="./result")
    parser.add_argument("--save_ply", default=True)

    # attack parameter
    parser.add_argument(
        "--attack_type", help="perturbation/addition", default="perturbation"
    )
    parser.add_argument("--clip", default=0.01, help="clip bound")
    parser.add_argument(
        "--lr_attack",
        type=float,
        default=0.01,
        help="learning rate for optimization based attack",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=250,
        help="attack optimization iteration",
    )
    parser.add_argument(
        "--h_dist_weight",
        type=float,
        default=50,
        help="lambda for distance loss",
    )
    parser.add_argument(
        "--class_loss_weight",
        type=float,
        default=1,
        help="lambda for classification loss",
    )
    parser.add_argument(
        "--count_weight",
        type=float,
        default=0.15,
        help="lamba for minimum loss",
    )
    args = parser.parse_args()
    print(args)

    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    args.save_dir = os.path.join(args.save_dir, args.attack_type, current_time)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("save_dir: ", args.save_dir)

    data, labels = datasets.dataloder(args)
    attack = Attack(args)
    attack.test(data, labels)
