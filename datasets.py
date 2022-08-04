import numpy as np
import h5py


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f["data"][:]
    label = f["label"][:]
    return data, label


def center_data(pcs):
    for pc in pcs:
        centroid = np.mean(pc, axis=0)
        pc[:, 0] -= centroid[0]
        pc[:, 1] -= centroid[1]
        pc[:, 2] -= centroid[2]
    return pcs


def normalize_data(pcs):
    for pc in pcs:
        # get furthest point distance then normalize
        d = max(np.sum(np.abs(pc) ** 2, axis=-1) ** (1.0 / 2))
        pc /= d

        # pc[:,0]/=max(abs(pc[:,0]))
        # pc[:,1]/=max(abs(pc[:,1]))
        # pc[:,2]/=max(abs(pc[:,2]))

    return pcs


def dataloder(args):
    if args.dataset_type == "modelnet40":
        test_files = getDataFiles(args.data_path)
        data = []
        labels = []

        for file in test_files:
            curr_x, curr_t = load_h5(file)
            data.append(curr_x[:, : args.num_point, :])
            labels.append(np.squeeze(curr_t))
        data = np.concatenate(data)
        labels = np.concatenate(labels)
    else:
        data, labels = load_h5(args.data_path)
        data = data[:, : args.num_point, :]

    if args.center_data:
        center_data(data)

    if args.normalized:
        normalize_data(data)

    print("total data len: ", len(labels))
    return data, labels
