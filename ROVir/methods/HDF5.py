import h5py


def extract_hf5_data(filename):
    with h5py.File(filename, "r") as f:
        group_key = list(f.keys())[1]

        data = f[group_key][:]

    return data
