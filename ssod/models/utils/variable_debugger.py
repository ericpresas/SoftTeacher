import pickle
import torch


def from_tensor_to_numpy(variable):
    return variable.cpu().detach().numpy()


def save_variables_pickle(keys, variable_list, file_path):
    obj_save = {}
    assert len(keys) == len(variable_list)

    for key, variable in zip(keys, variable_list):
        if torch.is_tensor(variable):
            new_variable = from_tensor_to_numpy(variable)
        elif isinstance(variable, list):
            new_variable = [from_tensor_to_numpy(value) for value in variable]
        elif isinstance(variable, dict):
            new_variable = {key:from_tensor_to_numpy(value) for key, value in variable.items()}

        obj_save[key] = new_variable

    with open(file_path, 'wb') as f:
        pickle.dump(obj_save, f)


def load_variables_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj