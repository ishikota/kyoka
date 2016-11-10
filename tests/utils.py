import os

def generate_tmp_dir_path(script_path):
    return os.path.join(os.path.dirname(script_path), "tmp")

def setup_tmp_dir(script_path):
    os.mkdir(generate_tmp_dir_path(script_path))

def teardown_tmp_dir(script_path, file_names):
    dir_path = generate_tmp_dir_path(script_path)
    if os.path.exists(dir_path):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            if os.path.exists(file_path): os.remove(file_path)
        os.rmdir(dir_path)

