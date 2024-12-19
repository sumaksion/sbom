import os

def find_lib_dot_files(src_dir, search_string, max_size, min_size):

    file_paths = []
    for file_name in os.listdir(src_dir): 
        if file_name.endswith('.dot'):
            file_path = os.path.join(src_dir, file_name)
            if min_size < os.path.getsize(file_path) // 1024 < max_size:
                with open(file_path, 'r', encoding='utf-8') as file:
                    if search_string in file.read():
                        file_paths.append(file_path)
    return file_paths


