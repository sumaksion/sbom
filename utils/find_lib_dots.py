import os

def find_lib_dot_files(src_dirs, search_string, max_size, min_size):

    file_paths = []

    for src_dir in src_dirs:
        if not os.path.isdir(src_dir):
            print(f"Skipping non-existent directory: {src_dir}")
            continue

        for file_name in os.listdir(src_dir):
            if file_name.endswith('.dot'):
                file_path = os.path.join(src_dir, file_name)
                file_size_kb = os.path.getsize(file_path) // 1024
                if min_size < file_size_kb < max_size:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            if search_string in file.read():
                                file_paths.append(file_path)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

    return file_paths


