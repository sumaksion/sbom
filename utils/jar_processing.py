import os
import subprocess
import zipfile
import shutil
import re


def process_lib_files(input_dir, detecting=True):
    joern_workspace = os.path.join(input_dir, 'workspace')
    out_dirs = []
    if not os.path.isdir(input_dir):
        raise ValueError(f"'{input_dir}' does not exist.")
    
    processed_dir = os.path.join(input_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith('.aar'):
            aar_path, aar_name = generate_new_name(input_dir, file)
            new_jar_path = os.path.join(input_dir, f"{aar_name}.jar")
            if os.path.exists(new_jar_path):
                continue

            with zipfile.ZipFile(aar_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    if member.filename.endswith('.jar'):
                        extracted_path = zip_ref.extract(member, input_dir)
                        os.rename(extracted_path, new_jar_path)
                        os.remove(aar_path)

        if file.endswith('.dex'):
            dex_path, dex_name = generate_new_name(input_dir, file)
            new_jar_path = os.path.join(input_dir, f"{dex_name}.jar")
            convert_dex_to_jar(dex_path, new_jar_path)
            processed_dir = os.path.join(input_dir, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            move_file_after_processing(file, input_dir, processed_dir)


    jar_files = [f for f in os.listdir(input_dir) if f.endswith('.jar')]

    for jar_file in jar_files:
        if detecting is False:
            if "-" in jar_file:
                jar_name, jar_number = jar_file.rsplit("-", 1)
                jar_number = os.path.splitext(jar_number)[0]
            else:
                print(f"Skipping '{jar_file}' as it doesn't follow the 'libname-libnumber' pattern.")
                bad_lib_name_dir = os.path.join(input_dir, 'incorrect_name_format')
                os.makedirs(bad_lib_name_dir, exist_ok=True)
                move_file_after_processing(jar_file, input_dir, bad_lib_name_dir)
                continue

            lib_dir = os.path.join(joern_workspace, jar_name)
            sub_dir = os.path.join(lib_dir, jar_number)

            if os.path.exists(os.path.join(sub_dir, 'out', '0-ast.dot')):
                out_dirs.append((os.path.join(sub_dir, 'out'), jar_name + '-' + jar_number))
                continue

        else:
            jar_name = os.path.splitext(jar_file)[0]
            lib_dir = os.path.join(joern_workspace, jar_name)
            sub_dir = lib_dir

            if os.path.exists(os.path.join(sub_dir, 'out', '0-ast.dot')):
                out_dirs.append((os.path.join(sub_dir, 'out'), jar_name))
                continue


        script = os.path.expanduser('~/sbom/utils/joern.sc')
        param_path = f"filePath={jar_file}"
        param_name = f"libName={jar_name}"

        joern_cli = os.path.expanduser("~/bin/joern/joern-cli/")
        joern_path = os.path.join(joern_cli, "joern")
        joern_export_path = os.path.join(joern_cli, "joern-export")

        command = f"{joern_path} --script {script} --param '{param_path}' --param '{param_name}'"
        try:
            subprocess.run(command, shell=True, cwd=input_dir, check=True)
            print(f"Exporting ASTs for '{jar_file}'...")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=False)
            export_dir = os.path.join(jar_number, "out") if detecting == False else 'out'
            #os.makedirs(os.path.join(lib_dir, export_dir), exist_ok=True)
            export_command = f"{joern_export_path} --repr ast --out {export_dir}"
            #cwd = lib_dir if detecting = False else 
            process_code = subprocess.run(export_command, cwd=lib_dir, check=True, shell=True)
            if process_code.returncode == 0:
                if detecting is False:
                    clean_up_lib_dir(lib_dir, sub_dir)
                    out_dirs.append((os.path.join(sub_dir, export_dir), jar_name + '-' + jar_number))
                else: out_dirs.append((os.path.join(sub_dir, export_dir), jar_name))
                print(f"Finished processing '{jar_file}'. ASTs in '{export_dir}'.")

        except subprocess.CalledProcessError as e:
            print(f"Error processing '{jar_file}' with return code {e.returncode}. Cleaning up...")
            shutil.rmtree(sub_dir, ignore_errors=True)
            if os.path.exists(lib_dir) and not os.listdir(lib_dir):
                os.rmdir(lib_dir)
    

    return out_dirs


def generate_new_name(input_dir, file):
    path = os.path.join(input_dir, file)
    name = os.path.splitext(file)[0]
    return path, name


def convert_dex_to_jar(dex_path, jar_path, dex2jar_path=os.path.expanduser('~/bin/dex-tools-v2.4/d2j-dex2jar.sh')):
    try:
        result = subprocess.run(
            [dex2jar_path, dex_path, "-o", jar_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"Error converting {dex_path}:\n{result.stderr}")
    except Exception as e:
        print(f"Exception while converting {dex_path}: {e}")


def clean_up_lib_dir(lib_dir, sub_dir):
    """Removes all files and folders in `lib_dir` except for `sub_dir` and those that match the pattern x.x.x."""
    # Define the pattern for matching versions like '1.2.3', '10.20.30', etc.
    version_pattern = re.compile(r'^\d+(\.\d+)*$')
    
    for item in os.listdir(lib_dir):
        item_path = os.path.join(lib_dir, item)
        
        # Skip the sub_dir
        if item_path == sub_dir:
            continue
        
        # Check if the item name matches the version pattern
        if not version_pattern.match(item) or item == 'out':
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)

def move_file_after_processing(file, input_dir, processed_dir):
    source_file = os.path.join(input_dir, file)
    destination_file = os.path.join(processed_dir, file)
    if os.path.exists(source_file) and not os.path.isdir(source_file):
        shutil.move(source_file, destination_file)
