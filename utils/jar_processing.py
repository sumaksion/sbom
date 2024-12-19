import os
import subprocess
import zipfile
import shutil


def process_lib_files(input_dir, joern_workspace="data/jars/workspace", detecting=True):

    out_dirs = []
    if not os.path.isdir(input_dir):
        raise ValueError(f"'{input_dir}' does not exist.")

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
                        new_jar_path = os.path.join(input_dir, f"{aar_name}.jar")
                        os.rename(extracted_path, new_jar_path)

        if file.endswith('.dex'):
            dex_path, dex_name = generate_new_name(input_dir, file)
            new_jar_path = os.path.join(input_dir, f"{dex_name}.jar")
            convert_dex_to_jar(dex_path, new_jar_path)

    jar_files = [f for f in os.listdir(input_dir) if f.endswith('.jar')]

    for jar_file in jar_files:
        if detecting is False:
            if "-" in jar_file:
                jar_name, jar_number = jar_file.rsplit("-", 1)
                jar_number = os.path.splitext(jar_number)[0]
            else:
                print(f"Skipping '{jar_file}' as it doesn't follow the 'libname-libnumber' pattern.")
                continue

            lib_dir = os.path.join(joern_workspace, jar_name)
            sub_dir = os.path.join(lib_dir, jar_number)

            if os.path.exists(sub_dir):
                out_dirs.append((os.path.join(sub_dir, 'out'), jar_name + '-' + jar_number))
                continue
            os.makedirs(sub_dir, exist_ok=False)
        else:
           jar_name = os.path.splitext(jar_file)[0]
           sub_dir = os.path.join(joern_workspace, jar_name)
           if os.path.exists(sub_dir):
               out_dirs.append((os.path.join(sub_dir, 'out'), jar_name))
               continue
           os.makedirs(sub_dir, exist_ok=False)

        script = os.path.expanduser('~/sbom/utils/joern.sc')
        param_path = f"filePath={jar_file}"
        param_name = f"libName={jar_name}"

        joern_cli = os.path.expanduser("~/bin/joern/joern-cli/")
        joern_path = os.path.join(joern_cli, "joern")
        joern_export_path = os.path.join(joern_cli, "joern-export")

        command = f"{joern_path} --script {script} --param '{param_path}' --param '{param_name}'"
        
        subprocess.run(command, shell=True, cwd=os.path.dirname(joern_workspace))

        print(f"Exporting ASTs for '{jar_file}'...")
        if detecting is False:
            export_dir = os.path.join(jar_number, "out")
            cwd = lib_dir
        else:
            export_dir = 'out'
            cwd = sub_dir

        export_command = f"{joern_export_path} --repr ast --out {export_dir}"
        
        try:
            subprocess.run(export_command, cwd=cwd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Command failed for '{jar_file}' with return code {e.returncode}. Cleaning up...")
            if os.path.exists(sub_dir):
                shutil.rmtree(sub_dir)
            if os.path.exists(lib_dir) and not os.listdir(lib_dir):  
                os.rmdir(lib_dir)

        #shutil.move(jar_file, os.path.join(input_dir, 'processed'))
        out_dirs.append((export_dir, jar_name))
        print(f"Finished processing '{jar_file}'. ASTs in '{export_dir}'.")

    return out_dirs

def generate_new_name(input_dir, file):
    path = os.path.join(input_dir, file)
    name = os.path.splitext(file)[0]  
    return path, name

def convert_dex_to_jar(dex_path, jar_path, dex2jar_path=os.path.expanduser('~/dex-tools-v2.4/d2j-dex2jar.sh')):
            
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


"""input_directory = "data/jars"
process_lib_files(input_directory)"""
