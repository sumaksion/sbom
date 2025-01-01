import os
import subprocess
import zipfile
import shutil
from utils.batch_baksmali import disassemble_classes_in_directory as baksmali
from detect_lib_by_file_structure import FileStructureLibDetector 

"""
requires a running android device with adb root and running frida server on device

frida-dexdump must be installed
https://github.com/hluwa/frida-dexdump

apk files should be in apk folder


"""
def run_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Success: {result.stdout}")
        else:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run command: {e}")
        return False

def install_apk(apk_path):
    print(f"Installing APK: {apk_path}")
    return run_command(['adb', 'install', apk_path])

def start_app(package_name):
    print(f"Starting app: {package_name}")
    for i in range(9):
        started = run_command(['adb', 'shell', 'monkey', '-p', package_name, '-c', 'android.intent.category.LAUNCHER', '1'])
        if started == True:
            break
    return started

def stop_app(package_name):
    print(f"Stopping app: {package_name}")
    return run_command(['adb', 'shell', 'am', 'force-stop', package_name])

def uninstall_apk(package_name):
    print(f"Uninstalling app: {package_name}")
    return run_command(['adb', 'uninstall', package_name])

def memory_dump(apk_name):
    
    out_dir = os.path.join('dumps', apk_name)
    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run(f'frida-dexdump -d -FU -o "{out_dir}"', shell=True, check=True)
        return out_dir
    except subprocess.CalledProcessError as e:
        print(f"Memory dump failed for {apk_name}: {e}")
        return None
def run_dynamic_detection():

    if os.name == "nt":
        try:
            subprocess.run([
                'wsl',
                '--cd', '~/sbom',
                '~/sbom/.conda/bin/python3', '~/sbom/dynamic_library_detection.py'
            ], check=True)

            return True
        except subprocess.CalledProcessError as e:
            print(e)
            return False
    else:
        try:
            subprocess.run(
                ["python3", os.path.expanduser("~/sbom/dynamic_library_detection.py")],
                cwd=os.path.expanduser("~/sbom"),
                check=True
)
            return True
        except subprocess.CalledProcessError as e:
            print(e)
            return False
  
def copy_directory(apk_name, user):
    source_dir = apk_name
    destination_dir = os.path.expanduser('~/sbom/data/dexs')

    if os.name == "nt":
        destination_dir = os.path.join(r'\\wsl.localhost', 'Ubuntu', 'home', user, 'sbom', 'data', 'dexs')

    destination_dir = os.path.join(destination_dir, os.path.basename(apk_name))

    try:
        shutil.copytree(source_dir, destination_dir)
        print(f"Copied directory {source_dir} to {destination_dir}")
        return True
    except FileNotFoundError:
        print(f"Error: Source directory {source_dir} not found.")
        return False
    except FileExistsError:
        print(f"Error: Destination directory {destination_dir} already exists.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


def process_apks(apk_directory):
    apk_files = [f for f in os.listdir(apk_directory) if f.endswith('.apk')]

    for apk_file in apk_files:
        apk_name = os.path.splitext(apk_file)[0]  
        apk_path = os.path.join(apk_directory, apk_file)

        print(f"Processing APK: {apk_name}")
        dump_dir = os.path.join('dumps', apk_name)
        if not os.path.exists(dump_dir):
            if install_apk(apk_path):
                package_name = apk_name  
                if start_app(package_name):
                    dump_dir = memory_dump(apk_name)
                    if dump_dir:
                        stop_app(package_name)
                uninstall_apk(package_name)
        
        target_directory = os.path.join('compiled_smali', apk_name)
        if not os.path.exists(os.path.join(target_directory, f'{apk_name}_classes03.dex' )):
            baksmali(dump_dir)
            detector = FileStructureLibDetector(dump_dir)
            target_directory = detector.detect()
        if copy_directory(target_directory, 'johannes'):
            run_dynamic_detection()
        src = os.path.join(directory, apk_file )
        dst_dir = os.path.join(directory, 'processed')
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(apk_file))
        try:
            shutil.move(src, dst)
            print(f"Moved {apk_file} to {dst}")
        except Exception as e: 
            print(f"Could not move {apk_file} to {dst_dir}: {e}")
        print(f"Finished processing {apk_name}\n")

if __name__ == "__main__":
    directory = input("Enter the directory path: ").strip()
    if os.path.isdir(directory):
        process_apks(directory)
        
