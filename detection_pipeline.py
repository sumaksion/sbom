import os
import subprocess
import zipfile
import shutil

"""
requires a running android device with adb root and running frida server on device

frida-dexdump must be installed
https://github.com/hluwa/frida-dexdump

apk files should be in apk folder

if running emulator on windows, have this file on windows and everything else in the given file structure in wsl
replace johannes with your linux username and replace Ubuntu with your linux distro if not running Ubuntu

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
    return run_command(['adb', 'shell', 'monkey', '-p', package_name, '-c', 'android.intent.category.LAUNCHER', '1'])

def stop_app(package_name):
    print(f"Stopping app: {package_name}")
    return run_command(['adb', 'shell', 'am', 'force-stop', package_name])

def uninstall_apk(package_name):
    print(f"Uninstalling app: {package_name}")
    return run_command(['adb', 'uninstall', package_name])

def memory_dump(apk_name):
    
    try:
        if not os.path.exists(apk_name):
            os.makedirs(apk_name, exist_ok=False)
        else:
            return True
    except Exception as e:
        print(f"Failed to create directory '{apk_name}': {e}")
        return False
    
    try:
        subprocess.run(f'frida-dexdump -d -FU -o "{apk_name}"', shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Memory dump failed for {apk_name}: {e}")
        return False

def extract_apk(apk_path, apk_name):

    zip_path = apk_path.replace(".apk", ".zip")
    os.rename(apk_path, zip_path)
    extract_dir = apk_name + "_unzipped"
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def compare_dex_files(apk_name):
    extracted_dir = apk_name+"_unzipped"
    dump_dir = apk_name  

    extracted_dex_files = [os.path.getsize(os.path.join(extracted_dir, f)) for f in os.listdir(extracted_dir) if f.endswith('.dex')]
    dump_dex_files = [os.path.getsize(os.path.join(dump_dir, f)) for f in os.listdir(dump_dir) if f.endswith('.dex')]

    if not extracted_dex_files or not dump_dex_files:
        return False
    
    dump_file_sizes = dump_dex_files.copy()

    all_matched = True

    for size in extracted_dex_files:
        if size in dump_file_sizes:
            dump_file_sizes.remove(size)  
        else:
            all_matched = False

    if all_matched:
        print("All matched")

    return all_matched

def copy_and_run_dynamic_detection(apk_name, user):
    try:
        source_file = os.path.join(apk_name, "classes03.dex")
    except FileNotFoundError as e:
        print(e)
        return False
    destination_dir = os.path.expanduser('~\sbom\data\dexs')
    if os.name == "nt":
        destination_dir = os.path.join(r'\\wsl.localhost', 'Ubuntu', 'home', user, 'sbom', 'data', 'dexs')
    destination_file = os.path.join(destination_dir, f"{apk_name}_classes03.dex")

    try:
        shutil.copy(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")
    except FileNotFoundError:
        print(f"Error: {source_file} not found.")
        return False
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


def process_apks(apk_directory):
    apk_files = [f for f in os.listdir(apk_directory) if f.endswith('.apk')]

    for apk_file in apk_files:
        apk_name = os.path.splitext(apk_file)[0]  
        apk_path = os.path.join(apk_directory, apk_file)

        print(f"Processing APK: {apk_name}")

        if install_apk(apk_path):
            package_name = apk_name  
            if start_app(package_name):
                if memory_dump(apk_name):
                    copy_and_run_dynamic_detection(apk_name, 'johannes')
                    stop_app(package_name)

            uninstall_apk(package_name)

        print(f"Finished processing {apk_name}\n")

if __name__ == '__main__':
    apk_directory = 'apks'
    if not os.path.exists(apk_directory):
        print(f"APK directory '{apk_directory}' not found.")
    else:
        process_apks(apk_directory)
