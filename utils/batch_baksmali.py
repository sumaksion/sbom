import os
import subprocess

BAKSMALI_PATH = "baksmali.jar"

def disassemble_class_file(class_file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Command to run baksmali
    command = [
        "java", "-jar", BAKSMALI_PATH,
        "disassemble", class_file_path,
        "-o", output_dir
    ]
    
    # Run the command
    subprocess.run(command)

# Function to process all .class files in a directory
def disassemble_classes_in_directory(input_directory):
    # Loop through each file in the directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
    
            if file.endswith(".dex"):
                # Full path to the .class file
                class_file_path = os.path.join(root, file)

                output_dir = os.path.join(root, os.path.splitext(file)[0])
                
                # Disassemble the class file
                disassemble_class_file(class_file_path, output_dir)
