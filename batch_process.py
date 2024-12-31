import os
import shutil
import subprocess

class JarBatchProcessor:
    def __init__(self, base_directory, target_count=30, to_process_subdir="to_process", script_to_run="process.py"):
        self.base_directory = base_directory
        self.target_count = target_count
        self.to_process_dir = os.path.join(base_directory, to_process_subdir)
        self.script_to_run = script_to_run

    def get_jar_files(self, directory):
        """Get a list of .jar files in the specified directory."""
        return [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.endswith(".jar")
        ]

    def move_files(self, files, destination):
        """Move the specified files to the destination directory."""
        for file in files:
            shutil.move(file, destination)

    def ensure_target_count(self):
        """Ensure the base directory contains the target number of .jar files."""
        current_jar_files = self.get_jar_files(self.base_directory)
        current_count = len(current_jar_files)

        if current_count < self.target_count:
            jars_needed = self.target_count - current_count
            to_process_jar_files = self.get_jar_files(self.to_process_dir)

            if jars_needed > len(to_process_jar_files):
                jars_needed = len(to_process_jar_files)

            jars_to_move = to_process_jar_files[:jars_needed]
            self.move_files(jars_to_move, self.base_directory)

    def process_batches(self):
        """Process .jar files in batches, launching a subprocess for each batch."""
        while True:
            self.ensure_target_count()
            current_jar_files = self.get_jar_files(self.base_directory)

            if not current_jar_files:
                print("No more files to process.")
                break

            print(f"Processing batch of {len(current_jar_files)} .jar files...")

            # Launch the subprocess
            process = subprocess.run(["/home/johannes/sbom/.conda/bin/python", "fit_gnn.py"])

            print(f"Subprocess finished with return code: {process.returncode}")

if __name__ == "__main__":
    base_directory = "data/jars"
    processor = JarBatchProcessor(base_directory)
    processor.process_batches()
