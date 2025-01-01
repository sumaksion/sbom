import os
import requests
import json
import shutil
import subprocess
import fnmatch


class FileStructureLibDetector:


    def __init__(self, directory):
        self.directory = directory
        self.start_words = {"com", "org"}
        self.potential_libraries = set()

    def compare_and_clean_directories(self, directory):
        """
        Compares all subdirectories in the given directory. If two subdirectories 
        match completely in path structure and contain identical files, the one 
        with the smaller total size is deleted.

        :param directory: The directory to process.
        """
        def get_total_size(path):
            """Calculate total size of all files in the directory and its subdirectories."""
            total_size = 0
            for root, _, files in os.walk(path):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            return total_size

        def are_dirs_identical(dir1, dir2):
            """
            Recursively check if two directories are identical in structure and content.
            Compares subdirectory names and `.smali` file names.
            """
            # Get lists of subdirectories and files in both directories
            subdirs1 = sorted([d for d in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, d))])
            subdirs2 = sorted([d for d in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, d))])
            
            files1 = sorted([f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f)) and f.endswith(".smali")])
            files2 = sorted([f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f)) and f.endswith(".smali")])

            # Check if subdirectory names and .smali file names match
            if subdirs1 != subdirs2 or files1 != files2:
                return False

            # Recursively compare all matching subdirectories
            for subdir in subdirs1:
                if not are_dirs_identical(os.path.join(dir1, subdir), os.path.join(dir2, subdir)):
                    return False

            return True

        def process_subdirectories(parent_dir):
            """
            Compare all pairs of subdirectories and delete redundant ones.
            """
            subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

            for i, dir1 in enumerate(subdirs):
                for dir2 in subdirs[i + 1:]:
                    if are_dirs_identical(dir1, dir2):
                        size1 = get_total_size(dir1)
                        size2 = get_total_size(dir2)

                        # Delete the smaller directory
                        if size1 <= size2:
                            print(f"Deleting smaller directory: {dir1}")
                            shutil.rmtree(dir1)
                            break
                        else:
                            print(f"Deleting smaller directory: {dir2}")
                            shutil.rmtree(dir2)
                            break

        process_subdirectories(directory)

    def analyze_directory_structure(self, directory):
        start_words = self.start_words
        three_segment_strings = set()

        for root, dirs, files in os.walk(directory):
            segments = root.split(os.sep)

            for i in range(len(segments) - 2):
                three_segment = f"{segments[i]}.{segments[i + 1]}.{segments[i + 2]}"
                first_segment = segments[i]
                mid_segment = segments[i + 1]
                last_segment = segments[i + 2]

                if first_segment in start_words and last_segment not in start_words and mid_segment != 'android':
                    three_segment_strings.add(three_segment)

            patterns = {"org.conscrypt.*", "org.w3c.dom", "org.apache.xalan", "org.apache.xml", "org.xml.*", "com.sun.net"}
            filtered_strings = {s for s in three_segment_strings if not any(fnmatch.fnmatch(s, pattern) for pattern in patterns)}
            three_segment_strings = filtered_strings

        return three_segment_strings

    def search_maven_repository(self, query):
        url = f'https://search.maven.org/solrsearch/select?q={query}&rows=20&wt=json'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if 60 > data['response']['numFound'] > 0:
                artifacts = [doc['a'] for doc in data['response']['docs'][:5]]
                return artifacts
        return None

    def process_three_segment_strings(self, three_segment_strings):
        valid_queries = {}
        for three_segment in three_segment_strings:
            if three_segment not in valid_queries:
                result = self.search_maven_repository(three_segment)
                if result is not None:
                    valid_queries[three_segment] = result
                else:
                    segments = three_segment.split('.')
                    if len(segments) == 3:
                        two_segment = f"{segments[0]}.{segments[1]}"
                        if two_segment not in valid_queries:
                            result = self.search_maven_repository(two_segment)
                            if result is not None:
                                valid_queries[two_segment] = result

        return valid_queries

    def copy_smali_files(self, directory, target_folder, segment_strings):

        for root, dirs, files in os.walk(directory):
            for segment in segment_strings:
                partial_path = segment.replace('.', os.sep)
                if partial_path in root:
                    target_subfolder = os.path.join(target_folder, os.path.relpath(root, directory))
                    os.makedirs(target_subfolder, exist_ok=True)
                    for file in files:
                        if file.endswith(".smali"):
                            shutil.copy(os.path.join(root, file), target_subfolder)

    def compile_smali_to_dex(self, target_directory, output_dir):
        smali_jar_path = "smali.jar" 
        output_dex_file = os.path.join(output_dir, f'{target_directory.name}.dex')
        cmd = ["java", "-jar", smali_jar_path, "assemble", target_directory.path, "-o", output_dex_file]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during dex compilation: {e}")
            print(e.stderr)  
        if not os.path.exists(output_dex_file):
            original_dex_path = os.path.join(self.directory, f'{target_directory.name}.dex')
            shutil.copy(original_dex_path, output_dir)
            print(f"Copied {original_dex_path} to {output_dir}")
        shutil.rmtree(target_directory.path)


    def detect(self):
        three_segment_strings = self.analyze_directory_structure(self.directory)
        valid_queries = self.process_three_segment_strings(three_segment_strings)
        
        print(f"Valid Queries: {valid_queries}")
        
        target_directory = os.path.join('compiled_smali', os.path.basename(self.directory))
        os.makedirs(target_directory, exist_ok=True)

        self.copy_smali_files(self.directory, target_directory, valid_queries)
        self.compare_and_clean_directories(target_directory)
        with os.scandir(target_directory) as entries:
            for entry in entries:
                if entry.is_dir():
                    self.compile_smali_to_dex(entry, target_directory)
        main_dex = 'classes03.dex'
        src = os.path.join(self.directory, main_dex)
        dst = os.path.join(target_directory, main_dex)
        shutil.copy(src, dst)

        if valid_queries:
            results = {
            'valid_queries': {query: artifacts for query, artifacts in valid_queries.items()}
            }
            with open(os.path.join(target_directory, 'results.json'), 'w') as f:
                json.dump(results, f, indent=4)

        for filename in os.listdir(target_directory):
            src = os.path.join(target_directory, filename)
            dst = os.path.join(target_directory, f"{os.path.basename(target_directory)}_{filename}")
            os.rename(src, dst)
        return target_directory
            

"""directory = "baksmali_output"
detector = FileStructureLibDetector(directory)
detector.process()"""
