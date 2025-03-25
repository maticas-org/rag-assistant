from os import walk, path, listdir

def get_files_paths_local(
                            directory_path: str, 
                            extensions: list,
                            file_prefix: str = None,
                            verbose: bool = False
                          ) -> list:
    """
        This function takes a directory path and returns a list of file paths
        inside the directory, that have a specific extension.

        This is basically a tree discovery function. That grabs the files.
    """

    if verbose:
        print(f"Found different directories: {listdir(directory_path)}")

    files = []
    for root, _, filenames in walk(directory_path):
        for filename in filenames:
            if filename.endswith(tuple(extensions)):
                files.append(path.join(root, filename))

    if file_prefix:
        files = [f for f in files if path.basename(f).startswith(file_prefix)]
    
    if verbose:
        print(f"Found {len(files)} files in {directory_path} with extensions {extensions}")
    
    return files