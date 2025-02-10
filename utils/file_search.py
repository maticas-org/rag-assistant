from os import walk, path, listdir

def get_files_paths_local(directory_path: str, 
                          extensions: list,
                          ) -> list:
    """
        This function takes a directory path and returns a list of file paths
        inside the directory, that have a specific extension.

        This is basically a tree discovery function. That grabs the files.
    """

    print(f"Found different directories: {listdir(directory_path)}")
    files = []
    for root, _, filenames in walk(directory_path):
        for filename in filenames:
            if filename.endswith(tuple(extensions)):
                files.append(path.join(root, filename))
    
    return files