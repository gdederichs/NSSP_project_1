def mkdir_no_exist(path:str)->None:
    """
    Create a directory if it doesn't exist

    Parameters
    ----------
    path: string
        Path name
    """
    import os
    if not os.path.isdir(path):
        os.makedirs(path)