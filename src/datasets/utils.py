import cv2


def check_image_load(path: str) -> bool:
    '''
    Checks that the picture on the specified path is correct
    :param path: path to image
    :return: is the path correct
    '''
    try:
        image = cv2.imread(path)
        _ = image.shape
        return True
    except Exception:
        return False
