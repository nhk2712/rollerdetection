import albumentations as A
from albumentations.pytorch import ToTensorV2


def scale_down_pipeline():
    """
    This function creates a scale down transform pipeline.
    The pipeline is designed to make people appear smaller in images.

    Parameters:
    None

    Returns:
    scale_down_transform (A.Compose): An Albumentations Compose object containing the scale down transform pipeline.

    The scale down transform pipeline includes the following operations:
    - A.ShiftScaleRotate(scale_limit=(-0.1, -0.05), rotate_limit=5, p=1): This operation shifts, scales, and rotates the image.
      The scale_limit parameter is set to (-0.1, -0.05) to reduce the size of the image by 10% to 5%.
      The rotate_limit parameter is set to 5 to allow a rotation of up to 5 degrees.
      The p parameter is set to 1 to apply this operation with a probability of 100%.

    The bbox_params and keypoint_params are set to handle bounding boxes and keypoints in the Pascal VOC format.
    """
    
    # Scale Down Pipeline to make People Smaller
    scale_down_transform = A.Compose([
        A.Resize(height=360, width=640),
        A.ShiftScaleRotate(scale_limit=(-0.1, -0.05), rotate_limit=5, p=1)
    ], 
    bbox_params=A.BboxParams(format='pascal_voc'), 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))
    return scale_down_transform
    

def resize_pipeline():
    """
    This function creates a resize transform pipeline.
    The pipeline is designed to make people slightly thinner in images.

    Parameters:
    None

    Returns:
    resize_transform (A.Compose): An Albumentations Compose object containing the resize transform pipeline.

    The resize transform pipeline includes the following operations:
    - A.Resize(height=512, width=512): Resizes the image to a fixed size of 512x512 pixels.

    The bbox_params and keypoint_params are set to handle bounding boxes and keypoints in the Pascal VOC format.
    """

    resize_transform = A.Compose([
        A.Resize(height=360, width=640)
    ], 
    bbox_params=A.BboxParams(format='pascal_voc'), 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False)
    )
    return resize_transform


def horizontal_flip_pipeline():
    """
    This function creates a horizontal flip transform pipeline.
    The pipeline is designed to make people mirrored in images.

    Parameters:
    None

    Returns:
    horizontal_flip_transform (A.Compose): An Albumentations Compose object containing the horizontal flip transform pipeline.

    The horizontal flip transform pipeline includes the following operations:
    - A.HorizontalFlip(p=1): Flips the image horizontally with a probability of 1.

    The bbox_params and keypoint_params are set to handle bounding boxes and keypoints in the Pascal VOC format.
    """
    
    horizontal_flip_transform = A.Compose([
        A.Resize(height=360, width=640),
        A.HorizontalFlip(p=1)
    ],
    bbox_params=A.BboxParams(format='pascal_voc'), 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False)
    )
    return horizontal_flip_transform


def shift_scale_rotate_pipeline():
    """
    This function creates a shifting and scale a little bit transform pipeline.
    The pipeline is designed for Scale Up to make People Bigger and Shift people Slightly

    Parameters:
    None

    Returns:
    horizontal_flip_transform (A.Compose): An Albumentations Compose object containing the horizontal flip transform pipeline.

    The horizontal flip transform pipeline includes the following operations:
    - A.HorizontalFlip(p=1): Flips the image horizontally with a probability of 1.

    The bbox_params and keypoint_params are set to handle bounding boxes and keypoints in the Pascal VOC format.
    """

    shift_scale_rotate_transform = A.Compose([
        A.Resize(height=360, width=640),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=1)
    ], 
    bbox_params=A.BboxParams(format='pascal_voc'), 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False)
    )
    return shift_scale_rotate_transform


def shift_scale_rotate_negative_pipeline():
    """
    This function creates a negative shifting and scale a little bit transform pipeline.
    The pipeline is designed for Scale Up to make People Bigger and Shift people Slightly

    Parameters:
    None

    Returns:
    horizontal_flip_transform (A.Compose): An Albumentations Compose object containing the horizontal flip transform pipeline.

    The horizontal flip transform pipeline includes the following operations:
    - A.HorizontalFlip(p=1): Flips the image horizontally with a probability of 1.

    The bbox_params and keypoint_params are set to handle bounding boxes and keypoints in the Pascal VOC format.
    """

    shift_scale_rotate_transform = A.Compose([
        A.Resize(height=360, width=640),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=-5, p=1)
    ], 
    bbox_params=A.BboxParams(format='pascal_voc'), 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False)
    )
    return shift_scale_rotate_transform