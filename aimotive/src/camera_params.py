import numpy as np


class CameraParams:
    """
    A class for storing camera parameters for a camera.

    Attributes
        intrinsic: camera to image matrix, shape: [3, 4]
        extrinsic: body to sensor matrix, shape: [4, 4]
        dist_coeffs: distortion coefficients
        camera_model: name of the camera model
        focal_length: list of focal length [f_x, f_y]
        principal_point: list of principal point [p_x, p_y]
        xi: xi parameter of a mei model, None otherwise
    """
    def __init__(self, intrinsic: np.array, extrinsic: np.array, dist_coeffs: np.array, yaw_pitch_roll: np.array, camera_model: str, xi=None):
        """
        Args::
            intrinsic: camera to image matrix, shape: [3, 4]
            extrinsic: body to sensor matrix, shape: [4, 4]
            dist_coeffs: distortion coefficients
            camera_model: name of the camera model
            xi: xi parameter of a mei model, None otherwise
        """
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.dist_coeffs = dist_coeffs
        self.yaw_pitch_roll = yaw_pitch_roll
        self.camera_model = camera_model
        self.focal_length = [intrinsic[0][0], intrinsic[1][1]]
        self.principal_point = [intrinsic[0][2], intrinsic[1][2]]
        self.xi = xi
