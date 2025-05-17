from .camera import Camera
from typing import List

class SystemResources:
    @staticmethod
    def get_cameras() -> List[Camera]:
        """Get list of all available cameras in the system"""
        available_ids = Camera.get_available_cameras()
        return [Camera.get_camera(device_id) for device_id in available_ids] 