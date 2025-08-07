from shapely.geometry.point import Point


class Tuna:
    def __init__(self, center_point: Point, track_id: int, clss: str, inner_roi: bool):
        self.center_point = center_point
        self.track_id = track_id
        self.clss = clss
        self.inner_roi = inner_roi

