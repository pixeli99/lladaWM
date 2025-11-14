from .observation import Observation
from .scoring import Scorer
from .visualization import Visualizer
import json


class Rewarder:
    def __init__(self):
        self._obseravation = Observation()
        self._scorer = Scorer()
        self._visualizer = Visualizer()

    def get_infos(self, raw_info: dict, vis: bool = False):
        self._obseravation.update(raw_info)
        result = self._obseravation.get_latest()
        return result
