import os
from dataclasses import dataclass, field
from typing import List
import webbrowser

@dataclass
class AppConfig:
    OUTPUT_DIR: str = "results"
    RANDOM_STATE: int = 2023
    TEST_SIZE: float = 0.3
    FEATURES_2D: List[int] = field(default_factory=lambda: [2, 3])
    FEATURES_3D: List[int] = field(default_factory=lambda: [1, 2, 3])
    COLORS: List[str] = field(default_factory=lambda: ['#FF6B6B', '#4ECDC4', '#45B7D1'])
    CLASS_NAMES: List[str] = field(default_factory=lambda: ['Setosa', 'Versicolor', 'Virginica'])
    BG_COLORS: List[str] = field(default_factory=lambda: ['#FFCCCC', '#CCFFCC', '#CCCCFF'])
    RES_2D: int = 200
    RES_3D: int = 40
    RES_VOL: int = 30

    def __post_init__(self):
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

    def open_file(self, filename):
        file_path = os.path.join(self.OUTPUT_DIR, filename)
        abs_path = os.path.abspath(file_path)
        try:
            if filename.endswith('.html'):
                webbrowser.open('file://' + abs_path)
            else:
                os.startfile(abs_path)
        except Exception:
            pass

cfg = AppConfig()