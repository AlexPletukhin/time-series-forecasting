#!/usr/bin/env python
import sys
from PySide6.QtWidgets import QApplication
from gui.ui_main import PipelineUI
from src import utils
import os, torch, numpy as np
os.environ["OMP_NUM_THREADS"] = "2"
torch.set_num_threads(2)
np.set_printoptions(legacy="1.25")

if __name__ == "__main__":
    utils.init_logging()
    app = QApplication(sys.argv)
    win = PipelineUI()
    win.show()
    sys.exit(app.exec())
