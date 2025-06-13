import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrope.models import SINDYc


model = SINDYc(**vars)
model.save_model("../data/Demo_tiegcm_SINDYc_hrd.pkl")