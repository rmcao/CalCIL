# Description:
#  
# Written by Ruiming Cao on March 25, 2022
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io


from dataclasses import dataclass
from typing import Tuple, Union, List

import numpy as np
from skimage import data, transform


def generate_shepp_logan(dim_yx):

    phantom = data.shepp_logan_phantom()
    phantom = transform.resize(phantom, dim_yx, anti_aliasing=True)

    return phantom
