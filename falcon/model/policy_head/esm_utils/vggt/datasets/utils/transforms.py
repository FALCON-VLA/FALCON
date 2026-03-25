# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
import torchvision.transforms as tvf
from vggt.utils.image import ImgNorm

# define the standard image transforms
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
