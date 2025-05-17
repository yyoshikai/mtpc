from .model import *
from .scheme import Scheme
from .barlowtwins import BarlowTwins, BarlowTwinsTransform
from .vicreg import VICReg, VICRegTransform
from .vicregl import VICRegL, VICRegLTransform

from .barlowtwins import BarlowTwins
from .vicreg import VICReg
from .vicregl import VICRegL
scheme_name2class: dict[str, Scheme] = {
    'bt': BarlowTwins, 
    'vicreg': VICReg,
    'vicregl': VICRegL
}
