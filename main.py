from flight_maneuvers.modules.resnet import ResNet
from flight_maneuvers.utils import *
from flight_maneuvers.tune import *

from flight_maneuvers.tune import *
from flight_maneuvers.modules.resnet import ResNet

s = Search(ResNet)
s.evolve()