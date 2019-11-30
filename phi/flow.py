# pylint: disable-msg = wildcard-import, unused-wildcard-import

from .physics.physics import *
from .physics.world import *

# For some reason the first import from the physics DomainStates cannot have "dependencies=DomainState.domain" in the struct.attr, as it cannot find such a dependency property? Not sure why this is, I am truly confused.
from .physics.gridliquid import *
from .physics.sdfliquid import *
from .physics.flipliquid import *
from .physics.schroedinger import *
from .physics.smoke import *
from .physics.burgers import *
from .physics.heat import *

from .physics.worldutil import *
from .physics.field import *
from .physics.obstacle import *
from .physics.material import *
from .physics.domain import *
from .physics.field.effect import *
from .data.fluidformat import *
from .data.dataset import *
from .data.stream import *
from .data.reader import *
from phi.geom import *
from .viz import display
from .viz.display import show
from .app import *
from phi import math, struct
