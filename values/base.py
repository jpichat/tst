import os
import uuid


class Identifier(uuid.UUID):
    def __init__(self, hexastring=None):
        """"""
        if hexastring is None:
            super().__init__(bytes=os.urandom(16), version=4)
        else:
            super().__init__(hex=str(hexastring))
