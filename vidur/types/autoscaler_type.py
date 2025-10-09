from vidur.types.base_int_enum import BaseIntEnum


class AutoscalerType(BaseIntEnum):
    DISABLED = 0
    INFERLINE = 1
    CUSTOM = 2
