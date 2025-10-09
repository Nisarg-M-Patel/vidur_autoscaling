from vidur.autoscaler.custom_autoscaler import CustomAutoscaler
from vidur.autoscaler.inferline_autoscaler import InferlineAutoscaler
from vidur.types.autoscaler_type import AutoscalerType
from vidur.utils.base_registry import BaseRegistry


class AutoscalerRegistry(BaseRegistry):
    pass


AutoscalerRegistry.register(AutoscalerType.INFERLINE, InferlineAutoscaler)
AutoscalerRegistry.register(AutoscalerType.CUSTOM, CustomAutoscaler)