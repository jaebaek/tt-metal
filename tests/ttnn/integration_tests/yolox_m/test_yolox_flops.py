from calflops import calculate_flops
import torch
from models.experimental.functional_yolox_m.reference.yolox_m import YOLOX

torch_model = YOLOX()
ank
torch_model.eval()
input_shape = (1, 3, 640, 640)
flops, macs, params = calculate_flops(
    model=torch_model, input_shape=input_shape, output_as_string=True, output_precision=4
)

print("FLOPs:%s    MACs:%s    Params:%s\n" % (flops, macs, params))
