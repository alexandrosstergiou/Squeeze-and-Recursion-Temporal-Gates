'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import coloredlogs, logging
coloredlogs.install()
from .MTnet import MTNet_32, MTNet_48, MTNet_64, MTNet_132, MTNet_200, MTNet_264
from .MTnet import MTNet_32_g8, MTNet_48_g8, MTNet_64_g8, MTNet_132_g8
from .srtg_resnet import r3d_18, r3d_34, r3d_50, r3d_101, r3d_152, r3d_200, r3dxt50_32x4d, r3dxt101_32x8d, wide_r3d50_2,wide_r3d101_2, r2plus1d_18, r2plus1d_34, r2plus1d_50, r2plus1d_101, r2plus1d_152, r2plus1d_200, r2plus1dxt50_32x4d, r2plus1dxt101_32x8d, wide_r2plus1d50_2,wide_r2plus1d101_2
from .srtg_resnet import srtg_r3d_18, srtg_r3d_34, srtg_r3d_50, srtg_r3d_101, srtg_r3d_152, srtg_r3d_200, srtg_r3dxt50_32x4d, srtg_r3dxt101_32x8d, srtg_wide_r3d50_2, srtg_wide_r3d101_2, srtg_r2plus1d_18, srtg_r2plus1d_34, srtg_r2plus1d_50, srtg_r2plus1d_101, srtg_r2plus1d_152, srtg_r2plus1d_200, srtg_r2plus1dxt50_32x4d, srtg_r2plus1dxt101_32x8d, srtg_wide_r2plus1d50_2, srtg_wide_r2plus1d101_2

from .config import get_config

'''
---  S T A R T  O F  F U N C T I O N  G E T _ S Y M B O L ---
    [About]
        Function for loading PyTorch models.
    [Args]
        - name: String for the network name.
        - print_net: Boolean for printing the architecture. Defaults to False.
    [Returns]
        - net: Module for the loaded Pytorch network.
        - config: Dictionary that includes a `mean` and `std` terms spcifying the mean and standard deviation.
                  See `network/config.py` for more info.
'''
def get_symbol(name, print_net=False, **kwargs):

    # Multi-Temporal net
    if "MTNET" in name.upper():
        if "MTNET_32" in name.upper():
            if "G8" in name.upper():
                net = MTNet_32_g8(**kwargs)
            else:
                net = MTNet_32(**kwargs)
        elif "MTNET_48" in name.upper():
            if "G8" in name.upper():
                net = MTNet_48_g8(**kwargs)
            else:
                net = MTNet_48(**kwargs)
        elif "MTNET_64" in name.upper():
            if "G8" in name.upper():
                net = MTNet_64_g8(**kwargs)
            else:
                net = MTNet_64(**kwargs)
        elif "MTNET_132" in name.upper():
            if "G8" in name.upper():
                net = MTNet_132_g8(**kwargs)
            else:
                net = MTNet_132(**kwargs)
        elif "MTNET_200" in name.upper():
            net = MTNet_200(**kwargs)
        elif "MTNET_264" in name.upper():
            net = MTNet_264(**kwargs)
        else:
            net = MTNet_64(**kwargs)
    # ResNet 3D
    elif "R3D" in name.upper():
        if "R3D_18" in name.upper():
            net = r3d_18(**kwargs)
        elif "R3D_34" in name.upper():
            net = r3d_34(**kwargs)
        elif "R3D_50" in name.upper():
            net = r3d_50(**kwargs)
        elif "R3D_101" in name.upper():
            net = r3d_101(**kwargs)
        elif "R3D_152" in name.upper():
            net = r3d_152(**kwargs)
        elif "R3D_200" in name.upper():
            net = r3d_200(**kwargs)
        elif "R3DXT50" in name.upper():
            net = r3dxt50_32x4d(**kwargs)
        elif "R3DXT101" in name.upper():
            net = r3dxt101_32x8d(**kwargs)
        elif "WIDE_R3D50" in name.upper():
            net = wide_r3d50_2(**kwargs)
        else:
            net = wide_r3d101_2(**kwargs)
    # Resnet (2+1)D
    elif "R2PLUS1D" in name.upper():
        if "R2PLUS1D_18" in name.upper():
            net = r2plus1d_18(**kwargs)
        elif "R2PLUS1D_34" in name.upper():
            net = r2plus1d_34(**kwargs)
        elif "R2PLUS1D_50" in name.upper():
            net = r2plus1d_50(**kwargs)
        elif "R2PLUS1D_101" in name.upper():
            net = r2plus1d_101(**kwargs)
        elif "R2PLUS1D_152" in name.upper():
            net = r2plus1d_152(**kwargs)
        elif "R2PLUS1D_200" in name.upper():
            net = r2plus1d_200(**kwargs)
        elif "R2PLUS1DXT50" in name.upper():
            net = r2plus1dxt50_32x4d(**kwargs)
        elif "R2PLUS1DXT101" in name.upper():
            net = r2plus1dxt101_32x8d(**kwargs)
        elif "WIDE_R2PLUS1D50" in name.upper():
            net = wide_r2plus1d50_2(**kwargs)
        else:
            net = wide_r2plus1d101_2(**kwargs)
    # Resnet 3D + SRTG
    elif "SRTG_R3D" in name.upper():
        if "SRTG_R3D_18" in name.upper():
            net = srtg_r3d_18(**kwargs)
        elif "SRTG_R3D_34" in name.upper():
            net = srtg_r3d_34(**kwargs)
        elif "SRTG_R3D_50" in name.upper():
            net = srtg_r3d_50(**kwargs)
        elif "SRTG_R3D_101" in name.upper():
            net = srtg_r3d_101(**kwargs)
        elif "SRTG_R3D_152" in name.upper():
            net = srtg_r3d_152(**kwargs)
        elif "SRTG_R3D_200" in name.upper():
            net = srtg_r3d_200(**kwargs)
        elif "SRTG_R3DXT50" in name.upper():
            net = srtg_r3dxt50_32x4d(**kwargs)
        elif "SRTG_R3DXT101" in name.upper():
            net = srtg_r3dxt101_32x8d(**kwargs)
        elif "SRTG_WIDE_R3D50" in name.upper():
            net = srtg_wide_r3d50_2(**kwargs)
        else:
            net = srtg_wide_r3d101_2(**kwargs)
    elif "SRTG_R2PLUS1D" in name.upper():
        if "SRTG_R2PLUS1D_18" in name.upper():
            net = srtg_r2plus1d_18(**kwargs)
        elif "SRTG_R2PLUS1D_34" in name.upper():
            net = srtg_r2plus1d_34(**kwargs)
        elif "SRTG_R2PLUS1D_50" in name.upper():
            net = srtg_r2plus1d_50(**kwargs)
        elif "SRTG_R2PLUS1D_101" in name.upper():
            net = srtg_r2plus1d_101(**kwargs)
        elif "SRTG_R2PLUS1D_152" in name.upper():
            net = srtg_r2plus1d_152(**kwargs)
        elif "SRTG_R2PLUS1D_200" in name.upper():
            net = srtg_r2plus1d_200(**kwargs)
        elif "SRTG_R2PLUS1DXT50" in name.upper():
            net = srtg_r2plus1dxt50_32x4d(**kwargs)
        elif "SRTG_R2PLUS1DXT101" in name.upper():
            net = srtg_r2plus1dxt101_32x8d(**kwargs)
        elif "SRTG_WIDE_R2PLUS1D50" in name.upper():
            net = srtg_wide_r2plus1d50_2(**kwargs)
        else:
            net = srtg_wide_r2plus1d101_2(**kwargs)
    else:
        logging.error("network '{}'' not implemented".format(name))
        raise NotImplementedError()

    if print_net:
        logging.debug("Symbol:: Network Architecture:")
        logging.debug(net)

    input_conf = get_config(name, **kwargs)
    return net, input_conf
'''
---  E N D  O F  F U N C T I O N  G E T _ S Y M B O L ---
'''
