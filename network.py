##Networks
import monai

#network present in each client
class UNet_custom(monai.networks.nets.UNet):
    def __init__(self, spatial_dims, in_channels, out_channels, channels,
                 strides, kernel_size, num_res_units, name, scaff=False, fed_rod=False):
        #call parent constructor
        super(UNet_custom, self).__init__(spatial_dims=spatial_dims,
                                          in_channels=in_channels,
                                          out_channels=out_channels, 
                                          channels=channels,
                                          strides=strides,
                                          kernel_size=kernel_size, 
                                          num_res_units=num_res_units)

        self.name = name
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        if scaff:
            #control variables for SCAFFOLD
            self.control = {}
            self.delta_control = {}
            self.delta_y = {}

        if fed_rod:
            #Unet params sets for FedRod
            self.encoder_generic = {}
            self.decoder_generic = {}
            self.decoder_personalized = {}