##Networks
import monai

def generate_nn(nn_name, name, dim="2D", scaff=False, fed_rod=False, **kwargs):
    if nn_name=="unet":
        #network used by Antoine
        if dim=="2D":
            nn=monai.networks.nets.UNet(spatial_dims=2,
                                      in_channels=1,
                                      out_channels=1,
                                      channels=(16, 32, 64, 128),
                                      strides=(2, 2, 2),
                                      kernel_size = (3,3),
                                      num_res_units=2)
        #network used by Seb/Jony
        elif dim=="3D":
            nn=monai.networks.nets.UNet(spatial_dims=3,
                                        in_channels=4,
                                        out_channels=3,
                                        channels=(16, 32, 64, 128, 256),
                                        strides=(2, 2, 2, 2),
                                        num_res_units=2)
        #custom definition using a dictionnary?
        else:
            nn = monai.networks.nets.UNet(**kwargs)

    elif nn_name=="unetr":
        nn = monai.networks.nets.UNETR(in_channels, out_channels,
                                  img_size=patch_site, 
                                  feature_size=16,
                                  hidden_size=768,
                                  mlp_dim=3072,
                                  num_heads=12,
                                  pos_embed='conv',
                                  norm_name='instance',
                                  conv_block=True,
                                  res_block=True,
                                  dropout_rate=0.0,
                                  spatial_dims=3)
    elif nn_name=="swin_unetr":
        nn = monai.networks.nets.SwinUNETR(img_size=patch_size,
                                           in_channels=4,
                                           out_channels=3,
                                           feature_size=48,
                                           use_checkpoint=True)
        #weight = torch.load("./models/model_swinvit.pt")
        #model.load_from(weights=weight)
        #print("Using pretrained self-supervied Swin UNETR backbone weights !") 

    elif nn_name=="segresnet":
        nn = monai.networks.nets.SegResNet(blocks_down=[1, 2, 2, 4],
                                           blocks_up=[1, 1, 1],
                                           init_filters=16,
                                           in_channels=4,
                                           out_channels=3,
                                           dropout_prob=0.2)

    if scaff:
        #control variables for SCAFFOLD
        nn.control = {}
        nn.delta_control = {}
        nn.delta_y = {}

    if fed_rod:
        #Unet params sets for FedRod
        nn.encoder_generic = {}
        nn.decoder_generic = {}
        nn.decoder_personalized = {}

    nn.name = name
    
    return nn