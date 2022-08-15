import monai

def generate_nn(nn_name, nn_class, nn_params={}, scaff=False, fedrod=False, fedprox=False):
    """ Generate the segmentation model based on the parameters passed in argument and additonnal 
        attributes depending on the federated framework used.
        This function heavily rely on the implementation of various models by the Monai library.
    """
    if nn_class.lower()=="unet":
        nn=monai.networks.nets.UNet(**nn_params)
        print("Using UNET as segmentation network")
    elif nn_class.lower()=="unetr":
        nn = monai.networks.nets.UNETR(**nn_params)
        print("Using UNETR as segmentation network")
    elif nn_class.lower()=="swin_unetr":
        nn = monai.networks.nets.SwinUNETR(**nn_params)
        #weight = torch.load("./models/model_swinvit.pt")
        #model.load_from(weights=weight)
        #print("Using pretrained self-supervied Swin UNETR backbone weights !")
        print("Using SWIN UNETR as segmentation network")
    elif nn_class.lower()=="segresnet":
        nn = monai.networks.nets.SegResNet(**nn_params)
        print("Using SEGRESNET as segmentation network")
    else:
        print("network name not supported", nn_class.lower())

    if scaff:
        #control variables for SCAFFOLD
        nn.control = {}
        nn.delta_control = {}
        nn.delta_y = {}

    elif fedrod:
        #params sets for FedRod
        nn.encoder_generic = {}
        nn.decoder_generic = {}
        nn.decoder_personalized = {}

    elif fedprox:
        print("fedprox don't require any special attribute")
        #params sets for FedRod

    nn.name = nn_name
    
    return nn