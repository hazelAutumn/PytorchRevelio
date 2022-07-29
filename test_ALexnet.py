import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from PytorchRevelio import PytorchRevelio
from utilities_PytorchRevelio import imagenet_labels
from imageio import imwrite


if __name__ == '__main__':

    # load pretrained Alexnet
    alexnet_net = torchvision.models.alexnet(pretrained=True)

    # choose GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # put network on device
    alexnet_net.to(device)

    # print name of modules
    for key, value in PytorchRevelio.layers_name_type(alexnet_net):
        print('+' * 10)
        print(key)
        print('-' * 10)
        print(value)

    # network transformer for input image
    img_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for different convolutional filter and neuron in fully connected layer
    # show representation
    count = 0
    no_feat_map ={1:64, 2:192, 3:384, 4:256, 5:256}
    first_layer_name = 'features.0'
    last_layer_name = 'features.11'
    for layer_name in alexnet_net.named_modules():

        layer_name = layer_name[0]

        # select convolutional and fully connected layers for visualization
        layer = PytorchRevelio.return_module_by_name(network=alexnet_net, module_name=layer_name)

        if isinstance(layer, nn.Conv2d):
            count += 1
            filter_neuron_num = layer.out_channels#no_feat_map[count]
            layer_type = 'Conv2d'
            #num_iter = 1000
            #lr = 1

            num_iter = 450
            lr = 0.09
            start_sigma = 2.5
            end_sigma = 0.5
        else:
            continue

        if count > 5:
            break

        # from each layer select 8 filter our neurons
        filters_neuron_indexs = [27, 60, 28, 33, 48] #for i in range(filter_neuron_num)], size=8)
        #[i for i in range(filter_neuron_num)] #np.random.choice([i for i in range(filter_neuron_num)], size=8)

        # for each selected filter or neuron, calculate representation
        plt.figure()
        for i, filter_neuron_index in enumerate(filters_neuron_indexs):
            #img = PytorchRevelio.activation_maximization(network=alexnet_net, img_transformer=img_transformer,
            #                                             in_img_size=(224, 224, 3),
            #                                             first_layer_name=first_layer_name, layer_name=layer_name,
            #                                             filter_or_neuron_index=filter_neuron_index, num_iter=num_iter,
            #                                             lr=lr, device=device)

            img = PytorchRevelio.activation_maximization_with_gaussian_blurring(
                network=alexnet_net, img_transformer=img_transformer,
                in_img_size=(224, 224, 3),
                first_layer_name=first_layer_name,
                layer_name=layer_name,
                filter_or_neuron_index=filter_neuron_index,
                num_iter=num_iter,
                start_sigma=start_sigma,
                end_sigma=end_sigma,
                lr=lr,
                device=device)

            # to cpu and normalize for illustration purpose
            img = PytorchRevelio.tensor_outputs_to_image(img)

            # Illustrate
            ax = plt.subplot(1, 5, i+1)
            plt.imshow(img)
            file_name = "./results/layer_"+str(count) +"_"+ str(filter_neuron_index) + 'gaussian.png'
            imwrite(file_name, img)
            if layer_name != last_layer_name:
                ax.set_title("{}".format(filter_neuron_index))
            else:
                ax.set_title("{}, {}".format(filter_neuron_index, imagenet_labels(class_number=filter_neuron_index)))

            plt.suptitle('Layer Name: {}, Type: {}'.format(layer_name, layer_type))
            ax.axis('off')
            print('Processing of layer {}, filter/neuron {} is done.'.format(layer_name, filter_neuron_index))

    plt.show()