#ifndef ROPE_COAE_H   
#define ROPE_COAE_H

#include <unordered_map>
#include <functional>
#include <iostream>

#include <torch/torch.h>

namespace COAE {

class Encoder : torch::nn::Module
{
    private:

        const double bottleneck_size;
        const double n_downsamples;
        const std::vector<double> n_layers;
        const std::vector<double> n_filters;
        const std::vector<string> activation_functions;

        std::vector<std::vector<double>> set_pools()
        // {
        //     std::vector<std::vector<double>> pools;
        //     switch(n_downsamples) {
        //         case 1:
        //             pools.push_back({12, 20, 16})
        //         case 2:
        //             pools.push_back({4, 5, 4})
        //             pools.push_back({3, 4, 4})
        //         case 3:
        //             pools.push_back({3, 5, 4})
        //             pools.push_back({2, 2, 2})
        //             pools.push_back({2, 2, 2})
        //         case 4:
        //             pools.push_back({3, 5, 2})
        //             pools.push_back({2, 2, 2})
        //             pools.push_back({2, 2, 2})
        //             pools.push_back({1, 1, 2})
        //         default:
        //             //error error error
        //     }
        //     // NOTE: these should not be hard-coded. Should be part of the input files.

        //     return pools;
        // }

        void build_layers()
            // def _build_layers(
            //     self,
            //     num_encoder_downsamples: int,
            //     num_layers: List[int],
            //     num_filters: List[int],
            //     activations: List[str],
            // ) -> None:
            //     """builds all layers depending on the configuration
        
            //     :param num_encoder_downsamples: number of times the dimensions are downsamples in the encoder
            //     :type num_encoder_downsamples: int
            //     :param num_layers: number of layers in each encoder section
            //     :type num_layers: List[int]
            //     :param num_filters: number of filters in each encoder section
            //     :type num_filters: List[int]
            //     :param activations: activation function for each encoder section
            //     :type activations: List[str]
            //     """
            //     self._layers = nn.ModuleList()
            //     pools = self._setup_pooling(num_encoder_downsamples)
            //     for i in range(num_encoder_downsamples):
            //         for j in range(num_layers[i]):
            //             self._layers.append(
            //                 nn.Conv3d(
            //                     in_channels=num_filters[i] if j == 0 else num_filters[i + 1],
            //                     out_channels=num_filters[i + 1],
            //                     kernel_size=(3, 3, 3),
            //                     stride=(1, 1, 1),
            //                     padding="same",
            //                 )
            //             )
            //             self._layers.append(self._activation_dict[activations[i].lower()])
            //         self._layers.append(nn.BatchNorm3d(num_filters[i + 1]))
            //         self._layers.append(nn.MaxPool3d(kernel_size=pools[i]))
            //     self._output_filters = num_filters[i + 1]
            //     self._layers.append(nn.Flatten())
            //     self._layers.append(nn.Linear(in_features=2 * num_filters[i + 1], out_features=self._bottleneck_size))

    public:

        Encoder(){}
        torch::Tensor forward(torch::Tensor& x)
            // for layer in self._layers:
            // x = layer(x)
            // return x

};

class Decoder : torch::nn::Module
{
    private:

        const double bottleneck_size;
        const double n_upsamples;
        const std::vector<double> n_layers;
        const std::vector<double> n_filters;
        const std::vector<string> activation_functions;

    public:

        Decoder(){}
        torch::Tensor forward(torch::Tensor& x)

};

class COAE : torch::nn::Module
{
    private:
        const Decoder decoder;
        const Encoder encoder;
        
    public:
        
        COAE(){}
        torch::Tensor forward(torch::Tensor& x)
}

};

#endif