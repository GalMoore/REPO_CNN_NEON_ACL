#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

class NEONCNNExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        ARM_COMPUTE_UNUSED(argc);
        ARM_COMPUTE_UNUSED(argv);

        // Create memory manager components
        // We need 2 memory managers: 1 for handling the tensors within the functions (mm_layers) and 1 for handling the input and output tensors of the functions (mm_transitions))
        auto lifetime_mgr0  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto lifetime_mgr1  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto pool_mgr0      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto pool_mgr1      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto mm_layers      = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0); // Create the memory manager for layers
        auto mm_transitions = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1); // Create the memory manager for transitions

        // The weights and biases tensors should be initialized with the values inferred with the training
        // Set memory manager where allowed to manage internal memory requirements
        conv0   = arm_compute::support::cpp14::make_unique<NEConvolutionLayer>(mm_layers);
        conv1   = arm_compute::support::cpp14::make_unique<NEConvolutionLayer>(mm_layers);
        fc0     = arm_compute::support::cpp14::make_unique<NEFullyConnectedLayer>(mm_layers);
        softmax = arm_compute::support::cpp14::make_unique<NESoftmaxLayer>(mm_layers);

        /* [Initialize NUMPY READERS] */
    	// instantiate object npy0 to read from .npy file into src open
        // file with this object load from .npy file to src
        NPYLoader npy0;
        npy0.open("/home/gal/code/compileACL_Neon_CNN/32x32x1_FP.npy");
        npy0.init_tensor(src, DataType::F32);

    	// for save to numpy
        is_fortran      = npy0.is_fortran();

        NPYLoader npy1;
        npy1.open("/home/gal/code/compileACL_Neon_CNN/weights_and_biases/weights_conv0.npy");
        npy1.init_tensor(weights0, DataType::F32);

        NPYLoader npy2;
        npy2.open("/home/gal/code/compileACL_Neon_CNN/weights_and_biases/bias_conv0.npy");
        npy2.init_tensor(biases0, DataType::F32);

        NPYLoader npy3;
        npy3.open("/home/gal/code/compileACL_Neon_CNN/weights_and_biases/weights_conv1.npy");
        npy3.init_tensor(weights1, DataType::F32);

        NPYLoader npy4;
        npy4.open("/home/gal/code/compileACL_Neon_CNN/weights_and_biases/bias_conv1.npy");
        npy4.init_tensor(biases1, DataType::F32);

        NPYLoader npy5;
        npy5.open("/home/gal/code/compileACL_Neon_CNN/weights_and_biases/weights_fc0.npy");
        npy5.init_tensor(weights2, DataType::F32);

        NPYLoader npy6;
        npy6.open("/home/gal/code/compileACL_Neon_CNN/weights_and_biases/bias_fc0.npy");
        npy6.init_tensor(biases2, DataType::F32);
        /* -----------------------End: [Initialize NUMPY READERS] */

        /* [Initialize tensors] */
        // Initialize src tensor
        constexpr unsigned int width_src_image  = 32; // const / constexpr (is not a type- it's an identifier)
        constexpr unsigned int height_src_image = 32; // you can not change this value in compile time
        constexpr unsigned int ifm_src_img      = 1;

        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

        // Initialize tensors of conv0
        constexpr unsigned int kernel_x_conv0 = 5;
        constexpr unsigned int kernel_y_conv0 = 5;
        constexpr unsigned int ofm_conv0      = 8;

        const TensorShape weights_shape_conv0(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);
        const TensorShape biases_shape_conv0(weights_shape_conv0[3]);
        const TensorShape out_shape_conv0(src_shape.x(), src_shape.y(), weights_shape_conv0[3]);

        weights0.allocator()->init(TensorInfo(weights_shape_conv0, 1, DataType::F32));
        biases0.allocator()->init(TensorInfo(biases_shape_conv0, 1, DataType::F32));
        out_conv0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

        // Initialize tensor of act0
        out_act0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

        // Initialize tensor of pool0
        TensorShape out_shape_pool0 = out_shape_conv0;
        out_shape_pool0.set(0, out_shape_pool0.x() / 2);
        out_shape_pool0.set(1, out_shape_pool0.y() / 2);
        out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType::F32));

        // Initialize tensors of conv1
        constexpr unsigned int kernel_x_conv1 = 3;
        constexpr unsigned int kernel_y_conv1 = 3;
        constexpr unsigned int ofm_conv1      = 16;

        const TensorShape weights_shape_conv1(kernel_x_conv1, kernel_y_conv1, out_shape_pool0.z(), ofm_conv1);

        const TensorShape biases_shape_conv1(weights_shape_conv1[3]);
        const TensorShape out_shape_conv1(out_shape_pool0.x(), out_shape_pool0.y(), weights_shape_conv1[3]);

        weights1.allocator()->init(TensorInfo(weights_shape_conv1, 1, DataType::F32));
        biases1.allocator()->init(TensorInfo(biases_shape_conv1, 1, DataType::F32));
        out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

        // Initialize tensor of act1
        out_act1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

        // Initialize tensor of pool1
        TensorShape out_shape_pool1 = out_shape_conv1;
        out_shape_pool1.set(0, out_shape_pool1.x() / 2);
        out_shape_pool1.set(1, out_shape_pool1.y() / 2);
        out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType::F32));

        // Initialize tensor of fc0
        constexpr unsigned int num_labels = 128;

        const TensorShape weights_shape_fc0(out_shape_pool1.x() * out_shape_pool1.y() * out_shape_pool1.z(), num_labels);
        const TensorShape biases_shape_fc0(num_labels);
        const TensorShape out_shape_fc0(num_labels);

        weights2.allocator()->init(TensorInfo(weights_shape_fc0, 1, DataType::F32));
        biases2.allocator()->init(TensorInfo(biases_shape_fc0, 1, DataType::F32));
        out_fc0.allocator()->init(TensorInfo(out_shape_fc0, 1, DataType::F32));

        // Initialize tensor of act2
        out_act2.allocator()->init(TensorInfo(out_shape_fc0, 1, DataType::F32));

        // Initialize tensor of softmax
        const TensorShape out_shape_softmax(out_shape_fc0.x());
        out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType::F32));

        constexpr auto data_layout = DataLayout::NCHW;
        /* -----------------------End: [Initialize tensors] */

        /* [Configure functions] */
        // in:32x32x1: 5x5 convolution, 8 output features maps (OFM)
        conv0->configure(&src, &weights0, &biases0, &out_conv0, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 2 /* pad_x */, 2 /* pad_y */));

        // in:32x32x8, out:32x32x8, Activation function: relu
        act0.configure(&out_conv0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // in:32x32x8, out:16x16x8 (2x2 pooling), Pool type function: Max
        pool0.configure(&out_act0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 2, data_layout, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));

        // in:16x16x8: 3x3 convolution, 16 output features maps (OFM)
        conv1->configure(&out_pool0, &weights1, &biases1, &out_conv1, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 1 /* pad_x */, 1 /* pad_y */));

        // in:16x16x16, out:16x16x16, Activation function: relu
        act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // in:16x16x16, out:8x8x16 (2x2 pooling), Pool type function: Average
        pool1.configure(&out_act1, &out_pool1, PoolingLayerInfo(PoolingType::AVG, 2, data_layout, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));

        // in:8x8x16, out:128
        fc0->configure(&out_pool1, &weights2, &biases2, &out_fc0);

        // in:128, out:128, Activation function: relu
        act2.configure(&out_fc0, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // in:128, out:128
        softmax->configure(&out_act2, &out_softmax);
        /* -----------------------End: [Configure functions] */

        /*[ Add tensors to memory manager ]*/
        // We need 2 memory groups for handling the input and output
        // We call explicitly allocate after manage() in order to avoid overlapping lifetimes
        memory_group0 = arm_compute::support::cpp14::make_unique<MemoryGroup>(mm_transitions);
        memory_group1 = arm_compute::support::cpp14::make_unique<MemoryGroup>(mm_transitions);

        memory_group0->manage(&out_conv0);
        out_conv0.allocator()->allocate();
        memory_group1->manage(&out_act0);
        out_act0.allocator()->allocate();
        memory_group0->manage(&out_pool0);
        out_pool0.allocator()->allocate();
        memory_group1->manage(&out_conv1);
        out_conv1.allocator()->allocate();
        memory_group0->manage(&out_act1);
        out_act1.allocator()->allocate();
        memory_group1->manage(&out_pool1);
        out_pool1.allocator()->allocate();
        memory_group0->manage(&out_fc0);
        out_fc0.allocator()->allocate();
        memory_group1->manage(&out_act2);
        out_act2.allocator()->allocate();
        memory_group0->manage(&out_softmax);
        out_softmax.allocator()->allocate();

        /* -----------------------End: [ Add tensors to memory manager ] */

        /* [Allocate tensors] */
        // Now that the padding requirements are known we can allocate all tensors
        src.allocator()->allocate();
        weights0.allocator()->allocate();
        weights1.allocator()->allocate();
        weights2.allocator()->allocate();
        biases0.allocator()->allocate();
        biases1.allocator()->allocate();
        biases2.allocator()->allocate();
        /* -----------------------End: [Allocate tensors] */

        /* [fill tensors] */
        // After allocation fill the tensor from the npy object
        npy0.fill_tensor(src);
        npy1.fill_tensor(weights0);
        npy2.fill_tensor(biases0);
        npy3.fill_tensor(weights1);
        npy4.fill_tensor(biases1);
        npy5.fill_tensor(weights2);
        npy6.fill_tensor(biases2);
        //std::cout<< "This is where I will print the biases matrix" << std::endl;
        //biases0.print(std::cout);
        /* -----------------------End: [fill tensors] */


        // Populate the layers manager. (Validity checks, memory allocations etc)
        mm_layers->populate(allocator, 1 /* num_pools */);

        // Populate the transitions manager. (Validity checks, memory allocations etc)
        mm_transitions->populate(allocator, 2 /* num_pools */);

        return true;
    }
    void do_run() override
    {
        // Acquire memory for the memory groups
        memory_group0->acquire();
        memory_group1->acquire();

        conv0->run();
    	save_to_npy(out_conv0,"out_conv0.npy",is_fortran);
        act0.run();
        save_to_npy(out_act0,"out_act0.npy",is_fortran);
        pool0.run();
        save_to_npy(out_pool0,"out_pool0.npy",is_fortran);
        conv1->run();
        save_to_npy(out_conv1,"out_conv1.npy",is_fortran);
        act1.run();
        save_to_npy(out_act1,"out_act1.npy",is_fortran);
        pool1.run();
        save_to_npy(out_pool1,"out_pool1.npy",is_fortran);
        fc0->run();
        save_to_npy(out_fc0,"out_fc0.npy",is_fortran);
        act2.run();
        save_to_npy(out_act2,"out_act2.npy",is_fortran);
        softmax->run();
//		out_softmax.print(std::cout);
    	save_to_npy(out_softmax,"out_softmax.npy",is_fortran);

        // Release memory
        memory_group0->release();
        memory_group1->release();
    }

private:
    // these variables for saving to numpy
    bool is_fortran{};
    // The src tensor should contain the input image
    Tensor src{};
    // Intermediate tensors used
    Tensor weights0{};
    Tensor weights1{};
    Tensor weights2{};
    Tensor biases0{};
    Tensor biases1{};
    Tensor biases2{};
    Tensor out_conv0{};
    Tensor out_conv1{};
    Tensor out_act0{};
    Tensor out_act1{};
    Tensor out_act2{};
    Tensor out_pool0{};
    Tensor out_pool1{};
    Tensor out_fc0{};
    Tensor out_softmax{};

    // NEON allocator
    Allocator allocator{};

    // Memory groups
    std::unique_ptr<MemoryGroup> memory_group0{};
    std::unique_ptr<MemoryGroup> memory_group1{};

    // Layers
    std::unique_ptr<NEConvolutionLayer>    conv0{};
    std::unique_ptr<NEConvolutionLayer>    conv1{};
    std::unique_ptr<NEFullyConnectedLayer> fc0{};
    std::unique_ptr<NESoftmaxLayer>        softmax{};
    NEPoolingLayer                         pool0{};
    NEPoolingLayer                         pool1{};
    NEActivationLayer                      act0{};
    NEActivationLayer                      act1{};
    NEActivationLayer                      act2{};
};

/** Main program for cnn test
 *
 * The example implements the following CNN architecture:
 *
 * Input -> conv0:5x5 -> act0:relu -> pool:2x2 -> conv1:3x3 -> act1:relu -> pool:2x2 -> fc0 -> act2:relu -> softmax
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEONCNNExample>(argc, argv);
}
