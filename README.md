# MultiGPU_Study
Build a Deep Learning (DL) Server and Test 
## Farhad Navid
This post has a recipe for building a DL server With some tidbits on the choices.  
### Part List
* Motherboard: Z10PE-08WS. 
* CPU: Xeon E5 V4 2620.  
* RAM: 128GB DDR4 2133 MHz speed.
* SSD: M.2 1TB 
* GPU: NVIDIA GFORCE 1080 TI 
* HDD: 10TB HDD, 7200 RPM with 256 M buffer 
* PSU: 1300 W power supply
* RAC: 4U Rack Mount Chassis. 

### Software Configuration 
The following recommendation are the a good starting point:
* OS: Ubuntu 16.04.5 LTS
* Python 3.5.2 
* Keras 2.2.4
* TensorFlow 1.12.0
* Pytorch 0.4.1
* Nvidia Driver 384.130 (Can be updated to 412.130)
* CUDA V9.0.176
* OpenCV 2.4.9.1
### Files
* FarhadNavid_final.pptx  contains details of the study with summary result.
* For the Mnist and Multi GPU 

# Conclusion
* Small Data set and small network the improvement in training time utilizing two GPU is about 1.6 improvement (248.3/154.2). 
* Larger Network and larger size dataset.  The improvement was 1.3 (49:12/38:16), 
* The majority of the over head in both cases could be attributed 
  * Communication between the GPU and CPU for parameter averaging.    
  * Some inefficiency due to the binary file of TensorFlow was not compiled for Xeon CPU. Identified a resource to compile the TensorFlow for Xeon processor.
  
# Refrences 
* Chen, K., & Huo, Q. (2016). Scalable training of deep learning machines by incremental block training with intra-block parallel optimization and blockwise model-update filtering. 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). doi:10.1109/icassp.2016.7472805
* Gupta, S., Zhang, W., & Wang, F. (2016, December 5). Retrieved from https://arxiv.org/pdf/1509.04210.pdf
* Su, H., Chen, H., & Xu, H. (2015, July 1). [1507.01239] Experiments on Parallel Training of Deep Neural Network using Model Averaging. Retrieved from https://arxiv.org/abs/1507.01239
* Zhang, W., Gupta, S., Lian, X., & Liu, J. (2016). Staleness-aware async-SGD for distributed deep learning. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI'16), 2350-2356. Retrieved from https://arxiv.org/pdf/1511.05950.pdf. 

[TensorFlow](https://www.tensorflow.org/tutorials/images/deep_cnn)

[PyTorch](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)

[Mnist-Multi](https://github.com/normanheckscher/mnist-multi-gpu) 
* Perfect reference for Multi GPU in Keras. Used only the Keras portion of the code and modified the model section to run a personal model. Result outlined in conclusion

[Cifar10-Multi](https://github.com/bearpaw/pytorch-classification)
* Excellent source for PyTorch classification utilizing multi GPU with both CIFAR10 data and imagenet. However, for this study only used the cifar10dataset and modified the directories to use only the vgg19_bn model (VGG 19 with Batch Normalization). The directory structure was modified locally on the server to run only the vgg19_bn model with One and two GPU.  Result Outlined in conclusion.    


 
