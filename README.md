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
### Consideration / Justification / Recommendation
**One can build a DL server with i7 seventh generation and two GPU a bit more economical, However The decision here was to build a system capable of expanding with more than two GPUs. 

* Motherboard: expandability, PCIe 3.0 ready, Fast SATA (NVMe or NVMe Express), # of PCIe slots.
* CPU: Power usage (85 W vs. 140 for i7), Cache size (min 2 Meg / GPU), Core/Tread (8), Cost, DDR4 memory speed. 
* RAM: Fastest common speed between the MB and CPU in this case that is ( min 2133 0r 2400 MHz)
* SSD: Fast boot device. Min 512 GB will do fine but if one can get 1 or 2 TB is preferred. The IOPS and seq throughput.
* GPU: Affordability, **Reasonable:** RTX 2070 TI and Gfroce 1080TI, **Good:** performance RTX 2080 TI.   
* HDD: Cost, Power consumption and transfer speed. 10 TB sweet spot now.
* PSU: Depending on number of GPU this case (3X250 + 200 mobo = 950. ) 1300 Watt EVGA
* RAC: Since this server will stay in data center . the 4 U was obvious choice, for the desktop a full tower with Min 8-10 prefered expansion slot is recommended. 

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
