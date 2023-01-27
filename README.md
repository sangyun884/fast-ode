# Fast-ODE

This is the codebase for our paper Minimizing Trajectory Curvature of ODE-based Generative Models.

![Teaser image](./images/main.jpg)

> **Minimizing Trajectory Curvature of ODE-based Generative Models**<br>
> Sangyun Lee<sup>1</sup>, Beomsu Kim<sup>2</sup>, ‪Jong Chul Ye<sup>2</sup>

> <sup>1</sup>Soongsil University, <sup>2</sup>KAIST

> Paper: https://arxiv.org/abs/coming-soon<br>

> **Abstract:** *Recent ODE/SDE-based generative models, such as diffusion models and flow matching, define a generative process as a time reversal of a fixed forward process. Even though these models show impressive performance on large-scale datasets, numerical simulation requires multiple evaluations of a neural network, leading to a slow sampling speed. We attribute the reason to the high curvature of the learned generative trajectories, as it is directly related to the truncation error of a numerical solver. Based on the relationship between the forward process and the curvature, here we present an efficient method of training the forward process to minimize the curvature of generative trajectories without any ODE/SDE simulation. Experiments show that our method achieves a lower curvature than previous models and, therefore, decreased sampling costs while maintaining competitive performance.*

## Usage
`train_reverse_2d_joint.py`: Train code on two mode Gaussian example.

`train_reverse_img_ddp`: Train code on image data.

`fid.py`: Calculate FID score.

### Train CIFAR-10
```python
 python train_reverse_img_ddp.py --gpu 0,1 --dir ./runs/cifar10-beta20/ --weight_prior 20 --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000 --optimizer adam --batchsize 128 --iterations 500000 --config_en configs\cifar10_en.json --config_de configs\cifar10_de.json
 ```

### Train MNIST
```python
 python train_reverse_img_ddp.py --gpu 0,1 --dir ./runs/mnist-beta20/ --weight_prior 20 --learning_rate 3e-4 --dataset mnist --warmup_steps 8000 --optimizer adam --batchsize 256 --iterations 60000 --config_en configs\mnist_en.json --config_de configs\mnist_de.json
 ```


### MNIST distillation
```python
 python distill.py --gpu 0 --config_de ./configs/mnist_de.json --dir test --im_dir C:\ML\learned-flow\mnist-learned-beta5\60000-N128-num100K\samples --im_dir_test C:\ML\learned-flow\mnist-learned-beta5\60000-N128-num100K\samples_test --z_dir C:\ML\learned-flow\mnist-learned-beta5\60000-N128-num100K\zs --z_dir_test C:\ML\learned-flow\mnist-learned-beta5\60000-N128-num100K\zs_test --batchsize 256 --ckpt D:\ML\learned-flows\runs\reverse\mnist-learned-beta5\flow_model_60000_ema.pth 
 ```

### Generate MNIST
```python
 python generate.py --gpu 0 --dir test --N 100 --res 28 --input_nc 1 --num_samples 10 --ckpt D:\ML\learned-flows\runs\reverse\mnist-learned-beta20\flow_model_60000_ema.pth --config_de configs\mnist_de.json 
 ```



### Generate MNIST from posterior
```python
 python generate.py --gpu 0 --dir test --N 100 --res 28 --input_nc 1 --num_samples 10 --ckpt D:\ML\learned-flows\runs\reverse\mnist-learned-beta20\flow_model_60000_ema.pth --encoder D:\ML\learned-flows\runs\reverse\mnist-learned-beta20\forward_model_60000_ema.pth --config_en configs\mnist_en.json --config_de configs\mnist_de.json --dataset mnist 
 ```


### Calcuate FID on cifar10
```python
python fid.py calc --images=runs\reverse\cifar10-learned-beta10-smallE\300000-N128\samples --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```

CIFAR-10 training roughly takes 9 days on 2x1080Ti.


## Environment
Tested environment: PyTorch 1.12.0 / 1.11.0, Python 3.8.5, Windows 10, CUDA 10.1

## Citation

If you find this work useful for your research, please cite our paper:

```
Coming soon
```