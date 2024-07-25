# Improving Adversarial Transferability via Model Alignment

This is the official repository of "Improving Adversarial Transferability via Model Alignment" accepted at the European Conference on Computer Vision (ECCV) 2024 ([paper](https://arxiv.org/pdf/2311.18495)).

![Improved Transferability](figures/improvement.png)
**Attacking the aligned source model for more transferable perturbations.** 
We compare the transferability of $\ell_\infty$-norm constrained perturbations (ϵ = 4/255) generated using the source model before and after performing model alignment. The result highlights the compatibility of model alignment with a wide range of attacks, as perturbations generated from the aligned source model become more transferable. Here, the source model is aligned using a witness model from the same architecture but is initialized and trained independently. Results are averaged over all target models.

## Requirements
To run the code, the following packages are needed:
- Python 3.9.15
- PyTorch 2.0.1
- torchvision 0.15.2
- numpy 1.22.4
- torchattacks 3.4.0
- timm 1.0.3

## Checkpoints
Access our model checkpoints [here](https://drive.google.com/drive/folders/).

## Model Alignment
- To align 
```
python3 main.py
```

## License
MIT License
