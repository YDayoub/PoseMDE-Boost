# PoseMDE-Boost

This repository provides the official code implementation for the paper **"Boosting Depth Estimation for Self-Driving in a Self-Supervised Framework via Improved Pose Network"**. The code integrates enhancements to the pose estimation networks for depth estimation in self-driving applications.

## Overview

This implementation improves depth estimation by integrating an enhanced pose network into a self-supervised framework.

## Prerequisites

To run this code, follow the prerequisites of the corresponding depth model:

- **MonodepthV2**: [MonodepthV2 GitHub Repository](https://github.com/nianticlabs/monodepth2)
- **VTDepth**: [VTDepth GitHub Repository](https://github.com/ahbpp/VTDepth)
- **MonoViT**: [MonoViT GitHub Repository](https://github.com/zxcqlf/MonoViT)
- **CADepth**: [CADepth GitHub Repository](https://github.com/kamiLight/CADepth-master)
- **Lite-Mono**: [Lite-Mono GitHub Repository](https://github.com/noahzn/Lite-Mono)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Install the required Python packages. It is recommended to use a virtual environment:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

For data preparation and usage, refer to the instructions in the [MonodepthV2 repository](https://github.com/nianticlabs/monodepth2).

## Usage

To use this implementation, follow these steps:

1. **Download and Prepare Data**: Follow the data preparation instructions in the MonodepthV2 repository.

2. **Replace Pose Networks**: Replace the pose network files in the corresponding depth model repository with the provided `pose_encoder.py` and `pose_decoder.py` from this repository.

3. **Run the Code**: Refer to the specific depth model repository for instructions on running the code with the new pose networks. Update any configuration files or scripts as needed to incorporate the modified pose networks.

## Files

- `pose_encoder.py`: Implementation of the PoseEncoder network.
- `pose_decoder.py`: Implementation of the PoseDecoder network.

## Acknowledgements

This work builds upon several open-source projects and depth estimation models. Special thanks to the authors and contributors of the following repositories:

- [MonodepthV2](https://github.com/nianticlabs/monodepth2)
- [VTDepth](https://github.com/ahbpp/VTDepth)
- [MonoViT](https://github.com/zxcqlf/MonoViT)
- [CADepth](https://github.com/kamiLight/CADepth-master)
- [Lite-Mono](https://github.com/noahzn/Lite-Mono)

## Contact

For questions or issues, please open an issue in this repository or contact [yazandayoubsyria@gmail.com](mailto:yazandayoubsyria@gmail.com).
