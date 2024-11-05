# PseR
official code for PseR: Pseudo-label Refinement for Point-Supervised Temporal Action Detection

# Install

- Install the python environment and get dataset of [HR-Pro](https://github.com/pipixin321/HR-Pro)
- `cd models/ops` and run: `python setup.py build_ext --inplace`
- `cd ../..`

# Generate seed proposals

PseR generates seed proposals based on existing methods. Take [LACP](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points), for example

- Run LACP to get the initial prediction
- Execute `python seed_process/ge_seed_proposal.py` to get `lacp_seed_final.json`
- [Optionally], you can use the [lacp_seed_final.json](https://www.alipan.com/s/MKxZBUcxNnF)[7vx4] we have already gotten

# Training PseR

- Run `python main.py` to get the pseudo-label of the PseR prediction: `lacp_pser.json`
- [Optionally], you can use the [lacp_pser.json](https://www.alipan.com/s/4WcAfX8z2QB)[ip52] we have already gotten

# Training TAD

We trained on the THUMOS'14 dataset based on [OpenTAD](https://github.com/sming256/OpenTAD) and lacp_pser.json

# Results

ActF stands for ActionFormer

| Model | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LACP  |  64.6  |  56.5  |  45.3  |  34.5  |  21.8  |  44.5   | | |
| LACP+ActF  |  77.1  |  68.5  |  56.2  |  41.8  |  23.3  |  53.4   | [config](opentad_dir/thumos_actf_i3d_lacp_pse/gpu1_id0/thumos_i3d_lacp_pse.py) | [log](https://www.alipan.com/s/HyMx892CZME)[k7b4] \| [model](https://pan.baidu.com/s/1iZ8TZslynw2ADRlJvqNsDA?pwd=9wjr) |
| Ours+ActF |  78.5  |  70.9  |  59.3  |  43.2  |  26.2  |  55.6   | [config](opentad_dir/thumos_actf_i3d_lacp_pser/gpu1_id0/thumos_i3d_lacp_pser.py) | [log](https://www.alipan.com/s/dV1hVGHeKxu)[a67a] \| [model](https://pan.baidu.com/s/1Si4JYKy9o_CKKmKuQkfkgg?pwd=qeh3) |

# Acknowledgement
Our code is based on [HR-Pro](https://github.com/pipixin321/HR-Pro), [OpenTAD](https://github.com/sming256/OpenTAD), [LACP](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points). We would like to express our gratitude for their outstanding work
