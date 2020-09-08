# Cell-Instance-Segmentation

## project plan
**Basic requirements:** <model, trainer, post> can be replaced
**Remember local, github, server should always keep consistent
1. Structure:
    - scripts
        - main.py
    - configs
        - config
        - .yaml
    - model
        - hover-net
        - seg + class
        - basic module
    - engine
        - trainer
    - utils
        - post-proc
        - moniter
2. Schedule
    - 2020/7/25 start the project
    - 2020/7/26 build framework in `main` and `trainer` and `config`
    - 2020/7/26 build finished `logging`, `model`, `config`. \
      to do `loss`, `moniter`, `data aug` 
    - 2020/7/27 HVDataset get id for training and testing
    - 2020/8/9 todo: aug, check hvdataset padding and calculation
    - 2020/9/1 todo: add independent augmentor file
    - 2020/9/3 finished `data` todo: `trainer`
    - 2020/9/4 finish `trainer`
    - 2020/9/5 test and run
    - 2020/9/8 finish `loss`