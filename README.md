
# Conditioned-U-Net-pytorch
An unofficial pytorch implementation of Conditioned-U-Net

## News

An [extension](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) of this model was released.



## Installation

```
conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge ffmpeg librosa
conda install -c anaconda jupyter
pip install musdb museval pytorch_lightning effortless_config tensorboard wandb pydub
pip install https://github.com/PytorchLightning/pytorch-lightning/archive/0.9.0rc12.zip --upgrade
```

## Evaluation Result

|Name                     |control_input_dim|control_n_layer|control_type|decoder_activation|encoder_activation|film_type|filters_layer_1|hop_length|input_channels|kernel_size|last_activation|lr   |n_fft|n_layers|num_frame|optimizer|stride|test_result/agg/bass_ISR|test_result/agg/bass_SAR|test_result/agg/bass_SDR|test_result/agg/bass_SIR|test_result/agg/drums_ISR|test_result/agg/drums_SAR|test_result/agg/drums_SDR|test_result/agg/drums_SIR|test_result/agg/other_ISR|test_result/agg/other_SAR|test_result/agg/other_SDR|test_result/agg/other_SIR|test_result/agg/vocals_ISR|test_result/agg/vocals_SAR|test_result/agg/vocals_SDR|test_result/agg/vocals_SIR|
|-------------------------|-----------------|---------------|------------|------------------|------------------|---------|---------------|----------|--------------|-----------|---------------|-----|-----|--------|---------|---------|------|------------------------|------------------------|------------------------|------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
|complex_2048_512_128eval |4                |4              |dense       |relu              |leaky_relu        |complex  |24             |512       |2             |[5,5]      |sigmoid        |0.001|2048 |6       |128      |adam     |[2,2] |8.84835                 |4.81325                 |2.795465                |4.114615                |9.69044                  |4.2979225                |3.492365                 |4.63526                  |6.93455                  |3.87871                  |1.85376                  |1.0855625                |6.0647475                 |2.2080925                 |2.49749                   |8.3487875                 |
|complex_32eval_          |4                |4              |dense       |relu              |leaky_relu        |complex  |32             |256       |2             |[5,5]      |sigmoid        |0.001|1024 |6       |256      |adam     |[2,2] |8.0865575               |4.79529                 |2.1145975               |2.6459025               |10.019905                |4.9158075                |3.795275                 |4.92333                  |7.5122025                |4.58683                  |1.705415                 |1.07406                  |7.470695                  |3.63371                   |2.415865                  |6.5487125                 |
|cunet_mme_sigmoid_32-eval|4                |4              |dense       |relu              |leaky_relu        |simple   |32             |256       |2             |[5,5]      |sigmoid        |0.001|1024 |6       |256      |adam     |[2,2] |7.6462525               |4.90194                 |1.84956                 |1.9313625               |9.4997225                |4.6694725                |3.327125                 |4.113235                 |7.648405                 |4.659825                 |1.500495                 |0.5541025                |6.710985                  |3.602105                  |2.12235                   |5.72728                   |

## How to use

### Training
- train.py 
    - parameters related to dataset
        - --**musdb_root**  ```your musdb path``` 
        - --**musdb_is_wav** ```True``` 
        - --**filed_mode** ```False```
    - parameters for the model configuration
        - --model_name ```cunet```
            - we only support ```cunet``` currently
        - stft parameters
            - --n_fft ```1024``` 
            - --hop_size ```512``` 
            - --num_frame ```256```
        - Condition generator parameters
            - --film_type ```simple```
            - --filters_layer_1 ```32```  
            - --control_type ```dense```
                - we only support dense currently.
                 - TODO: conv control
            - --control_n_layer ```4``` 
        - U-Net parameters
            - --n_layers ```6```
            - --stride ```(2,2)```
            - --kernel_size ```(5,5)```
            - --last_activation ```sigmoid```
            - --encoder_activation ```leaky_relu```
            - --decoder_activation ```relu```

    - parameters for the training env.
        - lr ```0.001``` 
        - optimizer ```adam```
        - --gpus ```1```
            - warn 1 (important): if you want to use multi gpus, then we recommend you to use ddp for the distributed_backend, i.e., --distributed_backend ddp
            - warn 2 (important): however, it seems that lightning currently does not support synchronized ```on_validation_epoch_end``` so that some log operations might be lost when you try to append logs for every instance in ```on_validation_epoch_end``` in ddp mode.
        - --batch_size ```your batch size```
        - --num_workers ```number of workders```
        - --pin_memory ```True```
        - --log_system ```True```
            - or you can use wandb
        - --patience ```20```
            - for early stop
        - --checkpoints_path ```your_path``
            - audio checkpoints are stored in here.
        - --save_top_k 
            - for audio checkpoint saving
        - --run_id ```run_id```
            - if you want to name this run, then use this. default: time stamp
        - --**dev_mode** ```True```
            - if True, then every dataset deals with 1~4 tracks, which are much smaller than those of counterparts.
        - --float16 ```True```
            - if True, then 16 precision training enabled
                        
#### example
```shell script
/train.py --musdb_root ../repos/musdb18_wav --filed_mode True --n_fft 2048 --hop_length 512 --num_frame 128 --filters_layer_1 24 --last_activation sigmoid --film_type complex --num_workers 20 --pin_memory True --log_system wandb --float16 True --batch_size 128 --gpus 2 --distributed_backend ddp --save_top_k 20 --patience 20
```
### Evaluation

- eval.py 
    - parameters related to dataset
        - --**musdb_root**  ```your musdb path``` 
        - --**musdb_is_wav** ```True``` 
        - --**filed_mode** ```False```
    - parameters for the model configuration
        - --model_name ```cunet```
            - we only support ```cunet``` currently
        - stft parameters
            - --n_fft ```1024``` 
            - --hop_size ```512``` 
            - --num_frame ```256```
        - Condition generator parameters
            - --film_type ```simple```
            - --filters_layer_1 ```32```  
            - --control_type ```dense```
                - we only support dense currently.
                 - TODO: conv control
            - --control_n_layer ```4``` 
        - U-Net parameters
            - --n_layers ```6```
            - --stride ```(2,2)```
            - --kernel_size ```(5,5)```
            - --last_activation ```sigmoid```
            - --encoder_activation ```leaky_relu```
            - --decoder_activation ```relu```

    - parameters for the Evaluation env.
        - --gpus ```1```
            - if use set gpus > 1, then automatically eval.py resets it to be 1 :(.
            - It seems that lightning currently does not support synchronized ```on_validation_epoch_end``` .
            - Although we have to log every single bbs metric for each track in musdb.test,
                -we found that some logs are lost when we use ddp.
            - I think that multiple-gpus with dp will work, but i have not tested it yet.            
            - To prevent ghost logs, we currently set gpus = 1.
        - --batch_size ```your batch size```
        - --num_workers ```number of workders```
        - --pin_memory ```True```
        - --log_system ```True```
            - or you can use wandb
        - --checkpoints_path ```your_path``
            - audio checkpoints are stored in here.
        - --run_id ```run_id you want to eval```
        - --epoch ```the epoch (int) you want to eval) ```
        - --**dev_mode** ```True```
            - if True, then every dataset deals with 1~4 tracks, which are much smaller than those of counterparts.
        - --float16 ```True```
            - if True, then 16 precision training enabled
            
#### example
```shell script
/eval.py --musdb_root ../repos/musdb18_wav --filed_mode True --n_fft 2048 --hop_length 512 --num_frame 128 --filters_layer_1 24 --last_activation sigmoid --film_type complex --num_workers 20 --pin_memory True --log_system wandb --float16 True --batch_size 128 --gpus 1 --run_id complex_2048_512_128 --model_name cunet --epoch 52
```

## Reference
[1] Meseguer-Brocal, Gabriel, and Geoffroy Peeters. "CONDITIONED-U-NET: INTRODUCING A CONTROL MECHANISM IN THE U-NET FOR MULTIPLE SOURCE SEPARATIONS." Proceedings of the 20th International Society for Music Information Retrieval Conference. 2019.

> @inproceedings{Meseguer-Brocal_2019, Author = {Meseguer-Brocal, Gabriel and Peeters, Geoffroy}, Booktitle = {20th International Society for Music Information Retrieval Conference}, Editor = {ISMIR}, Month = {November}, Title = {CONDITIONED-U-NET: Introducing a Control Mechanism in the U-net For Multiple Source Separations.}, Year = {2019}}

[2] Official Github Repository, (Tensorflow-based): Conditioned-U-Net [Conditioned-U-Net for multitask musical instrument source separations](https://github.com/gabolsgabs/cunet)

