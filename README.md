# Conditioned-U-Net-pytorch
An unofficial pytorch implementation of Conditioned-U-Net



## installation

```
conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge ffmpeg librosa
conda install -c anaconda jupyter
pip install musdb museval pytorch_lightning effortless_config tensorboard wandb pydub
pip install https://github.com/PytorchLightning/pytorch-lightning/archive/0.9.0rc12.zip --upgrade
```


## Training
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
                        
### example
```shell script
/train.py --musdb_root ../repos/musdb18_wav --filed_mode True --n_fft 2048 --hop_length 512 --num_frame 128 --filters_layer_1 24 --last_activation sigmoid --film_type complex --num_workers 20 --pin_memory True --log_system wandb --float16 True --batch_size 128 --gpus 2 --distributed_backend ddp --save_top_k 20 --patience 20
```
## Evaluation

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
            
### example
```shell script
/eval.py --musdb_root ../repos/musdb18_wav --filed_mode True --n_fft 2048 --hop_length 512 --num_frame 128 --filters_layer_1 24 --last_activation sigmoid --film_type complex --num_workers 20 --pin_memory True --log_system wandb --float16 True --batch_size 128 --gpus 1 --run_id complex_2048_512_128 --model_name cunet --epoch 52
```

## Reference
[1] Meseguer-Brocal, Gabriel, and Geoffroy Peeters. "CONDITIONED-U-NET: INTRODUCING A CONTROL MECHANISM IN THE U-NET FOR MULTIPLE SOURCE SEPARATIONS." Proceedings of the 20th International Society for Music Information Retrieval Conference. 2019.

> @inproceedings{Meseguer-Brocal_2019, Author = {Meseguer-Brocal, Gabriel and Peeters, Geoffroy}, Booktitle = {20th International Society for Music Information Retrieval Conference}, Editor = {ISMIR}, Month = {November}, Title = {CONDITIONED-U-NET: Introducing a Control Mechanism in the U-net For Multiple Source Separations.}, Year = {2019}}

[2] Official Github Repository, (Tensorflow-based): Conditioned-U-Net [Conditioned-U-Net for multitask musical instrument source separations](https://github.com/gabolsgabs/cunet)

