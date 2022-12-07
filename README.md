# Combining EfficientNet and Vision Transformers for Video Deepfake Detection
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/combining-efficientnet-and-vision/deepfake-detection-on-dfdc)](https://paperswithcode.com/sota/deepfake-detection-on-dfdc?p=combining-efficientnet-and-vision)


Code for Video Deepfake Detection model from "Combining EfficientNet and Vision Transformers for Video Deepfake Detection" available on Arxiv and presented at ICIAP 2021 [<a href="https://arxiv.org/abs/2107.02612">Pre-print PDF</a> | <a href="https://www.springerprofessional.de/en/combining-efficientnet-and-vision-transformers-for-video-deepfak/20403304">Springer</a>]. Using this repository it is possible to train and test the two main architectures presented in the paper, Efficient Vision Transformers and Cross Efficient Vision Transformers, for video deepfake detection.
The architectures exploits internally the <a href="https://github.com/lukemelas/EfficientNet-PyTorch">EfficientNet-Pytorch</a> and <a href="https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch">ViT-Pytorch</a> repositories.
# Setup
Clone the repository and move into it:
```
git clone https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection.git

cd Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection
```

Setup Python environment using conda:
```
conda env create --file environment.yml
conda activate deepfakes
export PYTHONPATH=.
```

# Get the data
Download and extract the dataset you want to use from:
- DFDC: https://dfdc.ai/
- FaceForensics++: https://github.com/ondyari/FaceForensics/blob/master/dataset/


# Preprocess the data
The preprocessing phase is based on <a href="https://github.com/selimsef/dfdc_deepfake_challenge">Selim Seferbekov implementation</a>.

In order to perform deepfake detection it is necessary to first identify and extract faces from all the videos in the dataset.
Detect the faces inside the videos:
```
cd preprocessing
python3 detect_faces.py --data_path "path/to/videos"
```
By default the consideted dataset structure will be the one of DFDC but you can customize it with the following parameter:
- --dataset: Dataset (DFDC / FACEFORENSICS)


The extracted boxes will be saved inside the "path/to/videos/boxes" folder.
In order to get the best possible result, make sure that at least one face is identified in each video. If not, you can reduce the threshold values of the MTCNN on line 38 of face_detector.py and run the command again until at least one detection occurs.
At the end of the execution of face_detector.py an error message will appear if the detector was unable to find faces inside some videos.

If you want to manually check that at least one face has been identified in each video, make sure that the number of files in the "boxes" folder is equal to the number of videos. To count the files in the folder use:
```
cd path/to/videos/boxes
ls | wc -l
```

Extract the detected faces obtaining the images:
```
python3 extract_crops.py --data_path "path/to/videos" --output_path "path/to/output"
```

By default the consideted dataset structure will be the one of DFDC but you can customize it with the following parameter:
- --dataset: Dataset (DFDC / FACEFORENSICS)

Repeat detection and extraction for all the different parts of your dataset.

After extracting all the faces from the videos in your dataset, organise the "dataset" folder as follows:
```
- dataset
    - training_set
        - Deepfakes
            - video_name_0
                0_0.png
                1_0.png
                2_0.png
                ...
                N_0.png
            ...
            - video_name_K
                0_0.png
                1_0.png
                2_0.png
                ...
                M_0.png
        - DFDC
        - Face2Face
        - FaceShifter
        - FaceSwap
        - NeuralTextures
        - Original
    - validation_set
        ...
            ...
                ...
                ...
    - test_set
        ...
            ...
                ...
                ...
```

We suggest to exploit the --output_path parameter when executing extract_crops.py to build the folders structure properly.

# Evaluate
Move into the choosen architecture folder you want to evaluate and download the pre-trained model:

(Efficient ViT)
```
cd efficient-vit
wget http://datino.isti.cnr.it/efficientvit_deepfake/efficient_vit.pth
```

(Cross Efficient ViT)
```
cd cross-efficient-vit
wget http://datino.isti.cnr.it/efficientvit_deepfake/cross_efficient_vit.pth
```


If you are unable to use the previous urls you can download the weights from [Google Drive](https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1?usp=sharing).


Then, issue the following commands for evaluating a given model giving the pre-trained model path and the configuration file available in the config directory:
```
python3 test.py --model_path "pretrained_models/[model]" --config "configs/architecture.yaml"
```

By default the command will test on DFDC dataset but you can customize the following parameters for both the architectures:
- --dataset: Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)
- --max_videos: Maximum number of videos to use for training (default: all)
- --workers: Number of data loader workers (default: 10)
- --frames_per_video: Number of equidistant frames for each video (default: 30)
- --batch_size: Prediction Batch Size (default: 32)



    
To evaluate a customized model trained from scratch with a different architecture you need to edit the configs/architecture.yaml file.

# Train
Only for DFDC dataset, prepare the metadata moving all of them (by default inside dfdc_train_part_X folders) into a subfolder:
```
mkdir data/metadata
cd path/to/videos/training_set
mv **/metadata.json ../../../data/metadata
```

In order to train the model using our architectures configurations use:

(Efficient ViT)
```
cd efficient-vit
python3 train.py --config configs/architecture.yaml
```

(Cross Efficient ViT)
```
cd cross-efficient-vit
python3 train.py --config configs/architecture.yaml
```

By default the commands will train on DFDC dataset but you can customize the following parameters for both the architectures:
- --num_epochs: Number of training epochs (default: 300)
- --workers: Number of data loader workers (default: 10)
- --resume: Path to latest checkpoint (default: none)
- --dataset: Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All) (default: All)
- --max_videos: Maximum number of videos to use for training (default: all)
- --patience: How many epochs wait before stopping for validation loss not improving (default: 5)
    

Only for the Efficient ViT model it's also possible to custom the patch extractor and use different versions of EfficientNet (only B0 and B7) by adding the following parameter:
- --efficient_net: Which EfficientNet version to use (0 or 7, default: 0)


# Reference
```
@InProceedings{10.1007/978-3-031-06433-3_19,
author="Coccomini, Davide Alessandro
and Messina, Nicola
and Gennaro, Claudio
and Falchi, Fabrizio",
editor="Sclaroff, Stan
and Distante, Cosimo
and Leo, Marco
and Farinella, Giovanni M.
and Tombari, Federico",
title="Combining EfficientNet and Vision Transformers for Video Deepfake Detection",
booktitle="Image Analysis and Processing -- ICIAP 2022",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="219--229",
isbn="978-3-031-06433-3"
}
```

