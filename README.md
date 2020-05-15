# Generating Stylized Dance Motions Using Motion Graphs

A common 3D animation problem is as follows: given states A and B and an unknown environment, generate a realitic motion between the two states. While this can be done easily for running and walking given enough data, dance motions have additional complexities in the form of audio-dependency and personal stylistic choice that relate to whether a generated motion is realistic or not. Here we use motion graphs to investigate the latter independently of the former as a proof of concept for adding more complexity in the future.

## Getting the Data

Data for one subject can be obtained by running the following command in console:
```
cd data
sh get_data.sh
```

## Constructing the Graph

Once the data is downloaded, run the following command to generate the graph:
```
python motion_graph.py
```

## Extracting Motion

To get the pose sequence, run the following commands:
```
python run_mg.py
```

Then to convert the sequence to pixels, we must apply a modified version of pix2pixHD as described here. Run the following commands to train the model and generate pixel mappings for the sequence for the desired style.

```
python train_encoder --label_nc 3 --no_html --dataroot PATH_TO_STYLE_IMAGES --name STYLE --resize_or_crop none --loadSize 640
python test_encoder.py --label_nc 3 --dataroot PATH_TO_POSES --name STYLE --resize_or_crop none --loadSize 640
```
