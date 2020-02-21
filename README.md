# EXPERT - Learning to Learn Words from Narrated Video

Code from the paper [Learning to Learn Words from Narrated Video](https://arxiv.org/pdf/1911.11237.pdf)

Website of the project in [expert.cs.columbia.edu](https://expert.cs.columbia.edu)

If you use the code, please cite the paper as:

```
@article{suris2019learning,
  title={Learning to learn words from narrated video},
  author={Surís, Dídac and Epstein, Dave and Ji, Heng and Chang, Shih-Fu and Vondrick, Carl},
  journal={arXiv preprint arXiv:1911.11237},
  year={2019}
}
```

An example of command line execution can be found in `scripts/run.sh`

Run `python main.py --help` for information on arguments.

Be sure to have the external libraries in _requirements.txt_ installed.


## Data
We work with the [Epic Kitchens dataset](https://epic-kitchens.github.io/2019) for this project. To run our code, you 
will need to download their images and annotations. 

Specifically, the **annotations** directory has to contain:
- [EPIC_train_object_labels.csv](https://github.com/epic-kitchens/annotations/blob/master/EPIC_train_object_labels.csv)
(provided by the Epic Kitchens dataset) 
- [EPIC_video_info.csv](https://github.com/epic-kitchens/annotations/blob/master/EPIC_video_info.csv)
(provided by the Epic Kitchens dataset) 
- [splits.pth](https://expert.cs.columbia.edu/data/splits.pth). File containing our train/test splits.
- [processed_EPIC_train_action_labels](https://expert.cs.columbia.edu/data/processed_EPIC_train_action_labels.pth)

The path to this directory has to be introduced in the `--annotation_root` argument. A compressed _.tar.gz_ file with 
these four files can be downloaded [here](https://expert.cs.columbia.edu/data/annotations.tar.gz).

The **images** directory has to be specified using `--img_root`. It contains all the images with the following 
subfolder structure: `path_to_img_root/participant_id/vid_id/frame_{frame_id:010d}.jpg`. This is the default structure 
if you download the data from the Epic Kitchens website 
([download from here](https://data.bris.ac.uk/data/dataset/3h91syskeag572hl6tvuovwv4d)). For this project we only use
the RGB images, not flow information.


## Pretrained models
The pretrained models reported in our paper can be found in the following links:
- [Bert baseline](https://expert.cs.columbia.edu/models/bert_baseline.tar.gz)
- [Model with isolated attention](https://expert.cs.columbia.edu/models/isolated_attn.tar.gz)
- [Model with target-to-reference attention](https://expert.cs.columbia.edu/models/tgt_to_ref_attn.tar.gz)
- [Model with via-vision attention](https://expert.cs.columbia.edu/models/via_vision_attn.tar.gz)
- [Model with via-vision attention and input pointing](https://expert.cs.columbia.edu/models/via_vision_attn_input_pointing.tar.gz)
- [Model with full attention](https://expert.cs.columbia.edu/models/full_attn.tar.gz)

Each one of these is a _.tar.gz_ file containing the files necessary to load the model (_checkpoint_best.pth_, 
_config.json_ and _tokenizer.pth_). 

To resume training or to test from one of these pretrained models, set the `--resume` to _True_. 
Extract the models under the `/path/to/your/checkpoints` directory you introduce in 
the `--checkpoint_dir` argument. Refer to the specific model using the `--resume_name` argument.
