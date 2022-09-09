"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""

AIDataPrefix = "/apdcephfs_cq2/share_1630463/portrait_matting/"
DATA_PATHS = {
    # Matting Datasets
    'videomatteSD': {
        'train': AIDataPrefix + 'VideoMatte240K_JPEG_SD/train',
        'valid': AIDataPrefix + 'VideoMatte240K_JPEG_SD/test',
    },
    'videomatte': {
        'train': AIDataPrefix + 'VideoMatte240K_JPEG_HD/train',
        'valid': AIDataPrefix + 'VideoMatte240K_JPEG_HD/test',
    },
    'imagematte': {
        'train': AIDataPrefix + 'ImageMatte/train',
        'valid': AIDataPrefix + 'ImageMatte/valid',
    },

    # Background Datasets
    'background_images': {
        'train': AIDataPrefix + 'BackgroundImages/train',
        'valid': AIDataPrefix + 'BackgroundImages/valid',
    },
    'background_videos': {
        'train': AIDataPrefix + 'BackgroundVideos/train',
        'valid': AIDataPrefix + 'BackgroundVideos/valid',
    },
    
    # Segmentation Datasets
    'coco_panoptic': {
        'imgdir': AIDataPrefix + 'coco/train2017/',
        'anndir': AIDataPrefix + 'coco/annotations/panoptic_train2017/',
        'annfile': AIDataPrefix + 'coco/annotations/panoptic_train2017.json',
    },
    'spd': {
        'imgdir': AIDataPrefix + 'SuperviselyPersonDataset/img',
        'segdir': AIDataPrefix + 'SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir': AIDataPrefix + 'YouTubeVIS/train/JPEGImages',
        'annfile': AIDataPrefix + 'YouTubeVIS/train/instances.json',
    }
    
}

import os
if __name__ == '__main__':
    for pkey, pval in DATA_PATHS.items():
        for stage, path in pval.items():
            if not os.path.exists(path):
                print("[%s][%s][%s] Not Exists" % (pkey, stage, path))


