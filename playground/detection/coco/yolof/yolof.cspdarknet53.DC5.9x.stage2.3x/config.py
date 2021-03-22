from cvpods.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # Backbone NAME: "build_darknet_backbone"
        WEIGHTS="../../../../../pretrained_models/YOLOF_CSP_D_53_DC5_9x.pth",
        # or
        # WEIGHTS="../yolof.cspdarknet53.DC5.9x/log/model_final.pth",
        DARKNET=dict(
            DEPTH=53,
            WITH_CSP=True,
            NORM="SyncBN",
            OUT_FEATURES=["res5"],
            RES5_DILATION=2
        ),
        ANCHOR_GENERATOR=dict(
            SIZES=[[16, 32, 64, 128, 256, 512]],
            ASPECT_RATIOS=[[1.0]]
        ),
        YOLOF=dict(
            ENCODER=dict(
                IN_FEATURES=["res5"],
                NUM_CHANNELS=512,
                BLOCK_MID_CHANNELS=128,
                NUM_RESIDUAL_BLOCKS=8,
                BLOCK_DILATIONS=[1, 2, 3, 4, 5, 6, 7, 8],
                NORM="SyncBN",
                ACTIVATION="LeakyReLU"
            ),
            DECODER=dict(
                IN_CHANNELS=512,
                NUM_CLASSES=80,
                NUM_ANCHORS=6,
                CLS_NUM_CONVS=2,
                REG_NUM_CONVS=4,
                NORM="SyncBN",
                ACTIVATION="LeakyReLU",
                PRIOR_PROB=0.01
            ),
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            ADD_CTR_CLAMP=True,
            CTR_CLAMP=32,
            MATCHER_TOPK=8,
            POS_IGNORE_THRESHOLD=0.1,
            NEG_IGNORE_THRESHOLD=0.8,
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.6
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    DATALOADER=dict(NUM_WORKERS=8),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(52500, 62500),
            MAX_ITER=67500,
            WARMUP_FACTOR=0.00066667,
            WARMUP_ITERS=1500
        ),
        OPTIMIZER=dict(
            NAME="D2SGD",
            BASE_LR=0.04,
            BIAS_LR_FACTOR=1.0,
            WEIGHT_DECAY=0.0001,
            WEIGHT_DECAY_NORM=0.0,
            MOMENTUM=0.9,
            BACKBONE_LR_FACTOR=1.0
        ),
        IMS_PER_BATCH=64,
        IMS_PER_DEVICE=8,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("JitterCrop", dict(jitter_ratio=0.3)),
                ("Resize", dict(shape=(640, 640), scale_jitter=(0.8, 1.2))),
                ("RandomDistortion2",
                 dict(hue=0.1, saturation=1.5, exposure=1.5)),
                ("RandomFlip", dict()),
                ("RandomShift", dict(max_shifts=32))
            ],
            TEST_PIPELINES=[
                ("Resize", dict(shape=(608, 608))),
            ],
        ),
        MOSAIC=dict(
            MIN_OFFSET=0.2,
            MOSAIC_WIDTH=640,
            MOSAIC_HEIGHT=640
        ),
        # Whether the model needs RGB, YUV, HSV etc.
        FORMAT="BGR",
    ),
)


class YOLOFConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = YOLOFConfig()
