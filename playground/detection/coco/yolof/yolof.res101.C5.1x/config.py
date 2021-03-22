from cvpods.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # Backbone NAME: "build_resnet_backbone"
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        RESNETS=dict(DEPTH=101, OUT_FEATURES=["res5"]),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32, 64, 128, 256, 512]],
            ASPECT_RATIOS=[[1.0]]
        ),
        YOLOF=dict(
            ENCODER=dict(
                IN_FEATURES=["res5"],
                NUM_CHANNELS=512,
                BLOCK_MID_CHANNELS=128,
                NUM_RESIDUAL_BLOCKS=4,
                BLOCK_DILATIONS=[2, 4, 6, 8],
                NORM="BN",
                ACTIVATION="ReLU"
            ),
            DECODER=dict(
                IN_CHANNELS=512,
                NUM_CLASSES=80,
                NUM_ANCHORS=5,
                CLS_NUM_CONVS=2,
                REG_NUM_CONVS=4,
                NORM="BN",
                ACTIVATION="ReLU",
                PRIOR_PROB=0.01
            ),
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            ADD_CTR_CLAMP=True,
            CTR_CLAMP=32,
            MATCHER_TOPK=4,
            POS_IGNORE_THRESHOLD=0.15,
            NEG_IGNORE_THRESHOLD=0.7,
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
            STEPS=(15000, 20000),
            MAX_ITER=22500,
            WARMUP_FACTOR=0.00066667,
            WARMUP_ITERS=1500
        ),
        OPTIMIZER=dict(
            NAME="D2SGD",
            BASE_LR=0.12,
            BIAS_LR_FACTOR=1.0,
            WEIGHT_DECAY=0.0001,
            WEIGHT_DECAY_NORM=0.0,
            MOMENTUM=0.9,
            BACKBONE_LR_FACTOR=0.334
        ),
        IMS_PER_BATCH=64,
        IMS_PER_DEVICE=8,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(800,), max_size=1333,
                    sample_style="choice")),
                ("RandomFlip", dict()),
                ("RandomShift", dict(max_shifts=32))
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=800, max_size=1333,
                    sample_style="choice")),
            ],
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
