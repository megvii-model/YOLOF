# How to Contribute

We'd love to accept your contributions to this project. There are just a few small guidelines you need to follow.

## How to add a model into examples

Here are some steps for your reference:

1. **Clone cvpods and cvpods_playground, switch to your own branch:**

   ```shell
   # fork and clone cvpods
   git clone https://github.com/Megvii-BaseDetection/cvpods.git
   git checkout -b [user]
   
   # cvpods_playground
   cd cvpods/playground 
   mkdir [username]
   ```

2. **Add the code for the new model to cvpods:**

   In general, you need to add the code files of the new model to the following two paths:

   1. **backbone:** [`cvpods/modeling/backbone`](https://git-core.megvii-inc.com/zhubenjin/cvpods/tree/megvii/cvpods/modeling/backbone)

      This step is optional because there are already many commonly used [backbones in cvpods]((https://git-core.megvii-inc.com/zhubenjin/cvpods/tree/megvii/cvpods/modeling/backbone)), for example: [ResNeSt](https://git-core.megvii-inc.com/zhubenjin/cvpods/blob/megvii/cvpods/modeling/backbone/resnet.py), [FPN](https://git-core.megvii-inc.com/zhubenjin/cvpods/blob/megvii/cvpods/modeling/backbone/fpn.py).  
      When the backbone needed for the new model does not exist, you need to add it.

   2. **meta_arch:** [`cvpods/modeling/meta_arch`](https://git-core.megvii-inc.com/zhubenjin/cvpods/tree/megvii/cvpods/modeling/meta_arch)

      You need to add the model architecture code (except backbone) here, for example：[RetinaNet](https://git-core.megvii-inc.com/zhubenjin/cvpods/blob/megvii/cvpods/modeling/meta_arch/retinanet.py), [FCOS](https://git-core.megvii-inc.com/zhubenjin/cvpods/blob/megvii/cvpods/modeling/meta_arch/fcos.py).

3. **Create a new project in the cvpods_playground, for example:**

   ```shell
   mkdir -p examples/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x
   ```

   > **Note:** When creating a new project, please follow this directory hierarchy: `examples/task/dataset/method_name/model_name` (model_name means the project name of the method in different configurations).

   A new project must contain the following two files：

   - `config.py`：configuration file, used for environment and hyperparameter setting.
   - `net.py`：network structure build file, used to create builders for each module of the model, such as backbone builder, anchor generator builder, etc.

4. **Train & test the model until it achieves your expectations:**

   ```shell
   # Train
   pods_train --num-gpus 8
   # Test
   pods_test --num-gpus 8
   ```

   > **Note:** After the model test is completed, please make sure that the project directory contains the `README.md` file with model evaluation metrics and corresponding scores, for example: [FCOS](https://github.com/Megvii-BaseDetection/cvpods/blob/master/playground/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x).

5. **Update MODEL_ZOO:**

   Add new model information in the [`README.md`](https://github.com/Megvii-BaseDetection/cvpods/blob/master/playground/README.md) file in playground.

6. **After all the above work is completed, push the code to Gitlab and submit merge request to cvpods and cvpods_playground respectively.**

   > **Note:** 
   >
   > 1. Before pushing, please make sure your code style conforms to [PEP8](https://www.python.org/dev/peps/pep-0008/).
   >    ```shell
   >    pip install flake8
   >    # Run the following command in the root of cvpods and cvpods_playgroun
   >    flake8 ./
   >    ```
   >
   > 2. When submitting the merge request to playground, **please attach the final checkpoint file of each model in merge request description**, for example: `/url/to/your/playground/examples/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x/model_final.pth`.
