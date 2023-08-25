# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector
from .. import build_detector
from mmcv.runner import load_checkpoint

import torch
from mmrotate.core import rbbox2result
import numpy as np
import torch.nn as nn


@ROTATED_DETECTORS.register_module()
class GroupKnowledgeDistillationRotatedFCOS(RotatedSingleStageDetector):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 group,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 eval_teacher=True,
                 ):
        super(GroupKnowledgeDistillationRotatedFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                          test_cfg, pretrained, init_cfg)

        self.eval_teacher = eval_teacher
        self.is_teacher  = group.is_teacher
        self.group_cfg = group.group_cfg
        self.group_ckpt = group.group_ckpt
        self.classmates=[]
        assert len(self.group_cfg) == len(self.group_ckpt)

        if not group.is_teacher:
            if len(self.group_cfg) != 0:
                for idx in range(len(self.group_cfg)):
                    if isinstance(self.group_cfg[idx], str):
                        # print('building model:', idx)
                        classmate_config = mmcv.Config.fromfile(self.group_cfg[idx])
                        # print('get model cfg success!')
                        # print(classmate_config['model'])
                        classmate = build_detector(classmate_config['model'])
                        # print('build model success!')
                        load_checkpoint(classmate, self.group_ckpt[idx], map_location='cpu')
                        classmate.eval()
                        self.classmates.append(classmate)
                        # print('finished:', idx)





    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)

        classmate_x = []
        classmate_out = []

        if len(self.classmates) != 0:
            with torch.no_grad():
                for idx in range(len(self.classmates)):
                    cm_x = self.classmates[idx].extract_feat(img)
                    classmate_x.append(cm_x)
                    classmate_out.append(self.classmates[idx].bbox_head(cm_x))

        # cx = []
        # for i in range(len(classmate_x[0])):
        #     tmp = []
        #     for j in range(len(classmate_x)):
        #         tmp.append(classmate_x[j][i])
        #     tmp = torch.cat(tmp, dim=1)
        #     cx.append(tmp)
        #
        # ca_result = 0
        # for i in range(len(cx)):
        #     ca_result += self.ca.forward(cx[i])
        # one = torch.sum(torch.tensor(ca_result))
        # weight_group = ca_result / one
        # weight_group = weight_group.reshape(-1, 3)

        if len(classmate_x) != 0:
            losses = self.bbox_head.forward_train(x=x,
                                                  group_x=classmate_x,
                                                  group_out=classmate_out,
                                                  # weight_group=weight_group,
                                                  gt_bboxes=gt_bboxes,
                                                  gt_labels=gt_labels,
                                                  gt_bboxes_ignore=gt_bboxes_ignore,
                                                  img_metas=img_metas,)
        else:
            losses = self.bbox_head.forward_train(x=x,
                                                  group_x=[],
                                                  group_out=[],
                                                  # weight_group=0,
                                                  gt_bboxes=gt_bboxes,
                                                  gt_labels=gt_labels,
                                                  gt_bboxes_ignore=gt_bboxes_ignore,
                                                  img_metas=img_metas,)

        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        if len(self.classmates) != 0:
            for i in range(len(self.group_cfg)):
                self.classmates[i].cuda(device=device)
        return super().cuda(device=device)

    # def train(self, mode=True):
    #     if self.eval_teacher:
    #         self.teacher.train(False)
    #     else:
    #         self.teacher.train(mode)
    #     super.train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results







