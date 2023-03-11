import math
import numpy as np
import cv2
from matplotlib import pyplot as plt, patches
import open3d as o3d
import os
import torch

class calcualte_force():
    def __init__(self, suction_score):
        # mapping score in percentage
        self.suction_score = suction_score*100

        self.curved_object_intercept = 22.26361025
        self.curved_object_slope =  -0.17844147

        # # Lasso for flat objects
        # self.flat_object_intercept = 2.7684182
        # self.flat_object_slope =  0.0

        # Ridge for flat objects
        self.flat_object_intercept = 7.66228198
        self.flat_object_slope =  -0.05622074

        self.device = 'cuda:0'

    def regression(self):
        if(self.suction_score < 0.8):
            prediction = self.suction_score*self.curved_object_slope + self.curved_object_intercept
            mu, sigma = prediction, 1.6678003887167776
            force = torch.normal(mu, sigma).to(self.device)
        else:
            prediction = self.suction_score*self.flat_object_slope + self.flat_object_intercept
            mu, sigma = prediction, 1.4155587192333443
            force = torch.normal(mu, sigma).to(self.device)
        return force