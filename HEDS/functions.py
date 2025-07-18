# -*- coding: utf-8 -*-

#------------------------- \cond COPYRIGHT --------------------------#
#                                                                    #
# Copyright (C) 2025 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
#                                                                    #
#----------------------------- \endcond -----------------------------#


import math

import HEDS
from hedslib.heds_types import *

def computeAiryBeam(dataWidth, dataHeight, centerX , centerY, innerRadius, rotAngleDeg, onedimensional=False):
    # Pre-calc. helper variables:
    phaseModulation = 2 * math.pi
    angleRad = rotAngleDeg / 360.0 * 2.0 * math.pi

    # Move white-black phase wraps out of the center:
    cyclicShift = - phaseModulation / 2.0

    # The phase data field to return:
    phaseData = HEDS.SLMDataField(dataWidth, dataHeight, HEDSDTFMT_FLOAT_32, 0, False)

    # Compute the airy beam phase function:
    for y in range(dataHeight):
        row = phaseData.data()[y]

        for x in range(dataWidth):
            x0Deg = x - dataWidth / 2 - centerX
            y0Deg = y - dataHeight / 2 + centerY

            # Rotate x/y coordinates before calculation to avoid rotating a pixelated data field:
            xDeg = math.cos(angleRad) * x0Deg - math.sin(angleRad) * y0Deg
            yDeg = math.sin(angleRad) * x0Deg + math.cos(angleRad) * y0Deg

            #  pre-compute the cubic coordinates:
            x3 = math.pow(xDeg, 3.0)
            y3 = math.pow(yDeg, 3.0)

            if onedimensional:
                val = x3 * math.pow(innerRadius, -3.0)
            else:
                val = (x3 + y3) * math.pow(innerRadius, -3.0)

            row[x] = val * phaseModulation + cyclicShift

    return phaseData


def computeAiryBeamNumPy(dataWidth, dataHeight, centerX, centerY, innerRadius, rotAngleDeg, onedimensional=False):
    import numpy as np

    # pre-calc. helper variables:
    phaseModulation = 2 * math.pi
    angleRad = rotAngleDeg / 360.0 * 2.0 * math.pi

    # Move white-black phase wraps out of the center:
    cyclicShift = - phaseModulation / 2.0

    # use numpy as phase data field:
    xvec = np.linspace(0, dataWidth-1, dataWidth, dtype=np.float32) - np.float32(dataWidth / 2) - np.float32(centerX)
    yvec = np.linspace(0, dataHeight-1, dataHeight, dtype=np.float32) - np.float32(dataHeight / 2) + np.float32(centerY)

    xvec = np.matrix(xvec)
    yvec = np.matrix(yvec).transpose()

    xmat = np.array(np.dot(np.ones([dataHeight, 1], np.float32), xvec))
    ymat = np.array(np.dot(yvec, np.ones([1, dataWidth], np.float32)))

    # Rotate x/y coordinates before calculation to avoid rotating a pixelated data field:
    xmat_rot = math.cos(angleRad) * xmat - math.sin(angleRad) * ymat
    ymat_rot = math.sin(angleRad) * xmat + math.cos(angleRad) * ymat

    xmat3_rot = np.power(xmat_rot, 3)
    ymat3_rot = np.power(ymat_rot, 3)

    if onedimensional:
        ar = xmat3_rot
    else:
        ar = xmat3_rot + ymat3_rot

    phaseData = ar * np.float32(phaseModulation) * np.power(innerRadius, -3, dtype=np.float32) + cyclicShift

    return phaseData

