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


import os, logging
import sys

## Stores if NumPy could be found.
# \ingroup SLMDisplayPython
supportNumPy = True

try:
    import numpy
except:
    supportNumPy = False

## \cond INTERNALDOC
## Stores if the current Python version is 3 or higher
isPython3 = sys.version_info[0] == 3
## \endcond

if isPython3:
    sys.path.append(os.path.dirname(__file__))

import detect_heds_module_path

# Define the path to the library file:
if os.name == "nt":
    platform = "win32" if sys.maxsize == (2 ** 31 - 1) else "win64"
    library_file_path = os.path.join(detect_heds_module_path.env_path, "bin", platform, "holoeye_slmdisplaysdk.dll")
else:
    platform = "linux"
    library_file_path = os.path.join(detect_heds_module_path.env_path, "bin", platform, "holoeye_slmdisplaysdk.so")

from hedslib import *

pyverstr = "Python " + "{}.{}.{} {} {}".format(*sys.version_info)

## The class describes a data field for a SLM rectangle geometry for phase data.
## \ingroup SLMDisplaySDK_CONVAPI_Python
class SLMDataField:

    ## Constructs a rectangular geometry to determine the size of SLM. Like 2D array.
    ## \param w is the width of the n horizontal direction.
    ## \param h is the position in vertical direction. The default value is set to 0.
    ## \param data_format is a \b heds_dataformat. Is the format of image or phase shift data.
    ## \param flags Load flags to change behavior when loading data. You can already pass show flags here.
    ## \param pitch If lines are longer than \p w * sizeof(element_type) in memory, please specify the memory length of each line manually.
    ##              If 0 (default) is used, pitch is computed automatically assuming no additional memory space between lines.
    ## \param useNumPy This enables numpy when pumpy is installed.
    def __init__(self, w, h,  data_format = HEDSDTFMT_INT_U8, flags=(HEDSLDF_Default | HEDSSHF_PresentAutomatic), pitch=0, useNumPy = True):

        assert w > 0 and h > 0
        ## Width of the datafield
        self._width = int(w)
        ## Heigth of the datafield.
        self._height = int(h)
        ## Holding the SLM data values as array. The SLM has rectangular geometry, like 2D array.
        self._values = None
        ## Holding the data format
        self._data_format = None

        if useNumPy and supportNumPy:
            elementType = None
            dimension = 1
            if data_format == HEDSDTFMT_INT_U8:
                elementType = numpy.ubyte
            if data_format == HEDSDTFMT_INT_RGB24:
                elementType = numpy.ubyte
                dimension = 3
            if data_format == HEDSDTFMT_INT_RGBA32:
                elementType = numpy.ubyte
                dimension = 4
            if data_format == HEDSDTFMT_FLOAT_32:
                elementType = numpy.single   # numpy.float32
            if data_format == HEDSDTFMT_FLOAT_64:
                elementType = numpy.double  # numpy.float64

            assert elementType is not None, "The given format is not supported."
            self._values = numpy.empty((self._height, self._width, dimension), elementType)

        else:
            elementType = None
            if data_format == HEDSDTFMT_INT_U8:
                elementType = ctypes.c_ubyte
            if data_format == HEDSDTFMT_INT_RGB24:
                elementType = heds_rgb24
            if  data_format == HEDSDTFMT_INT_RGBA32:
                elementType = heds_rgba32
            if data_format == HEDSDTFMT_FLOAT_32:
                elementType = ctypes.c_float
            if data_format == HEDSDTFMT_FLOAT_64:
                elementType = ctypes.c_double

            assert elementType is not None, "The given format is not supported."
            self._values = ((elementType * self._width) * self._height)()

        ## Holding data format
        self._data_format = data_format
        ## Setting of the load and show flags.
        self._flags = flags
        ## Pitch of lines.
        self._pitch = int(pitch)

    ## Clear the field data to be set to 0. Field size is not changed.
    ## \return None
    def clear(self):
        if supportNumPy:
            elementType = None
            dimension = 1
            if self._data_format == HEDSDTFMT_INT_U8:
                elementType = numpy.ubyte
            if self._data_format == HEDSDTFMT_INT_RGB24:
                elementType = numpy.ubyte
                dimension = 3
            if self._data_format == HEDSDTFMT_INT_RGBA32:
                elementType = numpy.ubyte
                dimension = 4
            if self._data_format == HEDSDTFMT_FLOAT_32:
                elementType = numpy.single  # numpy.float32
            if self._data_format == HEDSDTFMT_FLOAT_64:
                elementType = numpy.double  # numpy.float64

            assert elementType is not None, "The given format is not supported."
            self._values = numpy.zeros((self._height, self._width, dimension), elementType)
        else:
            elementType = None
            if self._data_format == HEDSDTFMT_INT_U8:
                elementType = ctypes.c_ubyte
            if self._data_format == HEDSDTFMT_INT_RGB24:
                elementType = heds_rgb24
            if self._data_format == HEDSDTFMT_INT_RGBA32:
                elementType = heds_rgba32
            if self._data_format == HEDSDTFMT_FLOAT_32:
                elementType = ctypes.c_float
            if self._data_format == HEDSDTFMT_FLOAT_64:
                elementType = ctypes.c_double

            assert elementType is not None, "The given format is not supported."
            self._values = ((elementType * self._width) * self._height)()

        return None

    ## Set the pixel value in x and y position.
    ## \param i The pixel position index in x-direction, i.e. along the width. Index 0 means the left column.
    ## \param j The pixel position index in y-direction, i.e. along the height. Index 0 means the top row.
    ## \param value The value provided in the same type the field was created for.
    ## \returns HEDSERR_NoError when the desired wait duration was reached. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setPixel(self, i, j, value):
        assert self._width > i or self._height > j
        if not self.checkDataFormat (value):
            return HEDSERR_InvalidArgument
        self._values[j][i] = value

        return HEDSERR_NoError

    ## Get the pixel value in x and y position.
    ## \param i The pixel position index in x-direction, i.e. along the width. Index 0 means the left column.
    ## \param j The pixel position index in y-direction, i.e. along the height. Index 0 means the top row.
    ## \return The value of the pixel at position \p i, \p j.
    def getPixel(self, i, j):
        return self._values[j][i]

    ## Check if SLMDataField is set with values.
    ## \return True when no data is available.
    def isEmpty(self):
        return len(self._values) == 0

    ## Check if SLMDataField is valid, i.e. is properly initialized, so that data can be accessed.
    ## \return True when this field has set proper data.
    def isValid(self):
        return not self.isEmpty()

    ## Getting the data values as a given template type.
    ## \return data pointer of values as a given type
    def data(self):
        return self._values

    ## Get the width of the data field, i.e. the pixel count in x-direction.
    ## \return The field width in number of pixel.
    def width(self):
        return int(self._width)

    ## Get the heigth of the data field, i.e. the pixel count in y-direction.
    ## \return The field heigth in number of pixel.
    def height(self):
        return int(self._height)

    ## Get the flag setting of the load and show flags.
    ## \return The load and show flags already set during construction.
    def flags(self):
        return self._flags

    ## Getting the pitch of lines.
    ## \return int of the pitch
    def pitch(self):
        return int(self._pitch)

    ## Getting the data format of the values.
    ## \return the data type given from the setting.
    def getDataFormat(self):
        return self._data_format

    ## Checking if the given value has the same the data format of the datafield
    ## \param value is the value with a certain dataFormat to check for.
    ## \return bool confirms that the data type is the same where given from.
    def checkDataFormat(self, value):
        return self._data_format ==  HoloeyeInputDataDescriptor(value).data_format

    ## A function to create a multi-line string for printing properties and some values of this field to the command line for debug output.
    ## \param name An optional parameter to give the field a name within the returned print string.
    ## \return A multi-line string containing properties and values of the field.
    def printString(self, name= ""):
        s = name + ":\n"
        s += str("  width:  %8d pixel.\n" % self._width)
        s += str("  height: %8d pixel.\n" % self._height)
        s += str("  size:   %8d bytes.\n" % self._pitch * self._height)
        s += str("  pitch:  %8d bytes.\n" % self._pitch)
        s += str("  flags:  %8d (%s).\n"  % (self._flags, SDK.libapi.heds_dataflags_string(self._flags)))
        s += str("  Pixel data: \n")
        s += str(self._values)
        return s

from holoeye_slmdisplaysdk_types import *
