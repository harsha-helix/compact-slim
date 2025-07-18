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


import os
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

## The class represents a the preview window for a [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) on s). It holds all the properties
## ingroup SLMDisplaySDK_CONVAPI_Python
class SLMPreview:
    ## Constructs a SLMPreview parameter / configuartion class , except the default errorcode.
    ## \param slmwindow_id as \b heds_slmwindow_id is the id of the SLM window
    ## \param geo as [HEDS.RectGeometry]( \see  HEDS::holoeye_slmdisplaysdk_types::RectGeometry). The new geometry (position and size) to the SLMPreview window.
    def __init__(self, slmwindow_id, geo=None):
        ## Describes a rectangular geometry of preview window.
        self._geo = geo
        ## Holds the address of the SLM window.
        self._slmw_id = slmwindow_id
        ## Holds the current error code.
        self._err = HEDSERR_NoError

    ## Open the SLM preview window. SLM window must be initialized before opening the corresponding SLM preview window.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def open(self):
        self._err = SDK.libapi.heds_slmpreview_open(self._slmw_id)
        SDK.LogErrorString(self._err)
        return int(self._err)

    ## Close the SLM preview window.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def close(self):
        self._err = SDK.libapi.heds_slmpreview_close(self._slmw_id)
        SDK.LogErrorString(self._err)
        return int(self._err)

    ## Check if the SLM preview window is open. SLM window must be initialized.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def isOpen(self):
        return SDK.libapi.heds_slmpreview_is_open(self._slmw_id)

    ## Sets settings flags and the zoom factor of the preview window.
    ## \param flags The preview flags to set. Refer to \b heds_slmpreviewflags_enum for details.
    ## \param zoom The zoom factor of the preview window. Use zero to make the data fit the screen.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setSettings(self, flags=HEDSSLMPF_None, zoom=1.0):
        self._err = SDK.libapi.heds_slmpreview_set_settings(self._slmw_id, flags, zoom)
        SDK.LogErrorString(self._err)
        return int(self._err)

    ## Get settings flags and the zoom factor of the preview window.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return flags The preview flags to set. Refer to \b heds_slmpreviewflags_enum for details.
    ## \return zoom The zoom factor of the preview window. Use zero to make the data fit the screen.
    def getSettings(self):
        self._err, flags, zoom = SDK.libapi.heds_slmpreview_get_settings(self._slmw_id)
        SDK.LogErrorString(self._err)
        return int(self._err), flags, zoom

    ## Allows switching the SLM preview capture (render) mode. This option may not be available depending on the implementation.
    ## Please check the returned error code to see if it is possible to apply that mode in the currently running implementation.
    ## \param mode The SLM preview capture mode to apply. See \b heds_slmpreview_capturemode_enum for available modes.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setCaptureMode(self, mode):
        self._err = SDK.libapi.heds_slmpreview_set_capturemode(self._slmw_id, mode)
        SDK.LogErrorString(self._err)
        return self._err

    ## Returns the currently selected capture (render) mode of the SLM preview window.
    ## \return The currently selected SLM preview capture mode. See \b heds_slmpreview_capturemode_enum for available modes.
    def getCaptureMode(self):
        self._err, capturemode = SDK.libapi.heds_slmpreview_get_capturemode(self._slmw_id)
        SDK.LogErrorString(self._err)
        return self._err, capturemode

    ## Get the position and size of the preview window.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def getGeo(self):
        return self._geo

    ## Set the position and size of the preview window.
    ## \param geo The new geometry (position and size) to apply to the SLMPreview window. [HEDS.RectGeometry]( \see  HEDS::holoeye_slmdisplaysdk_types::RectGeometry).
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setGeo(self, geo):
        self._geo = geo
        self._err = SDK.libapi.heds_slmpreview_move(self._slmw_id, geo.x(), geo.y(), geo.width(), geo.height())
        SDK.LogErrorString(self._err)
        return int(self._err)

    ## Changes the position and size of the preview window.
    ## \param posX The horizontal position of the window on the desktop.
    ## \param posY The vertical position of the window on the desktop.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def move(self, posX, posY):
        self._err = SDK.libapi.heds_slmpreview_move(self._slmw_id, posX, posY)
        SDK.LogErrorString(self._err)
        return int(self._err)

    ## Changes the size size of the preview window.
    ## \param width The width of the window. If \p width or \p height is zero, the size will not be changed.
    ## \param height The height of the window. If \p width or \p height is zero, the size will not be changed.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def resize(self, width, height):
        self._err = SDK.libapi.heds_slmpreview_resize(self._slmw_id, width, height)
        SDK.LogErrorString(self._err)
        return int(self._err)

    ## This function allows saving the SLM preview image data into an image file, like a *.png image file.
    ## Please provide a \p filename for saving the preview image into.
    ## \param filename Please provide a file name (and path) for saving the image file into.
    ##                 The function will try to use the file ending to determine the image file format.
    ##                 An already existing file will be overwritten forcefully only if \p force_overwrite is set to true.
    ##                 By default, \p force_overwrite is false and the error code HEDSERR_FileAlreadyExists will be returned if the file already exists.
    ## \param mode The SLM preview image capture mode to use for saving.
    ## \param force_overwrite If set to True, no error will be returned if the given \p filename already exists.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def savePreviewImageToFile(self, filename, mode=HEDSSLMPM_CaptureScreen, force_overwrite = False):
        self._err = SDK.libapi.heds_slmpreview_save_to_file(self._slmw_id, filename, mode, force_overwrite)
        SDK.LogErrorString(self._err)
        return self._err

    ## This function allows setting up the SLM preview position using a layout on the detected secondary monitor, which is not the primary
    ## and not an SLM, i.e. most probably a free monitor screen. The layout allows placing multiple SLM preview windows on that area easily.
    ## If no secondary monitor is found, the primary monitor is used instead.
    ## \param number_of_preview_columns Number of SLM preview columns (x-direction) within the layout.
    ## \param number_of_preview_rows Number of SLM preview rows (y-direction) within the layout.
    ## \param place_on_col_index The column index to place this SLM preview into within the layout.
    ## \param place_on_row_index The row index to place this SLM preview into within the layout.
    ## \param margin The number of pixels in all edges of the monitor screen area, which are not used to place the SLMPreview layout on.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def autoplaceLayoutOnSecondaryMonitor(self, number_of_preview_columns=1, number_of_preview_rows=1, place_on_col_index=0, place_on_row_index=0, margin=100):
        self._err, x, y, w, h = SDK.libapi.heds_info_monitor_get_geometry(SDK.libapi.heds_info_monitor_get_id_secondary())
        single_preview_width = (w - 2 * margin) / number_of_preview_columns
        single_preview_height = (h - 2 * margin) / number_of_preview_rows
        pos_x = x + margin + place_on_col_index * single_preview_width
        pos_y = y + margin + place_on_row_index * single_preview_height
        geo = RectGeometry(pos_x, pos_y, single_preview_width, single_preview_height)
        ##self._err = self.setGeo(geo)
        self._err = SDK.libapi.heds_slmpreview_move(self._slmw_id, geo.x(), geo.y(), geo.width(), geo.height())
        SDK.LogErrorString(self._err)
        return int(self._err)

    ## Provides the ID of the [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) this SLMPreview is created for.
    ## \return The SLMWindow ID.
    def getSLMWindowId(self):
        return self._slmw_id

    ## Provides the current error code SLM
    ## \return \b heds_errorcode errorCode.
    def getErrorCode(self):
        return self._err


from holoeye_slmdisplaysdk_types import *
