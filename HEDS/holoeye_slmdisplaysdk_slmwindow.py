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



## The class manage multiple SLM's for multiple devices. Multipile devices is case for Mosaic.
## \ingroup SLMDisplaySDK_CONVAPI_Python
class SLMWindow:

    ## Opens a new SLM window. If you pass a non-empty pre-selection string, the device will be
    ## selected without any GUI dialogs.
    ## The pre-selection string can contain device property filter(s) to automatically find and select the
    ## device to open the SLM window on.
    ## If the maximum number of concurrent SLM windows is already reached, this call will return the error
    ## code HEDSERR_MaxSLMWindowCountReached.
    ## If a remote host is opened using a preselection string like "connect://127.0.0.1:6230", a connection will be established and
    ## the remotely running process will be told to open its SLM window. The connection will persist until heds_slmwindow_close(slmwindow)
    ##  is called. The remote SLM window may stay open depending on the remote server configuration.
    ## \param preselect An optional string to automatically select a device to open the SLM window on.
    ##                  The string consists of a format "property:value", with properties separated by ";" Valid properties are "index", "name",
    ##                  and "serial". An example would be "name:pluto;serial:0001". In addition, if the string starts with "connect://", a manually
    ##                  started process can be addressed using the format "connect://ipv4:port", e.g. "connect://127.0.0.1:6230".
    ## \param openPreview Opens the [HEDS.SLMPreview](\ref HEDS::holoeye_slmdisplaysdk_preview::SLMPreview) window when creating the new SLMWindow.
    ## \param previewScale If \p openPreview is true, this parameter allows specifying the initial preview scale.
    ##                     Is used by most provided examples to specify whether each example needs the preview to be in 1:1 or in fit mode.
    def __init__(self, preselect="", openPreview=True, previewScale=1.0):
        ## Holds the address of the SLM window.
        self._id = HEDS_SLMWINDOW_ID_INVALID
        ## Holds the current error code.
        self._err = HEDSERR_NoError
        ## Holds all added SLM's in list for the SLMWindow container.
        self._slmlist = []

        self._err, self._id = SDK.libapi.heds_slmwindow_open(preselect)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        if not SDK.libapi.heds_slmwindow_isopen(self._id):
            self._err = HEDSERR_NoSLMConnected
            print(SDK.libapi.heds_error_string(self._err))
            return

        if openPreview:
            self._err = SDK.libapi.heds_slmpreview_open(self._id)
            assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

            self._err = SDK.libapi.heds_slmpreview_set_settings(self._id, HEDSSLMPF_None, previewScale)
            assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

    ##  The SLM window can in principle be spread over multiple SLM display devices,
    ##  when using NVIDIA Mosaic feature to set up such a mosaic screen out of multiple SLMs.
    ##  In such a case, all SLM display devices will be in the same mode (addressed resolution and framerate).
    ##  \return The number of SLM display devices in x direction within this SLM window.
    def deviceColumns(self):
        self._err, dev_cols, dev_rows = SDK.libapi.heds_slmwindow_device_layout(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return int(dev_cols)

    ## The SLM window can in principle be spread over multiple SLM display devices,
    ##  when using NVIDIA Mosaic feature to set up such a mosaic screen out of multiple SLMs.
    ##  In such a case, all SLM display devices will be in the same mode (addressed resolution and framerate).
    ##  \return The number of SLM display devices in y direction within this SLM window.
    def deviceRows(self):
        self._err, dev_cols, dev_rows = SDK.libapi.heds_slmwindow_device_layout(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return int(dev_rows)

    ## Adding a SLM setup in a given geometry to the SLMWindow.
    ## \param geo is [HEDS.RectGeometry](\ref HEDS::holoeye_slmdisplaysdk_types::RectGeometry)
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def slmSetupAdd(self, geo):
        if not self.isOpen():
            self._err = HEDSERR_NoSLMConnected
            print(SDK.libapi.heds_error_string(self._err))
            return self._err

        self._err = SDK.libapi.heds_slmwindow_slmsetup_add_screen(self._id, geo.x(), geo.y(), geo.width(), geo.height())
        return self._err

    ## Applies all added SLM setup to the SLMWindow.
    ## \return a list of all applied [HEDS.SLM](\ref HEDS::holoeye_slmdisplaysdk_slm::SLM)'s.
    def slmSetupApply(self):
        _err, slms = SDK.libapi.heds_slmwindow_slmsetup_apply(self._id)
        self._slmlist = []
        for slm in slms:
            self._slmlist.append(SLM (self, slm))

        return self._slmlist

    ## Clears all the current setups and applies a SLM or multiple SLMs to the SLMWindow.
    ## Mulitple SLMs where open for NVIDIA Mosaic, otherwise only one SLM.
    ## \return a list of all applied [HEDS.SLM](\ref HEDS::holoeye_slmdisplaysdk_slm::SLM)'s.
    def slmSetupApplyDefault(self):
        self._err = SDK.libapi.heds_slmwindow_slmsetup_clear(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        self._err, cols, rows = SDK.libapi.heds_slmwindow_device_layout(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        x=0
        y=0
        slm_w = self.width_px() / cols
        slm_h = self.height_px() / rows
        for i in range(cols):
            for j in range(rows):
                self._err = self.slmSetupAdd(RectGeometry(x, y, slm_w, slm_h))
                assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
                x += slm_w
            y += slm_h

        return self.slmSetupApply()

    ## Applies a setup os SLMs into the SLMWindow with a array layout.
    ## \param cols is the amount of colums of SLMs in the x direction.
    ## \param rows is the amount of rows of SLMs in the y direction.
    ## \return a list of all applied [HEDS.SLM](\ref HEDS::holoeye_slmdisplaysdk_slm::SLM)'s.
    def slmSetupApplyLayout(self, cols, rows):
        self.slmSetupClear()
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        x, y = 0, 0
        slm_w = int(self.width_px() / cols)
        slm_h = int(self.height_px() / rows)

        for i in range(rows):
            for j in range(cols):
                self._err = self.slmSetupAdd(RectGeometry(x, y, slm_w, slm_h))
                assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
                x += slm_w
            x = 0
            y += slm_h

        return self.slmSetupApply()

    ## Use this function to actually apply changes made with direct_apply=false on multiple SLMs within this SLMWindow.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def slmSetupApplyChanges(self):
        self._err = SDK.libapi.heds_slmwindow_slmsetup_apply_changes(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return self._err

    ## Clears all the current setups in the SLMWindow.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def slmSetupClear(self):
        self._err = SDK.libapi.heds_slmwindow_slmsetup_clear(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return

    ## Get all added SLM 's from the SLMWindow.
    ## \return a list of all applied [HEDS.SLM](\ref HEDS::holoeye_slmdisplaysdk_slm::SLM)'s.
    def getSLMs(self):
        return self._slmlist

    ## Adding a SLM to the SLWindow. All existing SLM's in the SLMWindow class will be removed and displayed again with the newly added SLM.
    ## \param x is the position in horizontal direction. The default value is set to 0.
    ## \param y is the position in vertical direction. The default value is set to 0.
    ## \param width is the length of the horizontal direction. The default value is set to 0.
    ## \param height is the length in vertical direction. The default value is set to 0.
    ## \return SLM as SLM object. [HEDS.SLM](\ref HEDS::holoeye_slmdisplaysdk_slm::SLM) is the newly added one.
    def addSLM_xywh(self, x=0, y=0, width=0, height=0):
        geo = RectGeometry(x, y, width, height)
        return self.addSLM(geo)

    ## Adding a SLM to the SLWindow. All existing SLM's in the SLMWindow class will be removed and displayed again with the newly added SLM.
    ## \param geo is [HEDS.RectGeometry](\ref HEDS::holoeye_slmdisplaysdk_types::RectGeometry) of the new aded SLM.
    ## \return SLM as SLM object. [HEDS.SLM](\ref HEDS::holoeye_slmdisplaysdk_slm::SLM) is the newly added one.
    def addSLM(self, geo):
        count = SDK.libapi.heds_slmwindow_slmsetup_count(self._id)
        self._err = self.slmSetupAdd(geo)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        slms = self.slmSetupApply()
        if count < len(slms):
            return slms[count]

        return SLM(self, HEDSERR_NoError)

    ## Provides the width of the SLMWindow in number of pixels.
    ## \return int as width of SLMWindow in number of pixel.
    def width_px(self):
        _err,  w, h = SDK.libapi.heds_slmwindow_size_px(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return w

    ## Provides the height of the SLMWindow in number of pixels.
    ## \return int as height of SLMWindow in number ofpixel.
    def height_px(self):
        _err, w, h = SDK.libapi.heds_slmwindow_size_px(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return h

    ## Provides the pixelsize of the connected SLM hardware device(s). The unit of the returned pixel size is micro-meter.
    ## If the SLM window is spread over multiple hardware SLMs, all hardware SLMs will have the same pixel size for sure.
    ## \return The SLM's pixelsize in micro meter. Returns zero when the SLM window was not opened yet.
    def pixelsize_um(self):
        return SDK.libapi.heds_slmwindow_pixelsize_um(self._id)

    ##  Provides the refresh rate of the connected device. The unit of the returned refresh rate is is derived SI unit Hz.
    ##  \return The SLM's current refreshrate. Returns zero when the SLMWindow was not opened yet.
    def refreshrate_hz(self):
        return SDK.libapi.heds_slmwindow_refreshrate_hz(self._id)

    ## Set all SLMs within this SLMWindow into phase modulation and apply a monochrome wavelength to all SLM screen areas of this SLMWindow.
    ## Using this function instead of setting each SLMs wavelengths separately is faster in case multiple SLMs are created inside this SLMWindow.
    ## \param laser_wavelength_nm Monochrome laser wavelength used on this SLMWindow in nano meter.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setWavelength(self, laser_wavelength_nm):
        for idx in range (0, len(self._slmlist)):
            self._err  = self._slmlist.setWavelength(laser_wavelength_nm, False)
            assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        return self.slmSetupApplyChanges()


    ## Set all SLMs within this SLMWindow into phase modulation and apply either one monochrome wavelength or a color field sequential (CFS-RGB)
    ## wavelength set of three values for red, green, and blue color channels to all SLM screen areas of this SLMWindow.
    ## Using this function instead of setting each SLMs wavelengths separately is faster in case multiple SLMs are created inside this SLMWindow.
    ## \param laser_wavelengths_nm A vector of either one or three laser wavelength used on this SLM in nano meter.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setWavelengths(self, laser_wavelengths_nm):
        for idx in range (0, len(self._slmlist)):
            self._err = self._slmlist.setWavelength(laser_wavelengths_nm, False)
            assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return self.slmSetupApplyChanges()

    ## Set all SLMs within this SLMWindow into color field sequential phase modulation and apply a set of three wavelengths for the red, green,
    ## and blue color channels to all SLM screen areas of this SLMWindow.
    ## Using this function instead of setting each SLMs wavelengths separately is faster in case multiple SLMs are created inside this SLMWindow.
    ## \param laser_wavelength_red_nm The laser wavelength used on the red color channel of this SLM.
    ## \param laser_wavelength_green_nm The laser wavelength used on the green color channel of this SLM.
    ## \param laser_wavelength_blue_nm The laser wavelength used on the blue color channel of this SLM.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setWavelengthCFS(self, laser_wavelength_red_nm, laser_wavelength_green_nm, laser_wavelength_blue_nm):
        for idx in range(0, len(self._slmlist)):
            self._err = self._slmlist.setWavelengthCFS (laser_wavelength_red_nm, laser_wavelength_green_nm, laser_wavelength_blue_nm, False)
            assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return self.slmSetupApplyChanges()

    ## Loads a wavefront compensation from a .h5 file provided by HOLOEYE and applies it to the given device area.
    ## The wavelength for converting optical path differences into phase shifts is taken from the given SLM screen
    ## properties, [heds_slm_set_modulation]. Therefore, the SLM screen layout must have been applied (\ref slmSetupApply())
    ## before loading any wavefront compensation, and then SLM screens not set to HEDSMOD_Phase and/or without a meaningful laser
    ## wavelength setting will not enable wavefront compensation.
    ## \param filename The wavefront compensation H5 file to load.
    ## \param dev_col_idx The column index of the hardware device behind the SLM window if there are multiple hardware
    ##        devices merged into one operating system screen.Not working yet.WFC is always applied to full SLMWindow atm.
    ## \param dev_row_idx The row index of the hardware device behind the SLMWindow if there are multiple hardware
    ##        devices merged into one operating system screen.Not working yet.WFC is always applied to full SLM
    ##        window atm.
    ## \param flip_x Flip the wavefront compensation field in x direction, i.e.flip left / right.
    ## \param flip_y Flip the wavefront compensation field in y direction, i.e.flip top / bottom.
    ## \param shift_x Shift the wavefront compensation field in x direction.
    ## \param shift_y Shift the wavefront compensation field in y direction.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def loadWavefrontCompensationFile(self, filename, dev_col_idx=0, dev_row_idx=0, flip_x=False, flip_y=False, shift_x=0, shift_y=0):
        self._err = SDK.libapi.heds_slmwindow_wavefrontcompensation_load_from_file(self._id, filename, dev_col_idx,
                                                                              dev_row_idx, flip_x, flip_y, shift_x,
                                                                              shift_y)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return self._err

    ## Clears a previously loaded wavefront compensation for a given device area.
    ## \param dev_col_idx The column index of the hardware device behind the SLM window if there are multiple hardware devices merged into one operating system screen. Not working yet. It always clears the WFC from the full SLM window atm.
    ## \param dev_row_idx The row index of the hardware device behind the SLM window if there are multiple hardware devices merged into one operating system screen. Not working yet. It always clears the WFC from the full SLM window atm.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def clearWavefrontCompensation(self, dev_col_idx=0, dev_row_idx=0):
        self._err = SDK.libapi.heds_slmwindow_wavefrontcompensation_clear(self._id, dev_col_idx, dev_row_idx)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return self._err

    ## Retrieve a class to access SLM preview window for this SLM window.
    ## \return An [HEDS.SLMPreview](\ref HEDS::holoeye_slmdisplaysdk_preview::SLMPreview) object assigned to this SLM window.
    def preview(self):
        return SLMPreview(self._id)

    ## Provides the current error code SLMWindow
    ## \return heds_errorcode errorCode.
    def errorCode(self):
        return self._err

    ## Checks if the given SLM window is already open. Can check for a response of the connected SLM window.
    ## \return true if the window is open and responds, and false if it is not open.
    def isOpen(self):
        return SDK.libapi.heds_slmwindow_isopen(self._id, True)

    ## Close the SLM window. If available, this call will close the SLM preview window and the related tray icon of that SLM window, too.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def close(self):
        return SDK.libapi.heds_slmwindow_close(self._id)

    ## Waits until the given duration has passed. If given duration is zero, it will wait forever.
    ##  The function checks if the SLMWindow is closed (e.g. through tray icon) during the wait.
    ##  Therefore, by default, a duration of zero will wait until the given SLMWindow is closed, either manually or due to an error.
    ##  \param duration_ms The number of milliseconds to wait before returning for sure. By default (0), the function waits forever until the given SLMWindow was closed.
    ##  \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def wait(self, duration_ms = 0 ):
        self._err = SDK.libapi.heds_slmwindow_wait(self._id, duration_ms)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return self._err

    ## Waits forever until all SLM's in the SLMWindow container where closed (e.g. through tray icon) during the wait, either manually or due to an error.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def waitUntilClosed(self):
        self._err = SDK.libapi.heds_slmwindow_wait(self._id, 0)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return self._err

    ## Provides the address/id  of the SLM Window.
    ## \return heds_slmwindow_id is the id of the SLM window.
    def id(self):
        return self._id

from holoeye_slmdisplaysdk_slm import *
from holoeye_slmdisplaysdk_preview import *
