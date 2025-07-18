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
#from warnings import deprecated  # needs Python 3.13 to work
from warnings import warn

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

# Test imported hedslib module without calling functions from the library file:
#print("HEDSAPI_Python = " + str(HEDSAPI_Python))
#print("pyverstr = " + pyverstr)


from ctypes import create_string_buffer



## This class represents an SLM screen area within an [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow). This is the main abstracted interface class to make use of an SLM.
##  Please do not mix up with hardware SLM display devices. The SLM window always represents a hardware SLM display device,
##  or in case of NVIDIA Mosaic, multiple hardware SLM display devices.
##  This SLM class instead defines an area within the SLM window to abstract the SLM from the hardware.
##  Normally, there is only one SLM display device, on which one SLM window is opened, and that SLM window creates one SLM to work with,
##  see \ref HEDS::SLMWindow::slmSetupApplyDefault().
##  See \ref HEDS::SLM::Init() function to select and open exactly one SLM on the detected hardware setup.
##  \ingroup SLMDisplaySDK_CONVAPI_Python
class SLM:
    ## Constructs an SLM screen.
    ## This SLM has no geo parameter setting, when initializing it uses the full screen of selected device for [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) .
    ## \param slmwindow is [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) a refernce to the SLM belongs to.
    ## \param slmid is is a refernce to the id wich will be constructed here.
    ## \param geo is [HEDS.RectGeometry](\ref HEDS::holoeye_slmdisplaysdk_types::RectGeometry) to the determine the position and size of the SLM.
    ## \param error_code is a \b heds_errorcode
    def __init__(self, slmwindow = None, slmid = None, geo = None, error_code = None ):
        ## Holds the address of the SLM.
        self._id = heds_slm_id()
        ## Holds the current error code.
        self._err = HEDSERR_NoError
        ## Holds the parent [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow)  container where the SLM is added.
        self._Wnd = None
        if slmwindow is None and slmid is None and geo is None and error_code is None:
            self._err = HEDSERR_NoSLMConnected
            return
        elif error_code is not None:
            self._err = error_code
            return
        elif slmwindow is not None and slmid is not None and geo is None and error_code is None:
            self._Wnd = slmwindow
            self._id = slmid
        elif slmwindow is not None and slmid is None and geo is not None :
            if not slmwindow.isOpen():
                self._err = HEDSERR_InvalidSLMWindow
                return
            self._Wnd = slmwindow
            self._id = self._Wnd.addSLM(geo).id()

    ## Opens a new [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) creating exactly one SLM on the full [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) screen.
    ## \param preselect A string to preselect an SLM device by its name, serial number, index, etc. to open the [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) on,
    ##        bypassing the EDID Device Detection GUI. When default parameter = "" is used, it opens the EDID Device
    ##        Detection GUI and uses its selection result. In case only one SLM is available, EDID Device Detection GUI will
    ##        select the first device automatically after a timeout, is configured properly.
    ## \param openPreview If true, opens the [HEDS.SLMPreview](\ref HEDS::holoeye_slmdisplaysdk_preview::SLMPreview) window of the created [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) and \ref SLM.
    ## \param previewScale If \p openPreview is true, allows initializing the [HEDS.SLMPreview](\ref HEDS::holoeye_slmdisplaysdk_preview::SLMPreview) window with a scale.
    ##        Default is 1.0f for pixelwise display.
    ## \return a \ref SLM class object within an [HEDS.SLMWindow](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow) class.
    @staticmethod
    def Init(preselect = "", openPreview = True, previewScale = 1.0):
        slmw = SLMWindow(preselect, openPreview, previewScale)
        return SLM(slmw, None, RectGeometry())

    ## Waits forever until SLM is closed (e.g. through tray icon) during the wait, either manually or due to an error.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def waitUntilClosed(self):
        if self._Wnd is not None:
            if self._Wnd.getErrorCode() is not HEDSERR_NoError:
                SDK.LogErrorString(self._Wnd.getErrorCode())
                self._err = self._Wnd.getErrorCode()
                return self._err
            self._err = self._Wnd.waitUntilClosed()
            return self._err

        self._err = HEDSERR_NoSLMConnected
        return self._err

    ## Set the SLM into phase modulation and apply a monochrome wavelength to this SLM screen area.
    ## \param laser_wavelength_nm Monochrome laser wavelength used on this SLM in nano meter.
    ## \param direct_apply Is true by default. If you set up the wavelength for multiple SLMs within the same SLM window, this can be set
    ##                     to false to apply the changes later on for all SLMs in a single call to [HEDS.SLMWindow::slmSetupApplyChanges()](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow::slmSetupApplyChanges()).
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setWavelength(self, laser_wavelength_nm, direct_apply=True):
        wls = None
        if isinstance(laser_wavelength_nm, list) or isinstance(laser_wavelength_nm, ctypes.Array):
            wls = (ctypes.c_float * len(laser_wavelength_nm))()
            for idx in range(len(laser_wavelength_nm)):
                wls[idx] = ctypes.c_float(laser_wavelength_nm[idx])
        elif isinstance(laser_wavelength_nm, float) or isinstance(laser_wavelength_nm, int):
            wls = (ctypes.c_float * 1)(laser_wavelength_nm)

        self._err = SDK.libapi.heds_slm_set_wavelength(self._id, wls, direct_apply)
        return self._err

    ## Set the SLM into color field sequential phase modulation and apply a set of three wavelengths for the red, green, and blue color channels
    ##  to this SLM screen area.
    ##  \param laser_wavelength_red_nm The laser wavelength used on the red color channel of this SLM.
    ##  \param laser_wavelength_green_nm The laser wavelength used on the green color channel of this SLM.
    ##  \param laser_wavelength_blue_nm The laser wavelength used on the blue color channel of this SLM.
    ##  \param direct_apply Is true by default. If you set up the wavelength for multiple SLMs within the same SLM window, this can be set
    ##                      to false to apply the changes later on for all SLMs in a single call to [HEDS.SLMWindow::slmSetupApplyChanges()](\ref HEDS::holoeye_slmdisplaysdk_slmwindow::SLMWindow::slmSetupApplyChanges()).
    ##  \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setWavelengthCFS(self, laser_wavelength_red_nm, laser_wavelength_green_nm, laser_wavelength_blue_nm, direct_apply=True):
        wls = [laser_wavelength_red_nm,laser_wavelength_green_nm,laser_wavelength_blue_nm]
        return self.setWavelength(wls,direct_apply)

    ## This factor allows to compensate for transverse (lateral) chromatic abberations within the optical setup when using color field sequential mode on the
    ## SLM screen area. By default, beam manipulation (\b heds_datahandle_set_beam_manipulation()) and hologram calculation use the wavelengths set for the
    ## SLM screen area to compute different phase fields for each color channel. In case the optical setup introduces chromatic abberations, the wavelengths
    ## alone would not result in best overlap of the three colored holograms. Therefore, this factor allows to multiply the wavelengths with this factor for
    ## each color channel separately, but only for diffraction angle dependent features like beam manipulation and hologram calculation.
    ## In monochrome operation, this factor is not needed typically.
    ## \param factor_red The chromatic abberation compensation factor for the red color channel.
    ## \param factor_green The chromatic abberation compensation factor for the green color channel.
    ## \param factor_blue The chromatic abberation compensation factor for the blue color channel.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setChromaticAbberationFactorsCFS(self, factor_red=1.0, factor_green=1.0, factor_blue=1.0):
        if self._err == HEDSERR_NoError:
            self._err = SDK.libapi.heds_slm_set_chromatic_abberation_factor(self._id, factor_red, HEDSCC_Red, direct_apply=False)
        if self._err == HEDSERR_NoError:
            self._err = SDK.libapi.heds_slm_set_chromatic_abberation_factor(self._id, factor_green, HEDSCC_Green, direct_apply=False)
        if self._err == HEDSERR_NoError:
            self._err = SDK.libapi.heds_slm_set_chromatic_abberation_factor(self._id, factor_blue, HEDSCC_Blue, direct_apply=True)
        return self._err

    ## Retrieve the wavelength setting of this SLM.
    ## \param color_channel_idx The color channel index to get the wavelength for. By default this is the monochrome color channel index (0) or in case this SLM is in CFS phase mode already, the wavelength of the red color channel index (0) is returned by default.
    ## \return The laser wavelength in nano meter for the requested color channel. If this SLM has no wavelength set or is not in phase modulation mode, 0.0f is returned.
    def getWavelength(self, color_channel_idx=HEDSCCI_Mono):
        wls = self.getWavelengths()
        if wls.len > color_channel_idx:
            return wls[color_channel_idx]
        # wavelength not available:
        return 0.0

    # Retrieve the wavelength settings of this SLM.
    ## \return A list of either one or three laser wavelengths in nano meter. If this SLM has no wavelength set or is not in phase modulation mode, an empty vector is returned.
    def getWavelengths(self):
        wls = [3]
        self._err, wls[0] = SDK.libapi.heds_slm_get_wavelength(self._id, HEDSCC_Red)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        self._err, wls[1] = SDK.libapi.heds_slm_get_wavelength(self._id, HEDSCC_Green)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        self._err, wls[0] = SDK.libapi.heds_slm_get_wavelength(self._id, HEDSCC_Blue)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        min_wl_nm = 1.0
        if wls[0] <= min_wl_nm:
            if wls[1] <= min_wl_nm and wls[2] <= min_wl_nm:
                 wls = wls[0]

        return wls

    ## Load Zernike parameters from a Zernike parameter text file (*.zernike.txt), for example saved by HOLOEYE SLM Pattern Generator.
    ## In case of color-field sequential operation, each color channel can have a separate Zernike parameter file. So this function only
    ## loads the data into the vector, which has up to HEDSZER_COUNT (15) values. The first value is always the Zernike radius in pixel.
    ## \param filename The file name and path to the Zernike text file to load.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return The loaded Zernike parameter array. An empty vector in any error case.
    def zernikeLoadParamsFromFile(self, filename):
        self._err, zernike_params = SDK.libapi.heds_slm_zernike_load_paramfile(self._id, filename)
        return self._err, zernike_params

    ## Applies an array of up to HEDSZER_COUNT (15) Zernike parameter values. The first value is always the Zernike radius in pixel.
    ## \param zernike_params A vector consisting of a list of Zernike parameters. The first parameter is the Zernike radius instead of the piston.
    ## \param color_channel Optional parameter to only apply the Zernike parameters into some color channels. Color channels can be combined using bitwise or.
    ##  \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def zernikeApplyParams(self, zernike_params, color_channel=HEDSCC_Mono):
        return SDK.libapi.heds_slm_zernike_set_parameters(self._id, zernike_params, color_channel)

    ## A convenience function to retrieve a string to print a vector of Zernike parameters on command line, including some info about the parameters.
    ## \param zernike_params A vector consisting of a list of Zernike parameters. The first parameter is the Zernike radius instead of the piston.
    ## \param name Optional parameter. If set, it changes the print string to give the data this specific \p name.
    ## \return An ASCII string containing multiple lines to be printed on a terminal, printing all the data of the Zernike vector in an informative way.
    @staticmethod
    def ZernikePrintString(zernike_params, name="Zernike parameters"):
        s = str(name + ": \n")
        if isinstance(zernike_params, ctypes.Array):
            zernike_params_size = len(zernike_params)
            if zernike_params_size > 0:
                s += str("  Zernike radius = " + str(zernike_params[HEDSZER_RadiusPx]) + " pixel \n")
                for i in range(1, len(zernike_params)):
                    polynom = SDK.libapi.heds_zernike_param_funcstr(i)
                    if polynom.find("+")  or polynom.find("-"):
                        polynom = "(" + polynom + ")"
                    while len(polynom) < 40: # just fill empty characters
                        polynom += " "

                    s += str("  C%02d " % i + " = %12.6f " % zernike_params[i] + " | phase(x,y) = C%02d " % i + " * %s" % polynom + " |  " + SDK.libapi.heds_zernike_param_name(i) + ".\n")
                return s
        s += str(" No valid data! \n")
        return s

    ## Loads a blank screen with a constant value. The SLM has to be initialized first.
    ## \param value The color value which will be adressed on the first side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadBlankScreen(self, value):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_blankscreen(self._id, value)
        SDK.LogErrorString(self._err)
        dh = SLMDataHandle(data_handle_id)
        self._err = dh.applyErrorCode(self._err)
        return self._err, dh

    ## Shows a blank screen with a constant value. The SLM has to be initialized first.
    ## \param value The color value which will be adressed on the first side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showBlankScreen(self, value):
        self._err = SDK.libapi.heds_slm_show_blankscreen(self._id, value)
        SDK.LogErrorString(self._err)
        return self._err

    ## Loads data to show two areas on the SLM with two different color values. The function is intended to be used for phase
    ## measurements of the SLM in which one half of the SLM can be used as a reference to the other half. The screen will be
    ## split along the vertical (y) axis. This means that the values a and b are painted to the upper and lower side of the SLM, resp.
    ## \param a_value The color value which will be adressed on the first side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \param b_value The color value which will be adressed on the second side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \param screen_divider The ratio by which the SLM screen should be divided. Meaningful values are between 0.0 and 1.0. [default: 0.5]
    ## \param flipped If set to true, the first side will addressed with \p b_value and the second side will be set to a_value. [default: false]
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadDividedScreenVertical(self, a_value, b_value, screen_divider=0.5, flipped=False):
        _err, data_handle_id = SDK.libapi.heds_slm_load_dividedscreen_vertical(self._id, a_value, b_value, screen_divider, flipped)
        SDK.LogErrorString(self._err)
        dh = SLMDataHandle(data_handle_id)
        self._err = dh.applyErrorCode(self._err)
        return self._err, dh

    ## Shows data to show two areas on the SLM with two different color values. The function is intended to be used for phase
    ## measurements of the SLM in which one half of the SLM can be used as a reference to the other half. The screen will be
    ## split along the vertical (y) axis. This means that the values a and b are painted to the upper and lower side  of the SLM, resp.
    ## \param a_value The color value which will be adressed on the first side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \param b_value The color value which will be adressed on the second side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \param screen_divider The ratio by which the SLM screen should be divided. Meaningful values are between 0.0 and 1.0. [default: 0.5]
    ## \param flipped If set to true, the first side will addressed with \p b_value and the second side will be set to a_value. [default: false]
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showDividedScreenVertical(self, a_value, b_value, screen_divider=0.5, flipped=False):
        self._err = SDK.libapi.heds_slm_show_dividedscreen_vertical(self._id, a_value, b_value, screen_divider, flipped)
        SDK.LogErrorString(self._err)
        return self._err

    ## Loads data to show two areas on the SLM with two different color values. The function is intended to be used for phase
    ## measurements of the SLM in which one half of the SLM can be used as a reference to the other half. The screen will be
    ## split along the horizontal (x) axis. This means that the values a and b are painted to the upper and lower side of the SLM, resp.
    ## \param a_value The color value which will be adressed on the first side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \param b_value The color value which will be adressed on the second side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \param screen_divider The ratio by which the SLM screen should be divided. Meaningful values are between 0.0 and 1.0. [default: 0.5]
    ## \param flipped If set to true, the first side will addressed with \p b_value and the second side will be set to a_value. [default: false]
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadDividedScreenHorizontal(self, a_value, b_value, screen_divider = 0.5, flipped = False):
        _err, data_handle_id = SDK.libapi.heds_slm_load_dividedscreen_horizontal(self._id, a_value, b_value, screen_divider, flipped)
        SDK.LogErrorString(self._err)
        dh = SLMDataHandle(data_handle_id)
        self._err = dh.applyErrorCode(self._err)
        return self._err, dh

    ## Shows data to show two areas on the SLM with two different color values. The function is intended to be used for phase
    ## measurements of the SLM in which one half of the SLM can be used as a reference to the other half. The screen will be
    ## split along the horizontal (x) axis. This means that the values a and b are painted to the upper and lower side  of the SLM, resp.
    ## \param a_value The color value which will be adressed on the first side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \param b_value The color value which will be adressed on the second side of the SLM. Values types of \b heds_dataformat are allowed.
    ## \param screen_divider The ratio by which the SLM screen should be divided. Meaningful values are between 0.0 and 1.0. [default: 0.5]
    ## \param flipped If set to true, the first side will addressed with \p b_value and the second side will be set to a_value. [default: false]
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showDividedScreenHorizontal(self,  a_value, b_value, screen_divider = 0.5, flipped = False):
        self._err = SDK.libapi.heds_slm_show_dividedscreen_horizontal(self._id, a_value, b_value, screen_divider, flipped)
        SDK.LogErrorString(self._err)
        return self._err

    ## Loads data to show a vertical binary grating. The grating consists of two values \p a_value and \p b_value, which will be
    ## addressed to the SLM pixel. The width of each area with the value \p a_value and \p b_value is defined by \p a_width
    ## and \p b_width, respectively. Each pair of values is repeated so that the SLM screen is completely filled.
    ## \param a_width The width of the first pixel block with the value of \p a_value. This parameter is mandatory.
    ## \param b_width The width of the second pixel block with the value of \p b_value. This parameter is mandatory.
    ## \param a_value The addressed value of the first pixel block.
    ## \param b_value The addressed value of the second pixel block.
    ## \param shift   The horizontal offset applied to both pixel blocks. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadGratingBinaryVertical(self, a_width, b_width, a_value, b_value, shift = 0):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_grating_binary_vertical(self._id, a_width, b_width, a_value, b_value, shift)
        SDK.LogErrorString(self._err)
        dh = SLMDataHandle(data_handle_id)
        self._err = dh.applyErrorCode(self._err)
        return self._err, dh

    ## Shows data of a vertical binary grating. The grating consists of two values \p a_value and \p b_value, which will be
    ## addressed to the SLM pixel. The width of each area with the value \p a_value and \p b_value is defined by \p a_width
    ## and \p b_width, respectively. Each pair of values is repeated so that the SLM screen is completely filled.
    ## \param a_width The width of the first pixel block with the value of \p a_value. This parameter is mandatory.
    ## \param b_width The width of the second pixel block with the value of \p b_value. This parameter is mandatory.
    ## \param a_value The addressed value of the first pixel block.
    ## \param b_value The addressed value of the second pixel block.
    ## \param shift The horizontal offset applied to both pixel blocks. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showGratingBinaryVertical(self, a_width, b_width, a_value, b_value, shift = 0):
        self._err = SDK.libapi.heds_slm_show_grating_binary_vertical(self._id, a_width, b_width, a_value, b_value, shift)
        SDK.LogErrorString(self._err)
        return self._err

    ## Loads data to show a horizontal blazed grating on the SLM.
    ## \param grating_period The grating period in SLM pixel. The value is mandatory. Can be either positive or negative for an inverted grating profile. Please note that the phase can also be inverted by the \p phase_scale. If both values are negative, the invertions will superimpose to non invertion.
    ## \param shift The horizontal offset applied to the grating in pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadGratingBinaryBlazeHorizontal(self, grating_period, shift = 0):
        _err, data_handle_id = SDK.libapi.heds_slm_load_grating_binary_horizontal(self._id, grating_period, shift)
        SDK.LogErrorString(self._err)
        dh = SLMDataHandle(data_handle_id)
        self._err = dh.applyErrorCode(self._err)
        return self._err, dh

    ## Shows data of a horizontal binary grating. The grating consists of two values \p a_value and \p b_value, which will be
    ## addressed to the SLM pixel. The width of each area with the value \p a_value and \p b_value is defined by \p a_width and
    ## \p b_width, respectively. Each pair of values is repeated so that the SLM screen is completely filled.
    ## \param a_width The width of the first pixel block with the value of \p a_value. This parameter is mandatory.
    ## \param b_width The width of the second pixel block with the value of \p b_value. This parameter is mandatory.
    ## \param a_value The addressed value of the first pixel block.
    ## \param b_value The addressed value of the second pixel block.
    ## \param shift The horizontal offset applied to both pixel blocks. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showGratingBinaryHorizontal(self, a_width, b_width, a_value, b_value, shift = 0):
        self._err = SDK.libapi.heds_slm_show_grating_binary_horizontal(self._id, a_width, b_width, a_value, b_value, shift)
        SDK.LogErrorString(self._err)
        return self._err

    ## Loads data to show a vertical blazed grating on the SLM.
    ## \param grating_period The grating period in SLM pixel. The value is mandatory. Can be either positive or negative for an inverted grating profile. Please note that the phase can also be inverted by the \p phase_scale. If both values are negative, the invertions will superimpose to non invertion.
    ## \param shift The vertical offset applied to the grating in pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadGratingBlazeVertical(self, grating_period, shift=0):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_grating_blaze_vertical(self._id, grating_period, shift)
        SDK.LogErrorString(self._err)
        dh = SLMDataHandle(data_handle_id)
        self._err = dh.applyErrorCode(self._err)
        return int(self._err), dh

    ## Show a vertical blazed grating on the SLM.
    ## \param grating_period The grating period in SLM pixel. The value is mandatory. Can be either positive or negative for an inverted grating profile. Please note that the phase can also be inverted by the \p phase_scale. If both values are negative, the invertions will superimpose to non invertion.
    ## \param shift The vertical offset applied to the grating in pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showGratingBlazeVertical(self, grating_period, shift=0) :
        self._err = SDK.libapi.heds_slm_show_grating_blaze_vertical(self._id, grating_period, shift)
        SDK.LogErrorString(self._err)
        return self._err

    ## Loads data to show a horizontal blazed grating on the SLM.
    ## \param handle A [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) is returned to identify the uploaded data.
    ## \param grating_period The grating period in SLM pixel. The value is mandatory. Can be either positive or negative for an inverted grating profile.
    ##        Please note that the phase can also be inverted by the \p phase_scale. If both values are negative, the invertions will superimpose to non invertion.
    ## \param shift The horizontal offset applied to the grating in pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadGratingBlazeHorizontal(self, handle, grating_period, shift=0):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_grating_blaze_horizontal(self._id, handle.get_handle(), grating_period, shift)
        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)

    ##  Loads data to show horizontal blazed grating on the SLM.
    ##  \param grating_period The grating period in SLM pixel. The value is mandatory. Can be either positive or negative for an inverted grating profile.
    ##         Please note that the phase can also be inverted by the \p phase_scale. If both values are negative, the invertions will superimpose to non invertion.
    ##  \param shift The horizontal offset applied to the grating in pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showGratingBlazeHorizontal(self, grating_period, shift=0):
        self._err = SDK.libapi.heds_slm_show_grating_blaze_horizontal(self._id, grating_period, shift)
        SDK.LogErrorString(self._err)
        return self._err

    ## Loads data to show an axicon phase function at the SLM screen. The phase has a conical shape.
    ## \param innerRadius The radius in number of SLM pixel where the axicon phase function reached 2pi for the first time in respect to the center of the axicon.
    ## \param centerX Horizontal shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \param centerY Vertical shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the axicon screen
    def loadPhaseFunctionAxicon(self, innerRadius, centerX=0, centerY=0):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_phasefunction_axicon(self._id, innerRadius, centerX, centerY)
        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)

    ## This function is left here during renaming, to allow backwards compatibility. This function is marked deprecated and will be deleted in next major release.
    #@deprecated("Use loadPhaseFunctionAxicon() instead.")  # needs Python 3.13 to work.
    def loadAxicon(self, innerRadius, centerX=0, centerY=0):
        warn("This function is deprecated. Use loadPhaseFunctionAxicon() instead.", DeprecationWarning)
        return self.loadPhaseFunctionAxicon(innerRadius, centerX, centerY)

    ## Shows an axicon phase function at the SLM screen. The phase has a conical shape.
    ## \param innerRadius The radius in number of SLM pixel where the axicon phase function reached 2pi for the first time in respect to the center of the axicon.
    ## \param centerX Horizontal shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \param centerY Vertical shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showPhaseFunctionAxicon(self, innerRadius, centerX=0, centerY=0):
        self._err = SDK.libapi.heds_slm_show_phasefunction_axicon(self._id, innerRadius, centerX, centerY)
        SDK.LogErrorString(self._err)
        return self._err

    ## This function is left here during renaming, to allow backwards compatibility. This function is marked deprecated and will be deleted in next major release.
    # @deprecated("Use showPhaseFunctionAxicon() instead.")  # needs Python 3.13 to work.
    def showAxicon(self, innerRadius, centerX=0, centerY=0):
        warn("This function is deprecated. Use showPhaseFunctionAxicon() instead.", DeprecationWarning)
        return self.showPhaseFunctionAxicon(innerRadius, centerX, centerY)

    ## Loads data to show a lens phase function. The phase has a parabolic shape.
    ##  The resulting focal length can be calculated as f [m] = (\p innerRadius * \b heds_slm_pixelsize_um()*1.0E-6) ^2 / (2.0*lambda),
    ##  with the incident optical wavelength lambda.
    ## \param innerRadius The radius in number of SLM pixel where the lens phase function reached 2pi for the first time in respect to the center of the lens. This value is related to the focal length f of the lens phase function by f = (inner_radius_px * \b heds_slm_pixelsize())^2 / (2*lambda).
    ## \param centerX Horizontal shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \param centerY Vertical shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadPhaseFunctionLens(self, innerRadius, centerX=0, centerY=0):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_phasefunction_lens(self._id, innerRadius, centerX, centerY)
        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)

    ## Shows data to show a lens phase function. The phase has a parabolic shape.
    ##  The resulting focal length can be calculated as f [m] = (\p innerRadius *  \b heds_slm_pixelsize_um() *1.0E-6) ^2 / (2.0*lambda),
    ##  with the incident optical wavelength lambda.
    ## \param innerRadius The radius in number of SLM pixel where the lens phase function reached 2pi for the first time in respect to the center of the lens. This value is related to the focal length f of the lens phase function by f = (inner_radius_px * \b heds_slm_pixelsize())^2 / (2*lambda).
    ## \param centerX Horizontal shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \param centerY Vertical shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showPhaseFunctionLens(self, innerRadius, centerX=0, centerY=0):
        self._err = SDK.libapi.heds_slm_show_phasefunction_lens(self._id, innerRadius, centerX, centerY)
        SDK.LogErrorString(self._err)
        return self._err

    ## Loads data to show an optical vortex phase function into and generates a handle. The phase has a helical shape.
    ## \param vortexOrder The order of the optical vortex. If the order is one, the phase goes from 0 to 2pi over the full angle of 360 degree.
    ##        For higher orders, 2pi phase shift is reached at angles of 360 degree divided by the given \p vortex_order. [default: 1].
    ## \param innerRadiusPx The radius at the sigularity which will be set to gray value 0 on the SLM. [default: 0].
    ## \param centerX Horizontal shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \param centerY Vertical shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the vortex screen to identify the uploaded data
    def loadPhaseFunctionVortex(self, vortexOrder, innerRadiusPx=0, centerX=0, centerY=0):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_phasefunction_vortex(self._id, vortexOrder, innerRadiusPx, centerX, centerY)
        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)

    ## This function is left here during renaming, to allow backwards compatibility. This function is marked deprecated and will be deleted in next major release.
    # @deprecated("Use loadPhaseFunctionVortex() instead.")  # needs Python 3.13 to work.
    def loadVortex(self, vortexOrder, innerRadiusPx=0, centerX=0, centerY=0):
        warn("This function is deprecated. Use loadPhaseFunctionVortex() instead.", DeprecationWarning)
        return self.loadPhaseFunctionVortex(vortexOrder, innerRadiusPx, centerX, centerY)

    ## Shows an optical vortex phase function at the SLM screen. The phase has a helical shape.
    ## \param vortexOrder The order of the optical vortex. If the order is one, the phase goes from 0 to 2pi over the full angle of 360 degree.
    ##        For higher orders, 2pi phase shift is reached at angles of 360 degree divided by the given \p vortex_order. [default: 1].
    ## \param innerRadiusPx The radius at the sigularity which will be set to gray value 0 on the SLM. [default: 0].
    ## \param centerX Horizontal shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \param centerY Vertical shift of the center of the optical function on the SLM in number of pixel. [default: 0].
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showPhaseFunctionVortex(self, vortexOrder, innerRadiusPx=0, centerX=0, centerY=0):
        self._err = SDK.libapi.heds_slm_show_phasefunction_vortex(self._id, vortexOrder, innerRadiusPx, centerX, centerY)
        SDK.LogErrorString(self._err)
        return self._err

    ## This function is left here during renaming, to allow backwards compatibility. This function is marked deprecated and will be deleted in next major release.
    # @deprecated("Use showPhaseFunctionVortex() instead.")  # needs Python 3.13 to work.
    def showVortex(self, vortexOrder, innerRadiusPx=0, centerX=0, centerY=0):
        warn("This function is deprecated. Use showPhaseFunctionVortex() instead.", DeprecationWarning)
        return self.showPhaseFunctionVortex(vortexOrder, innerRadiusPx, centerX, centerY)


    ## Load two-dimensional phase values data with the templated class [HEDS.SLMDataField](\ref HEDS::holoeye_slmdisplaysdk_datafield::SLMDataField) to allow multiple data formats.
    ## In addition, you can specify a phase value unit (e.g. 2 pi rad), at which the SDK can automatically wrap the values at.
    ## \param data can be filled [HEDS.SLMDataField](\ref HEDS::holoeye_slmdisplaysdk_datafield::SLMDataField). All \b heds_dataformat types are allowed.
    ## \param flags Optional load flags to change behavior when loading data. You can already pass show flags here to be stored in the returned [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle).
    ## \param phase_unit The unit of the given phase values. The default value is 2 pi radian. If necessary, the given phase values are automatically
    ##                    and efficiently wrapped into the given phase unit by the SDK.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the phase data.
    def loadPhaseData(self, data, flags=None, phase_unit=2.0 * math.pi):
        if isinstance(data, SLMDataField):
            if data.data() is None:
                return HEDSERR_InvalidDataPointer
            if flags is None:
                flags = data.flags()
            self._err, data_handle_id = SDK.libapi.heds_slm_load_phasedata(self._id, data.data(), flags, phase_unit)
        else:
            if flags is None:
                flags = HEDSLDF_Default | HEDSSHF_PresentAutomatic
            self._err, data_handle_id = SDK.libapi.heds_slm_load_phasedata(self._id, data, flags, phase_unit)

        SDK.LogErrorString(self._err)


        return self._err, SLMDataHandle(data_handle_id)

    ## Shows two-dimensional phase values data with the class [HEDS.SLMDataField](\ref HEDS::holoeye_slmdisplaysdk_datafield::SLMDataField) to allow multiple data formats.
    ## In addition, you can specify a phase value unit (e.g. 2 pi rad), at which the SDK can automatically wrap the values at.
    ## \param data can be filled [HEDS.SLMDataField](\ref HEDS::holoeye_slmdisplaysdk_datafield::SLMDataField). All \b heds_dataformat types are allowed.
    ## \param flags Optional show flags to change behavior when presenting data on screen.
    ## \param phase_unit The unit of the given phase values. The default value is 2 pi radian. If necessary, the given phase values are automatically
    ##                    and efficiently wrapped into the given phase unit by the SDK.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showPhaseData(self, data, flags=None, phase_unit=2.0 * math.pi):
        if isinstance(data, SLMDataField):
            if data.data() is None:
                return HEDSERR_InvalidDataPointer
            if flags is None:
                flags = data.flags()
            self._err = SDK.libapi.heds_slm_show_phasedata(self._id, data.data(), flags, phase_unit)
        else:
            if flags is None:
                flags = HEDSLDF_Default | HEDSSHF_PresentAutomatic
            self._err = SDK.libapi.heds_slm_show_phasedata(self._id, data, flags, phase_unit)

        SDK.LogErrorString(self._err)
        return self._err

    ## Load image data from memory into HOLOEYE SLM Display SDK. The SDK can pre-load the data into the GPU memory
    ## or any other device memory.
    ## Loading data as image data marks the data to behave differently compared to phase values.
    ## \param data can be filled [HEDS.SLMDataField](\ref HEDS::holoeye_slmdisplaysdk_datafield::SLMDataField). All \b heds_dataformat types are allowed.
    ## \param flags Optional load flags to change behavior when loading data. You can already pass show flags here to be stored in the returned [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle).
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadImageData(self, data, flags=None):
        if isinstance(data, SLMDataField):
            if data.data() is None:
                return HEDSERR_InvalidDataPointer
            if flags is None:
                flags = data.flags()
            self._err, data_handle_id = SDK.libapi.heds_slm_load_imagedata(self._id, data.data(), flags)
        else:
            if flags is None:
                flags = HEDSLDF_Default
            self._err, data_handle_id = SDK.libapi.heds_slm_load_imagedata(self._id, data, flags)

        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)


    ## Show image data into HOLOEYE SLM Display.
    ## Showing data as image data marks the data to behave differently compared to phase values.
    ##  Image data is shown as is and does not allow phase overlays.
    ## \param data can be filled [HEDS.SLMDataField](\ref HEDS::holoeye_slmdisplaysdk_datafield::SLMDataField). All \b heds_dataformat types are allowed.
    ## \param flags Optional show flags to change behavior when presenting data on screen.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showImageData(self, data, flags=None):
        if isinstance(data, SLMDataField):
            if data.data() is None:
                return HEDSERR_InvalidDataPointer
            if flags is None:
                flags = data.flags()
            self._err = SDK.libapi.heds_slm_show_imagedata(self._id, data.data(), flags)
        else:
            if flags is None:
                flags = HEDSSHF_PresentAutomatic
            self._err = SDK.libapi.heds_slm_show_imagedata(self._id, data, flags)

        SDK.LogErrorString(self._err)
        return self._err

    ## Load phase values data from a file on disk into HOLOEYE SLM Display SDK. The SDK can pre-load the data into the GPU memory
    ## or any other device memory, so that the handle can be shown fast as possible.
    ## Loading data as phase values marks the data to behave differently compared to image data.
    ## When loading phase values from image files containing integer gray values in range 0 to n_gray_levels-1, the phase values will be converted
    ## assuming a phase unit of 2*pi*rad for the whole file, i.e. each value will be read by doing
    ## > phase_val = (float)gray_level * (2*pi*rad) / (float)n_gray_levels,
    ## with n_gray_levels is typically 256 and gray_level is ranging from 0 to 255.
    ## If the image file stores multiple color channels, they are all converted into phase values separately.
    ## \param filename The file name and/or path to load the data from.
    ## \param flags Optional load flags to change behavior when loading data. You can already pass show flags here to be stored in the returned [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle).
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def loadPhaseDataFromFile(self, filename, flags = HEDSLDF_Default | HEDSSHF_PresentAutomatic):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_phasedata_from_file(self._id, filename, flags)
        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)

    ## Show phase values data from a file on disk into HOLOEYE SLM Display SDK. The phase data is immediately displayed on the SLM.
    ## Showing data as phase values marks the data to behave differently compared to image data.
    ## When showing phase values from image files containing integer gray values in range 0 to n_gray_levels-1, the phase values will be converted
    ## assuming a phase unit of 2*pi*rad for the whole file, i.e. each value will be read by doing
    ## phase_val = (float)gray_level * (2*pi*rad) / (float)n_gray_levels,
    ## with n_gray_levels is typically 256 and gray_level is ranging from 0 to 255.
    ## If the image file stores multiple color channels, they are all converted into phase values separately.
    ## \param filename The file name and/or path to load the data from.
    ## \param flags Optional show flags to change behavior when presenting data on screen.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return a [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) for the screen to identify the uploaded data.
    def showPhaseDataFromFile( self, filename, flags = HEDSLDF_Default | HEDSSHF_PresentAutomatic):
        self._err = SDK.libapi.heds_slm_show_phasedata_from_file(self._id, filename, flags)
        SDK.LogErrorString(self._err)
        return self._err

    ## Load image data from an image file on disk into HOLOEYE SLM Display SDK. The SDK can pre-load the data into the GPU memory
    ## or any other device memory, so that the handle can be shown as fast as possible.
    ## Loading data as image data marks the data to behave differently compared to phase values.
    ## Image data is shown as is and does not allow phase overlays.
    ## \param filename The file name and/or path to load the data from.
    ## \param flags Load flags to change behavior when loading data. You can already pass show flags here to be stored in the returned \p handle.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return The returned [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) gives access to the data. To check for errors, please check the [HEDS.SLMDataHandle.errorCode()](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle::errorCode()) return value.
    def loadImageDataFromFile(self, filename, flags=HEDSLDF_Default | HEDSSHF_PresentAutomatic):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_imagedata_from_file(self._id, filename, flags)
        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)

    ## Show image data from an image file on disk into SLM HOLOEYE SLM Display SDK. The image is immediately displayed on the SLM.
    ## Loading data as image data marks the data to behave differently compared to phase values.
    ## Image data is shown as is and does not allow phase overlays.
    ## \param filename The file name and/or path to load the data from.
    ## \param flags Load flags to change behavior when loading data. You can already pass show flags here to be stored in the returned \p handle.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showImageDataFromFile(self, filename, flags=HEDSLDF_Default | HEDSSHF_PresentAutomatic):
        self._err = SDK.libapi.heds_slm_show_imagedata_from_file(self._id, filename, flags)
        SDK.LogErrorString(self._err)
        return self._err

    ## Compute a Computer Generated Hologram (CGH) out of a signal image, given as image data from memory.
    ## The resulting phase data field is pre-loaded into the GPU memory or any other device memory, so that
    ## the returned data handle can be shown as fast as possible.
    ## \param data The data pointer to the beginning of the image data memory block. Each element can have different formats, please see \p datafmt.
    ##        The minimum reserved memory behind this pointer must be \p width * \p heigth * sizeof(T) bytes,
    ##        and if pitch is not 0 (default), it must be at least \p pitch * \p height * sizeof(T).
    ## \param flags Load flags to change behavior when loading data. You can already pass show flags here to be stored in the returned \p handle.
    ## \param image_size_factor This value represents the slider value found in our GUI programs to adjust the size of the resulting hologram image.
    ##                          Please use this value to adjust the resulting hologram image size relatively to the size of the diffraction unit cell size.
    ##                          Meaningful values for this scale factor are between 1% (0.01) and 100% (1.0). A value of 1.0 represents a fit into the
    ##                          unit cell size so that the whole input image is visible, and black bars are added to correct for the aspect ratio differences
    ##                          between the quadratic unit cell and the given input image.
    ##                          Technically, high resolution images are downscaled to fit into the selected maximum unit cell size (see \p max_unit_cell_size).
    ##                          For low resolution images, the actual used unit cell size is adapted to the input image and no scaling of the input image is done
    ##                          until necessary. The setting applies to all images in the batch process.
    ##                          The default scale factor is set to 0.7, because smaller holograms have better quality.
    ## \param max_unit_cell_size The unit cell size is calculated automatically to generate optimal holographic image quality. However, for large image files, a larger
    ##                           unit cell size is used which generates longer calculation times. If a faster calculation is desired, please select a smaller maximum
    ##                           unit cell size and limit the holographic quality.
    ##                           A smaller unit cell size should also be selected if the SLM height is not fully illuminated.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return The returned [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) gives access to the data. To check for errors, please check the [HEDS.SLMDataHandle.errorCode()](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle::errorCode()) return value.
    def loadCGHFromImageData(self, data, flags=HEDSLDF_Default | HEDSSHF_PresentTiledCentered, image_size_factor=0.7, max_unit_cell_size=0):
        if isinstance(data, SLMDataField):
            if data.data() is None:
                return HEDSERR_InvalidDataPointer
            if flags is None:
                flags = data.flags()
            self._err, data_handle_id = SDK.libapi.heds_slm_load_cgh_from_imagedata(self._id, data.data(), flags, image_size_factor, max_unit_cell_size )
        else:
            if flags is None:
                flags = HEDSLDF_Default
            self._err, data_handle_id = SDK.libapi.heds_slm_load_cgh_from_imagedata(self._id, data, flags, image_size_factor, max_unit_cell_size)

        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)

    ## Compute a Computer Generated Hologram (CGH) out of a signal image, given as image data from memory.
    ## The resulting phase data field is directly shown on the SLM screen.
    ## \param data The data pointer to the beginning of the image data memory block. Each element can have different formats, please see \p datafmt.
    ##        The minimum reserved memory behind this pointer must be \p width * \p heigth * sizeof(T) bytes,
    ##        and if pitch is not 0 (default), it must be at least \p pitch * \p height * sizeof(T).
    ## \param flags Show flags to change behavior when showing data. \see heds_showflags_enum.
    ## \param image_size_factor This value represents the slider value found in our GUI programs to adjust the size of the resulting hologram image.
    ##                          Please use this value to adjust the resulting hologram image size relatively to the size of the diffraction unit cell size.
    ##                          Meaningful values for this scale factor are between 1% (0.01) and 100% (1.0). A value of 1.0 represents a fit into the
    ##                          unit cell size so that the whole input image is visible, and black bars are added to correct for the aspect ratio differences
    ##                          between the quadratic unit cell and the given input image.
    ##                          Technically, high resolution images are downscaled to fit into the selected maximum unit cell size (see \p max_unit_cell_size).
    ##                          For low resolution images, the actual used unit cell size is adapted to the input image and no scaling of the input image is done
    ##                          until necessary. The setting applies to all images in the batch process.
    ##                          The default scale factor is set to 0.7, because smaller holograms have better quality.
    ## \param max_unit_cell_size The unit cell size is calculated automatically to generate optimal holographic image quality. However, for large image files, a larger
    ##                           unit cell size is used which generates longer calculation times. If a faster calculation is desired, please select a smaller maximum
    ##                           unit cell size and limit the holographic quality.
    ##                           A smaller unit cell size should also be selected if the SLM height is not fully illuminated.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showCGHFromImageData(self, data, flags=HEDSLDF_Default | HEDSSHF_PresentTiledCentered, image_size_factor=0.7, max_unit_cell_size=0):
        if isinstance(data, SLMDataField):
            if data.data() is None:
                return HEDSERR_InvalidDataPointer
            if flags is None:
                flags = data.flags()
            self._err = SDK.libapi.heds_slm_show_cgh_from_imagedata(self._id, data.data(), flags, image_size_factor, max_unit_cell_size )
        else:
            if flags is None:
                flags = HEDSLDF_Default | HEDSSHF_PresentTiledCentered
            self._err = SDK.libapi.heds_slm_show_cgh_from_imagedata(self._id, data, flags, image_size_factor, max_unit_cell_size)

        SDK.LogErrorString(self._err)
        return self._err


    ## Compute a Computer Generated Hologram (CGH) out of a signal image, given as an image file from disk.
    ## The resulting phase data field is pre-loaded into the GPU memory or any other device memory, so that
    ## the returned data handle can be shown as fast as possible.
    ## \param filename The file name and/or path to load the data from.
    ## \param flags Load flags to change behavior when loading data. You can already pass show flags here to be stored in the returned \p handle.
    ## \param image_size_factor This value represents the slider value found in our GUI programs to adjust the size of the resulting hologram image.
    ##                          Please use this value to adjust the resulting hologram image size relatively to the size of the diffraction unit cell size.
    ##                          Meaningful values for this scale factor are between 1% (0.01) and 100% (1.0). A value of 1.0 represents a fit into the
    ##                          unit cell size so that the whole input image is visible, and black bars are added to correct for the aspect ratio differences
    ##                          between the quadratic unit cell and the given input image.
    ##                          Technically, high resolution images are downscaled to fit into the selected maximum unit cell size (see \p max_unit_cell_size).
    ##                          For low resolution images, the actual used unit cell size is adapted to the input image and no scaling of the input image is done
    ##                          until necessary. The setting applies to all images in the batch process.
    ##                          The default scale factor is set to 0.7, because smaller holograms have better quality.
    ## \param max_unit_cell_size The unit cell size is calculated automatically to generate optimal holographic image quality. However, for large image files, a larger
    ##                           unit cell size is used which generates longer calculation times. If a faster calculation is desired, please select a smaller maximum
    ##                           unit cell size and limit the holographic quality.
    ##                           A smaller unit cell size should also be selected if the SLM height is not fully illuminated.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return The returned [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) gives access to the data. To check for errors, please check the [HEDS.SLMDataHandle.errorCode()](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle::errorCode()) return value.
    def loadCGHFromImageFile(self, filename, flags=HEDSLDF_Default | HEDSSHF_PresentTiledCentered, image_size_factor=0.7, max_unit_cell_size=0):
        self._err, data_handle_id = SDK.libapi.heds_slm_load_cgh_from_imagefile(self._id, filename, flags, image_size_factor, max_unit_cell_size)
        SDK.LogErrorString(self._err)
        return self._err, SLMDataHandle(data_handle_id)

    ## Compute a Computer Generated Hologram (CGH) out of a signal image, given as an image file from disk.
    ## The resulting phase data field is shown directly on the SLM screen.
    ## \param filename The file name and/or path to load the data from.
    ## \param flags Show flags to change behavior when showing data. \see heds_showflags_enum.
    ## \param image_size_factor This value represents the slider value found in our GUI programs to adjust the size of the resulting hologram image.
    ##                          Please use this value to adjust the resulting hologram image size relatively to the size of the diffraction unit cell size.
    ##                          Meaningful values for this scale factor are between 1% (0.01) and 100% (1.0). A value of 1.0 represents a fit into the
    ##                          unit cell size so that the whole input image is visible, and black bars are added to correct for the aspect ratio differences
    ##                          between the quadratic unit cell and the given input image.
    ##                          Technically, high resolution images are downscaled to fit into the selected maximum unit cell size (see \p max_unit_cell_size).
    ##                          For low resolution images, the actual used unit cell size is adapted to the input image and no scaling of the input image is done
    ##                          until necessary. The setting applies to all images in the batch process.
    ##                          The default scale factor is set to 0.7, because smaller holograms have better quality.
    ## \param max_unit_cell_size The unit cell size is calculated automatically to generate optimal holographic image quality. However, for large image files, a larger
    ##                           unit cell size is used which generates longer calculation times. If a faster calculation is desired, please select a smaller maximum
    ##                           unit cell size and limit the holographic quality.
    ##                           A smaller unit cell size should also be selected if the SLM height is not fully illuminated.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def showCGHFromImageFile(self, filename, flags=HEDSLDF_Default | HEDSSHF_PresentTiledCentered, image_size_factor=0.7, max_unit_cell_size=0):
        self._err = SDK.libapi.heds_slm_show_cgh_from_imagefile(self._id, filename, flags, image_size_factor, max_unit_cell_size)
        SDK.LogErrorString(self._err)
        return self._err

    ## Provides the width of the SLM in number of pixels.
    ## \return The width of SLM in number of pixel.
    def width_px(self):
        self._err, pxPosX, pxPosY, pxWidth, pxHeight = SDK.libapi.heds_slm_get_geometry(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return pxWidth

    ## Provides the width of the SLM in millimeter.
    ## \return The width of SLM in mm.
    def width_mm(self):
        return self.width_px() * self.pixelsize_um() / 1000.0

    ## Provides the height of the SLM in number of pixels.
    ## \return The height of SLM in number of pixel.
    def height_px(self):
        self._err, pxPosX, pxPosY, pxWidth, pxHeight = SDK.libapi.heds_slm_get_geometry(self._id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        return pxHeight

    ## Provides the height of the SLM in milli meter.
    ## \return The height of SLM in mm.
    def height_mm(self):
        return self.height_px() * self.pixelsize_um() / 1000.0

    ## Provides the pixel size/pitch relevant for diffraction patterns of the SLM. Pixel pitch is always the same for x- and y- direction.
    ## Can be used to calculate active area of the SLM, see getWidth()
    ##  \return The pixel size/pitch in micro meter (um).
    def pixelsize_um(self):
        return self._Wnd.pixelsize_um()

    ## Provides the refresh rate of the SLM screen in Hertz (1/s).
    ## \return The SLM's current refresh rate. Returns zero when the SLM was not initialized properly.
    def refreshrate_hz(self):
        return self._Wnd.refreshrate_hz()

    ## Provides the inverse refresh rate of the SLM screen in milliseconds. This is the duration of each video frame.
    ## \return The SLMs video output frame time in ms. Returns zero when the SLM was not initialized properly.
    def frametime_ms(self):
        rr = self.refreshrate_hz()
        if rr <= 0.0:
            return 0.0
        return 1000.0/rr

    ## Retrieve a class to access SLM preview window for this SLM.
    ## \return An [HEDS.SLMPreview](\ref HEDS::holoeye_slmdisplaysdk_preview::SLMPreview) object assigned to this SLM.
    def window(self):
        return self._Wnd

    ## Retrieve a class to access SLM preview window for this SLM.
    ## \return An SLMPreview object assigned to this SLM.
    def preview(self):
        return SLMPreview(self._id.slmwindow_id)

    ## Provides the current error code SLM
    ## \return \b heds_errorcode errorCode.
    def errorCode(self):
        return self._err

    ## Provides the address/id  of the SLM.
    ## \return \b heds_slm_id is the id of the SLM.
    def id(self):
        return self._id

    ##  Waits until the given duration has passed. If given duration is zero, it will wait forever.
    ##  The function checks if the SLM/SLM window is closed (e.g. through tray icon) during the wait.
    ##  Therefore, by default, a duration of zero will wait until the given SLM window is closed, either manually or due to an error.
    ##  \param duration_ms as int: The number of milliseconds to wait before returning for sure. By default (0), the function waits forever until the given SLM window was closed.
    ##  \returns HEDSERR_NoError when the desired wait duration was reached. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def wait(self, duration_ms=0):
        self._err = self._Wnd.wait(duration_ms)
        return self._err

from holoeye_slmdisplaysdk_slmwindow import *
from holoeye_slmdisplaysdk_preview import *
from holoeye_slmdisplaysdk_datafield import *
