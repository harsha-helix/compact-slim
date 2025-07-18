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


##  Class to provide main acces to the SLM Display SDK. Used to initialzing, closing, versioning etc.
## \ingroup SLMDisplaySDK_CONVAPI_Python
class SDK:
    ## Open the library file and save an instance into heds (HOLOEYE SLM Display SDK):
    libapi = holoeye_slmdisplaysdk_library(os.path.abspath(library_file_path))

    ## Initializes SLM Display SDK and checks for the given version numbers to be fulfilled to be able to
    ## detect API incompatibilities between the actually used SDK version and the user code version.
    ## \param major The major version number of SLM Display SDK your code is meant to be run with.
    ## \param minor The minor version number of SLM Display SDK your code is meant to be run with.
    ## \param base_port If set, the base port number on the internal network is changed to the given value. Default value 0 does not change the default port range.
    ## \return HEDSERR_NoError when there is no error.Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    @staticmethod
    def Init(major, minor, base_port=0):
        # Make sure the library file has the correct version required by this script file:
        err = SDK.libapi.heds_requires_version(major, minor)
        assert err == HEDSERR_NoError, SDK.libapi.heds_error_string(err)
        # Make sure library is initialized to be able to open SLM windows:
        err = SDK.libapi.heds_config_api(HEDSAPI_Python, pyverstr)
        assert err == HEDSERR_NoError, SDK.libapi.heds_error_string(err)
        # Apply the base port number for internal network. Default value 0 is automatically handled internally:
        err = SDK.libapi.heds_set_base_portnumber(base_port)
        assert err == HEDSERR_NoError, SDK.libapi.heds_error_string(err)

        return err

    ## Providing a version string of the SLM Display SDK.
    ## \return string This the version string of HOLOEYE SLM Display SDK. The string is always UTF_8.
    @staticmethod
    def Version():
        return SDK.libapi.heds_info_version_string()  # Make sure this returns a string

    @staticmethod
    def PrintVersion():
        printstr = "Using HOLOEYE SLM Display SDK version " + SDK.Version() + "+r" + str(SDK.libapi.heds_info_version_revision()) + ".\n"
        print(printstr)

    ## Getting the major version of the SLM Display SDK as a number.
    ## \return int major HOLOEYE SLM Display SDK version number.
    @staticmethod
    def Major():
        return SDK.libapi.heds_info_version_major ()

    ## Getting the minor version of the SLM Display SDK as a number.
    ## \return int minor HOLOEYE SLM Display SDK version number.
    @staticmethod
    def Minor():
        return SDK.libapi.heds_info_version_minor()

    ## Getting the revision of the SLM Display SDK as a number.
    ## \return int revision number of the HOLOEYE SLM Display SDK.
    @staticmethod
    def Revision():
        return SDK.libapi.heds_info_version_revision()

    ## Getting the hotfix of the SLM Display SDK SLM Display SDK as a number.
    ## \return int hotfix number of the SDK.
    @staticmethod
    def Hotfix():
        return SDK.libapi.heds_info_version_hotfix()

    ## Shows all given handles to view the data at all SLMs.
    ## \param handles
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    @staticmethod
    def ShowDataHandles(handles):
        assert handles is not None, "Missing argument handles."
        assert isinstance(handles, list), "Argument is not a list."
        assert len(handles) > 0, "Argument list is empty."

        slm_datahandle_id_list = (heds_datahandle_id * len(handles))()

        i = 0
        for dhid in handles:
            if isinstance(handles[0], heds_datahandle_id):
                slm_datahandle_id_list[i] = dhid
            elif isinstance(handles[0], SLMDataHandle):
                slm_datahandle_id_list[i] = dhid.id()
            else:
                assert False, "Unsupported argument type."
            i += 1

        err = SDK.libapi.heds_datahandles_show(slm_datahandle_id_list)
        SDK.LogErrorString(err)
        return err

    @staticmethod
    ## Releases all internal data (including data on GPU memory) for all handles uploaded into the given SLM window.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def ReleaseAllHandles():
        for window_id in range(HEDS_SLMWINDOW_ID_MAX):
            if SDK.libapi.heds_slmwindow_isopen(window_id, True):
                err = SDK.libapi.heds_handles_release_all(window_id)
                assert err == HEDSERR_NoError, SDK.libapi.heds_error_string(err)
        return

    @staticmethod
    ## Closes everything, i.e. closes all open SLMs and SLM windows including tray icons etc., and cleans up everything.
    ## \return None
    def Close():
        # close all SLMs
        SDK.libapi.heds_close()
        return None

    @staticmethod
    ## Waits until each open SLM window was closed manually by using the tray icon GUI.
    ## \return None
    def WaitAllClosed():
        # wait until all SLMs are closed
        for window_id in range(HEDS_SLMWINDOW_ID_MAX):
            if SDK.libapi.heds_slmwindow_isopen(window_id, True):
               err = SDK.libapi.heds_slmwindow_wait(window_id, 0)
               if err != HEDSERR_NoError:
                   print(SDK.libapi.heds_error_string(err))

        return None

    @staticmethod
    ## A convenience function to print detected monitor info from the system on standard output.
    ## Please note: When there is no SLM window open, the function will fail to detect monitors
    ## and will only print out some default values.
    ## \return None
    def PrintMonitorInfos():
        monitor_count =  SDK.libapi.heds_info_monitor_count()
        monitor_id_primary = SDK.libapi.heds_info_monitor_get_id_primary()
        monitor_id_secondary = SDK.libapi.heds_info_monitor_get_id_secondary()
        monitor_id_used_slm = []

        for slmwid in range(HEDS_SLMWINDOW_MAX_COUNT):
            monitor_id_used_slm.append(SDK.libapi.heds_info_monitor_get_id_used_slm(slmwid))

        detected_id_string_primary = " <-- Primary monitor"
        detected_id_string_secondary = " <-- Secondary monitor"
        detected_id_string_used_slm = " <-- Used by SLM window ID "
        for monitor_id in range(monitor_count):
            err, x, y, w, h = SDK.libapi.heds_info_monitor_get_geometry(monitor_id)
            assert err == HEDSERR_NoError, SDK.libapi.heds_error_string(err)
            framerate = SDK.libapi.heds_info_monitor_get_framerate(monitor_id)
            name = SDK.libapi.heds_info_monitor_get_name(monitor_id)
            detected_id_string = ""
            if monitor_id == monitor_id_primary:
                detected_id_string = detected_id_string_primary
            elif monitor_id == monitor_id_secondary:
                detected_id_string = detected_id_string_secondary
            else:
                for slmwid in range(HEDS_SLMWINDOW_MAX_COUNT):
                    if monitor_id == monitor_id_used_slm[slmwid]:
                        detected_id_string = detected_id_string_used_slm + str(slmwid)
                        break
            print("monitor_id = %2d: [%5d, %5d, %5d x %5d] @%.1f Hz - \"%s\"%s" % (monitor_id, x, y, w, h, framerate, name, detected_id_string))

        return None

    @staticmethod
    ## Get the errorstring for heds_errorcode.
    ## \param err is a heds_errorcode of the SLM Display SDK
    ## \return string of error description.
    def ErrorString(err):
        return SDK.libapi.heds_error_string(err)

    @staticmethod
    ## Logging the errorstring to console output.
    ## \param err is a heds_errorcode of the SLM Display SDK
    ## \return None
    def LogErrorString(err):
        if err != HEDSERR_NoError:
            logging.error(SDK.libapi.heds_error_string(err))

        return None

## The class describes a rectangular geometry. It is used to determine or maintain the size of an SLM.
## \ingroup SLMDisplaySDK_CONVAPI_Python
class RectGeometry:

    ## Constructs a rectangular geometry to determine the size of SLM.
    ## \param x is the position in horizontal direction. The default value is set to 0.
    ## \param y is the position in vertical direction. The default value is set to 0.
    ## \param width is the position of the horizontal direction. The default value is set to 0.
    ## \param height is the position in vertical direction. The default value is set to 0.
    def __init__(self, x=0, y=0, width=0, height=0):
        ## Position in horizontal direction of the rectangle geometry.
        self._x = int(x)
        ## Position in vertical direction of the rectangle geometry.
        self._y = int(y)
        ## Width of the rectangle geometry.
        self._width = int(width)
        ## Height of the the rectangle geometry.
        self._height = int(height)

    ## Returns the position in horizontal direction of the rectangle.
    ## \return the position in horizontal direction.
    def x(self):
        return int(self._x)

    ## Returns the position in vertical direction of the rectangle.
    ## \return the position in vertical direction.
    def y(self):
        return int(self._y)

    ## Returns the width of the rectangle.
    ## \return the width of the rectangle.
    def width(self):
        return int(self._width)

    ## Returns the height of the rectangle.
    ## \return the height of the rectangle.
    def height(self):
        return int(self._height)



from holoeye_slmdisplaysdk_datahandle import SLMDataHandle
