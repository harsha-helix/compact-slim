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


## Stores if NumPy could be found.
# \ingroup SLMDisplayPython
import HEDS

supportNumPy = True

try:
    import numpy
except:
    supportNumPy = False

from hedslib import *

## The class references the internal data which has been uploaded to the SLM.
## \ingroup SLMDisplaySDK_CONVAPI_Python
class SLMDataHandle:

    ## Constructs a default datahandle for SLM output data.
    ## \param handle_id An optional initial datahandle ID retrieved from the Library APIs load data functions.
    def __init__(self, handle_id = None):
        ## Holds the current error code.
        self._err = HEDSERR_NoError
        ## Holds the data handle of type \b heds_datahandle_id.
        self._handle_id = heds_datahandle_id()
        if handle_id is not None:
            self._handle_id = handle_id

    ## Shows the data, this data handle refers to, on its SLM screen. When uploading the data, the data is assigned to a data handle, which
    ## stores on which SLM the data handle was uploaded into, and that is where it is shown when using this function. The show uses the
    ## show flags stored in this data handle. Show flags can be overwritten using the parameter \p showflags.
    ## When the data handle is already visible, this function will show the data again, for its duration in frames, and all changes made
    ## to the handle are automatically applied, so there is no need to call \ref apply() before showing the data handle.
    ## \param showflags Optional parameter. If provided, show flags of the data handle are overwritten. Show flags define how the data is shown on the SLM.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def show(self, showflags=None):
        if self._handle_id is None:
            return HEDSERR_InvalidHandle
        if showflags is not None:
            self._err = SDK.libapi.heds_datahandle_set_showflags(self._handle_id, showflags)
            assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        dhids = (heds_datahandle_id * 1)()
        dhids[0] = self._handle_id
        self._err = SDK.libapi.heds_datahandles_show(dhids)
        return self._err

    ## If this data handle is visible already, setting new properties does not apply the new values until the data handle is shown again
    ## By calling this apply() function, the changes can be applied manually while the data is visible.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def apply(self):
        if self._handle_id is None:
            return HEDSERR_InvalidHandle
        self._err = SDK.libapi.heds_datahandle_apply_changes(self._handle_id)
        return self._err

    ## Releases the given data handle.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def release(self):
        if self._handle_id is None:
            return HEDSERR_InvalidHandle

        dhids = (heds_datahandle_id * 1)()
        dhids[0] = self._handle_id

        self._err = SDK.libapi.heds_datahandles_release(dhids)
        self._handle_id = heds_datahandle_id()
        return self._err

    ## Apply new show flags to this data handle. The new value is either applied when using the \ref show() or the \ref apply() function. \b heds_showflags_enum.
    ## \param showflags The new show flags to apply to this data handle.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setShowFlags(self, showflags):
        self._err = SDK.libapi.heds_datahandle_set_showflags(self._handle_id, showflags)
        return self._err

    ## Get the currently set show flags from this data handle. \b heds_showflags_enum for more info about show flags.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def getShowFlags(self):
        self._err, show_flags = SDK.libapi.heds_datahandle_get_showflags(self._handle_id)
        return self._err, show_flags

    ## Setting the show duration for this data in frames.
    ## \param duration_in_frames as duration in frames.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setDuration(self, duration_in_frames=1):
        self._err = SDK.libapi.heds_datahandle_set_duration(self._handle_id, duration_in_frames)
        return self._err

    ## Getting the show duration for this data in frames.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return uint as duration in frames.
    def getDuration(self):
        self._err, duration_in_frames = SDK.libapi.heds_datahandle_get_duration(self._handle_id)
        return int(self._err), duration_in_frames

    ## Set beam manipulation using the convenience class [HEDS.BeamManipulation](\ref HEDS::holoeye_slmdisplaysdk_beammanipulation::BeamManipulation).
    ##  This allows easier conversion between physical properties and generalized parameters.
    ##  \param bm Beam manipulation object containing at least the generalized beam manipulation parameters.
    ##            Wavelength etc. are not required to be set, but are need to be set to convert from physical
    ##            units if physical units were set.
    ##  \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setBeamManipulation(self, bm):
        self._err = SDK.libapi.heds_datahandle_set_beammanipulation(self._handle_id, bm.getBeamSteerX(), bm.getBeamSteerY(), bm.getBeamLens())
        return self._err

    ## Retrieve an [HEDS.BeamManipulation](\ref HEDS::holoeye_slmdisplaysdk_beammanipulation::BeamManipulation) object containing all relevant beam manipulation information.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return bm Returns a new [HEDS.BeamManipulation](\ref HEDS::holoeye_slmdisplaysdk_beammanipulation::BeamManipulation) object set with all relevant parameters and infos to be able to retrieve all information from it easily.
    def getBeamManipulation(self):
        self._err, beamSteerX, beamSteerY, beamLens = SDK.libapi.heds_datahandle_get_beammanipulation(self._handle_id)
        bm = HEDS.BeamManipulation(beamSteerX, beamSteerY, beamLens, 0, 0.0, 0.0, None, self._handle_id)
        return self._err, bm

    ## This feature allows scaling the data on the SLM screen by the \p scale factor. The data remains on its center position.
    ## \param scale as float value. The factor by which to resize the shown data on the SLM screen.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setTransformScale(self, scale):
        self._err = SDK.libapi.heds_datahandle_set_transform_scale(self._handle_id, scale)
        return self._err

    ## Get the current scale factor of this data handle.
    ## \return scaling as float value. Returns the current scale factor.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def getTransformScale(self):
        self._err, scale = SDK.libapi.heds_datahandle_get_transform_scale(self._handle_id)
        return int(self._err), scale

    ## Sets the horizontal and vertical shifts of the data on the SLM screen in number of pixel.
    ## \param shift_x_px as int value. Shift in x-direction (horizontal) in pixel.
    ## \param shift_y_px as int value. Shift in y-direction (vertical) in pixel.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setTransformShift(self, shift_x_px, shift_y_px):
        self._err = SDK.libapi.heds_datahandle_set_transform_shift(self._handle_id, int(shift_x_px), int(shift_y_px))
        return self._err

    ## Gets the horizontal and vertical shifts of the data on the SLM screen in number of shifted pixel.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return shift_x_px as int value. Shift in x-direction (horizontal) in pixel.
    ## \return shift y_px as int value. Shift in y-direction (vertical) in pixel.
    def getTransformShift(self):
        self._err, shift_x_px, shift_y_px = SDK.libapi.heds_datahandle_get_transform_shift(self._handle_id)
        return int(self._err), shift_x_px, shift_y_px

    ## Set the gamma factor to apply a gray value manipulation curve on the SLM the data is shown on.
    ## \param gamma as float value. The gamma factor to apply to the SLM the data is shown on. \b heds_datahandle_set_gamma() for more info.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setGamma(self, gamma):
        self._err = SDK.libapi.heds_datahandle_set_gamma(self._handle_id, gamma)
        return self._err

    ## Get the gamma factor currently set to this data handle.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return gamma as float value. Returns the currently set gamma factor.
    def getGamma(self):
        self._err, gamma = SDK.libapi.heds_datahandle_get_gamma(self._handle_id)
        return self._err, gamma

    ## Set a value offset to the data. The value offset can be given in all supported formats.
    ## \param offset The value offset added to the data.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def setValueOffset(self, offset):
        self._err = SDK.libapi.heds_datahandle_set_value_offset(self._handle_id, offset)
        return self._err

    ## Get the currently set value offset of this data handle.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return offset Returns the value offset in the requested format.
    ## \return fmt is the format of offset value. Please \b heds_dataformat_enum to retrieve further error information.
    def getValueOffset(self):
        self._err, offset, fmt = SDK.libapi.heds_datahandle_get_value_offset(self._handle_id)
        return self._err, offset, fmt

    ## Getting all properies for this data.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return fmt is the format of offset value. Please \b heds_dataformat_enum to retrieve further error information.
    ## \return width as uint value. The data width in number of pixel of the uploaded data.
    ## \return height as uint value. The data height in number of pixel of the uploaded data.
    ## \return pitch as uint value. The data pitch in number of bytes per line of the uploaded data.
    ## \return phasewrap as float value. The phase unit/wrap of the uploaded data.
    ## \return uint as flags. The data flags, i.e. load and show flags, of the uploaded data.
    def getDataProperties(self):
        self._err, fmt, width, height, pitch, phase_wrap, flags = SDK.libapi.heds_datahandle_get_data_properties(self._handle_id)
        return int(self._err), fmt, width, height, pitch, phase_wrap, flags

    ## Retrieve the current data processing state. \b heds_datahandle_state_enum for available state values.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return state pls.\b heds_datahandle_state_enum to retrieve further error information.
    def getState(self):
        self._err, state = SDK.libapi.heds_datahandle_get_state(self._handle_id)
        return int(self._err), state

    ## Retrieve the duration in milliseconds for how long a specified \p state was active within the processing pipeline.
    ## \param state The state for which to receive the duration for.
    ##        If this returns HOLOEYE_FALSE (0), \p duration_ms is 0, so that no special value would return invalid duration values when calculating with or visualizing the returned durations.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return duration_ms Returns the duration of the specified \p state in milliseconds.
    ## \return was_done returns if the state was ever reached so far. Some states may not be executed during the processing pipeline, depending on the uploaded data.
    def geStateDuration(self, state):
        self._err, duration_ms, was_done = SDK.libapi.heds_datahandle_get_state_duration_ms(self._handle_id, state)
        return int(self._err), duration_ms, was_done

    ## Returns a string for timing printout of this data handle when playing a slideshow of data handles.
    ## Please call after the data handle was visible and a new data handle is already visisble, otherwise timing measurement data may be incomplete.
    ## \param includePrepare If true, the prepare duration is printed as well. Prepare duration is only measured once when loading data from file, transferring data into VRAM, etc. .
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    ## \return A command line printout string containing relevant timing data of this data handle.
    def getTimingPrintString(self, includePrepare=False):
        printstr = ""
        self._err = SDK.libapi.heds_datahandle_read_internal_data(self._handle_id)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)

        printstr += "ID: %04u" % self._handle_id.datahandle_id

        duration_ms = 0
        was_done = 0
        if includePrepare:
            self._err, duration_ms, was_done = SDK.libapi.heds_datahandle_get_state_duration_ms(self._handle_id, HEDSDHST_ReadyToRender)
            assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
            printstr += " | Prepare Duration: %3dms" % duration_ms

        self._err, duration_ms, was_done = SDK.libapi.heds_datahandle_get_state_duration_ms(self._handle_id, HEDSDHST_Rendering)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        printstr += " | Render Duration: %3dms" % duration_ms

        self._err, duration_ms, was_done = SDK.libapi.heds_datahandle_get_state_duration_ms(self._handle_id, HEDSDHST_Visible)
        assert self._err == HEDSERR_NoError, SDK.libapi.heds_error_string(self._err)
        printstr += " | Visible Duration: %3dms" % duration_ms

        return printstr

    ## Request to block the current thread until a given \p state has been reached for this data handle.
    ## \param state The state to wait for to be reached.
    ## \return HEDSERR_NoError when there is no error. Please use \ref HEDS::SDK::ErrorString() to retrieve further error information.
    def waitforState(self, state=HEDSDHST_ReadyToRender):
        self._err = SDK.libapi.heds_datahandle_waitfor_state(self._handle_id, state)
        return self._err

    ## Getting the handle with a type of \b heds_datahandle_id.
    ## \return handle of \b heds_datahandle_id
    def id(self):
        return self._handle_id

    ## Get the ID of the SLM this data handle was uploaded into.
    ## \return ID of the SLM.
    def slmId(self):
        return heds_slm_id(self._handle_id.slmwindow_id, self._handle_id.slmscreen_id)

    ## Reads the error code from the data handle.
    ## \return \b heds_errorcode Returns the first occurred error code.
    def errorCode(self):
        return self._err

    ## A function to set an error code into this data handle only if the data handle does not have an error code set already.
    ## This is used by the Python convenience API implementation to return an SLMDataHandle object also containing all error codes combined.
    ## The function also uses the macro to print/call the error information in case there is any error.
    ## \param err The error code to apply into this data handle.
    ## \return The latest error code within this class.
    def applyErrorCode(self, err):
        if self._err == HEDSERR_NoError:
            self._err = err

        SDK.LogErrorString(self._err)
        return self._err

    ## Checks if the ID stored in this class is valid. Data is invalidated when calling the \ref release() function.
    ## \return True when data is still valid and can be shown on the SLM.
    def isValid(self):
        return self._handle_id.slmwindow_id != HEDS_SLMWINDOW_ID_INVALID

from holoeye_slmdisplaysdk_beammanipulation import *
