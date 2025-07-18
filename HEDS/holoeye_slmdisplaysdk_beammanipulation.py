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
import HEDS

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



## A class to conveniently convert generalized beam manipulation parameters to and from physical units.
## \ingroup SLMDisplaySDK_CONVAPI_Python
class BeamManipulation:

    ## Creates a new BeamManipulation object, optionally using initial generalized beam manipulation parameters already.
    ##  These generalized parameters can be directly applied without the need to provide SLM width, wavelength, and pixel size.
    ##  Please note: After using this constructor, physical units can not be used in setter/getter functions.
    ##  To make use of physical units, please provide either wavelength/pixelsize, an SLM, or an [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle), see other constructors.
    ##  \param beamSteerX as float.  Optional initial generalized beam manipulation value to steer the beam in x-direction.
    ##  \param beamSteerY Optional initial generalized beam manipulation value to steer the beam in y-direction.
    ##  \param beamLens Optional initial generalized beam manipulation value to defocus the beam.
    ##  \param slm_width_px The width of the SLM in pixel is needed to convert between focal length and beam lens values.
    ##  \param wavelength_nm The incident laser wavelength in nano meter. In CFS case, please provide the laser wavelength of the blue laser.
    ##  \param pixelsize_um The pixel pitch of the SLM in micro meter.
    ##  \param slm The SLM screen object to retrieve SLM width, wavelength, and pixel pitch from.
    ##  \param handle An [HEDS.SLMDataHandle](\ref HEDS::holoeye_slmdisplaysdk_datahandle::SLMDataHandle) object to retrieve the SLM from, to then get SLM width, wavelength, and pixel pitch from.
    def __init__(self, beamSteerX=0.0, beamSteerY=0.0, beamLens=0.0, slm_width_px=int(0), wavelength_nm=0.0, pixelsize_um=0.0, slm=None, handle=None):
        ## The generalized beam steer parameter for steering in x direction (horizontal steering).
        self._beamSteerX = beamSteerX
        ## The generalized beam steer parameter for steering in y direction (vertical steering).
        self._beamSteerY = beamSteerY
        ## The generalized beam lens parameter for defocus.
        self._beamLens = beamLens
        ## The width of the SLM.
        self._width_px = 0
        ## The incident laser wavelength.
        self._wavelength_nm = 0.0
        ## The pixel pitch of the SLM.
        self._pixelsize_um = 0

        if slm is None and handle is None :
            self._width_px = slm_width_px
            self._wavelength_nm = wavelength_nm
            self._pixelsize_um = pixelsize_um
            return

        if slm is not None and isinstance(slm, HEDS.SLM):
            self._width_px = slm.width_px()
            self._wavelength_nm = slm.getWavelength(HEDSCCI_Blue)
            self._pixelsize_um = slm.pixelsize_um()
            return

        if handle is not None and isinstance(handle, HEDS.SLMDataHandle):
            self._width_px=0
            self._wavelength_nm=0.0
            self._pixelsize_um=0.0
            slm_id = heds_slm_id()
            slm_id.slmwindow_id = handle.id().slmwindow_id
            slm_id.slmscreen_id = handle.id().slmscreen_id

            ## We need the blue wavelength in case of CFS mode, in monochrome mode, blue wavelength is the same like the monochrome one.
            err, self._wavelength_nm = SDK.libapi.heds_slm_get_wavelength(slm_id, HEDSCCI_Blue)
            assert err == HEDSERR_NoError, SDK.libapi.heds_error_string(err)

            self._pixelsize_um = SDK.libapi.heds_slmwindow_pixelsize_um(slm_id.slmwindow_id)

            err, x, y, self._width_px, h = SDK.libapi.heds_slm_get_geometry(slm_id)
            assert err == HEDSERR_NoError, SDK.libapi.heds_error_string(err)

    ## Set the beam steer in x-direction using an angle in unit degree.
    ## \param deviation_angle_deg The beam steer angle for steering in x-direction (horizontal steering) in degree.
    ## \return None
    def setBeamSteerXDegree(self, deviation_angle_deg = 0.0):
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        self._beamSteerX = SDK.libapi.heds_datahandle_beam_steer_from_angle_deg(self._pixelsize_um, self._wavelength_nm, deviation_angle_deg)
        return None

    ## Set the beam steer in x-direction using an angle in unit radian.
    ## \param deviation_angle_rad The beam steer angle for steering in x-direction (horizontal steering) in radian.
    ## \return None
    def setBeamSteerXRad(self, deviation_angle_rad=0.0):
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        self._beamSteerX = SDK.libapi.heds_datahandle_beam_steer_from_angle_rad(self._pixelsize_um, self._wavelength_nm, deviation_angle_rad)
        return None

    ## Set the beam steer in x-direction using the generalized value, independent on wavelength and pixel pitch.
    ## \param beam_steer The generalized beam steer parameter for steering in x-direction (horizontal steering).
    ## \return None
    def setBeamSteerX(self, beam_steer=0.0):
        self._beamSteerX = beam_steer
        return None

    ## Set the beam steer in y-direction using an angle in unit degree.
    ## \param deviation_angle_deg The beam steer angle for steering in y-direction (vertical steering) in degree.
    ## \return None
    def setBeamSteerYDegree(self, deviation_angle_deg=0.0):
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        self._beamSteerY = SDK.libapi.heds_datahandle_beam_steer_from_angle_deg(self._pixelsize_um, self._wavelength_nm, deviation_angle_deg)
        return None

    ## Set the beam steer in y-direction using an angle in unit radian.
    ## \param deviation_angle_rad The beam steer angle for steering in y-direction (vertical steering) in radian.
    ## \return None
    def setBeamSteerYRad(self, deviation_angle_rad=0.0):
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        self._beamSteerY = SDK.libapi.heds_datahandle_beam_steer_from_angle_rad(self._pixelsize_um, self._wavelength_nm, deviation_angle_rad)
        return None

    ## Set the beam steer in y-direction using the generalized value, independent on wavelength and pixel pitch.
    ## \param beam_steer The generalized beam steer parameter for steering in y-direction (vertical steering).
    ## \return None
    def setBeamSteerY(self, beam_steer=0.0):
        self._beamSteerY = beam_steer
        return None

    ## Set the beam lens using a focal length.
    ## \param lens_focal_length_mm The generalized beam lens parameter for defocus, given as the focal length of the Fresnel zone lens in milli-meter.
    ## \return None
    def setBeamLensFocalLengthMM(self, lens_focal_length_mm=0.0):
        assert self._width_px > 0, "Missing pixel width parameter."
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        self._beamLens = SDK.libapi.heds_datahandle_beam_lens_from_focal_length_mm(self._width_px, self._pixelsize_um, self._wavelength_nm, lens_focal_length_mm)
        return None

    ## Set the beam lens using the generalized value, independent on wavelength and pixel pitch.
    ## \param beam_lens The generalized beam lens parameter for defocus.
    ## \return None
    def setBeamLens(self, beam_lens=0.0):
        self._beamLens = beam_lens
        return None

    ## Get the beam steer in x-direction as an angle in unit degree.
    ## \return The horizontal beam steer deflection angle in degree.
    def getBeamSteerXDegree(self):
        assert self._width_px > 0, "Missing pixel width parameter."
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        return SDK.libapi.heds_datahandle_beam_steer_to_angle_deg(self._pixelsize_um, self._wavelength_nm, self._beamSteerX)

    ## Get the beam steer in x-direction as an angle in unit radian.
    ## \return The horizontal beam steer deflection angle in radian.
    def getBeamSteerXRad(self):
        assert self._width_px > 0, "Missing pixel width parameter."
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        return SDK.libapi.heds_datahandle_beam_steer_to_angle_rad(self._pixelsize_um, self._wavelength_nm, self._beamSteerX)

    ## Get the beam steer in x-direction as the generalized value, independent on wavelength and pixel pitch.
    ## \return The generalized beam steering value for horizontoal steering.
    def getBeamSteerX(self):
        return self._beamSteerX

    ## Get the beam steer in y-direction as an angle in unit degree.
    ## \return The vertical beam steer deflection angle in degree.
    def getBeamSteerYDegree(self):
        assert self._width_px > 0, "Missing pixel width parameter."
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        return SDK.libapi.heds_datahandle_beam_steer_to_angle_deg(self._pixelsize_um, self._wavelength_nm, self._beamSteerY)

    ## Get the beam steer in y-direction as an angle in unit radian.
    ## \return The vertical beam steer deflection angle in radian.
    def getBeamSteerYRad(self):
        assert self._width_px > 0, "Missing pixel width parameter."
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        return SDK.libapi.heds_datahandle_beam_steer_to_angle_rad(self._pixelsize_um, self._wavelength_nm, self._beamSteerY)

    ## Get the beam steer in y-direction as the generalized value, independent on wavelength and pixel pitch.
    ## \return The generalized beam steering value for vertical steering.
    def getBeamSteerY(self):
        return self._beamSteerY

    ## Set the beam lens as a focal length in unit milli meter.
    ## \return The focal length of the beam manipulation lens in milli-meter.
    def getBeamLensFocalLengthMM(self):
        assert self._width_px > 0, "Missing pixel width parameter."
        assert self._pixelsize_um > 0.0, "Missing pixel size parameter."
        assert self._wavelength_nm > 0.0, "Missing wave length parameter."
        return SDK.libapi.heds_datahandle_beam_lens_to_focal_length_mm(self._width_px, self._pixelsize_um, self._wavelength_nm, self._beamLens)

    ## Set the beam lens as the generalized value, independent on wavelength and pixel pitch.
    ## \return The generalized beam lens value.
    def getBeamLens(self):
        return self._beamLens


from holoeye_slmdisplaysdk_types import *




