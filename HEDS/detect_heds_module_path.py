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


# This file is part of the HEDS module, which needs to be copied into your script
# folder and can then be imported like this:
#
# import HEDS
#
# The HEDS module folder contains the Python Convenience API files of HOLOEYE SLM Display SDK.
# This file adds the installation folder into the Python system path, so that the Convenience
# API can import the Library API from the installation folder path.


import os
import sys

# Import the SLM Display SDK:
# define the module path for the Python API bindings:
envvar_name = "HEDS_4_1_PYTHON"
env_path = os.getenv(envvar_name)
if env_path is None or not os.path.isdir(env_path):
    env_path = os.path.abspath("../..")

importpath = ""
if env_path is not None:
    importpath = os.path.join(env_path, "api", "python")

asserterrmsg = "Failed to find HOLOEYE SLM Display SDK installation path from environment variable " + envvar_name + ". \n\nPlease relogin your Windows user account and try again. \nIf that does not help, please reinstall the SDK and then relogin your user account and try again. \nA simple restart of the computer might fix the problem, too."
assert os.path.isdir(importpath), asserterrmsg

sys.path.append(importpath)
#print(sys.path)
