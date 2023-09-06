# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#from .ur16e_manipulator import UR16eManipualtion
from .rl_ur16e_manipulator import RL_UR16eManipulation
from .rl_ur16e_manipulator_nocam import RL_UR16eManipulation_Nocam
from .rl_ur16e_manipulator_sparse import RL_UR16eManipulation as RL_UR16eManipulation_Full
from .rl_ur16e_manipulator_sparse_Nocam import RL_UR16eManipulation as RL_UR16eManipulation_Full_Nocam
#from .rl_ur16e_manipulator_sparse_Retract import RL_UR16eManipulation as RL_UR16eManipulation_Full_Retract

# Mappings from strings to environments
isaacgym_task_map = {
    #"UR16eManipualtion": UR16eManipualtion,
    "RL_UR16eManipulation": RL_UR16eManipulation,
    "RL_UR16eManipulation_Nocam": RL_UR16eManipulation_Nocam,
    "RL_UR16eManipulation_Full": RL_UR16eManipulation_Full,
    "RL_UR16eManipulation_Full_Nocam": RL_UR16eManipulation_Full_Nocam,
    #"RL_UR16eManipulation_Full_Retract": RL_UR16eManipulation_Full_Retract,
}