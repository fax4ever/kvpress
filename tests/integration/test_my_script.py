# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from playground import my_script


def test_main_runs_end_to_end(capsys):
    with capsys.disabled():
        my_script.main()
