# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deployment drivers for the multi-pod AFD E2E runner.

A driver creates the workload and collects its results. It never decides launch
order, polls readiness, evaluates, or sequences teardown -- those belong to the
pods.
"""
