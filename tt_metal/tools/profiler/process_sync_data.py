#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
from io import StringIO
import contextlib
from loguru import logger

import pandas as pd
from sklearn.linear_model import TheilSenRegressor, LinearRegression
from sklearn.datasets import make_regression

from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_BIN_DIR,
    TRACY_FILE_NAME,
    TRACY_CSVEXPROT_TOOL,
)

# RUN_TYPE = "runs_2min_apart_syncs"
RUN_TYPE = "runs_first_last_10ms_sync"

lsCmd = subprocess.run(
    f"cd {TT_METAL_HOME / RUN_TYPE}; ls",
    shell=True,
    check=True,
    capture_output=True,
)

runsList = lsCmd.stdout.decode().split("\n")[:-1]

runData = {}

pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_rows = 250

finalReport = pd.DataFrame()
for run in runsList:
    tracyCsv = ""
    logsFolder = TT_METAL_HOME / RUN_TYPE / run / ".logs"
    tracyFile = logsFolder / TRACY_FILE_NAME
    deviceCsvFile = logsFolder / PROFILER_DEVICE_SIDE_LOG
    syncInfoFile = logsFolder / "sync_device_info.csv"

    filesExist = True
    for file in [tracyFile, deviceCsvFile, syncInfoFile]:
        filesExist &= os.path.isfile(file)

    testPoints = 2
    if filesExist:
        tracyDF = pd.DataFrame()
        for zoneName in ["CQ-CONSUMER-MAIN", "HWCommandQueue_finish"]:
            csvExportCmd = subprocess.run(
                f"{PROFILER_BIN_DIR / TRACY_CSVEXPROT_TOOL} -u -f {zoneName} {tracyFile}",
                shell=True,
                check=True,
                capture_output=True,
            )

            tracyCsv = csvExportCmd.stdout.decode()
            if not tracyDF.empty:
                tracyDF = pd.concat([tracyDF, pd.read_csv(StringIO(tracyCsv))])
            else:
                tracyDF = pd.read_csv(StringIO(tracyCsv))

        tracyDF = tracyDF.reset_index()
        tracyDF = tracyDF.drop(list(range(0, testPoints * 2, 2)))  # remove CONSUMER_MAIN that is not relevant
        tracyDF = tracyDF.reset_index()
        tracyDF = tracyDF.drop(columns=["level_0", "index"])
        tracyDF["ns_end_time"] = tracyDF["ns_since_start"] + tracyDF["exec_time_ns"]

        syncinfoDF = pd.read_csv(syncInfoFile)

        tracyBaseTime_ns = syncinfoDF.iloc[0]["tracy_base_time"] * syncinfoDF.iloc[0]["tracy_ratio"]

        syncinfoDF["host_time_global [ns]"] = (syncinfoDF["host_tracy"] + syncinfoDF["host_start"]) * syncinfoDF[
            "tracy_ratio"
        ]

        X = [[x] for x in list(syncinfoDF["host_time_global [ns]"])]
        y = list(syncinfoDF["device"])
        with contextlib.redirect_stderr(StringIO()) as f:
            # reg = TheilSenRegressor(random_state=0).fit(X, y)
            reg = LinearRegression().fit(X, y)

        frequency = reg.coef_[0]
        delayCycle = reg.intercept_

        print(frequency, delayCycle)

        deviceDF = pd.read_csv(deviceCsvFile, skiprows=[0])
        deviceDF = deviceDF.loc[(deviceDF[" zone name"] == "CQ-CONSUMER-MAIN") & (deviceDF[" zone phase"] == "end")]
        deviceDF = deviceDF.reset_index()
        deviceDF = deviceDF.drop(list(range(0, testPoints * 2, 2)))  # remove CONSUMER_MAIN that is not relevant
        deviceDF = deviceDF.reset_index()

        deviceDF = deviceDF[[" time[cycles since reset]"]]
        deviceDF["time[ns since reset]"] = deviceDF[" time[cycles since reset]"] / frequency
        deviceDF["time[cycles host synced]"] = deviceDF[" time[cycles since reset]"] - delayCycle
        deviceDF["time[ns host synced]"] = deviceDF["time[cycles host synced]"] / frequency
        # deviceDF["corrected device time [ns]"] = round(deviceDF["time[ns host synced]"] + hostStart - tracyBaseTime_ns)

        testDF = deviceDF[["time[ns host synced]"]].copy()
        testDF["tracy device time [ns]"] = (
            tracyDF.loc[(tracyDF["name"] == "CQ-CONSUMER-MAIN")]["ns_end_time"] + tracyBaseTime_ns
        )
        testDF["tracy host time [ns]"] = (
            tracyDF.loc[(tracyDF["name"] == "HWCommandQueue_finish")].reset_index()["ns_end_time"] + tracyBaseTime_ns
        )
        testDF["original host device diff [ns]"] = testDF["tracy host time [ns]"] - testDF["tracy device time [ns]"]
        testDF["new host device diff [ns]"] = testDF["tracy host time [ns]"] - testDF["time[ns host synced]"]
        finalReportTmp = pd.DataFrame()
        finalReportTmp["C++ diff 1"] = [testDF.iloc[0]["original host device diff [ns]"]]
        finalReportTmp["C++ diff 2"] = testDF.iloc[1]["original host device diff [ns]"]
        finalReportTmp["Python diff 1"] = testDF.iloc[0]["new host device diff [ns]"]
        finalReportTmp["Python diff 2"] = testDF.iloc[1]["new host device diff [ns]"]
        finalReportTmp["frequency init"] = syncinfoDF.iloc[5]["frequency"] * 1e9
        finalReportTmp["frequency init + dump"] = syncinfoDF.iloc[260]["frequency"] * 1e9
        finalReportTmp["delay init"] = syncinfoDF.iloc[5]["delay"]
        finalReportTmp["delay init + dump"] = syncinfoDF.iloc[260]["delay"]

        finalReport = pd.concat([finalReport, finalReportTmp])

        print(finalReportTmp)

finalReport.reindex()
finalReport.to_excel("final_report.xlsx")
