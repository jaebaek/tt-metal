#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Debug shebang
#!/usr/bin/env -S python3 -m pdb

import os
import csv
from pathlib import Path
import json
import yaml
from datetime import datetime

import click
from loguru import logger

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import (
    PROFILER_LOGS_DIR,
    PROFILER_OPS_LOGS_DIR,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_HOST_SIDE_LOG,
    PROFILER_OUTPUT_DIR,
    TRACY_FILE_NAME,
    TRACY_OPS_TIMES_FILE_NAME,
    TRACY_OPS_DATA_FILE_NAME,
)

yaml.SafeDumper.ignore_aliases = lambda *args: True

OUT_NAME = "ops_perf_results"

OPS_CSV_HEADER = [
    "OP CODE",
    "OP TYPE",
    "GLOBAL CALL COUNT",
    "DEVICE ID",
    "ATTRIBUTES",
    "MATH FIDELITY",
    "CORE COUNT",
    "PARALLELIZATION STRATEGY",
    "HOST START TS",
    "HOST END TS",
    "HOST DURATION [ns]",
    "DEVICE FW START CYCLE",
    "DEVICE FW END CYCLE",
    "DEVICE FW DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "DEVICE ERISC KERNEL DURATION [ns]",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
    "INPUTS",
    "OUTPUTS",
    "COMPUTE KERNEL PATH",
    "COMPUTE KERNEL HASH",
    "DATA MOVEMENT KERNEL PATH",
    "DATA MOVEMENT KERNEL HASH",
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM REQ I BW",
    "PM REQ O BW",
]


def import_tracy_op_logs():
    logger.info(f"Importting ops logs")
    tracyOpTimesLog = os.path.join(PROFILER_LOGS_DIR, TRACY_OPS_TIMES_FILE_NAME)
    tracyOpDataLog = os.path.join(PROFILER_LOGS_DIR, TRACY_OPS_DATA_FILE_NAME)

    ops = {}
    signposts = {}
    signpostsCount = 0
    cached_ops = {}
    with open(tracyOpDataLog, "r", newline="") as csvFile:
        opDataDicts = csv.DictReader(csvFile, delimiter=";", quotechar="`")
        opsData = []
        for opDataDict in opDataDicts:
            opDataStr = opDataDict["MessageName"]
            opDataTime = opDataDict["total_ns"]
            if "TT_DNN" in opDataStr:
                tmpStrs = opDataStr.split(", uncached\n", 1)
                opData = {}
                if len(tmpStrs) > 1:
                    jsonStr = tmpStrs[-1]
                    opData = json.loads(jsonStr)
                    cached_ops[int(opData["op_hash"])] = opData.copy()
                    del cached_ops[int(opData["op_hash"])]["global_call_count"]
                else:
                    opDataList = opDataStr.split(":", 1)[-1].split(",")
                    opCode = opDataList[0].strip()
                    opHash = int(opDataList[1])
                    opID = int(opDataList[2])
                    opData = cached_ops[opHash].copy()
                    opData["global_call_count"] = opID
                opData["tracy_time"] = opDataTime
                opsData.append(opData)

            if "TT_SIGNPOST" in opDataStr:
                signpostsCount += 1
                signposts[f"sp_{signpostsCount}"] = {"data": opDataStr, "tracy_time": opDataTime}
    for opData in opsData:
        ops[opData["global_call_count"]] = opData

    with open(tracyOpTimesLog, "r") as csvFile:
        csvReader = csv.DictReader(csvFile)
        for op in csvReader:
            opID = int(op["zone_text"].split(":")[-1])
            assert opID in ops.keys(), f"Op time for op {opID} must present"
            ops[opID]["host_time"] = op

    return ops, signposts


# Generate a map of OP reference list per device.
def get_device_op_data(ops):
    logger.info(f"Getting device ops")
    deviceOps = {}
    for opID, opData in ops.items():
        if "device_id" in opData.keys():
            deviceID = opData["device_id"]
            if deviceID not in deviceOps.keys():
                deviceOps[deviceID] = [opData]
            else:
                deviceOps[deviceID].append(opData)

    def device_ops_compare(op):
        return int(op["global_call_count"])

    for deviceID in deviceOps:
        deviceOps[deviceID].sort(key=device_ops_compare)

    return deviceOps


# Append device data to device ops and return the list of mapped device op ref list
def append_device_data(ops, deviceLogFolder):
    deviceOps = get_device_op_data(ops)
    logger.info(f"Appending device data")
    deviceTimesLog = os.path.join(deviceLogFolder, PROFILER_DEVICE_SIDE_LOG)
    if os.path.isfile(deviceTimesLog):
        setup = device_post_proc_config.default_setup()
        setup.deviceInputLog = deviceTimesLog
        deviceData = import_log_run_stats(setup)
        freq = deviceData["deviceInfo"]["freq"]
        for device in deviceOps:
            assert device in deviceData["devices"].keys()
            deviceOpsTime = deviceData["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]
            assert len(deviceOps[device]) == len(
                deviceOpsTime
            ), f"Device data mismatch. Expected {len(deviceOps[device])} but recieved {len(deviceOpsTime)} ops on device {device}"
            for deviceOp, deviceOpTime in zip(deviceOps[device], deviceOpsTime):
                cores = set()
                for timeID, ts, statData, risc, core in deviceOpTime["timeseries"]:
                    if "zone_name" in timeID.keys() and "FW" in timeID["zone_name"]:
                        if core not in cores:
                            cores.add(core)
                deviceOp["core_usage"] = {"count": len(cores), "cores": [str(core) for core in cores]}
                deviceOp["device_time"] = {
                    analysis: data["series"] for analysis, data in deviceOpTime["analysis"].items()
                }
                for analysis, data in deviceOp["device_time"].items():
                    for sample in data:
                        sample["duration_ns"] = sample["duration_cycles"] * 1000 / freq
    return deviceOps


def generate_reports(ops, deviceOps, signposts, outputFolder, date, nameAppend):
    logger.info(f"OPs' perf analysis is finished! Generating reports ...")
    outFolder = PROFILER_OUTPUT_DIR
    if outputFolder:
        outFolder = outputFolder

    name = OUT_NAME
    outFolder = os.path.abspath(outFolder)

    if nameAppend:
        name += f"_{nameAppend}"
        outFolder = os.path.join(outFolder, nameAppend)

    if date:
        dateStr = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        name += f"_{dateStr}"
        outFolder = os.path.join(outFolder, dateStr)

    logger.info(f"Copying runtime artifacts")
    os.system(f"rm -rf {outFolder}; mkdir -p {outFolder}")
    if os.path.isfile(f"{PROFILER_LOGS_DIR / TRACY_FILE_NAME}"):
        os.system(f"cp {PROFILER_LOGS_DIR / TRACY_FILE_NAME} {outFolder}")
    if os.path.isfile(f"{PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG}"):
        os.system(f"cp {PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG} {outFolder}")

    # logger.info(f"Generating OPs yaml")
    # allOpsYAMLPath = os.path.join(outFolder, f"{name}_all_ops.yaml")
    # with open(allOpsYAMLPath, "w") as allOpsYAML:
    # yaml.safe_dump(ops, allOpsYAML, default_flow_style=False)
    # logger.info(f"OPs yaml generated at: {allOpsYAMLPath}")

    # logger.info(f"Generating Device OPs yaml")
    # deviceOpsYAMLPath = os.path.join(outFolder, f"{name}_devices_ops.yaml")
    # with open(deviceOpsYAMLPath, "w") as deviceOpsYAML:
    # yaml.safe_dump(deviceOps, deviceOpsYAML, default_flow_style=False)
    # logger.info(f"Device OPs yaml generated at: {deviceOpsYAMLPath}")

    logger.info(f"Generating OPs CSV")
    allOpsCSVPath = os.path.join(outFolder, f"{name}.csv")
    with open(allOpsCSVPath, "w") as allOpsCSV:
        rowDicts = []

        tensorCSVData = {
            "INPUT": {
                "maxCount": -1,
                "headers": [],
            },
            "OUTPUT": {
                "maxCount": -1,
                "headers": [],
            },
        }

        def io_tensor_to_csv(ioField, ioData):
            headers = []
            data = {}
            if ioField == "shape":
                for field in ["W", "Z", "Y", "X"]:
                    headers.append(field)
                    assert field in ioData.keys(), "Wrong io tensor shape data format"
                    data[field] = ioData[field]
            elif ioField == "dtype":
                headers = ["DATATYPE"]
                data["DATATYPE"] = ioData
            elif ioField == "layout":
                headers = ["LAYOUT"]
                data["LAYOUT"] = ioData
            elif ioField == "storage_type":
                headers = ["MEMORY"]
                if type(ioData) == str:
                    data["MEMORY"] = ioData
                else:
                    assert "device_id" in ioData.keys(), "Wrong io tensor memory data format"
                    deviceID = ioData["device_id"]
                    assert "memory_config" in ioData.keys(), "Wrong io tensor memory data format"
                    assert "buffer_type" in ioData["memory_config"].keys(), "Wrong io tensor memory data format"
                    bufferType = ioData["memory_config"]["buffer_type"].upper()
                    assert "memory_layout" in ioData["memory_config"].keys(), "Wrong io tensor memory data format"
                    memoryLayout = ioData["memory_config"]["memory_layout"].upper()
                    data["MEMORY"] = f"DEV_{deviceID}_{bufferType}_{memoryLayout}"

            return headers, data

        def add_io_data(tensors, ioType):
            ioFields = ["shape", "layout", "dtype", "storage_type"]
            for count, tensor in enumerate(tensors):
                for ioField in ioFields:
                    assert ioField in tensor.keys(), "Wrong io tensor fields"
                    ioData = tensor[ioField]
                    fields, data = io_tensor_to_csv(ioField, ioData)
                    for field in fields:
                        header = f"{ioType}_{count}_{field}".upper()
                        rowDict[header] = data[field]
                        if count > tensorCSVData[ioType]["maxCount"]:
                            tensorCSVData[ioType]["headers"].append(header)
                if count > tensorCSVData[ioType]["maxCount"]:
                    tensorCSVData[ioType]["maxCount"] = count

        def csv_header_format(header):
            return header.replace("_", " ").upper()

        def row_compare(row):
            ret = 0
            if type(row) is str and "sp" in row:
                ret = signposts[row]["tracy_time"]
            elif type(row) is int:
                ret = ops[row]["tracy_time"]
            ret = int(ret)
            return ret

        rowKeys = list(ops.keys()) + list(signposts.keys())
        rowKeys.sort(key=row_compare)
        for row in rowKeys:
            rowDict = {}
            if type(row) is str and "sp" in row:
                headerAndMessage = signposts[row]["data"].split(": ")[-1].split("\n")
                rowDict["OP CODE"] = headerAndMessage[0]
                rowDict["OP TYPE"] = "signpost"
                if len(headerAndMessage) > 1:
                    rowDict["ATTRIBUTES"] = headerAndMessage[1]
                rowDict["HOST START TS"] = int(signposts[row]["tracy_time"])
            elif type(row) is int:
                op = row
                opData = ops[op]
                for field, fieldData in opData.items():
                    headerField = csv_header_format(field)
                    if headerField in OPS_CSV_HEADER:
                        rowDict[headerField] = fieldData

                assert "host_time" in opData.keys(), "Corrupted op data"
                rowDict["HOST START TS"] = int(opData["host_time"]["ns_since_start"])
                rowDict["HOST END TS"] = int(opData["host_time"]["ns_since_start"]) + int(
                    opData["host_time"]["exec_time_ns"]
                )
                rowDict["HOST DURATION [ns]"] = int(opData["host_time"]["exec_time_ns"])

                if "kernel_info" in opData.keys():
                    rowDict["COMPUTE KERNEL PATH"] = []
                    rowDict["COMPUTE KERNEL HASH"] = []
                    rowDict["DATA MOVEMENT KERNEL PATH"] = []
                    rowDict["DATA MOVEMENT KERNEL HASH"] = []
                    for computeKernel in opData["kernel_info"]["compute_kernels"]:
                        rowDict["MATH FIDELITY"] = computeKernel["math_fidelity"]
                        rowDict["COMPUTE KERNEL PATH"].append(computeKernel["path"])
                        rowDict["COMPUTE KERNEL HASH"].append(computeKernel["name"])

                    for dmKernel in opData["kernel_info"]["datamovement_kernels"]:
                        rowDict["DATA MOVEMENT KERNEL PATH"].append(dmKernel["path"])
                        rowDict["DATA MOVEMENT KERNEL HASH"].append(dmKernel["name"])

                if "core_usage" in opData.keys():
                    rowDict["CORE COUNT"] = opData["core_usage"]["count"]

                if "device_time" in opData.keys():
                    for analysis, analysisData in opData["device_time"].items():
                        headerField = f"{csv_header_format(analysis)} [ns]"
                        assert len(analysisData) == 1, "Unexpected device data format"
                        rowDict[headerField] = f"{analysisData[0]['duration_ns']:.0f}"
                    rowDict["DEVICE FW START CYCLE"] = analysisData[0]["start_cycle"]
                    rowDict["DEVICE FW END CYCLE"] = analysisData[0]["end_cycle"]

                assert "input_tensors" in opData.keys(), "Ops must have input tensors"
                if "optional_input_tensors" in opData.keys():
                    add_io_data(opData["input_tensors"] + opData["optional_input_tensors"], "INPUT")
                else:
                    add_io_data(opData["input_tensors"], "INPUT")

                if "output_tensors" in opData.keys():
                    add_io_data(opData["output_tensors"], "OUTPUT")

                if "performance_model" in opData.keys():
                    rowDict["PM IDEAL [ns]"] = opData["performance_model"]["compute_ns"]
                    rowDict["PM COMPUTE [ns]"] = opData["performance_model"]["ideal_ns"]
                    rowDict["PM BANDWIDTH [ns]"] = opData["performance_model"]["bandwidth_ns"]
                    rowDict["PM REQ I BW"] = opData["performance_model"]["input_bws"]
                    rowDict["PM REQ O BW"] = opData["performance_model"]["output_bws"]

            rowDicts.append(rowDict)

        ioHeaderIndex = OPS_CSV_HEADER.index("INPUTS")
        allHeaders = (
            OPS_CSV_HEADER[:ioHeaderIndex]
            + tensorCSVData["INPUT"]["headers"]
            + tensorCSVData["OUTPUT"]["headers"]
            + OPS_CSV_HEADER[ioHeaderIndex + 2 :]
        )
        writer = csv.DictWriter(allOpsCSV, fieldnames=allHeaders)
        writer.writeheader()
        for rowDict in rowDicts:
            for field, fieldData in rowDict.items():
                rowDict[field] = str(fieldData).replace(",", ";")
            writer.writerow(rowDict)
    logger.info(f"OPs csv generated at: {allOpsCSVPath}")


def process_ops(output_folder, name_append, date):
    ops, signposts = import_tracy_op_logs()

    deviceOps = append_device_data(ops, PROFILER_LOGS_DIR)

    generate_reports(ops, deviceOps, signposts, output_folder, date, name_append)


@click.command()
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-n", "--name-append", type=str, help="Name to be appended to default csv name")
@click.option("--date", default=False, is_flag=True, help="Append date to output files")
def main(output_folder, name_append, date):
    process_ops(output_folder, name_append, date)


if __name__ == "__main__":
    main()
