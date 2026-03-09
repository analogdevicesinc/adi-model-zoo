# Copyright © 2025-2026 Analog Devices, Inc. All Rights Reserved. This software is proprietary and confidential to Analog Devices, Inc. and its licensors.

import os
import sys
import csv
import glob
import logging
import numpy
import argparse
import ai_edge_litert.interpreter as litert
from sklearn import metrics

def setup_logger():
    logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
    logger = logging.getLogger(' ')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def predict(interpreter, data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = numpy.array(data, dtype=numpy.float32)
    output_data = numpy.empty_like(data)
    for i in range(input_data.shape[0]):
        interpreter.set_tensor(input_details[0]['index'], input_data[i:i+1, :])
        interpreter.invoke()
        output_data[i:i+1, :] = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def load_model(tflite_file, machine_type):
    if not os.path.exists(tflite_file):
        logger.error("{} model not found: {}".format(machine_type, tflite_file))
        sys.exit(-1)
    interpreter = litert.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    return interpreter

def evaluate_machine_id(interpreter, machine_type, id_path, model_path, root_path):
    id_str = os.path.split(id_path)[1][:-4]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    result_dir = os.path.join(root_path, "data", "result")
    os.makedirs(result_dir, exist_ok=True)

    anomaly_score_csv = os.path.join(
        result_dir,
        "{model_name}_anomaly_score_{machine_type}_{id_str}.csv".format(
            model_name=model_name,
            machine_type=machine_type,
            id_str=id_str
        )
    )

    anomaly_score_list = []

    print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
    print("MACHINE ID: " + id_str)

    data_p = numpy.load(os.path.join(root_path, "data", "input", machine_type, id_str + ".npy"))
    y_true = numpy.load(os.path.join(root_path, "data", "input_label", machine_type, id_str + ".npy"))
    y_pred = [0. for k in range(data_p.shape[0])]
    pred_data = None
    
    for jdx in range(data_p.shape[0]):
        try:
            jdx_str = str(jdx).zfill(4)
            data = data_p[jdx, :, :]
            pred_data = predict(interpreter, data)
            errors = numpy.mean(numpy.square(data - pred_data), axis=1)
            y_pred[jdx] = numpy.mean(errors)
            anomaly_score_list.append([jdx_str, y_pred[jdx]])
        except Exception as e:
            logger.error("file broken!!: {}".format(jdx_str))
            print(e)

    save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
    logger.info("anomaly score result -> {}".format(anomaly_score_csv))
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    logger.info("AUC : {}".format(auc))
    logger.info("pAUC : {}".format(p_auc))

    print("\n============ END OF TEST FOR A MACHINE ID ============")

    output_dir = os.path.join(root_path, "data", "output", machine_type)
    os.makedirs(output_dir, exist_ok=True)
    np_pred = numpy.array(pred_data)
    
    with open(os.path.join(output_dir, id_str + ".npy"), 'wb') as f:
        numpy.save(f, np_pred)

    return id_str, auc, p_auc

def run_evaluation(model_path, root_path, input_dir=None):
    csv_lines = []

    if input_dir:
        dirs = [os.path.join(root_path, input_dir)]
    else:
        dirs = glob.glob(os.path.join(root_path, "data", "input", "*"))

    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        machine_type = os.path.split(target_dir)[1]
        print("============== MODEL LOAD ==============")

        tflite_file = os.path.join(root_path, "data", "model", os.path.basename(model_path))
        interpreter = load_model(tflite_file, machine_type)

        csv_lines.append([machine_type])
        csv_lines.append(["id", "AUC", "pAUC"])
        performance = []

        machine_id_list = glob.glob(os.path.join(target_dir, "*"))
        for id_path in machine_id_list:
            id_str, auc, p_auc = evaluate_machine_id(interpreter, machine_type, id_path, model_path, root_path)
            csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
            performance.append([auc, p_auc])

        averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
        csv_lines.append(["Average"] + list(averaged_performance))
        csv_lines.append([])

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    result_path = os.path.join(
        root_path, "data", "result",
        "{model_name}_result.csv".format(model_name=model_name)
    )
    logger.info("AUC and pAUC results -> {}".format(result_path))
    save_csv(save_file_path=result_path, save_data=csv_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="model/model_fan_quant.tflite",
        help="Path to the model file to evaluate"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Specific input folder to evaluate (e.g. data/input/fan)"
    )
    args = parser.parse_args()

    root_path = os.path.dirname(__file__)
    os.makedirs(os.path.join(root_path, "data", "result"), exist_ok=True)

    run_evaluation(args.model, root_path, args.input)

if __name__ == "__main__":
    main()
