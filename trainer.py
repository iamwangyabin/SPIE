import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
from utils.experiment_logger import ExperimentLogger


def _to_builtin(value):
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_average(values):
    return "{:.2f}".format(sum(values) / len(values))


def _average_float(values):
    return round(sum(values) / len(values), 2)


def _compute_forgetting(matrix, task_idx):
    if len(matrix) == 0 or task_idx <= 0:
        return None

    np_acctable = np.zeros([task_idx + 1, task_idx + 1])
    for idxx, line in enumerate(matrix):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    np_acctable = np_acctable.T
    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task_idx])[:task_idx])
    return np_acctable, float(forgetting)


def _build_accuracy_matrix(matrix, labels):
    if len(matrix) == 0 or len(labels) == 0:
        return None

    acc_matrix = np.zeros([len(matrix), len(labels)])
    for row_idx, row in enumerate(matrix):
        acc_matrix[row_idx, : len(row)] = np.array(row)
    return acc_matrix


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    experiment_logger = ExperimentLogger(args)
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    cnn_matrix_labels, nme_matrix_labels = [], []

    try:
        for task in range(data_manager.nb_tasks):
            all_params = count_parameters(model._network)
            trainable_params = count_parameters(model._network, True)
            logging.info("All params: {}".format(all_params))
            logging.info(
                "Trainable params: {}".format(trainable_params)
            )
            model.incremental_train(data_manager)
            task_logging = model.consume_task_logging()
            experiment_logger.log_train_history(task, task_logging["train_history"])
            experiment_logger.log_extra_history(task, task_logging["extra_history"])

            cnn_accy, nme_accy = model.eval_task()
            model.after_task()

            avg_cnn = _average_float(cnn_curve["top1"] + [cnn_accy["top1"]])
            avg_nme = None

            if nme_accy is not None:
                logging.info("CNN: {}".format(_to_builtin(cnn_accy["grouped"])))
                logging.info("NME: {}".format(_to_builtin(nme_accy["grouped"])))

                cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
                cnn_matrix.append(cnn_values)
                cnn_matrix_labels = cnn_keys

                nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
                nme_values = [nme_accy["grouped"][key] for key in nme_keys]
                nme_matrix.append(nme_values)
                nme_matrix_labels = nme_keys

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])

                nme_curve["top1"].append(nme_accy["top1"])
                nme_curve["top5"].append(nme_accy["top5"])

                logging.info("CNN top1 curve: {}".format(_to_builtin(cnn_curve["top1"])))
                logging.info("CNN top5 curve: {}".format(_to_builtin(cnn_curve["top5"])))
                logging.info("NME top1 curve: {}".format(_to_builtin(nme_curve["top1"])))
                logging.info("NME top5 curve: {}\n".format(_to_builtin(nme_curve["top5"])))

                avg_cnn = _average_float(cnn_curve["top1"])
                avg_nme = _average_float(nme_curve["top1"])
                print("Average Accuracy (CNN):", "{:.2f}".format(avg_cnn))
                print("Average Accuracy (NME):", "{:.2f}".format(avg_nme))

                logging.info("Average Accuracy (CNN): {:.2f}".format(avg_cnn))
                logging.info("Average Accuracy (NME): {:.2f}".format(avg_nme))
            else:
                logging.info("No NME accuracy.")
                logging.info("CNN: {}".format(_to_builtin(cnn_accy["grouped"])))

                cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
                cnn_matrix.append(cnn_values)
                cnn_matrix_labels = cnn_keys

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])

                logging.info("CNN top1 curve: {}".format(_to_builtin(cnn_curve["top1"])))
                logging.info("CNN top5 curve: {}\n".format(_to_builtin(cnn_curve["top5"])))

                avg_cnn = _average_float(cnn_curve["top1"])
                print("Average Accuracy (CNN):", "{:.2f}".format(avg_cnn))
                logging.info("Average Accuracy (CNN): {:.2f} \n".format(avg_cnn))

            experiment_logger.log_eval(
                cnn_accy=cnn_accy,
                nme_accy=nme_accy,
                step=task,
                avg_cnn=avg_cnn,
                avg_nme=avg_nme,
            )

        summary_metrics = {}
        if cnn_curve["top1"]:
            summary_metrics["summary/cnn/final_top1"] = cnn_curve["top1"][-1]
            summary_metrics["summary/cnn/final_avg_top1"] = _average_float(cnn_curve["top1"])
        if nme_curve["top1"]:
            summary_metrics["summary/nme/final_top1"] = nme_curve["top1"][-1]
            summary_metrics["summary/nme/final_avg_top1"] = _average_float(nme_curve["top1"])

        cnn_accuracy_matrix = _build_accuracy_matrix(cnn_matrix, cnn_matrix_labels)
        if cnn_accuracy_matrix is not None:
            experiment_logger.log_accuracy_matrix("cnn", cnn_accuracy_matrix, cnn_matrix_labels)

        nme_accuracy_matrix = _build_accuracy_matrix(nme_matrix, nme_matrix_labels)
        if nme_accuracy_matrix is not None:
            experiment_logger.log_accuracy_matrix("nme", nme_accuracy_matrix, nme_matrix_labels)

        if 'print_forget' in args.keys() and args['print_forget'] is True:
            cnn_forgetting_info = _compute_forgetting(cnn_matrix, task)
            if cnn_forgetting_info is not None:
                cnn_acctable, forgetting = cnn_forgetting_info
                print('Accuracy Matrix (CNN):')
                print(cnn_acctable)
                logging.info('Forgetting (CNN): {}'.format(forgetting))
                summary_metrics["summary/cnn/forgetting"] = forgetting

            nme_forgetting_info = _compute_forgetting(nme_matrix, task)
            if nme_forgetting_info is not None:
                nme_acctable, forgetting = nme_forgetting_info
                print('Accuracy Matrix (NME):')
                print(nme_acctable)
                logging.info('Forgetting (NME): {}'.format(forgetting))
                summary_metrics["summary/nme/forgetting"] = forgetting

        experiment_logger.log_summary(summary_metrics)
    finally:
        experiment_logger.finish()


def _set_device(args):
    if torch.cuda.is_available():
        args["device"] = [
            torch.device("cuda:{}".format(i)) for i in range(torch.cuda.device_count())
        ]
    else:
        args["device"] = [torch.device("cpu")]


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
