# custom handler file for kaggle house price prediction

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import json
import torch
import pandas as pd
import os
from ts.torch_handler.base_handler import BaseHandler
import logging


class HousePriceHandler(BaseHandler):
    def __init__(self):
        super().__init__()

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # x1y path: F:\Projects\data\torchserve_data\kaggle_house_price
        # Take the input data and make it inference ready
        # filename = request.get("filename")
        #
        # if filename is None:
        #     filename = request.get("body")
        # print(f'filename: {filename}')
        # cache_dir = 'F:/Projects/data/torchserve_data/kaggle_house_price'
        cache_dir = '/home/torchserve_data/kaggle_house_price'
        test_data = pd.read_csv(os.path.join(cache_dir, 'kaggle_house_pred_test.csv'))
        train_data = pd.read_csv(os.path.join(cache_dir, 'kaggle_house_pred_train.csv'))
        print(os.path.join(cache_dir, 'kaggle_house_pred_train.csv'))
        print(os.path.join(cache_dir, 'kaggle_house_pred_train.csv'))
        # TODO：此处切掉了不带任何预测信息的ID，以及我们要预测的值那一列
        all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

        test_features = self.data_process(all_features, train_data)
        # print(test_features)
        return test_features

    def data_process(self, all_features, train_data):
        # 1 处理数值型的缺失值
        # 1.0. 获取数字的列的index
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        # print(numeric_features)

        # 1.1. 归一化
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        # 1.2. 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
        all_features[numeric_features] = all_features[numeric_features].fillna(0)

        # 2 处理离散值（对象型）的缺失值，热独编码！！！
        # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指⽰符特征
        all_features = pd.get_dummies(all_features, dummy_na=True)
        # print(all_features.shape)

        # 3 整理数据
        # 3.0 获取训练集的行数
        n_train = train_data.shape[0]
        # print(n_train)
        # train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
        test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
        # print("train feature size: " + str(train_features.shape))
        print("test feature size: " + str(test_features.shape))

        # # 3.1 拿到最后一列作为标注
        # train_labels = torch.tensor(
        #     train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
        #
        # print("train label size: " + str(train_labels.shape))

        return test_features

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        responses = []
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        inference_output = inference_output.detach().numpy()
        # conv to json
        inference_output_list = {'pred_result': inference_output.tolist()}
        inference_output_json = json.dumps(inference_output_list)
        print(inference_output_json)
        postprocess_output = inference_output_json
        responses.append(postprocess_output)
        # test
        return responses

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
