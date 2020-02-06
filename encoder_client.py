import sys
import argparse
import requests
import grpc
import json
import numpy as np

from typing import Iterable, List

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class UniversalEncoder(object):
    # Length of returned vectors
    FEATURE_SIZE = 50
    BATCH_SIZE = 32

    def __init__(self, server):
        self.server_url = server

        if not server.startswith("http"):
            self.channel = grpc.insecure_channel(self.server_url)
            self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    @staticmethod
    def _create_rpc_callback():
        """Creates RPC callback function.
        Returns:
          The callback function.
        """
        def _callback(result_future):
            """Callback function.
            Calculates the statistics for the prediction result.
            Args:
              result_future: Result future of the RPC.
            """
            exception = result_future.exception()
            if exception:
                print("Error:", exception)
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
                print("Result =", result_future.result())
                response = np.array(
                    result_future.result().outputs["scores"].float_val)
                prediction = np.argmax(response)
        return _callback

    def encode(self, data: Iterable[str]):
        all_vectors: List[List[float]] = []
        for i in range(0, len(data), self.BATCH_SIZE):
            batch = data[i:i+self.BATCH_SIZE]
            print(batch)
            # -------------------------------------------------
            # gRPC mode:
            # request = predict_pb2.PredictRequest()
            # request.model_spec.name = "universal_encoder"
            # request.model_spec.signature_name = ""
            # request.input = batch
            # result_future = self.stub.Predict.future(request, 5.0)  # 5 seconds
            # result_future.add_done_callback(
            #     UniversalEncoder._create_rpc_callback())
            # -------------------------------------------------

            data = json.dumps({
                "inputs": batch
            })
            headers = {"content-type": "application/json"}
            print("************* server_url = " + self.server_url)
            response = requests.post(
                self.server_url,
                data=data,
                headers=headers
            )
            if not response.ok:
                print(response)
                raise Exception(response)
            print(response.json())
            # all_vectors += response.json()['embeddings']

        return np.array(all_vectors).astype(np.float32)


def main(args):
    server_url = args.server

    data = ["I am a hero", "test"]

    print(server_url)

    encoder = UniversalEncoder(server_url)
    result = encoder.encode(data)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server',
        type=str,
        help='TensorFlow serving host:port')

    args = parser.parse_args()
    main(args)

