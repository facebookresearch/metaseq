from metaseq.distributed.nccl_watched_comms.heartbeat.datetime_converter import DatetimeTensorConverter

import unittest
import torch

class ConverterTest(unittest.TestCase):
    def test_create_join_encode_decode(self):

        converter = DatetimeTensorConverter()
        now_tensor = converter.utc_tensor()
        now_matrix = converter.join_tensors([now_tensor, now_tensor])
        encoded_single = converter.encode_tensor(now_tensor) 
        encoded_matrix = converter.encode_tensor(now_matrix) 
        decoded_single = converter.decode_tensor(encoded_single) 
        decoded_matrix = converter.decode_tensor(encoded_matrix)
        self.assertTrue(torch.equal(decoded_matrix, now_matrix))
        self.assertTrue(torch.equal(decoded_single, now_tensor))
        now_datetime = converter.tensor_to_datetime(now_tensor)
        now_list = converter.matrix_to_datetimes(decoded_matrix)
        self.assertCountEqual(now_list, [now_datetime, now_datetime])
        
        
