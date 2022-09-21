import datetime
from operator import attrgetter

import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DatetimeTensorConverter:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.date_attrs = ("year", "month", "day", "hour", "minute", "second", "microsecond")
        encoding_multipliers = (12, 31, 24, 60, 60, 1000000)
        ecnoder = [1]
        p = 1
        for multiplier in reversed(encoding_multipliers):
            p *= multiplier
            ecnoder.append(p)
        ecnoder = tuple(reversed(ecnoder))
        self.torch_encoder = torch.tensor(ecnoder, dtype=torch.int64, device=self.device)

        self.div_decoder = self.torch_encoder
        torch_int64_max = 1000000000
        # what is the largest value int64 can hold? - no docs on this!!!
        self.remainder_decoder = torch.tensor((torch_int64_max, ) + encoding_multipliers, dtype=torch.int64) 

    def datetime_to_tensor(self, datetime):

        as_tuple = attrgetter(*self.date_attrs)(datetime)
        as_tensor = torch.tensor(as_tuple, dtype=torch.int64, device=self.device)
        return as_tensor

    def join_tensors(self, list_of_datetime_as_tensors):
        return torch.stack(list_of_datetime_as_tensors)

    def utc_tensor(self):
        now_datetime = datetime.datetime.utcnow()
        now_tensor = self.datetime_to_tensor(now_datetime)
        return now_tensor

    def tensor_to_datetime(self, tensor):
        # UTC assumed
        kwargs = {k: int(v) for k, v in zip(self.date_attrs, tensor)}
        kwargs["tzinfo"] = datetime.timezone.utc
        return datetime.datetime(**kwargs)

    def matrix_to_datetimes(self, tensor_2d):
        result = []
        for row in tensor_2d:
            result.append(self.tensor_to_datetime(row))
        return result

    def encode_tensor(self, datetime_as_tensor):
        """
            stacked vectors like 
            datetime_as_tensor = torch.stack([now_tensor, now_tensor])
        """
        return datetime_as_tensor @ self.torch_encoder

    def decode_tensor(self, datetime_encoding):
        if len(datetime_encoding.shape) > 0:
            dim = len(datetime_encoding.shape)
            datetime_encoding = datetime_encoding.unsqueeze(dim).repeat((1,)*dim + (len(self.div_decoder),))
        div_decoded = torch.div(datetime_encoding, self.div_decoder, rounding_mode="trunc")
        fully_decoded = torch.remainder(div_decoded, self.remainder_decoder) 
        return fully_decoded