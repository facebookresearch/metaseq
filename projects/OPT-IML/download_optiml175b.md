# Accessing OPT-IML 175B

After receiving an email with a presigned URL to access the model weights, follow the below set of instructions to get started with hosting the model.

## Download all parts

We used 8x tensor parallelism to train OPT-IML 175B. We further converted the resulting model into 16x tensor parallelism so that it can be evaluated on GPUs with less memory. Thus, the model we provide has 16 parts. You will have received two URLs, for OPT-IML regular and OPT-IML max respectively. The URLs would look like:

```
https://<cloudfront_url>/regular/checkpoint_1_8000-model_part-*.pt?&<super_long_query_string>
https://<cloudfront_url>/max/checkpoint_1_12000-model_part-*.pt?&<super_long_query_string>
```

To download all 16 parts, you need to replace the * with 0-15. This can be achieved with this bash command:
run:
```
for i in `seq 0 15` do; wget https://<cloudfront_url>/regular/checkpoint_1_8000-model_part-${i}.pt?&<super_long_query_string>; done
for i in `seq 0 15` do; wget https://<cloudfront_url>/max/checkpoint_1_12000-model_part-${i}.pt?&<super_long_query_string>; done
```

### md5sum check
```
a8809edad238abb920e1ca0d5b12cd04  checkpoint_1_8000-model_part-0.pt
7f0a4b4ec8f1a1f37c8be3d6f93e44d0  checkpoint_1_8000-model_part-1.pt
0e90c1a503ba3967cba334b6de20baa7  checkpoint_1_8000-model_part-10.pt
43c612e5a3259d28b1376f5c066610bd  checkpoint_1_8000-model_part-11.pt
87edcc793f3c3b2bb4a6050e30e3dba9  checkpoint_1_8000-model_part-12.pt
1179225d9b54169ba4bcbc8075af8974  checkpoint_1_8000-model_part-13.pt
cf4014baa49771d3963acaff23193ab4  checkpoint_1_8000-model_part-14.pt
2833ee83c2e36574a5709c7f23af2262  checkpoint_1_8000-model_part-15.pt
2a735db09730c9da8397371656607e0d  checkpoint_1_8000-model_part-2.pt
60a80be167efc83eb63c4a416f766ac9  checkpoint_1_8000-model_part-3.pt
78e0ca192545a56b25d04fa009075257  checkpoint_1_8000-model_part-4.pt
eb3d89b5dfe3cc27832627975dc108f3  checkpoint_1_8000-model_part-5.pt
81899420c1541bd9ca034321145e21a0  checkpoint_1_8000-model_part-6.pt
6b8e8a440f4cc5db312f84ec3190195d  checkpoint_1_8000-model_part-7.pt
4f222a45f62906269d7918574637abca  checkpoint_1_8000-model_part-8.pt
d6061fa25bf32af09db3efa364629f72  checkpoint_1_8000-model_part-9.pt

4ffe8d95096228cc422a69639ebc1c76  checkpoint_1_12000-model_part-0.pt
036f3a04c668f53f4b423ab313641b6a  checkpoint_1_12000-model_part-1.pt
56ffe0da9aa2acad8f110496358a30b8  checkpoint_1_12000-model_part-10.pt
4916de70218e56e8789efe0b86d66bcb  checkpoint_1_12000-model_part-11.pt
0cd947260e8bf3b4dad64e886120bf51  checkpoint_1_12000-model_part-12.pt
a45a6d61bf48d40da0da5e7ac1c503fe  checkpoint_1_12000-model_part-13.pt
ebe01faf5fce59a5944952827947d652  checkpoint_1_12000-model_part-14.pt
80e8f31b90fe2db2607c955db637e9ee  checkpoint_1_12000-model_part-15.pt
d445a8df023705962259da22d4872ba9  checkpoint_1_12000-model_part-2.pt
49010d67e70d7ba7f9a23f8e57bf1f50  checkpoint_1_12000-model_part-3.pt
acfc906cb599477ccf6e693a284cd820  checkpoint_1_12000-model_part-4.pt
f999a3234e3e84aa0f9b5203ecebcb91  checkpoint_1_12000-model_part-5.pt
3e84250c799315679b77814995a4c869  checkpoint_1_12000-model_part-6.pt
c005e0113a1586439415ff4a4a45a270  checkpoint_1_12000-model_part-7.pt
7de79b7e446e5b6403d2efeff091a0e5  checkpoint_1_12000-model_part-8.pt
e63005b42c386c3f052b00ff7f76462a  checkpoint_1_12000-model_part-9.pt
```

## Run the API
The vocabulary files are the same as the ones used by [OPT](../OPT/assets/). Use the [same instructions provided with OPT](../../docs/api.md) to run an API.
