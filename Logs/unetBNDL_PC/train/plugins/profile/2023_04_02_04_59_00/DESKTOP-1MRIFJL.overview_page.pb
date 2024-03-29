�	���<"c@���<"c@!���<"c@      ��!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'���<"c@���9d@1��
���`@I�&���-@r0*efffng�@)      �=2f
/Iterator::Root::Prefetch::FlatMap[0]::Generator��k	��W@!Y��d��X@)��k	��W@1Y��d��X@:Preprocessing2O
Iterator::Root::Prefetch�St$���?!x0�c��?)�St$���?1x0�c��?:Preprocessing2E
Iterator::RootX9��v��?!�)���?)��H�}�?1Q7:�Î?:Preprocessing2X
!Iterator::Root::Prefetch::FlatMap�b�=�W@!��:-��X@)�J�4q?1E�J��q?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�Iw�%:(@Q��B��U@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���9d@���9d@!���9d@      ��!       "	��
���`@��
���`@!��
���`@*      ��!       2      ��!       :	�&���-@�&���-@!�&���-@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�Iw�%:(@y��B��U@�"d
8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter̓�<��?!̓�<��?0"d
8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��4a�?!p���8}�?0"-
IteratorGetNext/_2_Recv�V
oC�?!^Z90-��?"b
7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput֮1���?!���a��?0"3
model/conv2d_6/Conv2DConv2D�0hD��?!���e�?0"b
7gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropInputConv2DBackpropInput�A�NȢ?!LF]�?0"d
8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter���J3��?!c��	]�?0"3
model/conv2d_8/Conv2DConv2D��dr2�?!VQ��0P�?0"3
model/conv2d_5/Conv2DConv2D�OL��?!�A�q�?0"b
7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputConv2DBackpropInput#11�?!�rӁ,��?0Q      Y@Y��t��@a�2Y� !X@"�

device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 