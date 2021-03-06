?	:?m?=?q@:?m?=?q@!:?m?=?q@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-:?m?=?q@?ӹ???@1O<g?p@A?ݓ??Z??I???	?"@*	???(\?b@2U
Iterator::Model::ParallelMapV2E?a????!?1??Z5@)E?a????1?1??Z5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR?y9쾣?!?؆?e?9@)Χ?UJ??1z??H?44@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_?R#?3??!|s~&*b;@)a?4???1?????S3@:Preprocessing2F
Iterator::Model]¡?xx??!H;?LpRD@)/j?? ߝ?1?D??J3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice`???f???!??U? @)`???f???1??U? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor'ݖ?g??!?Q?7/@)'ݖ?g??1?Q?7/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS?[????!??Y???M@)???֪}?1c?Y?V(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???E???!n@?ܣ=@)X歺?d?1!?|a????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI@?7??@Q̃<?5?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ӹ???@?ӹ???@!?ӹ???@      ??!       "	O<g?p@O<g?p@!O<g?p@*      ??!       2	?ݓ??Z???ݓ??Z??!?ݓ??Z??:	???	?"@???	?"@!???	?"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@?7??@ỹ<?5?W@?"M
&Adam/Adam/update_212/ResourceApplyAdamResourceApplyAdam_?=$z??!_?=$z??"/
model/fc_1/MatMulMatMuln ?a????!z????8??0"=
gradient_tape/model/fc_1/MatMulMatMul?????3??!-O?x???0"=
!gradient_tape/model/fc_1/MatMul_1MatMul?~j???!?f?c???"o
Cgradient_tape/model/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter	??0^??!b??PU???0"o
Cgradient_tape/model/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?{D?"C??!A?-enE??0"o
Cgradient_tape/model/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter6d??8??!?ܠ4???0"f
:gradient_tape/model/conv1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterw???Y)??!??%?~x??0"o
Cgradient_tape/model/conv5_block1_0_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter"??!?!E?????0"m
Bgradient_tape/model/conv5_block1_0_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput`@-??|?!?r?ie??0Q      Y@Y>???>??a|???X@q'?21TH"@y?[?\??^?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 